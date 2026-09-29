"""PDF text extraction and cleaning for audio narration."""

import re
from dataclasses import dataclass, field

import pymupdf
import pymupdf4llm

from pdf2audio import equations


@dataclass
class TextSegment:
    text: str
    segment_type: str  # 'heading' or 'paragraph'
    heading_level: int = 0


def extract_pdf(pdf_path, start_page=None, end_page=None):
    """Return the document as markdown, with a marker where each display equation
    stood, and the list of equations marked."""
    doc = pymupdf.open(pdf_path)
    kwargs = {}
    if start_page is not None or end_page is not None:
        start = (start_page or 1) - 1
        end = (end_page or len(doc))
        kwargs["pages"] = list(range(start, end))

    parts, found = [], []
    for chunk in pymupdf4llm.to_markdown(doc, page_chunks=True, **kwargs):
        page = doc[chunk["metadata"]["page_number"] - 1]
        text, page_eqs = equations.mark_page(chunk, page, next_id=len(found))
        parts.append(text)
        found += page_eqs
    doc.close()
    return "".join(parts), found


def _strip_references(md):
    """Remove references/bibliography section and everything after it."""
    heading_pattern = re.compile(
        r"^(#{1,3})\s*(?:References|Bibliography|Works\s+Cited|Literature\s+Cited)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    m = heading_pattern.search(md)
    if m:
        level = len(m.group(1))
        rest = md[m.end() :]
        next_heading = re.search(rf"^#{{1,{level}}}\s+\S", rest, re.MULTILINE)
        return md[: m.start()] + (rest[next_heading.start() :] if next_heading else "")

    # Try bold-text or plain-text patterns (common in two-column papers)
    for pattern in [
        r"^\*{2}(?:References|Bibliography|Works\s+Cited)\*{2}\s*$",
        r"^(?:References|Bibliography|REFERENCES|BIBLIOGRAPHY)\s*$",
    ]:
        m = re.search(pattern, md, re.MULTILINE | re.IGNORECASE)
        if m:
            return md[: m.start()]

    return md


# Review-draft margin numbers (ICLR/NeurIPS style), which pymupdf4llm emits in
# bold: "**076**", "**079 080 081 082**", "**1002 1003**".
_NUM = r"\d{3,4}"
_LINE_NUMBER_RUN = re.compile(rf"\*\*({_NUM}(?:\s+{_NUM})*)\*\*")
_PICTURE_TEXT = re.compile(
    r"<!-- Start of picture text -->(.*?)<!-- End of picture text -->", re.DOTALL
)


def _has_line_numbers(md):
    """True when bold three- or four-digit numbers form a mostly ascending sequence."""
    nums = [int(n) for run in _LINE_NUMBER_RUN.findall(md) for n in run.split()]
    if len(nums) < 20:
        return False
    ascending = sum(b > a for a, b in zip(nums, nums[1:]))
    return ascending >= 0.8 * (len(nums) - 1)


def _rejoin(m, md):
    """Join a word split by line-break hyphenation. Keep the hyphen only if the
    paper spells the compound that way elsewhere ("per-run", not "coop-eration")."""
    left, right = m.group(1), m.group(2)
    hyphenated = f"{left}-{right}"
    if re.search(rf"\b{re.escape(hyphenated)}\b", md, re.IGNORECASE) and not re.search(
        rf"\b{re.escape(left + right)}\b", md, re.IGNORECASE
    ):
        return hyphenated
    return left + right


def _drop_figure_debris(md):
    """Drop axis ticks and labels that precede a caption in the same paragraph."""

    def clean(para):
        cap = re.search(r"\b(?:Figure|Fig\.|Table)\s*\d+[.:]", para)
        if cap and cap.start() > 0 and re.search(r"(?:-?\d+(?:\.\d+)?\s+){3,}", para[: cap.start()]):
            return para[cap.start() :]
        return para

    return "\n\n".join(clean(p) for p in re.split(r"\n\n+", md))


def _strip_line_numbers(md):
    """Remove margin line numbers and rejoin words they split across a line break."""

    md = re.sub(
        rf"(\w+)-\s*\*\*{_NUM}(?:\s+{_NUM})*\*\*\s*(\w+)", lambda m: _rejoin(m, md), md
    )
    md = _LINE_NUMBER_RUN.sub(" ", md)
    # Numbers merged into a neighbouring bold span: "**090 091 Contributions.**"
    md = re.sub(rf"\*\*(?:{_NUM}\s+)+(?=\S)", "**", md)
    md = re.sub(rf"(?:\s+{_NUM})+\*\*", "**", md)
    return re.sub(r"[ \t]{2,}", " ", md)


def _is_prose(line):
    words = line.split()
    alpha = sum(bool(re.fullmatch(r"[A-Za-z][A-Za-z'’,.;:()-]*", w)) for w in words)
    return len(words) >= 6 and alpha >= 0.7 * len(words)


def _clean_picture_text(md, line_numbers):
    """Drop axis ticks and labels from figure text, keeping any prose or caption
    pymupdf4llm swept into the figure block."""

    def keep_prose(m):
        lines = [l.strip() for l in m.group(1).split("<br>")]
        if line_numbers:
            lines = [re.sub(rf"^(?:{_NUM}\s+)+", "", l) for l in lines]
        # A line repeated within the block is a panel label, not prose.
        kept = [l for l in lines if _is_prose(l) and lines.count(l) == 1]
        return "\n\n" + " ".join(kept) + "\n\n"

    return _PICTURE_TEXT.sub(keep_prose, md)


def clean_text(
    md,
    skip_references=True,
    skip_equations=False,
    skip_captions=False,
    keep_footnotes=False,
):
    line_numbers = _has_line_numbers(md)
    if line_numbers:
        md = _strip_line_numbers(md)
    md = _clean_picture_text(md, line_numbers)
    md = _drop_figure_debris(md)

    # A hyphenated word split across a paragraph break: "Bayes- \n\n> adaptive"
    md = re.sub(r"(\w+)-\s*\n\s*\n?\s*>?\s*([a-z]\w*)", lambda m: _rejoin(m, md), md)
    # Strikethrough markers
    md = md.replace("~~", "")

    # pymupdf4llm italicises single symbols inside numbers: "0 _._ 15", "2 _×_ 2"
    md = re.sub(r"(\d)\s*_\._\s*(\d)", r"\1.\2", md)
    md = re.sub(r"\s*_([×±])_\s*", r" \1 ", md)

    # Image references
    md = re.sub(r"!\[.*?\]\(.*?\)", "", md)

    # Blockquote markers, which pymupdf4llm puts on some paragraphs
    md = re.sub(r"^[ \t]*>[ \t]?", "", md, flags=re.MULTILINE)

    # Page separator rules
    md = re.sub(r"^-{3,}$", "", md, flags=re.MULTILINE)
    md = re.sub(r"^\*{3,}$", "", md, flags=re.MULTILINE)

    # Equations
    if skip_equations:
        md = re.sub(r"\$\$.*?\$\$", "", md, flags=re.DOTALL)
        md = re.sub(r"(?<!\$)\$(?!\$)[^$]+\$(?!\$)", "", md)
    else:
        md = re.sub(r"\$\$.*?\$\$", " [equation] ", md, flags=re.DOTALL)
        md = re.sub(r"(?<!\$)\$(?!\$)[^$]+\$(?!\$)", " [equation] ", md)

    # Figure/table captions
    if skip_captions:
        md = re.sub(
            r"^(?:Figure|Fig\.|Table|Tab\.)\s*\d+[.:].+$",
            "",
            md,
            flags=re.MULTILINE | re.IGNORECASE,
        )

    # Footnotes and numeric citations: [3], [30, 31], [3–5]
    if not keep_footnotes:
        md = re.sub(r"\[[1-9]\d*(?:\s*[,–-]\s*\d+)*\]", "", md)  # never [0, 1], an interval
        md = re.sub(r"^\s*\[\d+\][.:].+$", "", md, flags=re.MULTILINE)
        # Footnote markers: a superscript number after punctuation, "direction.<sup>5</sup>"
        md = re.sub(r"(?<=[.,;:])<sup>\d+</sup>", "", md)

    # Superscript, subscript and underline tags are read aloud ("sup 2 slash sup"): keep
    # only their content. The script ell is spelt out by its code point ("letter 2113").
    md = re.sub(r"</?(?:sup|sub|u)>", "", md)
    md = md.replace("ℓ", "l")

    # References section
    if skip_references:
        md = _strip_references(md)

    # Standalone page numbers
    md = re.sub(r"^\s*\d{1,4}\s*$", "", md, flags=re.MULTILINE)

    # URLs
    md = re.sub(r"https?://\S+", "", md)

    # Markdown links -> keep text only
    md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)

    # Bold/italic markers
    md = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", md)
    md = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", md)

    # Markdown table formatting -> just keep cell text
    md = re.sub(r"^\|[-:| ]+\|$", "", md, flags=re.MULTILINE)  # separator rows
    md = re.sub(r"\|", " ", md)  # cell dividers

    # Collapse whitespace
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def parse_segments(text):
    segments = []
    for block in re.split(r"\n\n+", text):
        block = block.strip()
        if not block:
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", block, re.MULTILINE)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            if heading_text:
                segments.append(
                    TextSegment(
                        text=heading_text,
                        segment_type="heading",
                        heading_level=level,
                    )
                )
        else:
            clean = " ".join(block.split())
            if len(clean) > 2:
                segments.append(TextSegment(text=clean, segment_type="paragraph"))

    return segments


def extract_and_clean(
    pdf_path,
    start_page=None,
    end_page=None,
    skip_references=True,
    skip_equations=False,
    skip_captions=False,
    keep_footnotes=False,
    explain_equations=False,
    llm_model=equations.DEFAULT_MODEL,
    on_equation_page=None,
):
    """Extract and clean a PDF for narration.

    Display equations are replaced by their number ("Equation 3.6."), or with
    explain_equations by a spoken explanation from a local model. on_equation_page,
    if given, is called as on_equation_page(pages_done, pages_total, explained, total)
    while explanations are generated.
    """
    md, eqs = extract_pdf(pdf_path, start_page, end_page)
    cleaned = clean_text(
        md, skip_references, skip_equations, skip_captions, keep_footnotes
    )

    explanations = {}
    if explain_equations and not skip_equations:
        live = {int(i) for i in equations.MARKER_RE.findall(cleaned)}
        by_page = {}
        for eq in eqs:
            if eq.id in live:
                by_page.setdefault(eq.page, []).append(eq)
        if by_page:
            equations.check_model(llm_model)
            doc = pymupdf.open(pdf_path)
            for done, (page_no, page_eqs) in enumerate(sorted(by_page.items()), 1):
                explanations.update(equations.explain_page(doc, page_no, page_eqs, llm_model))
                if on_equation_page:
                    on_equation_page(done, len(by_page), len(explanations), len(live))
            doc.close()

    cleaned = equations.substitute(cleaned, eqs, explanations, skip=skip_equations)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    segments = parse_segments(cleaned)
    return segments, cleaned
