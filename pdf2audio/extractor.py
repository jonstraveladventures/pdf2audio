"""PDF text extraction and cleaning for audio narration."""

import re
from dataclasses import dataclass, field

import pymupdf
import pymupdf4llm

from pdf2audio import equations, fallback


@dataclass
class TextSegment:
    text: str
    segment_type: str  # 'heading' or 'paragraph'
    heading_level: int = 0


TABLE_OMITTED = "Table omitted."


def _blank_missed_tables(page):
    """Replace tables the layout model misses with the words "Table omitted.", in the
    in-memory document. find_tables also fires on boxed banners and diagrams, so a hit
    counts only with two or more rows and columns, mostly filled cells, and some numbers."""
    rects = []
    for table in page.find_tables().tables:
        cells = [str(c).strip() for row in table.extract() for c in row if c is not None]
        filled = [c for c in cells if c]
        numbers = re.findall(r"\d+(?:\.\d+)?", " ".join(filled))
        if table.row_count >= 2 and table.col_count >= 2 and len(filled) >= 0.5 * len(cells) and len(numbers) >= 3:
            rects.append(pymupdf.Rect(table.bbox))
    if not rects:
        return
    for r in rects:
        page.add_redact_annot(r)
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
    for r in rects:
        page.insert_text((r.x0, r.y0 + 10), TABLE_OMITTED, fontsize=9)


def extract_pdf(pdf_path, start_page=None, end_page=None):
    """Return the document as markdown, with a marker where each display equation
    stood and "Table omitted." for each table, and the list of equations marked."""
    doc = pymupdf.open(pdf_path)
    start = (start_page or 1) - 1
    end = end_page or len(doc)
    for page_no in range(start, end):
        _blank_missed_tables(doc[page_no])

    parts, found, page_definitions = [], [], {}
    for chunk in pymupdf4llm.to_markdown(doc, page_chunks=True, pages=list(range(start, end))):
        page = doc[chunk["metadata"]["page_number"] - 1]
        if fallback.layout_failed(chunk):
            chunk = fallback.rebuild(chunk, page)
        tables = [
            (*box["pos"], f"\n\n{TABLE_OMITTED}\n\n")
            for box in chunk.get("page_boxes", [])
            if box["class"] == "table"
        ]
        page_definitions[page.number] = equations.definitions(chunk["text"])
        text, page_eqs = equations.mark_page(chunk, page, next_id=len(found), extra_edits=tables)
        parts.append(text)
        found += page_eqs
    doc.close()
    equations.attach_glossaries(found, page_definitions)
    return "".join(parts), found


_REFERENCE_WORDS = r"(?:References|Bibliography|Works\s+Cited|Literature\s+Cited)"

# An appendix heading, as a markdown heading ("# A. Proofs", "## Appendix B") or, where
# the layout model has run it into the text, in capitals ("A STATE REPRESENTATIONS AND").
_APPENDIX = re.compile(
    r"^#{1,6}\s*(?:\*\*)?(?:Appendix\b|[A-H](?:\.\d+)*\.?\s)"
    r"|(?<![\w,])(?:Appendix\s+)?[A-H]\.?\s+[A-Z][A-Z-]{2,}(?:\s+[A-Z&][A-Z-]*)+\b",
    re.MULTILINE,
)


# A table or figure caption; manuscripts often put these after the references.
_CAPTION = re.compile(r"^(?:\*\*)?(?:Table|Figure|Fig\.)\s*\d+", re.MULTILINE)


def _strip_references(md):
    """Remove the references section, keeping any appendices, tables and figure
    captions that follow it."""
    m = re.search(rf"^(#{{1,6}})\s*(?:\*\*)?{_REFERENCE_WORDS}(?:\*\*)?\s*$", md, re.MULTILINE | re.IGNORECASE)
    if m:
        level = len(m.group(1))
        ends = [
            e.start()
            for e in (
                re.compile(rf"^#{{1,{level}}}\s+\S", re.MULTILINE).search(md, m.end()),
                _CAPTION.search(md, m.end()),
            )
            if e
        ]
        return md[: m.start()] + ("\n\n" + md[min(ends) :] if ends else "")

    # A bold or plain line, or, where the layout model has merged the page into one
    # block, "REFERENCES" mid-paragraph followed by entries with years.
    m = re.search(rf"^(?:\*\*)?{_REFERENCE_WORDS}(?:\*\*)?\s*$", md, re.MULTILINE | re.IGNORECASE)
    if not m:
        for m in re.finditer(r"\b(?:REFERENCES|BIBLIOGRAPHY)\b", md):
            if len(re.findall(r"\b(?:19|20)\d{2}\b", md[m.end() : m.end() + 800])) >= 3:
                break
        else:
            return md
    ends = [e.start() for e in (_APPENDIX.search(md, m.end()), _CAPTION.search(md, m.end())) if e]
    return md[: m.start()] + ("\n\n" + md[min(ends) :] if ends else "")


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
    # The table placeholder can pick up heading markup from the text around it; it must
    # not end a section or be read as a heading.
    md = re.sub(rf"(?m)^#*\s*(?:\*\*|<u>)*{re.escape(TABLE_OMITTED)}(?:\*\*|</u>)*", TABLE_OMITTED, md)

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
    skip_appendices=False,
    check_equations=False,
):
    """Extract and clean a PDF for narration.

    Display equations are replaced by their number ("Equation 3.6."), or with
    explain_equations by a spoken explanation from a local model; check_equations adds
    a second pass in which the model checks each explanation against the equation and
    corrects it. on_equation_page, if given, is called as
    on_equation_page(pages_done, pages_total, explained, total, corrected, unchecked)
    while explanations are generated, unchecked counting pages the check could not finish.
    """
    md, eqs = extract_pdf(pdf_path, start_page, end_page)
    cleaned = clean_text(
        md, skip_references, skip_equations, skip_captions, keep_footnotes
    )
    if skip_appendices:
        # Before explaining equations, so none is spent on the appendices.
        appendix = _APPENDIX.search(cleaned)
        if appendix:
            cleaned = cleaned[: appendix.start()]

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
            corrected = unchecked = 0
            for done, (page_no, page_eqs) in enumerate(sorted(by_page.items()), 1):
                got = equations.explain_page(doc, page_no, page_eqs, llm_model)
                if check_equations and got:
                    got, fixed, completed = equations.check_page(doc, page_no, page_eqs, got, llm_model)
                    corrected += len(fixed)
                    unchecked += not completed
                explanations.update(got)
                if on_equation_page:
                    on_equation_page(done, len(by_page), len(explanations), len(live), corrected, unchecked)
            doc.close()

    cleaned = equations.substitute(cleaned, eqs, explanations, skip=skip_equations)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    segments = parse_segments(cleaned)
    return segments, cleaned
