"""PDF text extraction and cleaning for audio narration."""

import re
from dataclasses import dataclass, field

import pymupdf
import pymupdf4llm


@dataclass
class TextSegment:
    text: str
    segment_type: str  # 'heading' or 'paragraph'
    heading_level: int = 0


def extract_pdf(pdf_path, start_page=None, end_page=None):
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    kwargs = {}
    if start_page is not None or end_page is not None:
        start = (start_page or 1) - 1
        end = (end_page or total_pages)
        kwargs["pages"] = list(range(start, end))

    return pymupdf4llm.to_markdown(pdf_path, **kwargs)


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


def clean_text(
    md,
    skip_references=True,
    skip_equations=False,
    skip_captions=False,
    keep_footnotes=False,
):
    # Image references
    md = re.sub(r"!\[.*?\]\(.*?\)", "", md)

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

    # Footnotes
    if not keep_footnotes:
        md = re.sub(r"\[\d+\]", "", md)
        md = re.sub(r"^\s*\[\d+\][.:].+$", "", md, flags=re.MULTILINE)

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
):
    md = extract_pdf(pdf_path, start_page, end_page)
    cleaned = clean_text(
        md, skip_references, skip_equations, skip_captions, keep_footnotes
    )
    segments = parse_segments(cleaned)
    return segments, cleaned
