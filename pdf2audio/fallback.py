"""Rebuild a page's text from the PDF's text layer when the layout model has failed.

On some pages (review drafts with margin line numbers, in particular) pymupdf4llm's
layout model returns nested boxes, one covering most of the page, and prints the same
lines from each: whole sentences come out twice. The text layer has every line once.
This module rebuilds such a page in the same markdown pymupdf4llm writes, so the rest
of the pipeline works unchanged: maths in italics ("_ϕ_"), section headings as "##",
and formula boxes positioned in the text for equation markers.

It also finds display equations the layout model missed, numbered or not, from their
fonts and position: a row set in maths fonts and indented from the text column.
"""

import re
from collections import Counter


TEXT_LIKE = {"text", "page-header", "page-footer", "list-item", "section-header", "caption", "footnote"}

_MATH_FONT = re.compile(
    r"^(?:CMMI|CMSY|CMEX|CMBSY|MSBM|MSAM|EUFM|EUSM|RSFS|MTMI|MTSY|MTEX|txmi|txsy|txex|pxmi|pxsy|pxex)"
    r"|Math|Symbol",
    re.IGNORECASE,
)
_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*|[A-H](?:\.\d+)*)\s+[A-Z]")


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inside(a, b):
    """Fraction of box a that lies inside box b."""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return _area((x0, y0, x1, y1)) / _area(a) if _area(a) else 0.0


def layout_failed(chunk):
    """True when a text box sits inside another text box or inside a picture box: the
    layout model then prints the same lines twice."""
    boxes = chunk.get("page_boxes", [])
    text = [b["bbox"] for b in boxes if b["class"] in TEXT_LIKE]
    containers = text + [b["bbox"] for b in boxes if b["class"] == "picture"]
    return any(a is not b and _inside(a, b) > 0.5 for a in text for b in containers)


def _center_in(bbox, boxes):
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes)


def rebuild(chunk, page):
    """Return a replacement chunk ({"text", "page_boxes", "metadata"}) built from the
    page's text layer, or the original chunk if the page has two text columns, which
    this simple top-to-bottom rebuild cannot order."""
    boxes = chunk.get("page_boxes", [])
    skip = [b["bbox"] for b in boxes if b["class"] in ("picture", "table")]
    formulas = [b["bbox"] for b in boxes if b["class"] == "formula"]
    tables = [b["bbox"] for b in boxes if b["class"] == "table"]
    text_boxes = [b["bbox"] for b in boxes if b["class"] in TEXT_LIKE]

    lines = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            kept = []
            for s in line["spans"]:
                s["text"] = "".join(c["c"] for c in s["chars"])
                # Drop figure labels and table cells, but keep a caption the layout
                # model has drawn inside a picture box.
                in_figure = _center_in(s["bbox"], skip) and not _center_in(s["bbox"], text_boxes)
                if s["text"].strip() and not in_figure and not _center_in(s["bbox"], formulas):
                    kept.append(s)
            if kept:
                lines.append(kept)
    spans = [s for line in lines for s in line]
    if not spans:
        return chunk

    # The body font is the commonest; Computer Modern roman is maths only when it is not.
    fonts = Counter()
    for s in spans:
        fonts[s["font"]] += len(s["text"])
    body = fonts.most_common(1)[0][0]
    is_math = lambda s: bool(_MATH_FONT.search(s["font"])) or (s["font"].startswith("CMR") and not body.startswith("CMR"))

    # Rows: text lines at the same height, which gathers a line with its margin number
    # and a display equation's pieces. A line joins a row only if its centre falls
    # inside the row, so a subscript hanging onto the next line does not merge them.
    def extent(line):
        return min(s["bbox"][1] for s in line), max(s["bbox"][3] for s in line)

    rows = []
    for line in sorted(lines, key=lambda ln: sum(extent(ln)) / 2):
        y0, y1 = extent(line)
        centre = (y0 + y1) / 2
        if rows and rows[-1]["y0"] - 1 <= centre <= rows[-1]["y1"] + 1:
            rows[-1]["spans"] += line
            rows[-1]["y0"], rows[-1]["y1"] = min(rows[-1]["y0"], y0), max(rows[-1]["y1"], y1)
        else:
            rows.append({"spans": list(line), "y0": y0, "y1": y1})

    # The text column: where rows of real prose start and end. Margin line numbers,
    # digits left of the column, are dropped.
    full = [r for r in rows if len(" ".join(s["text"] for s in r["spans"]).split()) >= 8]
    if not full:
        return chunk
    left = Counter(round(min(s["bbox"][0] for s in r["spans"] if not s["text"].strip().isdigit())) for r in full).most_common(1)[0][0]
    right = max(max(s["bbox"][2] for s in r["spans"]) for r in full)
    if right - left < 0.5 * page.rect.width:
        return chunk  # two columns
    for r in rows:
        r["spans"] = [s for s in r["spans"] if not (s["text"].strip().isdigit() and s["bbox"][2] < left - 2)]
        r["spans"].sort(key=lambda s: s["bbox"][0])
    rows = [r for r in rows if r["spans"]]

    def math_row(r):
        chars = sum(len(s["text"].strip()) for s in r["spans"])
        maths = sum(len(s["text"].strip()) for s in r["spans"] if is_math(s))
        x0 = min(s["bbox"][0] for s in r["spans"])
        return chars and maths >= 0.7 * chars and x0 >= left + 20

    def markdown(r):
        """The row's text with spaces restored from the gaps between characters (this
        PDF, like many, places words without space characters) and maths in italics."""
        runs, prev = [], None  # runs of [italic?, text]
        for s in r["spans"]:
            italic = bool(is_math(s) or s["flags"] & 2)
            for c in s["chars"]:
                ch = c["c"]
                if prev is not None and c["bbox"][0] - prev > 0.2 * s["size"] and ch != " ":
                    ch = " " + ch
                prev = c["bbox"][2]
                if runs and runs[-1][0] == italic:
                    runs[-1][1] += ch
                else:
                    runs.append([italic, ch])
        out = ""
        for italic, t in runs:
            core = t.strip()
            if italic and core:
                t = t.replace(core, f"_{core}_", 1)
            out += t
        return " ".join(out.split())

    # Items in reading order: prose rows, display-equation regions, layout formula and
    # table boxes. An item is (y, kind, payload).
    items = [(b[1], "formula", tuple(b)) for b in formulas]
    items += [(b[1], "table", tuple(b)) for b in tables]
    i = 0
    while i < len(rows):
        if math_row(rows[i]):
            j = i
            while j + 1 < len(rows) and math_row(rows[j + 1]) and rows[j + 1]["y0"] - rows[j]["y1"] < 14:
                j += 1
            group = [s for r in rows[i : j + 1] for s in r["spans"]]
            bbox = (
                min(s["bbox"][0] for s in group), min(s["bbox"][1] for s in group),
                max(s["bbox"][2] for s in group), max(s["bbox"][3] for s in group),
            )
            items.append((bbox[1], "formula", bbox))
            i = j + 1
        else:
            items.append((rows[i]["y0"], "row", rows[i]))
            i += 1
    items.sort(key=lambda it: it[0])

    # Assemble text. Prose rows join into paragraphs; a paragraph ends at a wide gap, a
    # heading, or an equation or table.
    text, page_boxes, para, para_box, last_y1 = "", [], [], None, None
    gaps = [b["y0"] - a["y1"] for a, b in zip(rows, rows[1:]) if 0 < b["y0"] - a["y1"] < 20]
    gap = sorted(gaps)[len(gaps) // 2] if gaps else 3.0

    def flush():
        nonlocal text, para, para_box
        if para:
            start = len(text)
            text += "\n".join(para) + "\n\n"
            page_boxes.append({"class": "text", "bbox": para_box, "pos": (start, len(text))})
        para, para_box = [], None

    for y, kind, payload in items:
        if kind == "row":
            line = markdown(payload)
            bbox = (min(s["bbox"][0] for s in payload["spans"]), payload["y0"],
                    max(s["bbox"][2] for s in payload["spans"]), payload["y1"])
            heading = len(line) < 100 and _HEADING.match(line.replace("_", "")) and (
                payload["spans"][0]["font"] != body or line.replace("_", "").isupper()
            )
            if heading or (last_y1 is not None and y - last_y1 > 1.8 * gap):
                flush()
            if heading:
                start = len(text)
                text += f"## {line.replace('_', '')}\n\n"
                page_boxes.append({"class": "section-header", "bbox": bbox, "pos": (start, len(text))})
            else:
                para.append(line)
                para_box = bbox if para_box is None else (
                    min(para_box[0], bbox[0]), min(para_box[1], bbox[1]),
                    max(para_box[2], bbox[2]), max(para_box[3], bbox[3]),
                )
            last_y1 = payload["y1"]
        else:
            flush()
            start = len(text)
            text += "\n\n"
            page_boxes.append({"class": kind, "bbox": payload, "pos": (start, len(text))})
            last_y1 = payload[3]
    flush()
    return {"text": text, "page_boxes": page_boxes, "metadata": chunk["metadata"]}
