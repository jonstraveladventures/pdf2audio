"""Display equations: find them, mark them in the text, and explain them in words.

pymupdf4llm's layout model returns display equations as "formula" boxes and drops their
text. We put a marker at each box's position so the narration can say something there.
Some numbered equations are misclassified as text and come through as symbol soup; we
cut those out by their trailing equation number and mark them too.

Explanations come from a local vision model through Ollama. Each page with equations is
rendered with the equations outlined and labelled, and the model explains each label.
Nothing leaves the machine.
"""

import base64
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import pymupdf

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:35b-a3b"
MARKER = "[[EQ{}]]"
MARKER_RE = re.compile(r"\[\[EQ(\d+)\]\]")

# An equation number at the right end of a display equation: (2), (3.6), (A.4), (2.3a)
_TAG = r"\(((?:[A-Z]\.)?\d{1,3}(?:\.\d{1,3})*[a-z]?)\)"

PROMPT = """You are preparing an academic paper to be listened to as audio. A listener \
cannot see the equations, so each one will be replaced by a short spoken explanation.

The image is one page of the paper. {where}

For each equation, write one to three plain sentences that say what the equation actually \
states: which quantity equals, depends on or is bounded by which others, and how (for \
example proportional to, the sum of, the rate of change of, integrated over, evaluated at, \
with a minus sign). Name each quantity by what it means in the paper, using the paper's \
context: "the temperature", "the charge density", "the horizon radius". A bare letter such as \
"the function U" is fine only when the paper gives it no other name. Do not stop at the \
equation's role, such as "this defines the ansatz": say what it contains.

Do not read the equation aloud term by term, and do not spell out symbols or subscripts \
("U sub h comma one", "script Q", "d times G"): a listener cannot hold that. Explain what the \
equation means, mentioning a specific number or factor only when it matters. Say only what \
the equation states and what its quantities mean: do not add claims about its significance, \
its consequences or where it is used, even when the surrounding text makes them, because the \
listener will hear that text anyway. Your text will be \
read by a speech synthesiser, so use no LaTeX and no symbols. Do not begin with "Equation N" or \
"This equation": start directly with what it says. If you are not sure what a quantity means \
or how the terms combine, say less rather than guess.

Reply with only a JSON object mapping each label to its explanation, for example \
{{"E1": "...", "E2": "..."}}."""


@dataclass
class Equation:
    id: int
    page: int  # 0-based page number in the document
    numbers: list = field(default_factory=list)  # printed equation numbers, e.g. ["3.6"]
    bbox: tuple = None  # formula box on the page; None for equations recovered from text

    def lead_in(self):
        if not self.numbers:
            return "An equation, in words:"
        if len(self.numbers) == 1:
            return f"Equation {self.numbers[0]}, in words:"
        return f"Equations {', '.join(self.numbers[:-1])} and {self.numbers[-1]}, in words:"

    def placeholder(self):
        """What to say when there is no explanation: the number, if it has one."""
        if not self.numbers:
            return ""
        if len(self.numbers) == 1:
            return f"Equation {self.numbers[0]}."
        return f"Equations {', '.join(self.numbers[:-1])} and {self.numbers[-1]}."


def _lines(page, clip=None):
    """Words grouped by text line, each line sorted left to right as (x0, x1, y0, word)."""
    lines = {}
    for x0, y0, x1, y1, word, block, line, _ in page.get_text("words", clip=clip):
        lines.setdefault((block, line), []).append((x0, x1, y0, word))
    return [sorted(ws) for ws in lines.values()]


def _column_edges(page):
    """Right edges of the page's text columns. Justified prose lines all end at the same
    x, so the edges are the line ends shared by several full lines."""
    ends = {}
    for words in _lines(page):
        if len(words) >= 6:
            x = round(words[-1][1])
            ends[x] = ends.get(x, 0) + 1
    return [x for x, n in ends.items() if n >= 3]


def _at_edge(x1, edges, fallback):
    return any(abs(x1 - e) <= 4 for e in edges) if edges else x1 >= fallback


def _numbers_in_box(page, bbox, edges):
    """Equation numbers in a formula box: the rightmost word on a line, at the right edge
    of a text column. That excludes "(0)" in a superscript and "(2.11)" cited on an arrow
    inside the maths, and works when a box spans both columns of a two-column page."""
    rect = pymupdf.Rect(bbox)
    numbers = []
    for words in sorted(_lines(page, rect), key=lambda ws: ws[-1][2]):
        m = re.fullmatch(_TAG, words[-1][3])
        if m and _at_edge(words[-1][1], edges, rect.x1 - 20):
            numbers.append(m.group(1))
    return numbers


def _display_tags(page, edges):
    """Equation numbers printed where display equations put them: the rightmost word on
    its line at a column's right edge, either set well apart from the rest of the line or
    following maths rather than a word. A citation such as "in (3)" or "Eq. (3)" at the end
    of a justified line follows a word at normal spacing and is excluded."""
    tags = set()
    for words in _lines(page):
        m = re.fullmatch(_TAG, words[-1][3])
        if not m or not _at_edge(words[-1][1], edges, 0):
            continue
        if len(words) == 1 or words[-1][0] - words[-2][1] > 15 or not re.search(r"[A-Za-z]{2}", words[-2][3]):
            tags.add(m.group(1))
    return tags


def _plain(token):
    return re.fullmatch(r"[A-Za-z][a-z]{2,}[,.;:]?", token) is not None


def _equation_start(text, end):
    """Where the equation ending at `end` begins: after the last pair of plain words of
    three or more letters in the same paragraph. pymupdf4llm italicises maths ("_ϕ_"),
    so the equation's own symbols do not count as plain words."""
    floor = text.rfind("\n\n", 0, end) + 1
    tokens = list(re.finditer(r"\S+", text[floor:end]))
    for i in range(len(tokens) - 1, 0, -1):
        if _plain(tokens[i].group()) and _plain(tokens[i - 1].group()):
            return floor + tokens[i].end()
    return floor


def _equation_end(text, start):
    """Where the equation continues past its number (a tag set beside the middle of a
    multi-line equation): over tokens that hold no word of two or more letters, stopping
    at the first word or at the paragraph's end."""
    stop = text.find("\n\n", start)
    stop = len(text) if stop < 0 else stop
    end = start
    for tok in re.finditer(r"\S+", text[start:stop]):
        bare = re.sub(r"</?(?:sup|sub|u)>|[_*]", "", tok.group())
        if re.search(r"[A-Za-z]{2}", bare):
            break
        end = start + tok.end()
    return end


def _leaked_equations(text, tags):
    """Display equations that pymupdf4llm left in running text as symbol soup: each tag
    in `tags` preceded by maths. Returns [(start, end, number)]."""
    found = []
    for m in re.finditer(_TAG, text):
        if m.group(1) not in tags:
            continue
        start = _equation_start(text, m.start())
        if re.search(r"[=<>≤≥∝≈]|_[^_\s][^_]*_", text[start : m.start()]):
            found.append((start, _equation_end(text, m.end()), m.group(1)))
    return found


def mark_page(chunk, page, next_id, extra_edits=()):
    """Insert markers into one page chunk's text, along with any other (start, end,
    replacement) edits the caller needs made at the same positions. Returns
    (new_text, equations)."""
    text = chunk["text"]
    boxes = chunk.get("page_boxes", [])
    edits, equations, covered = list(extra_edits), [], set()
    edges = _column_edges(page)

    for box in boxes:
        if box["class"] == "formula":
            s, e = box["pos"]
            if re.fullmatch(r"[\d\s.–-]*", page.get_text("text", clip=pymupdf.Rect(box["bbox"]))):
                continue  # a page number the layout model took for a formula
            eq = Equation(next_id + len(equations), page.number, _numbers_in_box(page, box["bbox"], edges), tuple(box["bbox"]))
            equations.append(eq)
            covered.update(eq.numbers)
            edits.append((s, e, f"\n\n{MARKER.format(eq.id)}\n\n"))

    tags = _display_tags(page, edges)
    for box in boxes:
        if box["class"] in ("formula", "picture", "table"):
            continue
        s, e = box["pos"]
        for start, end, number in _leaked_equations(text[s:e], tags):
            if number in covered:
                # The same equation also has a formula box: drop the soup, keep one marker.
                edits.append((s + start, s + end, ""))
                continue
            eq = Equation(next_id + len(equations), page.number, [number])
            equations.append(eq)
            covered.add(number)
            edits.append((s + start, s + end, f"\n\n{MARKER.format(eq.id)}\n\n"))

    for s, e, new in sorted(edits, reverse=True):
        text = text[:s] + new + text[e:]
    return text, equations


def _labelled_page_png(doc, page_no, equations, dpi=200):
    """Render the page with each formula box outlined and labelled E<n> in the margin."""
    tmp = pymupdf.open()
    tmp.insert_pdf(doc, from_page=page_no, to_page=page_no)
    page = tmp[0]
    for n, eq in enumerate(equations, 1):
        if eq.bbox:
            r = pymupdf.Rect(eq.bbox) + (-3, -3, 3, 3)
            page.draw_rect(r, color=(0.85, 0, 0), width=1.2)
            page.insert_text((max(4, r.x0 - 24), r.y0 + 10), f"E{n}", fontsize=10, color=(0.85, 0, 0))
    png = page.get_pixmap(dpi=dpi).tobytes("png")
    tmp.close()
    return base64.b64encode(png).decode()


def _describe_labels(equations):
    outlined = [f"E{n}" for n, eq in enumerate(equations, 1) if eq.bbox]
    parts = []
    if outlined:
        parts.append(
            f"The equations to explain are outlined in red and labelled {', '.join(outlined)} in the "
            "left margin. An equation's printed number, if it has one, is at the right end of its box."
        )
    for n, eq in enumerate(equations, 1):
        if not eq.bbox:
            parts.append(f"E{n} is the equation printed with the number ({eq.numbers[0]}) on this page; it is not outlined.")
        elif len(eq.numbers) > 1:
            listed = " and ".join(f"({x})" for x in eq.numbers)
            parts.append(f"E{n} contains more than one equation, {listed}; explain each of them in its answer.")
    return " ".join(parts)


def _chat(model, prompt, image, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image]}],
        "stream": False,
        "think": True,
        "options": {"temperature": 0.6, "num_ctx": 16384, "num_predict": 8192},
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["message"]["content"]


def check_model(model):
    """Raise RuntimeError with a readable message if Ollama or the model is unavailable."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            names = [m["name"] for m in json.load(r)["models"]]
    except (urllib.error.URLError, OSError):
        raise RuntimeError("Ollama is not running. Start it with `ollama serve`.")
    if model not in names:
        raise RuntimeError(f"Ollama has no model '{model}'. Pull it with `ollama pull {model}`.")


def explain_page(doc, page_no, equations, model=DEFAULT_MODEL, timeout=600):
    """Ask the model to explain one page's equations. Returns {equation id: text}."""
    prompt = PROMPT.format(where=_describe_labels(equations))
    image = _labelled_page_png(doc, page_no, equations)
    for _attempt in range(2):
        try:
            reply = _chat(model, prompt, image, timeout)
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
        m = re.search(r"\{.*\}", reply, re.DOTALL)
        try:
            answers = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            answers = None
        if isinstance(answers, dict):
            out = {}
            for n, eq in enumerate(equations, 1):
                text = str(answers.get(f"E{n}", "")).strip()
                if text:
                    out[eq.id] = text
            return out
    return {}


def substitute(text, equations, explanations=None, skip=False):
    """Replace markers with spoken explanations, placeholders, or nothing."""
    by_id = {eq.id: eq for eq in equations}
    explanations = explanations or {}

    def repl(m):
        eq = by_id.get(int(m.group(1)))
        if eq is None or skip:
            return ""
        if eq.id in explanations:
            # The model sometimes writes "chi_Phi" or "r_h"; the synthesiser would read
            # the marks, and explanations do not pass through the text cleaner.
            spoken = " ".join(re.sub(r"[_^\\$]", " ", explanations[eq.id]).split())
            return f"{eq.lead_in()} {spoken}"
        return eq.placeholder()

    return MARKER_RE.sub(repl, text)
