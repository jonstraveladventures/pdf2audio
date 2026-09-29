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

The image is one page of the paper. {where}{glossary}

For each equation, write one to three plain sentences that say what the equation actually \
states: which quantity equals, depends on or is bounded by which others, and how (for \
example proportional to, the sum of, the rate of change of, integrated over, evaluated at, \
with a minus sign). Name each quantity by what it means in the paper, using the paper's \
context: "the temperature", "the charge density", "the horizon radius". A bare letter such as \
"the function U" is fine only when the paper gives it no other name. Do not stop at the \
equation's role, such as "this defines the ansatz": say what it contains.

Do not read the equation aloud term by term, and do not spell out symbols or subscripts \
("U sub h comma one", "script Q", "d times G"): a listener cannot hold that. When a power or \
factor applies to a group, make the grouping audible: "the square of one minus two epsilon", \
not "one minus two epsilon squared". Explain what the \
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
    symbols: set = field(default_factory=set)  # maths symbols the equation uses
    context: str = ""  # the paper's definitions of those symbols, for the model

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
        # A citation follows a word ("Condition (2.25) is", "in (A.5)"); a display tag
        # follows maths. Cutting at a citation would delete the prose before it.
        before = re.search(r"(\S+)\s*$", text[: m.start()])
        if before and re.fullmatch(r"\(?[A-Za-z][A-Za-z.]+[,;:]?", before.group(1)):
            continue
        start = _equation_start(text, m.start())
        if re.search(r"[=<>≤≥∝≈]|_[^_\s][^_]*_", text[start : m.start()]):
            found.append((start, _equation_end(text, m.end()), m.group(1)))
    return found


_GREEK = "αβγδεϵζηθϑικλμµνξπϖρϱσςτυφϕχψωΓΔΘΛΞΠΣΥΦΨΩ"
# A symbol as pymupdf4llm writes it: italic ("_Q_", "_rh_") or a Greek letter, with an
# optional superscript.
_SYM = rf"(?:_[^_\n]{{1,8}}_|[{_GREEK}])(?:<sup>[^<]{{0,6}}</sup>)?"
_MEANING_END = r"(?=\s*(?:,|\.|;|:|\(|\)| and | with | where | which | that |$))"
# "Φ is the anisotropisation density", "µ denotes the chemical potential"
_SYMBOL_IS = re.compile(
    rf"({_SYM})\s+(?:is|are|denotes|stands for|represents)\s+((?:the|an?)\s+[^,.;:()]{{3,60}}?){_MEANING_END}"
)
# "the charge density Q", "the anisotropy parameter a"
_NAMED_SYMBOL = re.compile(
    rf"\b(?:the|an?)\s+((?:[a-z][a-z-]*\s+){{0,4}}[a-z][a-z-]{{2,}})\s+({_SYM})"
    r"(?=\s*(?:,|\.|;|:|\)|∈|=|<|>|≤|≥| and | with | where | which | that | is | are |$))"
)


def _symbols_in_markdown(text):
    """Maths symbols in pymupdf4llm markdown: Greek letters anywhere, and short tokens
    inside italics ("_V_", "_rh_"), since maths is italicised and prose words are not."""
    found = set(re.findall(f"[{_GREEK}]", text))
    for span in re.findall(r"_([^_\n]+)_", text):
        found.update(re.findall(r"\b[A-Za-z][A-Za-z0-9]{0,2}\b", span))
    return found


def _symbols_in_box(page, bbox):
    """Maths symbols in a formula box's text layer: Greek letters and single letters."""
    text = page.get_text("text", clip=pymupdf.Rect(bbox))
    return set(re.findall(f"[{_GREEK}]", text)) | set(re.findall(r"\b[A-Za-z]\b", text))


def _unmark(markdown):
    return " ".join(re.sub(r"</?(?:sup|sub|u)>|[_*]", "", markdown).split())


# Words that show a "the ... X" phrase is not a name for X: prepositions ("the expansion
# in small a"), conjunctions and verbs ("the corrections cancel and b").
_NOT_A_NAME = re.compile(
    r"\b(?:in|of|at|on|for|along|with|to|from|by|over|under|between|after|before|per|across|"
    r"within|into|through|about|than|against|and|or|but|if|then|as|so|when|while|thus|hence|"
    r"is|are|was|were|be|been|cancel|cancels|gives|becomes)\b"
)


def definitions(text):
    """Symbol definitions stated in a page's markdown: [(symbol, meaning)]."""
    text = re.sub(r"\*\*\d{3,4}(?:\s+\d{3,4})*\*\*", " ", text)  # review-draft line numbers
    found = [(_unmark(m.group(1)), _unmark(m.group(2))) for m in _SYMBOL_IS.finditer(text)]
    for m in _NAMED_SYMBOL.finditer(text):
        name = m.group(1)
        # "the expansion in small a", "the extensive quantities S, V": not definitions
        if _NOT_A_NAME.search(name) or re.search(r"[^su]s$", name):
            continue
        found.append((_unmark(m.group(2)), _unmark(name)))
    return [(sym, meaning) for sym, meaning in found if sym and len(meaning) > 3]


def attach_glossaries(equations, page_definitions):
    """Give each page's equations the paper's definition of each of their Greek symbols:
    the nearest one stated on that page or before it, else on the next page (a "where"
    clause can run over).

    Greek only. Tested on two papers, every naming fix came from a Greek symbol (Φ, χ,
    µ), and the only errors came from Latin capitals reused between sections: "U is the
    energy density" from the thermodynamics section was applied to the metric function
    U(r), turning two correct explanations wrong."""
    by_page = {}
    for eq in equations:
        by_page.setdefault(eq.page, []).append(eq)
    for page, eqs in by_page.items():
        wanted = {sym for sym in set().union(*(eq.symbols for eq in eqs)) if sym in _GREEK}
        order = [p for p in sorted(page_definitions, reverse=True) if p <= page] + [page + 1]
        chosen = {}
        for p in order:
            for sym, meaning in page_definitions.get(p, []):
                if sym in wanted and sym not in chosen:
                    chosen[sym] = (p, meaning)
        context = "\n".join(f"- {sym} (page {p + 1}): {meaning}" for sym, (p, meaning) in sorted(chosen.items()))
        for eq in eqs:
            eq.context = context


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
            eq = Equation(next_id + len(equations), page.number, _numbers_in_box(page, box["bbox"], edges), tuple(box["bbox"]),
                          _symbols_in_box(page, box["bbox"]))
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
            eq = Equation(next_id + len(equations), page.number, [number],
                          symbols=_symbols_in_markdown(text[s + start : s + end]))
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
        # The model thinks before answering; checking a dense page can take 20,000 tokens
        # of thought, and a reply cut off at the limit has no answer in it.
        "options": {"temperature": 0.6, "num_ctx": 32768, "num_predict": 24576},
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


CHECK_PROMPT = """You are checking spoken explanations of equations before they are read \
to a listener who cannot see the page.

The image is one page of a paper. {where}{glossary}

Below is the explanation written for each labelled equation. Check each one against the \
equation in the image, term by term: every sign; which quantity is subtracted from, divided \
by or compared with which; what is squared, inverted, summed, integrated or evaluated where; \
and whether each quantity is named as the paper names it. Do not rewrite an explanation for \
style.

For each label, reply "OK" if its explanation is correct. If anything in it is wrong, reply \
with a corrected explanation that keeps to the same rules: one to three plain sentences for a \
speech synthesiser, no symbols or LaTeX, grouping made audible ("the square of one minus two \
epsilon"), only what the equation states, and not beginning with "Equation N" or "This \
equation".

Explanations:
{explanations}

Reply with only a JSON object mapping each label to "OK" or to its corrected explanation."""


def _glossary(equations):
    glossary = equations[0].context if equations else ""
    if not glossary:
        return ""
    return (
        "\n\nThe paper names some of the symbols in these equations as follows. Use these "
        "names, but papers reuse symbols between sections, so if the page itself gives a "
        "symbol another meaning, follow the page:\n" + glossary
    )


def _ask(model, prompt, image, timeout):
    """Send a page prompt and return the JSON object in the reply, or None. Tries twice."""
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
            return answers
    return None


def explain_page(doc, page_no, equations, model=DEFAULT_MODEL, timeout=900):
    """Ask the model to explain one page's equations. Returns {equation id: text}."""
    prompt = PROMPT.format(where=_describe_labels(equations), glossary=_glossary(equations))
    answers = _ask(model, prompt, _labelled_page_png(doc, page_no, equations), timeout) or {}
    out = {}
    for n, eq in enumerate(equations, 1):
        text = str(answers.get(f"E{n}", "")).strip()
        if text:
            out[eq.id] = text
    return out


def check_page(doc, page_no, equations, explanations, model=DEFAULT_MODEL, timeout=900):
    """Have the model check one page's explanations against the equations and correct
    any that are wrong. Returns ({equation id: text}, [ids it corrected], completed),
    where completed is False if the model gave no usable answer."""
    labelled = {f"E{n}": explanations[eq.id] for n, eq in enumerate(equations, 1) if eq.id in explanations}
    if not labelled:
        return dict(explanations), [], True
    prompt = CHECK_PROMPT.format(
        where=_describe_labels(equations),
        glossary=_glossary(equations),
        explanations=json.dumps(labelled, indent=1, ensure_ascii=False),
    )
    answers = _ask(model, prompt, _labelled_page_png(doc, page_no, equations), timeout)
    if answers is None:
        return dict(explanations), [], False
    checked, corrected = dict(explanations), []
    for n, eq in enumerate(equations, 1):
        verdict = str(answers.get(f"E{n}", "")).strip()
        if eq.id in explanations and verdict and verdict.strip(" .\"'").upper() != "OK":
            checked[eq.id] = verdict
            corrected.append(eq.id)
    return checked, corrected, True


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
