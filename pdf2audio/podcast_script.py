"""podcast-script: write a two-host podcast script about a document, locally, with an LLM.

  podcast-script paper.pdf -o script.txt
  podcast-script notes.md -o script.txt --minutes 20

The source can be LaTeX (a .tex file, or a folder holding one, as arXiv supplies), a PDF,
Markdown or plain text; prefer LaTeX where there is a choice, since PDF text loses
superscripts and scrambles author blocks. A local model (through Ollama) first
plans the episode as sections, then writes each section as a conversation: host A
explains, host B asks what a listener would ask. Every number and name in a section must
appear in the source; a section that fails is written again with the problems pointed
out, up to twice. A section that passes is then edited against the source and the
episode so far, to strike claims the source does not make, points already made, and
equations read out symbol by symbol; the edit is kept only if it still passes. Numbers and acronyms are then written out as they should be said, and
the script is ready for `podcast`. A report of each section's check goes beside the
script as <script>.check.json; the exit code is non-zero if any section still fails.

The check covers numbers and names only. Read the script against the source before
rendering it: a host can still say something the source does not.
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:35b-a3b"
WORDS_PER_MINUTE = 150

STYLE = """How the conversation should sound:
- Host A explains. Host B asks what a curious listener would ask, reacts ("Wow.", "Okay.",
  "Right."), sometimes finishes A's sentence, and checks jargon ("Sorry, what's a
  benchmark?"). Most turns are short; A runs to three or four sentences at most.
- Use the openers and fillers of speech ("So,", "Right,", "I mean,"), an occasional echo
  ("38." / "38 models."), and [laugh] once every few minutes at most,
  only where something is funny.
- The hosts are presenters, not the authors: they say "the authors" or "they", never "we"
  or "our" about the work, and B never asks A about "your" study.
- Every fact comes from the source, as the source states it. No comparison, ranking,
  judgement or consequence the source does not state ("matters as much as", "a hard
  limit", "the problem is solved"). B's questions must not assert anything either. If the
  source does not say, the hosts do not say.
- Each point once: no host repeats what the other has just said, except a short echo.
- Never read an equation out symbol by symbol. Say in words what it means and what it
  predicts ("quality rises with the log of model size and of topic frequency").
- Write every number in digits (38, 8,900, 60%, 0.74) so it can be checked; it is turned
  into words later. Name people, places and organisations exactly as the source does.
- Plain words. No bullet points, headings, stage directions, sound effects or markdown.

A short example of the style (its subject is unrelated; copy the manner, not the facts):

A: So, the question today is a simple one. Why is the sky blue?
B: Which sounds like a children's question.
A: It does, and the answer is lovely. Sunlight has every colour in it, and air scatters short wavelengths much more than long ones.
B: Short being blue.
A: Short being blue. So blue light gets bounced around the whole sky, and that's what you see when you look up.
B: Okay, then why are sunsets red?
A: Right, because at sunset the light comes through far more air, so most of the blue has been scattered away before it reaches you."""

PLAN = """You are planning a two-host podcast episode of about {words} words in total \
about the document below. Plan between 4 and 8 sections that take a listener from what \
the document is and why it matters, through its substance in a sensible order, to what it \
concludes. Give each section a short title, a word budget, and the specific points it must \
cover, each stated as the document states it, with numbers in digits. Leave out reference \
lists, acknowledgements and formatting.

Reply with only a JSON object:
{{"sections": [{{"title": "...", "words": 300, "points": ["...", "..."]}}]}}

The document:
<<<
{source}
>>>"""

WRITE = """You are writing one section of a two-host podcast episode about the document \
below. Write about {words} words of conversation covering these points:
{points}

{position}
{previous}
{style}
{feedback}
Reply with only the dialogue, one turn per line, each line starting "A: " or "B: ".

The document:
<<<
{source}
>>>"""


EDIT = """You are editing one section of a two-host podcast episode about the document \
below, before it is recorded. Correct the draft section in these ways and change nothing else:

1. Delete or correct any statement, by either host, that the document does not make: a \
comparison, ranking, judgement, consequence or prediction it does not state, or a \
generalisation of one case (one model becoming "models", one topic becoming "topics").
2. Delete any point already made earlier in this section or in the episode so far, except \
a short echo.
3. Where an equation or derivation is read out symbol by symbol, replace it with one or \
two sentences saying in words what it means.
4. The hosts are presenters: "the authors", never "we" or "you" for the people who did \
the work.

Leave every line that needs none of these changes exactly as it is. Keep the exchange \
whole: when you delete a question, delete or rephrase the answer that depends on it, and \
when you delete an answer, delete its question. The hosts alternate; never leave one host \
speaking twice in a row. You may add a short bridging line where a deletion leaves a gap, \
if it states nothing new. Keep numbers in \
digits. Reply with only the edited section, one turn per line, each line starting "A: " \
or "B: ".

The episode so far:
<<<
{so_far}
>>>

The draft section:
<<<
{draft}
>>>

The document:
<<<
{source}
>>>"""


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------- model ----------

def chat(model, prompt, timeout=1800):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": True,
        "options": {"temperature": 0.7, "num_ctx": 40960, "num_predict": 16384},
    }
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["message"]["content"]


def check_model(model):
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            names = [m["name"] for m in json.load(r)["models"]]
    except (urllib.error.URLError, OSError):
        sys.exit("podcast-script: Ollama is not running. Start it with `ollama serve`.")
    if model not in names:
        sys.exit(f"podcast-script: Ollama has no model '{model}'. Pull it with `ollama pull {model}`.")


# ---------- source ----------

def read_tex(path):
    """Plain text of a LaTeX paper through pandoc: abstract kept, appendices dropped."""
    import subprocess

    if path.is_dir():
        mains = [f for f in sorted(path.glob("*.tex")) if "\\documentclass" in f.read_text(errors="ignore")]
        if not mains:
            sys.exit(f"podcast-script: no .tex file with \\documentclass in {path}")
        path = mains[0]
    tex = path.read_text(encoding="utf-8", errors="ignore")
    tex = re.split(r"\\appendix\b", tex)[0] + ("\n\\end{document}\n" if "\\appendix" in tex else "")

    def plain(latex, standalone):
        cmd = ["pandoc", "-f", "latex", "-t", "plain", "--wrap=none"] + (["-s"] if standalone else [])
        r = subprocess.run(cmd, input=latex, capture_output=True, text=True, cwd=path.parent)
        if r.returncode:
            sys.exit(f"podcast-script: pandoc could not read {path.name}: {r.stderr[:300]}")
        return r.stdout

    text = plain(tex, True)
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if m:
        abstract = "Abstract\n\n" + plain(m.group(1), False).strip() + "\n\n"
        head, sep, rest = text.partition("\n\n\n")
        text = head + "\n\n" + abstract + rest if sep else abstract + text
    return text


def read_source(path):
    path = Path(path).expanduser()
    if path.is_dir() or path.suffix.lower() == ".tex":
        text = read_tex(path)
    elif path.suffix.lower() == ".pdf":
        from pdf2audio.extractor import extract_and_clean

        _, text = extract_and_clean(str(path), skip_appendices=True)
    else:
        text = path.read_text(encoding="utf-8")
    text = re.sub(r"[\u00a0\u2000-\u200a\u202f]", " ", text)  # pandoc writes thin spaces around "="
    # PDF text often splits a number at its thousands comma ("9 , 120").
    return re.sub(r"(?<=\d) , (?=\d{3}\b)", ",", text)


# ---------- checks ----------

def norm_number(tok):
    return tok.replace(",", "").rstrip(".")


def source_terms(source):
    numbers = {norm_number(t) for t in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", source)}
    words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'À-ɏ-]*", source)}
    return numbers, words


ALWAYS_OK = {"i", "i'm", "i've", "i'd", "i'll", "ok", "okay", "wow", "a", "b"}


def unsupported(turns, numbers, words):
    """Numbers and capitalised names in the dialogue that the source does not contain."""
    missing = []
    for _, text in turns:
        for tok in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text):
            n = norm_number(tok)
            if n not in numbers and n.lstrip("0") not in numbers:
                missing.append(tok)
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            toks = re.findall(r"[A-Za-z][A-Za-z'À-ɏ-]*", sentence)
            for tok in toks[1:]:
                if not tok[0].isupper() or tok.lower() in words:
                    continue
                parts = [p for p in tok.split("-") if len(p) > 1 and p[0].isupper()]
                for part in parts:
                    low = part.lower()
                    if low not in ALWAYS_OK and low not in words and low.rstrip("s") not in words \
                            and low.removesuffix("'s") not in words:
                        missing.append(tok)
                        break
    return sorted(set(missing))


def shape_problems(turns, budget, keep=()):
    """Faults a numbers-and-names check cannot see: loops, runaway length, symbols left."""
    problems = []
    seen = set()
    for _, text in turns:
        key = re.sub(r"\W+", " ", text.lower()).strip()
        if len(key.split()) >= 6 and key in seen:
            problems.append("repeated turn: " + text[:60])
        seen.add(key)
    run = 1
    for (h0, _), (h1, _) in zip(turns, turns[1:]):
        run = run + 1 if h1 == h0 else 1
        if run == 3:
            problems.append(f"host {h1} speaks three or more times in a row")
    words = sum(len(t.split()) for _, t in turns)
    if words > 1.6 * budget:
        problems.append(f"{words} words against a budget of {budget}")
    for _, text in turns:
        left = set(re.findall(r"[^\w\s.,;:!?'’\"“”()\[\]-]|_", spoken(text, keep).replace("[laugh]", "")))
        if left:
            problems.append("symbols that cannot be spoken: " + " ".join(sorted(left)))
    return sorted(set(problems))


def parse_dialogue(reply):
    turns = []
    for line in reply.splitlines():
        line = line.strip().replace("**", "")
        m = re.match(r"^(?:Host\s+)?([AB])\s*:\s*(.+)$", line)
        if m:
            turns.append((m.group(1), m.group(2).strip()))
    return turns


# ---------- speech form ----------

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def int_words(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + int_words(rest))
    for size, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= size:
            rest = n % size
            head = int_words(n // size) + " " + name
            if rest == 0:
                return head
            return head + (" and " if rest < 100 else " ") + int_words(rest)


def number_words(tok, year_ok=True):
    tok = tok.replace(",", "")
    if "." in tok:
        whole, frac = tok.split(".", 1)
        return int_words(int(whole or 0)) + " point " + " ".join(ONES[int(d)] for d in frac)
    n = int(tok)
    if year_ok and len(tok) == 4 and 1100 <= n <= 2099 and "," not in tok:
        hi, lo = divmod(n, 100)
        if lo == 0:
            return int_words(hi) + " hundred"
        if 2000 <= n <= 2009:
            return "two thousand and " + int_words(lo)
        return int_words(hi) + " " + ("oh " + ONES[lo] if lo < 10 else int_words(lo))
    return int_words(n)


GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "κ": "kappa", "λ": "lambda",
         "μ": "mu", "ρ": "rho", "σ": "sigma", "τ": "tau", "φ": "phi", "χ": "chi", "ω": "omega"}


def spoken(text, keep=()):
    """Numbers, symbols and acronyms written out as they should be said."""
    num = r"\d+(?:,\d{3})*(?:\.\d+)?"
    text = re.sub(rf"({num})\s*[–-]\s*({num})\s*%",
                  lambda m: f"{number_words(m.group(1), False)} to {number_words(m.group(2), False)} per cent", text)
    text = re.sub(rf"({num})\s*%", lambda m: number_words(m.group(1), False) + " per cent", text)
    text = re.sub(rf"({num})\s*[–]\s*({num})", lambda m: f"{m.group(1)} to {m.group(2)}", text)
    text = re.sub(r"~\s*(?=\d)", "about ", text)
    text = re.sub(r"(?:(?<=\s)|^)[−-]\s*(?=\d)", "minus ", text)
    text = re.sub(r"\s*±\s*", " plus or minus ", text)
    text = re.sub(r"\s*(?:=|≈)\s*", lambda m: " is about " if "≈" in m.group(0) else " equals ", text)
    text = re.sub(r"\s*[<≤]\s*(?=\d)", " under ", text)
    text = re.sub(r"\s*[>≥]\s*(?=\d)", " over ", text)
    text = text.replace("²", " squared").replace("³", " cubed")
    text = re.sub(r"(?<=[A-Za-z])\s*×\s*(?=[A-Za-z])", " by ", text)
    for letter, name in GREEK.items():
        text = text.replace(letter, f" {name} ")
    text = re.sub(r"\b([A-Za-z]+)(\d)", r"\1 \2", text)  # V4 -> V 4, Qwen3 -> Qwen 3
    text = re.sub(r"\s*[—–]\s*", ", ", text)  # dashes left after ranges are pauses
    text = re.sub(r"√\s*", "the square root of ", text)
    text = re.sub(r"\s*\+\s*", " plus ", text)
    text = re.sub(r"(?<=\d)\s*/\s*", " over ", text)
    text = re.sub(r"\s*/\s*", " or ", text)
    text = re.sub(r"(?<=\d)\s*[x×](?![A-Za-z])", " times", text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)B\b", lambda m: number_words(m.group(1), False) + " billion", text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)M\b", lambda m: number_words(m.group(1), False) + " million", text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)T\b", lambda m: number_words(m.group(1), False) + " trillion", text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)K\b", lambda m: number_words(m.group(1), False) + " thousand", text)
    text = re.sub(num, lambda m: number_words(m.group(0)), text)

    def letters(m):
        word = m.group(0)
        if word in keep or word in ("OK", "I"):
            return word
        plural = word.endswith("s") and word[:-1].isupper()
        body = word[:-1] if plural else word
        return " ".join(body) + ("s" if plural else "")
    text = re.sub(r"\b[A-Z]{2,6}s?\b", letters, text)
    return re.sub(r"\s{2,}", " ", text).strip()


# ---------- main ----------

def plan(model, source, words):
    for _ in range(3):
        reply = chat(model, PLAN.format(words=words, source=source))
        m = re.search(r"\{.*\}", reply, re.DOTALL)
        try:
            sections = json.loads(m.group(0))["sections"] if m else None
        except (json.JSONDecodeError, KeyError):
            sections = None
        if sections:
            return sections
    sys.exit("podcast-script: the model did not return a usable plan")


def edit_section(model, source, turns, so_far, budget, numbers, words, keep=()):
    """The draft edited against the source and the episode so far, or None if the edit fails a check."""
    draft = "\n".join(f"{h}: {t}" for h, t in turns)
    edited = parse_dialogue(chat(model, EDIT.format(so_far=so_far or "(this is the first section)", draft=draft,
                                                    source=source)))
    if not edited or unsupported(edited, numbers, words) or shape_problems(edited, budget, keep):
        return None
    return edited


def write_section(model, source, section, i, n, previous, numbers, words, keep=(), so_far=""):
    points = "\n".join(f"- {p}" for p in section["points"])
    if i == 0:
        position = ("This is the opening section: A welcomes the listener and says what the document is and who wrote "
                    "it, naming every author and institution the document lists.")
    elif i == n - 1:
        position = "This is the closing section: end with what the document concludes, then a brief goodbye."
    else:
        position = "This section continues the episode: no greeting, no goodbye."
    prev = ("The conversation so far ended like this; carry on from it without repeating it:\n" + previous) if previous else ""
    feedback, best, budget = "", None, int(section.get("words", 300))
    for attempt in range(3):
        reply = chat(model, WRITE.format(words=budget, points=points, position=position,
                                         previous=prev, style=STYLE, feedback=feedback, source=source))
        turns = parse_dialogue(reply)
        missing = unsupported(turns, numbers, words) if turns else ["(no dialogue in the reply)"]
        shape = shape_problems(turns, budget, keep)
        result = dict(title=section["title"], attempt=attempt + 1, words=sum(len(t.split()) for _, t in turns),
                      unsupported=missing, problems=shape)
        log(f"section {i + 1}/{n} '{section['title']}': {result['words']} words"
            + (f"; not in the source: {', '.join(missing)}" if missing else "")
            + (f"; {'; '.join(shape)}" if shape else "")
            + ("; checks pass" if not missing and not shape else ""))
        score = len(missing) + len(shape) + (0 if turns else 1000)
        if best is None or score < best[2]:
            best = (turns, result, score)
        if not missing and not shape:
            edited = edit_section(model, source, turns, so_far, budget, numbers, words, keep)
            result["edit"] = "kept" if edited else "rejected: the edited version failed a check"
            if edited:
                result.update(words_before_edit=result["words"], words=sum(len(t.split()) for _, t in edited),
                              lines_before_edit=len(turns), lines=len(edited))
                turns = edited
            log(f"section {i + 1}/{n}: edit {result['edit']}"
                + (f", {result['words_before_edit']} -> {result['words']} words" if edited else ""))
            best = (turns, result, 0)
            break
        feedback = "A previous draft of this section had these faults; avoid them:\n"
        if missing:
            feedback += ("- numbers or names the document does not contain: " + ", ".join(missing)
                         + ". Use only numbers and names the document states.\n")
        if shape:
            feedback += "".join(f"- {p}\n" for p in shape)
            feedback += "Write each point once, keep to the word budget, and write symbols as words.\n"
    return best[:2]


def main():
    ap = argparse.ArgumentParser(prog="podcast-script", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="the document: .tex (or a folder holding one), .pdf, .md or .txt")
    ap.add_argument("-o", "--output", required=True, help="script to write, for `podcast`")
    ap.add_argument("--minutes", type=float, default=15, help="target episode length (default 15)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default {DEFAULT_MODEL})")
    a = ap.parse_args()

    check_model(a.model)
    from pdf2audio.podcast import load_say

    source = read_source(a.source)
    numbers, words = source_terms(source)
    target = int(a.minutes * WORDS_PER_MINUTE)
    log(f"source: {len(source.split())} words; planning about {target} words")
    sections = plan(a.model, source, target)
    log(f"plan: {len(sections)} sections: " + "; ".join(s["title"] for s in sections))

    keep = set(load_say())
    lines, report, previous, so_far = [], [], "", []
    for i, section in enumerate(sections):
        turns, result = write_section(a.model, source, section, i, len(sections), previous, numbers, words, keep,
                                      "\n".join(so_far))
        report.append(result)
        so_far += [f"# {section['title']}"] + [f"{h}: {t}" for h, t in turns]
        previous = "\n".join(f"{h}: {t}" for h, t in turns[-4:])
        lines.append(f"# {section['title']}")
        lines += [f"{h}: {spoken(t, keep)}" for h, t in turns]
        lines.append("")
    out = Path(a.output).expanduser()
    out.write_text("\n".join(lines), encoding="utf-8")
    failed = [r["title"] for r in report if r["unsupported"] or r["problems"]]
    out.with_name(out.name + ".check.json").write_text(json.dumps(
        dict(source=str(a.source), model=a.model, plan=sections, sections=report, failed=failed), indent=2))
    total = sum(r["words"] for r in report)
    log(f"wrote {out}: {total} words, about {total / WORDS_PER_MINUTE:.0f} minutes"
        + (f"; sections that still fail their checks: {failed}" if failed else "; every section passes its checks"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
