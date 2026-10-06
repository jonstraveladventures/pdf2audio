"""podcast-script: write a two-host podcast script about a document, locally, with an LLM.

  podcast-script paper.pdf -o script.txt
  podcast-script notes.md -o script.txt --minutes 20

The source can be a PDF, Markdown or plain text. A local model (through Ollama) first
plans the episode as sections, then writes each section as a conversation: host A
explains, host B asks what a listener would ask. Every number and name in a section must
appear in the source; a section that fails is written again with the problems pointed
out, up to twice. Numbers and acronyms are then written out as they should be said, and
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

def read_source(path):
    path = Path(path).expanduser()
    if path.suffix.lower() == ".pdf":
        from pdf2audio.extractor import extract_and_clean

        _, text = extract_and_clean(str(path), skip_appendices=True)
    else:
        text = path.read_text(encoding="utf-8")
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
                if tok[0].isupper() and tok.lower() not in ALWAYS_OK and tok.lower() not in words \
                        and tok.lower().rstrip("s") not in words and tok.lower().removesuffix("'s") not in words:
                    missing.append(tok)
    return sorted(set(missing))


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
    text = re.sub(r"\s*±\s*", " plus or minus ", text)
    text = re.sub(r"\s*(?:=|≈)\s*", lambda m: " is about " if "≈" in m.group(0) else " equals ", text)
    text = re.sub(r"\s*[<≤]\s*(?=\d)", " under ", text)
    text = re.sub(r"\s*[>≥]\s*(?=\d)", " over ", text)
    text = text.replace("²", " squared").replace("³", " cubed")
    text = re.sub(r"(?<=[A-Za-z])\s*×\s*(?=[A-Za-z])", " by ", text)
    for letter, name in GREEK.items():
        text = text.replace(letter, f" {name} ")
    text = re.sub(r"\b([A-Za-z])(\d)", r"\1 \2", text)  # V4 -> V 4
    text = re.sub(r"(?<=\d)\s*[x×](?![A-Za-z])", " times", text)
    text = re.sub(r"\b(\d+)B\b", lambda m: number_words(m.group(1), False) + " billion", text)
    text = re.sub(r"\b(\d+)M\b", lambda m: number_words(m.group(1), False) + " million", text)
    text = re.sub(r"\b(\d+)T\b", lambda m: number_words(m.group(1), False) + " trillion", text)
    text = re.sub(r"\b(\d+)K\b", lambda m: number_words(m.group(1), False) + " thousand", text)
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


def write_section(model, source, section, i, n, previous, numbers, words):
    points = "\n".join(f"- {p}" for p in section["points"])
    if i == 0:
        position = ("This is the opening section: A welcomes the listener and says what the document is and who wrote "
                    "it, naming every author and institution the document lists.")
    elif i == n - 1:
        position = "This is the closing section: end with what the document concludes, then a brief goodbye."
    else:
        position = "This section continues the episode: no greeting, no goodbye."
    prev = ("The conversation so far ended like this; carry on from it without repeating it:\n" + previous) if previous else ""
    feedback, best = "", None
    for attempt in range(3):
        reply = chat(model, WRITE.format(words=section.get("words", 300), points=points, position=position,
                                         previous=prev, style=STYLE, feedback=feedback, source=source))
        turns = parse_dialogue(reply)
        missing = unsupported(turns, numbers, words) if turns else ["(no dialogue in the reply)"]
        result = dict(title=section["title"], attempt=attempt + 1, words=sum(len(t.split()) for _, t in turns),
                      unsupported=missing)
        log(f"section {i + 1}/{n} '{section['title']}': {result['words']} words"
            + (f"; not in the source: {', '.join(missing)}" if missing else "; numbers and names check"))
        if best is None or len(missing) < len(best[1]["unsupported"]):
            best = (turns, result)
        if not missing:
            break
        feedback = ("A previous draft of this section used these numbers or names, which the document does not "
                    "contain: " + ", ".join(missing) + ". Use only numbers and names the document states.\n")
    return best


def main():
    ap = argparse.ArgumentParser(prog="podcast-script", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="the document: .pdf, .md or .txt")
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
    lines, report, previous = [], [], ""
    for i, section in enumerate(sections):
        turns, result = write_section(a.model, source, section, i, len(sections), previous, numbers, words)
        report.append(result)
        previous = "\n".join(f"{h}: {t}" for h, t in turns[-4:])
        lines.append(f"# {section['title']}")
        lines += [f"{h}: {spoken(t, keep)}" for h, t in turns]
        lines.append("")
    out = Path(a.output).expanduser()
    out.write_text("\n".join(lines), encoding="utf-8")
    failed = [r["title"] for r in report if r["unsupported"]]
    out.with_name(out.name + ".check.json").write_text(json.dumps(
        dict(source=str(a.source), model=a.model, plan=sections, sections=report, failed=failed), indent=2))
    total = sum(r["words"] for r in report)
    log(f"wrote {out}: {total} words, about {total / WORDS_PER_MINUTE:.0f} minutes"
        + (f"; sections with unsupported numbers or names: {failed}" if failed else "; every section checks"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
