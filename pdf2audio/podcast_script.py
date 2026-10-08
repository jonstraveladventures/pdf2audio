"""podcast-script: write a two-host podcast script about a document, locally, with an LLM.

  podcast-script paper.pdf -o script.txt
  podcast-script notes.md -o script.txt --minutes 20

The source can be LaTeX (a .tex file, or a folder holding one, as arXiv supplies), a PDF,
Markdown or plain text; prefer LaTeX where there is a choice, since PDF text loses
superscripts and scrambles author blocks. A local model (through Ollama) first plans the
episode as sections and reads off the title and authors; the opening, which names them,
and the goodbye are written by this program, not the model. Each section is then written
as a conversation, asked for its planned length and no less than four fifths of it: host A
explains, and every line of A's that states anything carries the passage of the source it
rests on, which is not spoken; host B only asks and restates what A has said. A may also
explain a general term the source uses but does not explain (what a sigmoid is) in a
BACKGROUND line, at most two in a section, with no numbers or names.

Every line is checked. A line is faulty if:
  - a number or name is not in the source, or a symbol is left that cannot be spoken;
  - a passage A cites is not word for word in the source (spacing aside, and a bracketed
    aside of the paper's sentence may be left out), lacks a number A says, or comes from a
    sentence of the source an earlier section cited;
  - an A line copies more than 12 words in a row from its passage (--max-copied N to
    change; 0 only measures and reports it);
  - a BACKGROUND line names a term the source does not use or one an earlier section
    explained, holds a number or name, says what the document or its authors found, or is
    the third in the section;
  - B says a number or name A has not said;
  - the model, asked about each line on its own, finds an A line its passage does not
    support, an A line with no passage that says something about the document, a
    BACKGROUND line that is not general knowledge, or a B line that brings in something new.
A faulty line is rewritten on its own, up to twice, with its faults pointed out (for a
passage not in the source, the source's nearest sentence is quoted); a B line still faulty
after that is replaced by one that only invites A to go on ("Go on."), and the A line after
it is rewritten to follow on, the rewrite kept only if it passes its checks. The section is
written again, up to three times, if a fault cannot be mended line by line (one host three
times running, or over 1.6 times its budget), if A's faulty lines are more than a third of
its lines, or if A's lines are still faulty after their repairs.
A section that passes is edited against the source and the episode so far; the edit is
kept only if it passes the same checks, cuts no more than a third of the section, and
keeps the numbers of the section's planned points. Numbers and acronyms are then written
out as they should be said.

Beside the script go <script>.review.md (each line of A's beside its passage, with any
check that failed and any long copied run), <script>.check.json (each section's checks) and <script>.calls.json
(tokens and time for each model call). The exit code is non-zero if any section still
fails. The checks are made by the same kind of model that wrote the script; reading the
review sheet against the source is still the way to be sure of an episode.
"""
import argparse
import functools
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:27b-mxfp8"
ATTEMPTS = 4  # writes of a section before the best one is kept
REPAIR_ROUNDS = 2  # repairs of a faulty line within one write
# Only the plan is made with thinking: it used nearly all the output of the writing calls,
# 7 to 15 minutes each, and once the whole allowance with no reply (2026-10-07).
WORDS_PER_MINUTE = 150
NUM_CTX = 40960  # the same for every call: a change makes Ollama reload the model
NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"

# Every prompt starts with the document, so Ollama can reuse its cache of it between calls.
PREFIX = """You are helping to make a two-host podcast episode about the document below. \
Read the document; your task follows it.

The document:
<<<
{source}
>>>

"""

STYLE = """How the conversation should sound:
- Host A explains the document. Host B asks what a curious listener would ask, checks
  jargon ("Sorry, what's a benchmark?"), and restates what A has just said in other words
  ("So the warmer water holds less oxygen."). B never brings in a fact, number,
  name, example or judgement that A has not said: everything about the document comes
  from A. Most turns are short; A runs to three or four sentences at most.
- Use the openers and fillers of speech ("So,", "Right,", "I mean,"), an occasional echo
  of what A has just said ("38." / "38 models."), and plain reactions ("Okay.", "I see.").
- Neither host judges the work or its numbers ("impressive", "nearly perfect", "huge",
  "rigorous", "scary") unless the document does, and neither adds a comparison, ranking,
  consequence or generalisation the document does not state. If the document does not
  say it, the hosts do not say it.
- The hosts are presenters, not the authors: they say "the authors" or "they", never "we"
  or "our" about the work, and B never asks A about "your" study.
- Each point once: no host repeats what the other has just said, except a short echo.
- Never read an equation out symbol by symbol. Say in words what it means and what it
  predicts ("quality rises with the log of model size and of topic frequency").
- Write every number in digits (38, 8,900, 60%, 0.74) so it can be checked; it is turned
  into words later. Name people, places and organisations exactly as the document does.
- Plain words. No bullet points, headings, stage directions, sound effects or markdown.

A short example of the style and the SOURCE and BACKGROUND lines (its subject and its source are
unrelated to this document; copy the manner, not the facts):

A: So, the question in this piece is why the sky is blue.
SOURCE: Why is the sky blue?
B: Okay. Where does the answer start?
A: With sunlight, which has every colour in it. Air scatters the short wavelengths, the blue end, much more than the long ones.
SOURCE: Sunlight contains every colour, and air scatters short wavelengths, at the blue end of the spectrum, far more strongly than long ones.
B: Sorry, what's a wavelength?
A: It's the distance from one crest of a wave to the next, so a short wavelength means the crests come close together.
BACKGROUND: wavelength
B: So the blue end gets bounced around the most.
A: Right. Blue light is scattered across the whole sky, and that's the colour you see when you look up.
SOURCE: Blue light is therefore scattered across the whole sky, which is the colour seen overhead.
B: Then why are sunsets red?
A: Because at sunset the light comes through far more air, so most of the blue has been scattered away before it reaches you.
SOURCE: At sunset, light crosses far more air, and most of the blue is scattered out before it reaches the observer."""

PLAN = """Plan a two-host podcast episode of about {words} words in total about the \
document. Plan between 4 and 8 sections that take a listener from what the document is \
and why it matters, through its substance in a sensible order, to what it concludes. Give \
each section a short title, a word budget, and the specific points it must cover, each \
stated as the document states it, with numbers in digits. Give each point to one section \
only. Leave out reference lists, acknowledgements and formatting.

Also give the document's title and its authors with their affiliations, copied exactly as \
the document gives them (an empty list if it names no authors; an empty affiliation if it \
gives none).

Reply with only a JSON object:
{{"title": "...", "authors": [{{"name": "...", "affiliation": "..."}}],
 "sections": [{{"title": "...", "words": 300, "points": ["...", "..."]}}]}}"""

WRITE = """Write one section of the episode: about {words} words of conversation, and no fewer \
than {min_words}, covering these points:
{points}

{position}
{covered}
{previous}
{style}
{feedback}
Reply with only the dialogue, one turn per line, each line starting "A: " or "B: ". After \
every A line that states anything from the document, add a line starting "SOURCE: " that \
copies, character for character, the sentence or part of a sentence of the document that \
the A line is based on: at least 5 words, holding every number the A line says. If the line \
draws on two places, give two SOURCE lines. A short A line that states nothing ("Exactly.", \
"Right, so let's look at the method.") needs no SOURCE line. B lines never have one.

The SOURCE is not spoken, so A says things in A's own words, as a presenter explaining the \
work to a listener, not by reading the document aloud: an A line takes no more than 8 words \
in a row from its SOURCE.

If A needs to explain a general term the document uses but does not explain itself (what a \
sigmoid or a parameter is), A may do it in one line followed by a line "BACKGROUND: " and \
only the term itself, in one to three words ("BACKGROUND: sigmoid"), instead of a SOURCE line. \
The explanation goes in A's line, not in the BACKGROUND line. A BACKGROUND line explains the term in general and says \
nothing about the document's own work or results; it holds no numbers and no names. At most \
{max_background} in this section."""

EDIT = """Edit one section of the episode before it is recorded. Correct the draft \
section in these ways and change nothing else:

1. Delete or correct any statement by A that its SOURCE does not make: a comparison, \
ranking, judgement, consequence or prediction it does not state, or a generalisation of one \
case (one model becoming "models", one topic becoming "topics").
2. B only asks and restates: delete or rephrase any B line that brings in a fact, number, \
name, example or judgement A has not said.
3. Delete any point already made earlier in this section or in the episode so far, except \
a short echo.
4. Where an equation or derivation is read out symbol by symbol, replace it with one or \
two sentences saying in words what it means.
5. The hosts are presenters: "the authors", never "we" or "you" for the people who did \
the work.

Leave every line that needs none of these changes exactly as it is, with its SOURCE or \
BACKGROUND line. Keep each SOURCE or BACKGROUND line under the A line it belongs to; if you \
change an A line it must still say only what its SOURCE says (or, for a BACKGROUND line, \
explain the term in general and nothing about the document's work), and if you delete an A \
line, delete its SOURCE or BACKGROUND line too. Keep \
the exchange whole: when you delete a question, delete or rephrase the answer that depends \
on it, and when you delete an answer, delete its question. The hosts alternate; never leave \
one host speaking twice in a row. You may add a short bridging line where a deletion leaves \
a gap, if it states nothing new. Keep numbers in digits. Reply with only the edited section \
in the same form as the draft.

The episode so far:
<<<
{so_far}
>>>

The draft section:
<<<
{draft}
>>>"""

CHECK_A = """Check one line of the script against the document.

The line, spoken by host A: "{line}"
The passage of the document it is based on: "{quote}"

Does the document, at that passage, state everything the line says? The line fails if it \
adds a comparison, judgement, consequence or generalisation the document does not state, \
says something the document says about a different part of the work, or changes a number \
or a name. Reply with one word, SUPPORTED or UNSUPPORTED, then a colon and at most one \
sentence of reason."""

REPAIR = """Rewrite one line of the episode to mend the faults listed under it. Keep what it says \
where that is allowed, keep it in the same host's voice, and keep it fitting between the lines \
around it.

The lines before it:
<<<
{before}
>>>
The line to rewrite:
<<<
{line}
>>>
The lines after it:
<<<
{after}
>>>
Its faults:
{faults}

{rules}
Reply with only the rewritten line, starting "{host}: "."""

REPAIR_RULES = {
    "A": """An A line that states anything from the document is followed by a line "SOURCE: " that copies, \
character for character, the passage of the document it rests on: at least 5 words, holding every \
number the line says. The SOURCE is not spoken, so the line says it in A's own words, taking no \
more than 8 words in a row from the SOURCE. A line explaining a general term the document uses \
can instead be followed by "BACKGROUND: " and the term alone; it then holds no numbers or names and \
says nothing about the document's own work. A short line that states nothing needs neither.""",
    "B": """B only asks questions and restates what A has already said: no fact, number, name, example \
or judgement that A has not said. A B line has no SOURCE line.""",
}

CHECK_BG = """Check one line in which host A explains a general term the document uses: \
"{term}".

The line: "{line}"

Is the line a correct general explanation of the term, which says nothing about the \
document's own work, results or claims? Reply with one word, GENERAL or FAILS, then a colon \
and at most one sentence of reason."""

CHECK_LINK = """Check one line spoken by host A that cites no passage of the document.

The conversation just before it:
<<<
{context}
>>>
A's line: "{line}"

A line that only links the conversation passes: agreeing, answering yes or no, or naming \
the next topic without saying anything about it. A line that says anything about the \
document's work, its methods, results, claims or conclusions, fails, since such a line \
needs a passage. Reply with one word, LINK or CLAIM, then a colon and at most one sentence \
of reason."""

CHECK_B = """Check one line spoken by host B, whose part is only to ask questions and to \
restate what host A has already said.

The conversation just before it:
<<<
{context}
>>>
B's line: "{line}"

The line fails if it states or presupposes a fact, number, name, example, judgement or \
comparison that the conversation before it does not contain. Repeating or rewording what A \
has already said, numbers and names included, does not fail. A question that only asks \
about something does not fail. Reply with one word, FINE or FAILS, then a colon and at \
most one sentence of reason."""


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------- model ----------

CALLS = []
_calls_lock = threading.Lock()


def chat(model, prompt, label, think=True, num_predict=16384, temperature=0.7, timeout=1800):
    """The model's reply; tokens and times for the call go to CALLS."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": think,
        "options": {"temperature": temperature, "num_ctx": NUM_CTX, "num_predict": num_predict},
    }
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        reply = json.load(r)
    message = reply.get("message", {})
    content, thinking = message.get("content", ""), message.get("thinking", "") or ""
    out = reply.get("eval_count", 0)
    if not content.strip() and reply.get("done_reason") == "length":
        log(f"{label}: the reasoning used the whole allowance of {num_predict} tokens and left no reply")
    share = len(thinking) / max(1, len(thinking) + len(content))
    call = dict(label=label, kind=label.split()[0], start=time.strftime("%H:%M:%S", time.localtime(start)),
                seconds=round(time.time() - start, 1),
                prompt_tokens=reply.get("prompt_eval_count", 0),
                prompt_seconds=round(reply.get("prompt_eval_duration", 0) / 1e9, 1),
                output_tokens=out, thinking_tokens_est=round(out * share),
                output_seconds=round(reply.get("eval_duration", 0) / 1e9, 1),
                load_seconds=round(reply.get("load_duration", 0) / 1e9, 1), think=think,
                done_reason=reply.get("done_reason", ""))
    if label.startswith("check"):
        call.update(asked=prompt.split("\n>>>\n\n", 1)[-1][-4000:], reply=content[:300])
    with _calls_lock:
        CALLS.append(call)
    return content


def describe(call):
    return (f"{call['label']}: prompt {call['prompt_tokens']} tokens in {call['prompt_seconds']} s, "
            f"output {call['output_tokens']} tokens (about {call['thinking_tokens_est']} thinking) "
            f"in {call['output_seconds']} s")


def call_totals():
    totals = {}
    for c in CALLS:
        t = totals.setdefault(c["kind"], dict(calls=0, seconds=0.0, prompt_tokens=0, output_tokens=0,
                                              thinking_tokens_est=0))
        t["calls"] += 1
        for k in ("seconds", "prompt_tokens", "output_tokens", "thinking_tokens_est"):
            t[k] += c[k]
    for t in totals.values():
        t["seconds"] = round(t["seconds"], 1)
    return totals


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
        # pandoc spaces a comma in maths, so $8{,}913$ would come out "8, 913": hold the thousands
        # separators aside and put them back after. A bare comma in maths separates, as in F_{1,381}.
        latex = re.sub(r"(?<=\d)\{,\}(?=\d{3}(?!\d))", "QQTHSEPQQ", latex)
        cmd = ["pandoc", "-f", "latex", "-t", "plain", "--wrap=none"] + (["-s"] if standalone else [])
        r = subprocess.run(cmd, input=latex, capture_output=True, text=True, cwd=path.parent)
        if r.returncode:
            sys.exit(f"podcast-script: pandoc could not read {path.name}: {r.stderr[:300]}")
        return r.stdout.replace("QQTHSEPQQ", ",")

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
    """A number as compared with the source's: no thousands commas, and no trailing zeros after the
    point, so that "0.9" matches the paper's "0.90"."""
    tok = tok.replace(",", "").rstrip(".")
    return tok.rstrip("0").rstrip(".") if "." in tok else tok


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



# ---------- dialogue and its sources ----------

MAX_BACKGROUND = 2  # BACKGROUND lines allowed in a section
STAND_INS = ["Go on.", "Tell me more.", "Carry on.", "Go on, then."]  # for a B line still failing after repair
COPY_REPORT = 8  # copied runs longer than this (the prompt's limit) are marked in the review sheet


def parse_cited(reply):
    """Turns as (host, text); for each turn, the SOURCE passages under it and the term its
    BACKGROUND line names (None if it has none)."""
    turns, quotes, bg = [], [], []
    for line in reply.splitlines():
        line = line.strip().replace("**", "")
        m = re.match(r"^(?:Host\s+)?([AB])\s*:\s*(.+)$", line)
        if m:
            turns.append((m.group(1), m.group(2).strip()))
            quotes.append([])
            bg.append(None)
            continue
        m = re.match(r"^(SOURCE|BACKGROUND)\s*:\s*(.+)$", line, re.I)
        if m and turns:
            text = m.group(2).strip().strip("\"“”'‘’").strip()
            if m.group(1).upper() == "SOURCE":
                quotes[-1].append(text)
            else:
                bg[-1] = text
    return turns, quotes, bg


def cited_text(turns, quotes, bg):
    out = []
    for (h, t), qs, term in zip(turns, quotes, bg):
        out.append(f"{h}: {t}")
        out += [f"SOURCE: {q}" for q in qs]
        if term:
            out.append(f"BACKGROUND: {term}")
    return "\n".join(out)


def norm_text(s):
    s = s.lower().translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-",
                                           "∼": "~"}))
    s = re.sub(r"[*_`#]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def quote_in_source(quote, src_norm):
    """True if the passage, or each piece of it either side of an ellipsis, is word for word in the source.
    Spacing is ignored (the source may have "  ∼ 40%" where the model writes "~40%"), and a passage may
    leave out the bracketed asides of the paper's sentence ("Model size alone accounts for 42.1%" where
    the paper has "Model size alone (log₁₀P) accounts for 42.1%"); a passage may not add one."""
    pieces = [norm_text(p).strip(" .,;:\"'") for p in re.split(r"\.\.\.|…", quote)]
    pieces = [p for p in pieces if p]
    if not pieces or sum(len(p.split()) for p in pieces) < 5:
        return False
    flat, flat_bare = flat_forms(src_norm)
    return all(p.replace(" ", "") in flat or ("(" not in p and p.replace(" ", "") in flat_bare) for p in pieces)


@functools.lru_cache(maxsize=4)
def flat_forms(src_norm):
    """The source without spaces, and the same with every bracketed aside taken out."""
    return flat_pair(src_norm)


def flat_pair(text):
    bare = text
    while True:
        cut = re.sub(r"\s*\([^()]*\)", "", bare)
        if cut == bare:
            break
        bare = cut
    return text.replace(" ", ""), bare.replace(" ", "")


@functools.lru_cache(maxsize=4)
def source_sentences(src_norm):
    """The source's sentences, each as (text, without spaces, without spaces or bracketed asides)."""
    return [(x, *flat_pair(x)) for x in re.split(r"(?<=[.!?])\s+", src_norm) if x.strip()]


@functools.lru_cache(maxsize=4096)
def passage_sentences(passage, src_norm):
    """The indices of the source sentences a passage is taken from (a piece running across a sentence
    end counts for both sentences)."""
    sents, out = source_sentences(src_norm), set()
    for p in [norm_text(x).strip(" .,;:\"'") for x in re.split(r"\.\.\.|…", passage)]:
        if not p:
            continue
        fp = p.replace(" ", "")
        hit = {k for k, (_, f, b) in enumerate(sents) if fp in f or ("(" not in p and fp in b)}
        if not hit:
            for k in range(len(sents) - 1):
                if fp in sents[k][1] + sents[k + 1][1]:
                    hit |= {k, k + 1}
        out |= hit
    return frozenset(out)


@functools.lru_cache(maxsize=4)
def printed_sentences(source):
    """The source's sentences as printed, by their normalised form."""
    return {norm_text(x): x for x in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", source)) if x.strip()}


def nearest_sentence(passage, src_norm):
    """The source sentence sharing the longest run of words with the passage, or None if no run of
    three words is shared."""
    best, run = None, 2
    for text, _, _ in source_sentences(src_norm):
        r = copied_run(passage, [text])
        if r > run:
            best, run = text, r
    return best


def states_something(text):
    """Whether an A line needs a SOURCE: anything longer than a short bridge, or holding a number or name."""
    if re.search(r"\d", text) or len(text.split()) > 8:
        return True
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        toks = re.findall(r"[A-Za-z][A-Za-z'À-ɏ-]*", sentence)
        if any(t[0].isupper() and t.lower() not in ALWAYS_OK for t in toks[1:]):
            return True
    return False


def names_in(text):
    """Capitalised words after the start of a sentence, read as unsupported() reads them: the parts of
    a hyphenated word longer than one letter ("S-shaped" holds none)."""
    names = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for tok in re.findall(r"[A-Za-z][A-Za-z'À-ɏ-]*", sentence)[1:]:
            names += [p for p in tok.split("-") if len(p) > 1 and p[0].isupper() and p.lower() not in ALWAYS_OK]
    return names


def words_of(text):
    return re.findall(r"[a-z0-9]+(?:[.,'][a-z0-9]+)*", norm_text(text))


def copied_run(line, passages):
    """The longest run of consecutive words the line shares with any of its passages."""
    a, best = words_of(line), 0
    for passage in passages:
        b = words_of(passage)
        prev = [0] * (len(b) + 1)
        for x in a:
            cur = [0] * (len(b) + 1)
            for j, y in enumerate(b, 1):
                if x == y:
                    cur[j] = prev[j - 1] + 1
                    best = max(best, cur[j])
            prev = cur
    return best


def term_key(term):
    return " ".join(w.rstrip("s") for w in words_of(term))


def cite_faults(turns, quotes, bg, src_norm, max_copied=0, used=(), explained=(), source=None):
    """For each A line that breaks a rule, the reasons: its SOURCE is missing, not in the source, short of
    a number the line says, or from a sentence an earlier section cited (`used`); it copies more than
    max_copied words of its passage; or, for a BACKGROUND line, the term is not the paper's, not
    explained, already explained in an earlier section (`explained`), or the section has too many."""
    faults, background = {}, 0
    src_words = None
    for i, ((h, t), qs, term) in enumerate(zip(turns, quotes, bg)):
        if h != "A":
            continue
        why = []
        if term and not qs:
            background += 1
            asked = turns[i - 1][1] if i else ""
            if len(term.split()) > 4:
                why.append(f'a BACKGROUND line must give only the term it explains, in one to three words '
                           f'("BACKGROUND: sigmoid"), with the explanation in A\'s line, not "{term[:50]}"')
            else:
                if src_words is None:
                    src_words = set(words_of(src_norm))
                if norm_text(term) not in src_norm and not all(said(w, src_words) for w in words_of(term)):
                    why.append(f'"{term}" is not a term the document uses')
                elif norm_text(term) not in norm_text(t + " " + asked):
                    why.append(f'the line does not explain its BACKGROUND term "{term}"')
                if term_key(term) in explained:
                    why.append(f'"{term}" was already explained in an earlier section; do not explain it again')
            if re.search(r"\d", t) or [n for n in names_in(t) if n.lower() not in norm_text(term)]:
                why.append("a BACKGROUND line may hold no number or name")
            if re.search(r"\bthe authors\b|\bth(e|is) (study|paper)\b(?! of)|\bthey (found|find|report|reported)\b"
                         r"|\bthey show(ed)? that\b", t, re.I):
                why.append("a BACKGROUND line explains a general term; it may not say what the document or its "
                           "authors did or found, which needs a SOURCE")
            if background > MAX_BACKGROUND:
                why.append(f"more than {MAX_BACKGROUND} BACKGROUND lines in this section: cite a SOURCE instead, "
                           "or leave the explanation out")
        elif not qs:
            if states_something(t):
                why.append("the line states something but has no SOURCE line")
        else:
            bad = [q for q in qs if not quote_in_source(q, src_norm)]
            if bad:
                near = nearest_sentence(bad[0], src_norm)
                if near and source:
                    near = printed_sentences(source).get(near, near)
                why.append(f'its SOURCE is not word for word in the document, or is under 5 words: "{bad[0][:70]}"'
                           + (f'; the nearest sentence in the document is: "{" ".join(near.split()[:80])}"'
                              if near else ""))
            else:
                have = {norm_number(x) for q in qs for x in re.findall(NUM, q)}
                extra = [x for x in re.findall(NUM, t) if norm_number(x) not in have]
                if extra:
                    why.append(f"it says numbers its SOURCE does not hold: {', '.join(extra)}")
                run = copied_run(t, qs)
                if max_copied and run > max_copied:
                    why.append(f"it copies {run} words in a row from its SOURCE; say it in A's own words, taking "
                               "no more than 8 words in a row")
                if used and any(passage_sentences(q, src_norm) & passage_sentences(u, src_norm)
                                for q in qs for u in used):
                    why.append("its SOURCE was already cited in an earlier section: say something not yet said, "
                               "or refer back in a few words without restating it")
        if why:
            faults[i] = why
    return faults


def cite_problems(turns, quotes, bg, src_norm, max_copied=0):
    """The faults of cite_faults as a flat list, each with its line."""
    return [f'{r}: "{turns[i][1][:60]}"' for i, rs in cite_faults(turns, quotes, bg, src_norm, max_copied).items()
            for r in rs]


def heard(lines):
    """Numbers A has said, and every word either host has said, in lines of the form 'H: text'."""
    numbers, words = set(), set()
    for line in lines:
        h, _, text = line.partition(": ")
        if h == "A":
            numbers |= {norm_number(x) for x in re.findall(NUM, text)}
        words |= spoken_words(text)
    return numbers, words


def spoken_words(text):
    """Every word in the text, lower case, with each part of a hyphenated word as well."""
    toks = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'À-ɏ-]*", text)]
    return set(toks) | {p for t in toks for p in t.split("-") if p}


def said(word, words):
    """Whether a word has been said, allowing for a plural ("LLM" after "LLMs")."""
    w = word.lower()
    return w in words or w + "s" in words or w.rstrip("s") in words or w.removesuffix("'s") in words


def b_faults(turns, so_far_lines):
    """For each B line with a number or name not said before it, the reason."""
    numbers, words = heard(so_far_lines)
    faults = {}
    for i, (h, t) in enumerate(turns):
        if h == "B":
            new = [x for x in re.findall(NUM, t) if norm_number(x) not in numbers]
            new += [w for w in names_in(t) if not said(w, words)]
            if new:
                faults[i] = [f"B says {', '.join(new)} before A has; B only asks and restates what A has said"]
        else:
            numbers |= {norm_number(x) for x in re.findall(NUM, t)}
        words |= spoken_words(t)
    return faults


def b_problems(turns, so_far_lines):
    return [f'{r}: "{turns[i][1][:60]}"' for i, rs in b_faults(turns, so_far_lines).items() for r in rs]


def line_faults(ctx, turns, quotes, bg, so_far_lines):
    """Every fault found without the model, by line: numbers and names not in the source, symbols that
    cannot be spoken, a turn repeated, and the SOURCE, BACKGROUND and B rules."""
    faults = {}

    def add(i, reasons):
        faults.setdefault(i, []).extend(reasons)
    seen = set()
    for i, turn in enumerate(turns):
        missing = unsupported([turn], ctx.numbers, ctx.words)
        if missing:
            add(i, [f"numbers or names the document does not contain: {', '.join(missing)}"])
        left = set(re.findall(r"[^\w\s.,;:!?'’\"“”()\[\]-]|_", spoken(turn[1], ctx.keep).replace("[laugh]", "")))
        if left:
            add(i, ["symbols that cannot be spoken: " + " ".join(sorted(left)) + "; write them as words"])
        key = re.sub(r"\W+", " ", turn[1].lower()).strip()
        if len(key.split()) >= 6 and key in seen:
            add(i, ["it repeats an earlier line"])
        seen.add(key)
    for i, rs in cite_faults(turns, quotes, bg, ctx.src_norm, ctx.max_copied, ctx.used_passages,
                             ctx.explained, getattr(ctx, "source", None)).items():
        add(i, rs)
    for i, rs in b_faults(turns, so_far_lines).items():
        add(i, rs)
    return faults


def section_faults(turns, budget):
    """Faults no repair of one line can mend: no dialogue, one host three times running, runaway length."""
    if not turns:
        return ["no dialogue in the reply"]
    problems = []
    run = 1
    for (h0, _), (h1, _) in zip(turns, turns[1:]):
        run = run + 1 if h1 == h0 else 1
        if run == 3:
            problems.append(f"host {h1} speaks three or more times in a row")
            break
    words = sum(len(t.split()) for _, t in turns)
    if words > 1.6 * budget:
        problems.append(f"{words} words against a budget of {budget}")
    return problems


def verdict(reply, good):
    m = re.match(r"\W*([A-Za-z]+)\W*(.*)", reply.strip(), re.S)
    if not m:
        return False, "no verdict in the reply"
    return m.group(1).upper() == good, m.group(2).strip()[:200]


def line_checks(ctx, turns, quotes, bg, so_far_lines, label, skip=()):
    """Each A line against its passage (or, for a BACKGROUND line, as general knowledge) and each B line
    against what came before, asked one at a time without thinking, several at once; lines in `skip`
    are left out. Returns the failures as (index, reason) and the number of lines checked."""
    jobs = []
    for i, (h, t) in enumerate(turns):
        if i in skip:
            continue
        context = "\n".join((so_far_lines + [f"{hh}: {tt}" for hh, tt in turns[:i]])[-8:]) or "(nothing yet)"
        if h == "A" and quotes[i]:
            jobs.append((i, CHECK_A.format(line=t, quote=" / ".join(quotes[i])), "SUPPORTED", ""))
        elif h == "A" and bg[i]:
            jobs.append((i, CHECK_BG.format(term=bg[i], line=t), "GENERAL", ""))
        elif h == "A" and len(t.split()) >= 4:
            jobs.append((i, CHECK_LINK.format(context=context, line=t), "LINK",
                         "it says something about the document but has no SOURCE line: "))
        elif h == "B" and len(t.split()) >= 4:
            jobs.append((i, CHECK_B.format(context=context, line=t), "FINE", ""))
    todo = list({p: good for _, p, good, _ in jobs if p not in ctx.cache}.items())
    if todo:
        start = time.time()

        def ask(job):
            p, good = job
            return p, verdict(chat(ctx.model, ctx.prefix + p, f"check {label}", think=False, num_predict=120,
                                   temperature=0), good)
        with ThreadPoolExecutor(max_workers=ctx.parallel) as pool:
            for p, v in pool.map(ask, todo):
                ctx.cache[p] = v
        log(f"{label}: checked {len(todo)} lines in {time.time() - start:.0f} s")
    fails = [(i, why + ctx.cache[p][1]) for i, p, _, why in jobs if not ctx.cache[p][0]]
    return fails, len(jobs)


def all_faults(ctx, turns, quotes, bg, so_far_lines, budget, label):
    """The section's faults: those no line repair can mend, and by line those found without the model
    and, for lines with none of those, by the model's check."""
    section = section_faults(turns, budget)
    if section:
        return section, {}, 0
    faults = line_faults(ctx, turns, quotes, bg, so_far_lines)
    fails, checked = line_checks(ctx, turns, quotes, bg, so_far_lines, label, skip=set(faults))
    for i, reason in fails:
        faults.setdefault(i, []).append(f"the model's check: {reason}")
    return [], faults, checked


def repair_line(ctx, turns, quotes, bg, i, reasons, so_far_lines, label):
    """The line rewritten to mend its faults, as (turn, quotes, term), or None if the reply has no line
    for the same host."""
    host = turns[i][0]
    before = (so_far_lines + [f"{h}: {t}" for h, t in turns[:i]])[-4:]
    after = [f"{h}: {t}" for h, t in turns[i + 1:i + 3]]
    reply = chat(ctx.model, ctx.prefix + REPAIR.format(
        before="\n".join(before) or "(the start of the episode)", line=cited_text([turns[i]], [quotes[i]], [bg[i]]),
        after="\n".join(after) or "(the end of the section)", faults="\n".join(f"- {r}" for r in reasons),
        rules=REPAIR_RULES[host], host=host), f"repair {label}", think=False, num_predict=400)
    new, nq, nbg = parse_cited(reply)
    for turn, q, term in zip(new, nq, nbg):
        if turn[0] == host:
            return turn, q, term
    return None


def settle(ctx, turns, quotes, bg, so_far_lines, budget, label, rounds=REPAIR_ROUNDS):
    """Check every line and repair the faulty ones one at a time, up to `rounds` times, unless a fault is
    one no line repair can mend or A's faulty lines are more than a third of all the lines. A B line still faulty
    after the repairs becomes a stand-in that only invites A to go on ("Go on."), and the A line after it
    is repaired to follow on from it. Returns the section as it
    stands, its section faults, its line faults, the number of lines checked by the model, the number
    of repairs made and the B lines replaced, as (stand-in, line, faults)."""
    turns, quotes, bg = list(turns), list(quotes), list(bg)
    repairs, stand_ins = 0, []
    for r in range(rounds + 1):
        section, faults, checked = all_faults(ctx, turns, quotes, bg, so_far_lines, budget, label)
        if r == rounds and rounds and not section:
            # The A line after a stand-in was written to answer the line it replaced, so it is repaired to
            # follow on; a repair that leaves it faulty is undone. The lines after a change follow
            # something new, so the section is checked again until no B line is faulty.
            for _ in range(3):
                bad = [i for i in sorted(faults) if turns[i][0] == "B"]
                if not bad:
                    break
                followers = {}
                for i in bad:
                    stand_ins.append((STAND_INS[len(stand_ins) % len(STAND_INS)], turns[i][1], faults[i]))
                    turns[i], quotes[i], bg[i] = ("B", stand_ins[-1][0]), [], None
                    if i + 1 < len(turns) and turns[i + 1][0] == "A":
                        followers[i + 1] = (turns[i + 1], quotes[i + 1], bg[i + 1])
                log(f"{label}: {len(bad)} B lines still faulty after repair replaced by a stand-in")

                def follow(j):
                    why = [f'the line before it is now B\'s "{turns[j - 1][1]}", put in for a line of B\'s that failed '
                           "its checks: make this line follow on from it, not answer a question or remark B no "
                           "longer makes, and keep what it says"]
                    return j, repair_line(ctx, turns, quotes, bg, j, why, so_far_lines, label)
                with ThreadPoolExecutor(max_workers=ctx.parallel) as pool:
                    for j, mended in pool.map(follow, sorted(followers)):
                        if mended and (turns[j], quotes[j], bg[j]) != mended:
                            turns[j], quotes[j], bg[j] = mended
                            repairs += 1
                section, faults, checked = all_faults(ctx, turns, quotes, bg, so_far_lines, budget, label)
                undo = [j for j in followers if j in faults and (turns[j], quotes[j], bg[j]) != followers[j]]
                for j in undo:
                    turns[j], quotes[j], bg[j] = followers[j]
                    repairs -= 1
                if undo:
                    section, faults, checked = all_faults(ctx, turns, quotes, bg, so_far_lines, budget, label)
        if section or not faults or r == rounds or sum(turns[i][0] == "A" for i in faults) > len(turns) / 3:
            return turns, quotes, bg, section, faults, checked, repairs, stand_ins
        log(f"{label}: repairing {len(faults)} of {len(turns)} lines")

        def fix(i):
            return i, repair_line(ctx, turns, quotes, bg, i, faults[i], so_far_lines, label)
        with ThreadPoolExecutor(max_workers=ctx.parallel) as pool:
            for i, mended in pool.map(fix, sorted(faults)):
                if mended:
                    repairs += (turns[i], quotes[i], bg[i]) != mended  # a line sent back unchanged is no repair
                    turns[i], quotes[i], bg[i] = mended


def copy_runs(turns, quotes):
    """For each A line with a passage, the longest run of words it copies from it."""
    return [copied_run(t, qs) for (h, t), qs in zip(turns, quotes) if h == "A" and qs]


# ---------- frame ----------

def in_block(text, src_norm):
    """Whether the source holds the text, ignoring spaces, commas and semicolons: an author block's line
    breaks may come back from the plan as either ("Centre, Canada" for "Centre" and "Canada" on two
    lines)."""
    return re.sub(r"[\s,;]", "", norm_text(text)).strip(".") in re.sub(r"[\s,;]", "", src_norm)


def frame(meta, src_norm):
    """The opening and the goodbye, from the title and authors the plan gives, kept only where the
    source holds them word for word."""
    title = (meta.get("title") or "").strip()
    if title and not in_block(title, src_norm):
        log(f"title not found in the source, left out: {title}")
        title = ""
    authors = []
    for a in meta.get("authors") or []:
        # The paper's author block can carry line breaks and footnote marks ("Smith[1]").
        name = re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]|[*†‡§¶]|(?<=[a-z])\d+$", "", a.get("name") or "")).strip()
        printed = re.sub(r"\[[^\]]*\]|^\d+\s*", "", (a.get("affiliation") or "").strip())
        aff = re.sub(r"\s+", " ", re.sub(r"\s*\n\s*", ", ", printed)).strip(" ,")  # a line break is said as a pause
        if not name or not in_block(name, src_norm):
            log(f"author not found in the source, left out: {name}")
            continue
        if aff and not in_block(printed, src_norm):
            log(f"affiliation not found in the source, left out: {aff}")
            aff = ""
        authors.append(name + (f", of {aff}" if aff else ""))
    opening = [("A", f"So, welcome. Today we're looking at {title}." if title
                else "So, welcome. Today we're looking at a new document.")]
    if authors:
        if len(authors) == 1:
            who = f"The author is {authors[0]}."
        elif len(authors) <= 6:
            who = "The authors are " + "; ".join(authors[:-1]) + "; and " + authors[-1] + "."
        else:
            who = f"The first author is {authors[0]}, with {len(authors) - 1} co-authors."
        opening += [("B", "Who wrote it?"), ("A", who)]
    closing = [("A", "That's all for this episode. Thanks for listening."), ("B", "Bye for now.")]
    return opening, closing


def closing_after(last_host, closing):
    """The goodbye, opened by B if A spoke last, so neither host speaks twice running."""
    if last_host == "A":
        return [("B", "Thanks for taking me through it."), ("A", "That's all for this episode. Thanks for listening.")]
    return closing


# ---------- main ----------

def plan(ctx, words):
    for _ in range(3):
        reply = chat(ctx.model, ctx.prefix + PLAN.format(words=words), "plan")
        log(describe(CALLS[-1]))
        m = re.search(r"\{.*\}", reply, re.DOTALL)
        try:
            meta = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            meta = None
        if meta and meta.get("sections"):
            return meta
    sys.exit("podcast-script: the model did not return a usable plan")


def edit_section(ctx, turns, quotes, bg, so_far_lines, budget, label, point_numbers):
    """The draft edited against the source and the episode so far, or the reason the edit was refused."""
    reply = chat(ctx.model, ctx.prefix + EDIT.format(so_far="\n".join(so_far_lines) or "(this is the first section)",
                                                     draft=cited_text(turns, quotes, bg)), f"edit {label}",
                 think=False)
    log(describe(CALLS[-1]))
    edited, eq, ebg = parse_cited(reply)
    if not edited:
        return None, "no dialogue in the reply"
    before, after = sum(len(t.split()) for _, t in turns), sum(len(t.split()) for _, t in edited)
    if after < before * 2 / 3:
        return None, f"cut the section from {before} to {after} words"
    kept = {norm_number(x) for _, t in edited for x in re.findall(NUM, t)}
    dropped = sorted({norm_number(x) for _, t in turns for x in re.findall(NUM, t)} & point_numbers - kept)
    if dropped:
        return None, "dropped planned points (" + ", ".join(dropped) + ")"
    edited, eq, ebg, section, faults, _, _, _ = settle(ctx, edited, eq, ebg, so_far_lines, budget,
                                                       f"edit {label}", rounds=0)
    if section or faults:
        return None, "; ".join(section + [f'"{edited[i][1][:50]}": {"; ".join(rs)}' for i, rs in sorted(faults.items())])
    return (edited, eq, ebg), None


def taken_text(ctx):
    """What earlier sections have used, for the writer to leave alone."""
    out = ""
    if ctx.used_passages:
        out += ("Earlier sections have already cited these passages; do not cite them again or restate what they "
                "say:\n" + "\n".join(f"- {' '.join(q.split()[:14])} ..." for q in ctx.used_passages) + "\n")
    if ctx.explained:
        out += ("Terms already explained in earlier sections; do not explain them again: "
                + ", ".join(sorted(ctx.explained.values())) + "\n")
    return out


def write_section(ctx, section, i, n, previous, so_far_lines, covered, later=()):
    points = "\n".join(f"- {p}" for p in section["points"])
    point_numbers = {norm_number(x) for p in section["points"] for x in re.findall(NUM, p)}
    if i == 0:
        position = ("This is the first section. The hosts have already greeted the listener and named the document "
                    "and its authors: do not greet, and do not name the authors again.")
    elif i == n - 1:
        position = "This is the closing section: end with what the document concludes. Do not say goodbye; that is added afterwards."
    else:
        position = "This section continues the episode: no greeting, no goodbye."
    prev = ("The conversation so far ended like this; carry on from it without repeating it:\n" + previous) if previous else ""
    cov = ("Earlier sections have already covered these points; do not explain them again (a brief reference back "
           "is fine):\n" + "\n".join(f"- {p}" for p in covered) + "\n") if covered else ""
    cov += ("Later sections will cover these points; leave them to those sections:\n"
            + "\n".join(f"- {p}" for p in later) + "\n") if later else ""
    cov += taken_text(ctx)
    feedback, best, budget = "", None, int(section.get("words", 300))
    label = f"section {i + 1}/{n}"
    for attempt in range(ATTEMPTS):
        reply = chat(ctx.model, ctx.prefix + WRITE.format(words=budget, min_words=int(budget * 0.8), points=points,
                                                          position=position, covered=cov,
                                                          previous=prev, style=STYLE, feedback=feedback,
                                                          max_background=MAX_BACKGROUND),
                     f"write {label}", think=False)
        log(describe(CALLS[-1]))
        turns, quotes, bg = parse_cited(reply)
        turns, quotes, bg, sect, faults, checked, repairs, stand_ins = settle(ctx, turns, quotes, bg, so_far_lines,
                                                                              budget, label)
        result = dict(title=section["title"], attempt=attempt + 1, words=sum(len(t.split()) for _, t in turns),
                      section_faults=sect, line_checks=checked, repairs=repairs,
                      stand_ins=[dict(stand_in=a, line=b, faults=c) for a, b, c in stand_ins],
                      failed_index=sorted(faults),
                      line_faults=[dict(line=turns[j][1], faults=faults[j]) for j in sorted(faults)])
        log(f"{label} '{section['title']}', attempt {attempt + 1}: {result['words']} words, {len(turns)} lines, "
            f"{repairs} repaired" + (f", {len(stand_ins)} B lines replaced by a stand-in" if stand_ins else "")
            + (f"; {'; '.join(sect)}" if sect else "")
            + (f"; {len(faults)} lines still faulty: " + "; ".join(
                f'"{turns[j][1][:40]}": {faults[j][0]}' for j in sorted(faults)) if faults else "")
            + ("; checks pass" if not sect and not faults else ""))
        score = len(faults) + 5 * len(sect) + (0 if turns else 1000)
        if best is None or score < best[4]:
            best = (turns, quotes, bg, result, score)
        if not sect and not faults:
            edited, refused = edit_section(ctx, turns, quotes, bg, so_far_lines, budget, label, point_numbers)
            result["edit"] = "kept" if edited else f"refused: {refused}"
            if edited:
                result.update(words_before_edit=result["words"], lines_before_edit=len(turns))
                turns, quotes, bg = edited
                result.update(words=sum(len(t.split()) for _, t in turns), lines=len(turns))
            log(f"{label}: edit {result['edit']}"
                + (f", {result['words_before_edit']} -> {result['words']} words" if edited else ""))
            best = (turns, quotes, bg, result, 0)
            break
        feedback = "A previous draft of this section had these faults; avoid them:\n"
        if not turns and CALLS[-1].get("done_reason") == "length":
            feedback += "- the reasoning ran so long that no dialogue was written. Reason briefly, then write.\n"
        feedback += "".join(f"- {p}\n" for p in sect)
        feedback += "".join(f'- the line "{turns[j][1][:60]}": {"; ".join(faults[j])}\n' for j in sorted(faults))
        feedback += ("Copy each SOURCE exactly from the document, let A say only what its SOURCE says, in A's own "
                     "words, let B only ask and restate, write each point once, keep to the word budget, and write "
                     "symbols as words.\n")
    turns, quotes, bg, result, _ = best
    if turns and result["section_faults"] and not result["line_checks"]:
        # A version with a fault no line repair can mend was never checked line by line; check it now, so
        # the review sheet never shows an unchecked line as if it had passed.
        faults = line_faults(ctx, turns, quotes, bg, so_far_lines)
        fails, checked = line_checks(ctx, turns, quotes, bg, so_far_lines, f"{label} kept", skip=set(faults))
        for j, reason in fails:
            faults.setdefault(j, []).append(f"the model's check: {reason}")
        result.update(line_checks=checked, failed_index=sorted(faults),
                      line_faults=[dict(line=turns[j][1], faults=faults[j]) for j in sorted(faults)])
    runs = copy_runs(turns, quotes)
    result.update(copied_runs=runs, longest_copied=max(runs, default=0),
                  background_lines=sum(1 for (h, _), qs, term in zip(turns, quotes, bg) if h == "A" and term and not qs))
    log(f"{label}: longest run copied from a passage {result['longest_copied']} words; "
        f"{sum(r > COPY_REPORT for r in runs)} of {len(runs)} cited lines over {COPY_REPORT}; "
        f"{result['background_lines']} background lines")
    return turns, quotes, bg, result


def review_sheet(sections_out, model):
    out = ["# Review sheet", "", f"Model: {model}. Each line of A's is followed by the passage it cites, which is "
           "not spoken. Read the passage, not the line, as the claim; a line marked with a fault did not pass its "
           f"checks. A background line explains a general term and cites nothing; a line copying more than "
           f"{COPY_REPORT} words in a row from its passage is marked.", ""]
    for title, turns, quotes, bg, result in sections_out:
        out += [f"## {title}", ""]
        faults = dict(zip(result.get("failed_index", []), (f["faults"] for f in result.get("line_faults", []))))
        for j, ((h, t), qs, term) in enumerate(zip(turns, quotes, bg)):
            out.append(f"**{h}:** {t}")
            out += [f"> {q}" for q in qs]
            if term and not qs:
                out.append(f"> background: {term}")
            run = copied_run(t, qs) if qs else 0
            if run > COPY_REPORT:
                out.append(f"> copies {run} words in a row from its passage")
            out += [f"> **fault:** {r}" for r in faults.get(j, [])]
            out.append("")
        for s in result.get("stand_ins", []):
            out += [f'B line replaced by "{s["stand_in"]}" after failing its checks: "{s["line"]}" '
                    f'({"; ".join(s["faults"])})', ""]
        if result.get("section_faults"):
            out += ["Faults in this section as a whole: " + "; ".join(result["section_faults"]), ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(prog="podcast-script", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="the document: .tex (or a folder holding one), .pdf, .md or .txt")
    ap.add_argument("-o", "--output", required=True, help="script to write, for `podcast`")
    ap.add_argument("--minutes", type=float, default=15, help="target episode length (default 15)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default {DEFAULT_MODEL})")
    ap.add_argument("--parallel", type=int, default=4,
                    help="line checks sent at once (default 4; Ollama runs them together only if "
                         "OLLAMA_NUM_PARALLEL allows)")
    ap.add_argument("--max-copied", type=int, default=12, metavar="N",
                    help="repair an A line that copies more than N words in a row from its passage "
                         "(default 12; 0 measures and reports only)")
    a = ap.parse_args()

    check_model(a.model)
    from pdf2audio.podcast import load_say

    started = time.time()
    source = read_source(a.source)
    numbers, words = source_terms(source)
    ctx = SimpleNamespace(model=a.model, prefix=PREFIX.format(source=source), source=source, src_norm=norm_text(source),
                          numbers=numbers, words=words, keep=set(load_say()), parallel=max(1, a.parallel),
                          max_copied=max(0, a.max_copied), cache={}, used_passages=[], explained={})
    target = int(a.minutes * WORDS_PER_MINUTE)
    log(f"source: {len(source.split())} words; planning about {target} words")
    meta = plan(ctx, target)
    sections = meta["sections"]
    log(f"plan: {len(sections)} sections: " + "; ".join(s["title"] for s in sections))
    opening, closing = frame(meta, ctx.src_norm)

    so_far = [f"{h}: {t}" for h, t in opening]
    lines = [f"{h}: {spoken(t, ctx.keep)}" for h, t in opening] + [""]
    report, done, previous = [], [], "\n".join(so_far)
    for i, section in enumerate(sections):
        covered = [p for s in sections[:i] for p in s["points"]]
        later = [p for s in sections[i + 1:] for p in s["points"]]
        turns, quotes, bg, result = write_section(ctx, section, i, len(sections), previous, so_far, covered, later)
        report.append(result)
        done.append((section["title"], turns, quotes, bg, result))
        for (h, _), qs, term in zip(turns, quotes, bg):
            if h == "A":
                ctx.used_passages += qs
                if term and not qs:
                    ctx.explained[term_key(term)] = term
        so_far += [f"# {section['title']}"] + [f"{h}: {t}" for h, t in turns]
        previous = "\n".join(f"{h}: {t}" for h, t in turns[-4:])
        lines.append(f"# {section['title']}")
        lines += [f"{h}: {spoken(t, ctx.keep)}" for h, t in turns]
        lines.append("")
    last = next((line[0] for line in reversed(so_far) if line[:2] in ("A:", "B:")), "B")
    lines[-1:] = [f"{h}: {spoken(t, ctx.keep)}" for h, t in closing_after(last, closing)] + [""]

    out = Path(a.output).expanduser()
    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_name(out.name + ".review.md").write_text(review_sheet(done, a.model), encoding="utf-8")
    failed = [r["title"] for r in report if r["section_faults"] or r["failed_index"]]
    runs = [x for r in report for x in r["copied_runs"]]
    copying = dict(cited_lines=len(runs), over_report_limit=sum(x > COPY_REPORT for x in runs),
                   longest=max(runs, default=0), median=sorted(runs)[len(runs) // 2] if runs else 0)
    out.with_name(out.name + ".check.json").write_text(json.dumps(
        dict(source=str(a.source), model=a.model, title=meta.get("title"), authors=meta.get("authors"),
             max_copied=ctx.max_copied, plan=sections, copying=copying, sections=report, failed=failed), indent=2))
    elapsed = time.time() - started
    out.with_name(out.name + ".calls.json").write_text(json.dumps(
        dict(model=a.model, elapsed_seconds=round(elapsed), totals=call_totals(), calls=CALLS), indent=2))
    total = sum(r["words"] for r in report)
    for kind, t in call_totals().items():
        log(f"{kind}: {t['calls']} calls, {t['seconds']:.0f} s, {t['prompt_tokens']} prompt and "
            f"{t['output_tokens']} output tokens (about {t['thinking_tokens_est']} thinking)")
    log(f"copying: {copying['over_report_limit']} of {copying['cited_lines']} cited lines copy more than "
        f"{COPY_REPORT} words in a row; median {copying['median']}, longest {copying['longest']}")
    log(f"wrote {out}: {total} words, about {total / WORDS_PER_MINUTE:.0f} minutes, in {elapsed / 60:.0f} min"
        + (f"; sections that still fail their checks: {failed}" if failed else "; every section passes its checks"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
