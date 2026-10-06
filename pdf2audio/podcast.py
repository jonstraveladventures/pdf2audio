"""podcast: turn a two-host script into a podcast episode, locally, with MOSS-TTSD.

  podcast script.txt -o episode.m4a
  podcast script.txt -o episode.mp3 --part-words 1200

Script: one turn per line, "A: ..." for the first host and "B: ..." for the second.
A line starting with "#" marks a section; it is not spoken, and a long episode is split
into parts only at section marks; a short tail joins the part before it. "[laugh]" makes a host laugh. Write numbers and
symbols as they should be said.

Voices: two fixed synthetic voices in ~/models/podcast/voices (host_a.wav and
host_b.wav, each with its exact words in a .txt beside it), which MOSS-TTSD copies.

Each part of about ten minutes is rendered in one pass and checked before it is used:
transcribed locally with whisper and compared with its script, and diarised to confirm
two voices. A part that fails is rendered again with a new seed, up to twice. The duller
host's turns are then brightened and set slightly louder than the other's, the parts
are joined, and the episode is levelled to -16 LUFS. Work goes in <output>.parts/, so a
rerun reuses parts that already passed. The exit code is non-zero if any part failed.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

MODEL = "OpenMOSS-Team/MOSS-TTSD-v1.0"
CODEC = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
VOICES = Path.home() / "models/podcast/voices"
TRANSCRIBE = Path.home() / "bin/transcribe"
TAGS = {"A": "[S1]", "B": "[S2]"}
# Words MOSS-TTSD misreads, respelt the way they should sound: one WORD=SPOKEN per line
# in a local file outside the repository, extended with --say.
SAY_FILE = Path.home() / "models/podcast/say.txt"


def load_say(path=SAY_FILE):
    if not path.exists():
        return {}
    pairs = [l.split("=", 1) for l in path.read_text(encoding="utf-8").splitlines() if "=" in l and not l.startswith("#")]
    return {w.strip(): s.strip() for w, s in pairs}
MAX_WER = 0.08
PAUSE_S = 1.0
PRESENCE_DB, OFFSET_DB = 6.0, 2.0  # chosen by ear, 2026-10-06


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------- script ----------

def read_script(path):
    """Sections as [(title, [(host, text), ...]), ...]."""
    sections = [("", [])]
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            sections.append((line.lstrip("#").strip(), []))
            continue
        m = re.match(r"^([AB])\s*:\s*(.+)$", line)
        if not m:
            sys.exit(f"podcast: line {n} is neither 'A: ...', 'B: ...' nor a '#' section mark: {line[:60]}")
        sections[-1][1].append((m.group(1), m.group(2)))
    return [s for s in sections if s[1]]


def part_size(turns):
    return sum(len(t.split()) for _, t in turns)


def plan_parts(sections, part_words):
    """Group whole sections into parts of at most about part_words words.

    A last part under a quarter of part_words is folded into the one before it, as long
    as that part stays within a quarter over part_words."""
    parts, current, count = [], [], 0
    for _, turns in sections:
        n = part_size(turns)
        if current and count + n > part_words:
            parts.append(current)
            current, count = [], 0
        current += turns
        count += n
    if current:
        parts.append(current)
    if len(parts) > 1 and part_size(parts[-1]) < part_words / 4 and \
            part_size(parts[-2]) + part_size(parts[-1]) <= 1.25 * part_words:
        parts[-2:] = [parts[-2] + parts[-1]]
    return parts


def moss_text(turns, say):
    from pdf2audio.moss_text import normalize_text

    out = []
    for host, text in turns:
        for word, spoken in say.items():
            text = re.sub(rf"\b{re.escape(word)}\b", spoken, text)
        # MOSS's clean-up deletes hyphens, joining the words ("co-leads" -> "coleads").
        text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
        out.append(f"{TAGS[host]} {text}")
    return normalize_text("\n".join(out))


# ---------- audio helpers ----------

def write_wav(path, x, sr):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def read_wav(path):
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
    return x, sr


def loudness(x, sr):
    """Integrated loudness, LUFS (EBU R128), measured by ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        write_wav(tmp.name, x, sr)
        err = subprocess.run(["ffmpeg", "-v", "info", "-i", tmp.name, "-af", "ebur128", "-f", "null", "-"],
                             capture_output=True, text=True).stderr
    return float([l for l in err.splitlines() if l.strip().startswith("I:")][-1].split()[1])


def presence(x, sr):
    """Energy in 2-5 kHz relative to 0.1-1 kHz, dB; low values sound muffled."""
    s = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return 10 * np.log10(s[(f > 2000) & (f < 5000)].sum() / s[(f > 100) & (f < 1000)].sum())


def biquad(kind, f0, gain_db, q, sr):
    """RBJ cookbook biquad as one second-order section."""
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    if kind == "peak":
        b = [1 + alpha * a, -2 * cw, 1 - alpha * a]
        den = [1 + alpha / a, -2 * cw, 1 - alpha / a]
    else:
        sa = 2 * np.sqrt(a) * alpha
        if kind == "high":
            b = [a * ((a + 1) + (a - 1) * cw + sa), -2 * a * ((a - 1) + (a + 1) * cw), a * ((a + 1) + (a - 1) * cw - sa)]
            den = [(a + 1) - (a - 1) * cw + sa, 2 * ((a - 1) - (a + 1) * cw), (a + 1) - (a - 1) * cw - sa]
        else:
            b = [a * ((a + 1) - (a - 1) * cw + sa), 2 * a * ((a - 1) - (a + 1) * cw), a * ((a + 1) - (a - 1) * cw - sa)]
            den = [(a + 1) + (a - 1) * cw + sa, -2 * ((a - 1) + (a + 1) * cw), (a + 1) + (a - 1) * cw - sa]
    return np.array(b + den) / den[0]


def compress(x, sr, threshold_db=-28.0, ratio=2.0, attack_s=0.01, release_s=0.12):
    """Gentle feed-forward compressor on an attack/release level follower."""
    level = np.abs(x)
    env = np.empty_like(level)
    ga, gr = np.exp(-1 / (attack_s * sr)), np.exp(-1 / (release_s * sr))
    e = 0.0
    for i, v in enumerate(level):
        g = ga if v > e else gr
        e = g * e + (1 - g) * v
        env[i] = e
    over = np.maximum(20 * np.log10(np.maximum(env, 1e-6)) - threshold_db, 0)
    return x * 10 ** (-over * (1 - 1 / ratio) / 20)


# ---------- checks ----------

def norm_words(text):
    text = text.replace("[laugh]", " ").lower().replace("-", " ").replace("programme", "program")
    text = re.sub(r"\bokay\b", "ok", text)
    text = re.sub(r"\b([a-z]) (?=[a-z]\b)", r"\1", text)  # spelled-out letters: "u s a" -> "usa"
    text = re.sub(r"is(e|es|ed|ing)\b", r"iz\1", text)
    text = re.sub(r"\b[a-z]*\d[\d.,]*(st|nd|rd|th|s)?\b", " ", text)  # numbers: whisper writes digits
    text = re.sub(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
                  r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
                  r"eighty|ninety|hundred|thousand|million|billion|point|first|second|third|fifth|sixth|"
                  r"fifteenth)\b", " ", text)
    text = re.sub(r"\b(um|uh|mm|hm|mmhm|mhm)\b", " ", text)  # fillers, which whisper drops
    return re.findall(r"[a-z0-9']+", text)


def word_errors(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[-1]


def vocabulary(turns):
    """Capitalised names and acronyms, to steer the transcriber."""
    seen = []
    for _, text in turns:
        for w in re.findall(r"\b[A-Z][A-Za-z]+(?:-[A-Z]+)?\b", text):
            if w not in seen and w not in ("A", "I", "So", "And", "But", "The", "That", "This", "It", "Okay", "Right", "Yes", "Yeah", "Oh"):
                seen.append(w)
    return ", ".join(seen[:40])


def transcribe(wav, turns):
    with tempfile.TemporaryDirectory() as d:
        copy = Path(d) / "part.wav"
        copy.write_bytes(Path(wav).read_bytes())
        subprocess.run([str(TRANSCRIBE), "-l", "en", "-p", vocabulary(turns), str(copy)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return (Path(d) / "part.txt").read_text()


_diariser = None


def diarise(x, sr):
    global _diariser
    import torch
    from pyannote.audio import Pipeline
    from scipy.signal import resample_poly

    if _diariser is None:
        _diariser = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
        _diariser.to(torch.device("mps"))
    audio = {"waveform": torch.from_numpy(resample_poly(x, 16000, sr).astype(np.float32)).unsqueeze(0), "sample_rate": 16000}
    try:
        out = _diariser(audio)
    except Exception as e:  # as in the meeting pipeline: fall back to the CPU
        log(f"diarisation on the GPU failed ({type(e).__name__}); using the CPU")
        _diariser.to(torch.device("cpu"))
        out = _diariser(audio)
    ann = getattr(out, "exclusive_speaker_diarization", None) or getattr(out, "speaker_diarization", out)
    return [(s.start, s.end, spk) for s, _, spk in ann.itertracks(yield_label=True)]


def check(x, sr, wav, turns):
    words = sum(len(t.split()) for _, t in turns)
    ref = norm_words(" ".join(t for _, t in turns))
    hyp = norm_words(transcribe(wav, turns))
    wer = word_errors(ref, hyp) / max(len(ref), 1)
    segs = diarise(x, sr)
    talk = {}
    for s, e, spk in segs:
        talk[spk] = talk.get(spk, 0) + e - s
    voices = [k for k, v in talk.items() if v / max(sum(talk.values()), 1e-9) > 0.05]
    secs = len(x) / sr
    rate = words / secs
    problems = []
    if wer > MAX_WER:
        problems.append(f"word error {wer:.1%} over {MAX_WER:.0%}")
    if len(voices) != 2:
        problems.append(f"{len(voices)} voices, not 2")
    if not 1.8 <= rate <= 3.6:
        problems.append(f"{rate:.2f} words a second")
    return dict(seconds=round(secs, 1), wer=round(wer, 4), voices=len(voices), words_per_s=round(rate, 2),
                problems=problems), segs


# ---------- voice fix ----------

def fix_voices(x, sr, segs):
    """Brighten the duller host's turns and set them OFFSET_DB above the other host."""
    from scipy.signal import sosfiltfilt

    spans = {}
    for s, e, spk in segs:
        spans.setdefault(spk, []).append((int(s * sr), int(e * sr)))
    hosts = sorted(spans, key=lambda k: -sum(e - s for s, e in spans[k]))[:2]
    if len(hosts) < 2:
        return x, {}
    clip = lambda sig, h: np.concatenate([sig[s:e] for s, e in spans[h]])
    pres = {h: presence(clip(x, h), sr) for h in hosts}
    dull, other = sorted(hosts, key=lambda h: pres[h])
    mask = np.zeros_like(x)
    for s, e in spans[dull]:
        mask[s:e] = 1.0
    ramp = int(0.03 * sr)
    kernel = np.hanning(2 * ramp + 1)
    mask = np.clip(np.convolve(mask, kernel / kernel.sum(), mode="same"), 0, 1)
    sos = np.vstack([biquad("peak", 3000, PRESENCE_DB, 0.9, sr), biquad("high", 6000, 3.0, 0.7, sr),
                     biquad("low", 180, -2.0, 0.7, sr)])
    bright = compress(sosfiltfilt(sos, x), sr)
    bright *= 10 ** ((loudness(clip(x, other), sr) + OFFSET_DB - loudness(clip(bright, dull), sr)) / 20)
    y = (1 - mask) * x + mask * bright
    return y, dict(presence_before=round(pres[dull], 1), presence_other=round(pres[other], 1),
                   presence_after=round(presence(clip(y, dull), sr), 1))


# ---------- rendering ----------

class Moss:
    def __init__(self):
        import torch
        import torchaudio
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, codec_path=CODEC)
        self.proc.audio_tokenizer = self.proc.audio_tokenizer.to("mps").eval()
        self.model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, attn_implementation="sdpa",
                                               torch_dtype=torch.bfloat16).to("mps").eval()
        self.sr = int(self.proc.model_config.sampling_rate)
        wavs, self.ref_text = [], {}
        for k in ("a", "b"):
            x, sr = read_wav(VOICES / f"host_{k}.wav")
            w = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
            wavs.append(torchaudio.functional.resample(w, sr, self.sr) if sr != self.sr else w)
            self.ref_text[k] = (VOICES / f"host_{k}.txt").read_text().strip()
        self.ref_codes = self.proc.encode_audios_from_wav(wavs, sampling_rate=self.sr)
        self.prompt = self.proc.encode_audios_from_wav([torch.cat(wavs, dim=-1)], sampling_rate=self.sr)[0]

    def render(self, text, words, seed):
        torch = self.torch
        conv = [
            self.proc.build_user_message(text=f"[S1] {self.ref_text['a']} [S2] {self.ref_text['b']} {text}",
                                         reference=self.ref_codes),
            self.proc.build_assistant_message(audio_codes_list=[self.prompt]),
        ]
        torch.manual_seed(seed)
        with torch.no_grad():
            batch = self.proc([conv], mode="continuation")
            out = self.model.generate(input_ids=batch["input_ids"].to("mps"),
                                      attention_mask=batch["attention_mask"].to("mps"),
                                      max_new_tokens=min(7 * words + 600, 16000))
        segs = [a.detach().cpu().to(torch.float32).numpy() for m in self.proc.decode(out) for a in m.audio_codes_list]
        return np.concatenate(segs).astype(np.float64)


def export(x, sr, path):
    fmt = path.suffix.lower().lstrip(".")
    codec = {"m4a": ["-c:a", "aac", "-b:a", "160k"], "mp3": ["-c:a", "libmp3lame", "-b:a", "192k"],
             "wav": ["-c:a", "pcm_s16le"], "flac": ["-c:a", "flac"]}
    if fmt not in codec:
        sys.exit(f"podcast: unsupported output format .{fmt}; use m4a, mp3, wav or flac")
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        write_wav(tmp.name, x, sr)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp.name, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                        "-ar", "44100", *codec[fmt], str(path)], check=True)


def main():
    ap = argparse.ArgumentParser(prog="podcast", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script", help="two-host script: 'A: ...' and 'B: ...' lines, '#' section marks")
    ap.add_argument("-o", "--output", required=True, help="episode file (.m4a, .mp3, .wav or .flac)")
    ap.add_argument("--part-words", type=int, default=1600, help="words per part, about ten minutes (default 1600)")
    ap.add_argument("--say", action="append", default=[], metavar="WORD=SPOKEN", help="respell a word, e.g. Nguyen=Win")
    a = ap.parse_args()

    out = Path(a.output).expanduser().resolve()
    work = out.with_name(out.name + ".parts")
    work.mkdir(parents=True, exist_ok=True)
    say = dict(load_say(), **dict(s.split("=", 1) for s in a.say))
    parts = plan_parts(read_script(a.script), a.part_words)
    total_words = sum(len(t.split()) for p in parts for _, t in p)
    log(f"{len(parts)} part(s), {total_words} words, about {total_words / 2.5 / 60:.0f} minutes")

    moss, report, pieces, sr = None, [], [], None
    for i, turns in enumerate(parts, 1):
        text = moss_text(turns, say)
        words = sum(len(t.split()) for _, t in turns)
        digest = hashlib.sha256((text + json.dumps(say, sort_keys=True)).encode()).hexdigest()[:16]
        raw, info_path = work / f"part{i:02d}.wav", work / f"part{i:02d}.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        if raw.exists() and info.get("digest") == digest and not info.get("problems"):
            log(f"part {i}: reusing the checked render")
            x, sr = read_wav(raw)
            segs = [tuple(s) for s in info["segments"]]
        else:
            moss = moss or Moss()
            sr = moss.sr
            best = None
            for attempt in range(3):
                seed = 42 + 1000 * attempt + i
                t0 = time.time()
                log(f"part {i}: rendering {words} words (seed {seed})")
                x = moss.render(text, words, seed)
                write_wav(raw, x, sr)
                result, segs = check(x, sr, raw, turns)
                result.update(seed=seed, render_s=round(time.time() - t0, 1))
                log(f"part {i}: {result['seconds']:.0f}s, word error {result['wer']:.1%}, {result['voices']} voices"
                    + (f"; problems: {'; '.join(result['problems'])}" if result["problems"] else ""))
                if best is None or len(result["problems"]) < len(best[0]["problems"]) or \
                        (len(result["problems"]) == len(best[0]["problems"]) and result["wer"] < best[0]["wer"]):
                    best = (result, x, segs)
                if not result["problems"]:
                    break
            info, x, segs = best[0], best[1], best[2]
            write_wav(raw, x, sr)
            info.update(digest=digest, segments=segs)
            info_path.write_text(json.dumps(info))
        fixed, voice = fix_voices(x, sr, segs)
        report.append({k: v for k, v in info.items() if k not in ("segments", "digest")} | {"part": i, "voice_fix": voice})
        pieces += [fixed, np.zeros(int(PAUSE_S * sr))]

    episode = np.concatenate(pieces[:-1])
    if np.abs(episode).max() > 0.98:
        episode *= 0.98 / np.abs(episode).max()
    export(episode, sr, out)
    failed = [r["part"] for r in report if r["problems"]]
    summary = dict(output=str(out), minutes=round(len(episode) / sr / 60, 1), parts=report, failed_parts=failed)
    out.with_name(out.name + ".report.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote {out} ({summary['minutes']} minutes)" + (f"; parts that failed their checks: {failed}" if failed else "; every part passed its checks"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
