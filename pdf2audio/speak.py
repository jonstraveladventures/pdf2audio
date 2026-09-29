"""tts: turn short text into speech audio files, locally, with Kokoro on MLX.

Kokoro-82M's weights and its text front end are Apache-2.0, so the audio can ship in
apps. Nothing leaves this machine.

  tts "Welcome back." -o assets/welcome.wav
  tts -f script.txt -o intro.mp3 --voice bm_george
  echo "Timer finished." | tts -o done.m4a
  tts --batch prompts.json --out-dir assets/audio --format wav
  tts --list-voices

Batch files: JSON, either a list of {"id", "text"[, "voice"]} objects or an {id: text}
object; or CSV with id and text columns (and an optional voice column). One file per
entry, named <id>.<format>. The format follows the output file's extension.
"""

import argparse
import contextlib
import csv
import io
import json
import logging
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

# Finder and some shells start with a bare PATH; ffmpeg (MP3, M4A, OGG) lives here.
os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

FORMATS = {"wav", "mp3", "m4a", "ogg", "flac"}


def load_synthesiser(voice, speed):
    # The model and its libraries print progress and warnings; keep the output clean.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from pdf2audio.synthesiser import Synthesiser, SAMPLE_RATE, VOICES

        if voice not in VOICES:
            sys.exit(f"tts: unknown voice '{voice}'. See tts --list-voices.")
        return Synthesiser(voice=voice, speed=speed), SAMPLE_RATE


def trim(audio, rate, threshold=0.01, pad_ms=30):
    """Cut leading and trailing silence, keeping a few milliseconds of padding."""
    import numpy as np

    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        return audio
    pad = int(rate * pad_ms / 1000)
    return audio[max(0, loud[0] - pad) : loud[-1] + pad]


def write(audio, rate, path, out_rate=None):
    """Write audio to path; the format follows the extension. Returns the duration."""
    import numpy as np
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = path.suffix.lower().lstrip(".")
    if fmt not in FORMATS:
        sys.exit(f"tts: unsupported format '.{fmt}'; use one of {', '.join(sorted(FORMATS))}")
    if fmt in ("wav", "flac") and not out_rate:
        sf.write(path, audio, rate)
    else:
        from pydub import AudioSegment

        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        seg = AudioSegment(pcm.tobytes(), frame_rate=rate, sample_width=2, channels=1)
        if out_rate:
            seg = seg.set_frame_rate(out_rate)
        # .m4a is AAC in an MP4 container; .ogg is Opus in Ogg (Homebrew's ffmpeg has no
        # Vorbis encoder, and Opus is the better codec for apps anyway).
        container = {"m4a": "ipod"}.get(fmt, fmt)
        options = {"mp3": {"bitrate": "192k"}, "m4a": {"bitrate": "192k"}, "ogg": {"codec": "libopus", "bitrate": "96k"}}
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                seg.export(path, format=container, **options.get(fmt, {}))
            except Exception as e:
                path.unlink(missing_ok=True)
                sys.exit(f"tts: ffmpeg could not write {path.name}: {str(e).splitlines()[0]}")
    return len(audio) / rate


def speak(synth, rate, text, do_trim):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        audio = synth.synthesise(" ".join(text.split()))
    if audio.size == 0:
        sys.exit("tts: no audio was produced (empty text?)")
    return trim(audio, rate) if do_trim else audio


def read_batch(path):
    """Entries from a JSON or CSV batch file: [(id, text, voice or None)]."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            return [(r["id"], r["text"], r.get("voice") or None) for r in csv.DictReader(f)]
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [(k, v, None) for k, v in data.items()]
    return [(d["id"], d["text"], d.get("voice")) for d in data]


def safe_name(entry_id):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(entry_id)).strip("._") or "clip"


def main():
    ap = argparse.ArgumentParser(prog="tts", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="text to speak (or use -f, or pipe it in)")
    ap.add_argument("-f", "--file", help="read the text from a file")
    ap.add_argument("-o", "--output", help="output file; the extension sets the format (default speech.wav)")
    ap.add_argument("--batch", help="JSON or CSV of id/text entries; writes one file per entry")
    ap.add_argument("--out-dir", default=".", help="folder for --batch output (default: current)")
    ap.add_argument("--format", default="wav", help="format for --batch output (default wav)")
    ap.add_argument("--voice", default="af_heart", help="Kokoro voice (default af_heart)")
    ap.add_argument("--speed", type=float, default=1.0, help="speaking speed (default 1.0)")
    ap.add_argument("--rate", type=int, help="resample to this rate, e.g. 44100 or 48000 (native is 24000)")
    ap.add_argument("--no-trim", action="store_true", help="keep Kokoro's leading and trailing silence")
    ap.add_argument("--play", action="store_true", help="play the result with afplay")
    ap.add_argument("--list-voices", action="store_true", help="list voices and exit")
    args = ap.parse_args()

    if args.list_voices:
        from pdf2audio.synthesiser import VOICES

        for vid, desc in sorted(VOICES.items()):
            print(f"{vid:14s} {desc}")
        return

    if args.batch:
        entries = read_batch(args.batch)
        fmt = args.format.lower().lstrip(".")
        synths = {}
        for entry_id, text, voice in entries:
            voice = voice or args.voice
            if voice not in synths:
                synths[voice] = load_synthesiser(voice, args.speed)
            synth, rate = synths[voice]
            path = Path(args.out_dir) / f"{safe_name(entry_id)}.{fmt}"
            secs = write(speak(synth, rate, text, not args.no_trim), rate, path, args.rate)
            print(f"{path}\t{secs:.2f}s")
        return

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text is not None:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        ap.error("give the text as an argument, with -f, or on stdin")
    if not text.strip():
        sys.exit("tts: the text is empty")

    synth, rate = load_synthesiser(args.voice, args.speed)
    path = Path(args.output or "speech.wav")
    secs = write(speak(synth, rate, text, not args.no_trim), rate, path, args.rate)
    print(f"{path}\t{secs:.2f}s")
    if args.play:
        subprocess.run(["/usr/bin/afplay", str(path)], check=False)


if __name__ == "__main__":
    main()
