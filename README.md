# pdf2audio

Convert PDF documents to audio narration, running entirely locally on Apple Silicon Macs (M1 or later). Uses [Kokoro](https://github.com/hexgrad/kokoro) (82M-parameter TTS model), run on the Mac's GPU through [MLX](https://github.com/Blaizzy/mlx-audio) — no cloud APIs, nothing leaves your machine.

## Quick start — Gradio app (drag-and-drop)

If you have [Claude Code](https://claude.ai/claude-code) installed, you can have it build a drag-and-drop GUI for you:

1. **Install system dependencies:**
   ```bash
   brew install espeak-ng ffmpeg
   ```

2. **Clone this repo:**
   ```bash
   git clone https://github.com/jonstraveladventures/pdf2audio.git
   cd pdf2audio
   ```

3. **Ask Claude Code to build the app:**
   ```
   Read CLAUDE_CODE_INSTRUCTIONS.md and build the Gradio app for me.
   Install the Python dependencies first.
   ```

4. **Run:**
   ```bash
   python app.py
   ```
   Opens a browser UI at `http://localhost:7860` where you can drag a PDF, pick a voice, toggle options, and get an MP3.

## Quick start — CLI

You can also install and use this directly as a command-line tool:

```bash
brew install espeak-ng ffmpeg
pip install .
pdf2audio paper.pdf -o paper.mp3
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--voice ID` | `af_heart` | TTS voice (see `--list-voices`) |
| `--speed N` | `1.0` | Playback speed multiplier |
| `--keep-references` | off | Keep bibliography section (stripped by default) |
| `--no-appendices` | off | Stop at the first appendix |
| `--skip-equations` | off | Remove equations entirely (by default each is announced by its number, "Equation 3.6") |
| `--explain-equations` | off | Replace each display equation with a spoken explanation from a local vision model (see below) |
| `--check-equations` | off | With `--explain-equations`, have the model check each explanation against the equation and correct it (about doubles the time) |
| `--llm-model NAME` | `qwen3.5:35b-a3b` | Ollama model used by `--explain-equations` |
| `--skip-captions` | off | Remove figure/table captions |
| `--keep-footnotes` | off | Include footnotes inline |
| `--output-text` | off | Also save cleaned text to `.txt` |
| `--start-page N` | | Start page (1-indexed) |
| `--end-page N` | | End page (1-indexed, inclusive) |
| `--format mp3\|wav` | `mp3` | Output format |
| `--paragraph-pause MS` | `500` | Silence between paragraphs (ms) |
| `--section-pause MS` | `1500` | Silence between sections (ms) |
| `--list-voices` | | Print available voices and exit |

## Short text: `tts`

The package also installs `tts`, which turns short text into an audio file, for example
voice prompts for an app. Kokoro-82M's weights and its text front end are Apache-2.0, so
the audio can ship.

```bash
tts "Welcome back." -o assets/welcome.wav
echo "Timer finished." | tts -o done.m4a --voice bm_george
tts --batch prompts.json --out-dir assets/audio --format wav
```

The output format follows the extension (`wav`, `flac`, `mp3`, `m4a`, `ogg` as Opus).
Leading and trailing silence is trimmed unless `--no-trim` is given; `--rate 48000`
resamples from Kokoro's native 24 kHz. A batch file is JSON (a list of
`{"id", "text", "voice"}` or an `{id: text}` object) or CSV with `id` and `text` columns.

## Two-host episodes: `podcast-script` and `podcast`

`podcast-script` turns a document (PDF, Markdown or text) into a conversation between two
hosts, written by a local model through Ollama. The model plans the episode in sections
and writes each one; every number and name in a section must appear in the source, and a
section that fails is rewritten with the problems pointed out. Numbers and acronyms are
then written out as they are said. The check covers numbers and names only, so read the
script against the source before rendering it.

`podcast` renders the script with MOSS-TTSD, copying two reference voices kept in
`~/models/podcast/voices` (`host_a.wav` and `host_b.wav`, each with its words in a `.txt`
beside it). Each part of about ten minutes is transcribed back with whisper, diarised to
confirm two voices, and rendered again if it fails.

```bash
podcast-script paper.pdf -o script.txt --minutes 15
podcast script.txt -o episode.m4a --say Nguyen=Win
```

A script is one turn per line, `A: ...` or `B: ...`, with `#` lines marking sections; a
long episode is split into parts only at those marks. Respellings for words the voice
model misreads can also go in `~/models/podcast/say.txt`, one `WORD=SPOKEN` per line.

## Requirements

- macOS with Apple Silicon (M1 or later). Other platforms fall back to the PyTorch build of Kokoro.
- Python 3.10–3.12
- `espeak-ng` and `ffmpeg` via Homebrew
- For `--explain-equations` and `podcast-script`: [Ollama](https://ollama.com) with `qwen3.5:35b-a3b` (`ollama pull qwen3.5:35b-a3b`, about 23 GB)

## How it works

1. **Extract** — `pymupdf4llm` converts PDF pages to markdown, then regex-based cleaning strips headers, footers, page numbers, review-draft line numbers, figure axis labels, references (keeping any appendices), and other non-narration content. Each table is replaced by the words "Table omitted."; its caption is kept. Pages where the layout analysis fails, which then prints lines twice (common in review drafts with line numbers), are rebuilt from the PDF's text layer instead.
2. **Synthesise** — Kokoro generates 24 kHz audio for each text segment, on the GPU via MLX on Apple Silicon (about 50× real time on an M5 Max) and via PyTorch elsewhere. 28 English voices available (American and British, male and female).
3. **Explain equations** (optional) — each page with display equations is rendered with the equations outlined and sent to a local vision model through Ollama, which writes a short spoken explanation of each. The narration then says, for example, "Equation 3.6, in words: …". On an M5 Max this adds roughly 20 to 110 seconds per page with equations.
4. **Export** — Segments are concatenated with configurable silence gaps between paragraphs and sections, then exported as MP3 (192 kbps) or WAV.

The first run downloads model weights and voices (~750 MB) automatically from HuggingFace. After that, everything runs fully offline — no data leaves your machine.

## License

AGPL-3.0 — required by the PyMuPDF dependency. See [LICENSE](LICENSE) for details.
