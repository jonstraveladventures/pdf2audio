# pdf2audio

Convert PDF documents to audio narration, running entirely locally on Apple Silicon Macs (M1–M4). Uses [Kokoro](https://github.com/hexgrad/kokoro) (82M-parameter TTS model) — no cloud APIs, nothing leaves your machine.

## Quick start — Gradio app (drag-and-drop)

If you have [Claude Code](https://claude.ai/claude-code) installed, you can have it build a drag-and-drop GUI for you:

1. **Install system dependencies:**
   ```bash
   brew install espeak-ng ffmpeg
   ```

2. **Clone this repo:**
   ```bash
   git clone https://github.com/jonathanshock/pdf2audio.git
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
| `--skip-equations` | off | Remove equations entirely (replaced with "equation" by default) |
| `--skip-captions` | off | Remove figure/table captions |
| `--keep-footnotes` | off | Include footnotes inline |
| `--output-text` | off | Also save cleaned text to `.txt` |
| `--start-page N` | | Start page (1-indexed) |
| `--end-page N` | | End page (1-indexed, inclusive) |
| `--format mp3\|wav` | `mp3` | Output format |
| `--paragraph-pause MS` | `500` | Silence between paragraphs (ms) |
| `--section-pause MS` | `1500` | Silence between sections (ms) |
| `--list-voices` | | Print available voices and exit |

## Requirements

- macOS with Apple Silicon (M1–M4)
- Python 3.10–3.12 (Kokoro does not yet support 3.13)
- `espeak-ng` and `ffmpeg` via Homebrew

## How it works

1. **Extract** — `pymupdf4llm` converts PDF pages to markdown, then regex-based cleaning strips headers, footers, page numbers, equations, references, and other non-narration content.
2. **Synthesise** — Kokoro generates 24 kHz audio for each text segment. 28 English voices available (American and British, male and female).
3. **Export** — Segments are concatenated with configurable silence gaps between paragraphs and sections, then exported as MP3 (192 kbps) or WAV.

The first run downloads model weights (~330 MB) automatically from HuggingFace. After that, everything runs fully offline — no data leaves your machine.

## License

AGPL-3.0 — required by the PyMuPDF dependency. See [LICENSE](LICENSE) for details.
