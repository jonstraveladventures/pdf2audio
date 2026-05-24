# Instructions for Claude Code: Build a local PDF-to-Audio app

Paste everything below this line into Claude Code on an M-series Mac (M1–M4).

---

## Task

Build a local, browser-based app that converts PDF documents to MP3 audiobook narrations. The user drags a PDF onto the interface, picks options from toggles, clicks "Convert", and gets a playable/downloadable MP3. Everything runs locally — no API calls, no cloud services.

## Prerequisites to install first

Before writing any code, run these:

```bash
brew install espeak-ng ffmpeg
pip install pymupdf4llm 'kokoro>=0.9.4' soundfile pydub rich gradio
```

Kokoro requires Python >=3.10 and <3.13. Verify with `python --version`. If using conda/miniforge and the version is outside that range, create an env:

```bash
conda create -n pdf2audio python=3.12 -y && conda activate pdf2audio
pip install pymupdf4llm 'kokoro>=0.9.4' soundfile pydub rich gradio
```

## Architecture

Build a single-directory project at `~/pdf2audio/` with these files:

```
pdf2audio/
├── app.py           # Gradio UI — the only file the user runs
├── extractor.py     # PDF text extraction and cleaning
├── synthesiser.py   # TTS wrapper around Kokoro
└── audio.py         # Audio concatenation and export
```

The user launches with `python app.py` and opens `http://localhost:7860` in their browser.

---

## 1. extractor.py — PDF text extraction and cleaning

Uses `pymupdf4llm` to extract markdown from a PDF, then cleans it for narration.

### Key API

```python
import pymupdf
import pymupdf4llm

# Get total page count
doc = pymupdf.open("file.pdf")
total = len(doc)
doc.close()

# Extract markdown (pages is 0-indexed list)
md = pymupdf4llm.to_markdown("file.pdf", pages=[0, 1, 2])
```

### Data structure

```python
from dataclasses import dataclass

@dataclass
class TextSegment:
    text: str
    segment_type: str   # 'heading' or 'paragraph'
    heading_level: int = 0
```

### Cleaning pipeline

The function `extract_and_clean(pdf_path, start_page, end_page, skip_references, skip_equations, skip_captions, keep_footnotes)` should:

1. Extract markdown from the PDF (convert 1-indexed inclusive page range to 0-indexed list for pymupdf4llm).
2. Clean the markdown:
   - Remove image references: `![...](...)` patterns
   - Remove horizontal rules (`---`, `***`) which are page separators
   - Handle equations: if `skip_equations` is True, remove `$$...$$` and `$...$` entirely. Otherwise replace them with the spoken word " [equation] ".
   - If `skip_captions`: remove lines matching `Figure N:`, `Fig. N:`, `Table N:`, `Tab. N:` etc.
   - If not `keep_footnotes`: remove `[N]` reference markers and footnote definition lines.
   - If `skip_references`: find the References/Bibliography section (look for heading patterns like `## References`, bold `**References**`, or plain-text `References` / `REFERENCES` on its own line) and strip everything from there to the end of the document (or to the next heading of equal/higher level).
   - Remove standalone page numbers (lines that are just 1–4 digits).
   - Remove URLs.
   - Strip markdown link syntax but keep the link text: `[text](url)` → `text`.
   - Strip bold/italic markers (`*`, `_`).
   - Strip markdown table formatting (separator rows of `|---|---|`, pipe characters) but keep cell text.
   - Collapse 3+ consecutive newlines into 2.
3. Parse the cleaned text into `TextSegment` objects:
   - Split on double-newlines.
   - Lines starting with `#` are headings (record the level from the number of `#` chars).
   - Everything else is a paragraph (join whitespace into a single line, skip if 2 chars or fewer).

Return `(segments: list[TextSegment], cleaned_text: str)`.

---

## 2. synthesiser.py — TTS wrapper

Uses Kokoro (https://github.com/hexgrad/kokoro), an 82M-parameter local TTS model.

### Critical: Set MPS fallback BEFORE importing

```python
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from kokoro import KPipeline
```

The env var must be set before torch is imported or MPS operations will fail on Apple Silicon.

### Kokoro API

```python
pipeline = KPipeline(lang_code='a')  # 'a' = American English, 'b' = British English

generator = pipeline(
    text,
    voice='af_heart',    # voice ID string
    speed=1.0,           # speed multiplier
    split_pattern=r'\n+'
)

for graphemes, phonemes, audio in generator:
    # audio is a torch tensor or numpy array at 24000 Hz sample rate
    # may need .cpu().numpy() if it's a torch tensor
```

### Voice list

Include ALL of these as a dict mapping voice_id → description:

**American English (20 voices):**
af_heart, af_alloy, af_aoede, af_bella, af_jessica, af_kore, af_nicole, af_nova, af_river, af_sarah, af_sky, am_adam, am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx, am_puck, am_santa

**British English (8 voices):**
bf_alice, bf_emma, bf_isabella, bf_lily, bm_daniel, bm_fable, bm_george, bm_lewis

Naming convention: first letter = language (a=American, b=British), second letter = gender (f=female, m=male), then underscore + name. Use this to generate human-readable descriptions like "American Female - Heart".

### Language code derivation

The first character of the voice ID gives the lang_code to pass to `KPipeline(lang_code=...)`.

### Synthesiser class

```python
class Synthesiser:
    def __init__(self, voice='af_heart', speed=1.0):
        # Create pipeline with correct lang_code derived from voice
        # Store voice and speed

    def synthesise(self, text: str) -> np.ndarray:
        # Feed text to pipeline, collect all audio chunks
        # Handle both torch tensors (.cpu().numpy()) and numpy arrays
        # Flatten and concatenate into a single float32 array
        # Return empty array if text is empty or no audio generated
```

Sample rate is always 24000 Hz.

---

## 3. audio.py — Audio concatenation and export

### numpy to pydub conversion

```python
from pydub import AudioSegment
import numpy as np

SAMPLE_RATE = 24000

def numpy_to_segment(audio_array, sample_rate=SAMPLE_RATE):
    audio_int16 = np.clip(audio_array * 32767, -32768, 32767).astype(np.int16)
    return AudioSegment(
        data=audio_int16.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )
```

### Concatenation

Join a list of AudioSegments with silence gaps:
- Between paragraphs: configurable, default 500ms
- Before/after headings: configurable, default 1500ms

### Export

Export to MP3 at 192kbps or WAV using `AudioSegment.export()`.

---

## 4. app.py — Gradio UI

This is the main file. The user runs `python app.py`.

### UI Layout

Use `gr.Blocks` for the layout. The interface should have:

**Top section — Input:**
- `gr.File` for PDF upload (drag-and-drop), file_types=[".pdf"], label="Drop your PDF here"

**Middle section — Options (use `gr.Row` and `gr.Column` to lay these out neatly):**
- `gr.Dropdown` for voice selection — populate with all 28 English voices, display as "American Female - Heart (af_heart)" etc., default to "af_heart"
- `gr.Slider` for speed — min 0.5, max 2.0, step 0.1, default 1.0
- `gr.Slider` for start page — min 1, max 999, step 1, default 1, label="Start page"
- `gr.Slider` for end page — min 1, max 999, step 1, default 999, label="End page (use 999 for all)"
- `gr.Checkbox` for "Skip references section" — default True
- `gr.Checkbox` for "Skip equations" — default False (they get replaced with "[equation]" when unchecked)
- `gr.Checkbox` for "Skip figure/table captions" — default False
- `gr.Checkbox` for "Keep footnotes" — default False
- `gr.Radio` for output format — choices ["mp3", "wav"], default "mp3"

**Bottom section — Output:**
- `gr.Audio` for playback of the result (type="filepath")
- `gr.File` for downloading the output file
- `gr.Textbox` for the cleaned text (collapsible/accordion, so it doesn't dominate the page)
- A status/progress `gr.Textbox` showing what step is happening

**Convert button** between options and output.

### Processing function

The convert button triggers a function that:
1. Validates the uploaded file exists and is a PDF.
2. Calls `extract_and_clean(...)` with the selected options.
3. Instantiates `Synthesiser(voice=..., speed=...)`.
4. Loops through segments, synthesising each one. Update the status textbox with progress like "Synthesising segment 5/42...".
5. Concatenates audio with `concatenate_segments(...)`.
6. Exports to a temp file (use `tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)`).
7. Returns the audio file path (for playback), the file path (for download), and the cleaned text.

Use `gr.Progress` if possible to show a progress bar during synthesis.

### Error handling

Wrap the whole pipeline in try/except. If a single segment fails TTS, log it and continue. Show errors in the status textbox, not as crashes.

### Launch

```python
if __name__ == "__main__":
    demo.launch()  # opens browser automatically at localhost:7860
```

---

## Important notes

- The first run will be slow — Kokoro downloads its model weights (~330 MB) on first use. This is automatic via HuggingFace Hub.
- Test with a short PDF first (even a 1-page document) to verify everything works before trying a long document.
- For Apple Silicon GPU acceleration, the env var `PYTORCH_ENABLE_MPS_FALLBACK=1` MUST be set before torch is imported. Set it at the top of synthesiser.py using `os.environ.setdefault(...)`.
- The audio sample rate from Kokoro is always 24000 Hz.
- Gradio will create a local web server. The app is entirely local — nothing leaves the machine.
- If you get an error about espeak-ng not found, make sure `brew install espeak-ng` was run.
- If you get ffmpeg errors on MP3 export, make sure `brew install ffmpeg` was run.
