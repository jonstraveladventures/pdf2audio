#!/usr/bin/env python3
"""Gradio UI for pdf2audio — drag-and-drop PDF to audio conversion."""

import os
import tempfile

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gradio as gr
import numpy as np

from pdf2audio.extractor import extract_and_clean
from pdf2audio.synthesiser import Synthesiser, VOICES, SAMPLE_RATE
from pdf2audio.audio import numpy_to_segment, concatenate_segments, export_audio

VOICE_CHOICES = [(f"{desc} ({vid})", vid) for vid, desc in sorted(VOICES.items())]


def convert(
    pdf_file,
    voice_id,
    speed,
    start_page,
    end_page,
    skip_references,
    skip_equations,
    skip_captions,
    keep_footnotes,
    output_format,
    progress=gr.Progress(track_tqdm=False),
):
    if pdf_file is None:
        raise gr.Error("Please upload a PDF file.")

    pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name

    # --- Extract ---
    progress(0, desc="Extracting text from PDF...")
    end_pg = None if end_page >= 999 else int(end_page)
    start_pg = int(start_page)

    segments, cleaned_text = extract_and_clean(
        pdf_path,
        start_page=start_pg,
        end_page=end_pg,
        skip_references=skip_references,
        skip_equations=skip_equations,
        skip_captions=skip_captions,
        keep_footnotes=keep_footnotes,
    )

    if not segments:
        raise gr.Error("No text could be extracted from the PDF.")

    # --- Synthesise ---
    progress(0, desc=f"Loading TTS model (voice: {voice_id})...")
    synth = Synthesiser(voice=voice_id, speed=speed)

    audio_parts = []
    seg_types = []
    total = len(segments)

    for i, seg in enumerate(segments):
        progress((i + 1) / total, desc=f"Synthesising segment {i + 1}/{total}...")
        try:
            arr = synth.synthesise(seg.text)
            if arr.size > 0:
                audio_parts.append(numpy_to_segment(arr))
                seg_types.append(seg.segment_type)
        except Exception as e:
            print(f"Warning: skipped segment {i + 1}: {e}")

    if not audio_parts:
        raise gr.Error("No audio was generated — all segments failed.")

    # --- Concatenate and export ---
    progress(1, desc="Exporting audio...")
    final = concatenate_segments(audio_parts, seg_types)

    suffix = f".{output_format}"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    export_audio(final, tmp.name, fmt=output_format)

    duration = len(final) / 1000
    minutes, seconds = int(duration // 60), int(duration % 60)
    size_kb = os.path.getsize(tmp.name) / 1024
    status = f"Done — {minutes}m {seconds}s, {size_kb:.0f} KB, {total} segments, voice: {voice_id}"

    return tmp.name, tmp.name, cleaned_text, status


with gr.Blocks(title="pdf2audio") as demo:
    gr.Markdown("# pdf2audio\nDrop a PDF, pick a voice, get an audiobook. Runs entirely locally.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Drop your PDF here", file_types=[".pdf"])

            voice = gr.Dropdown(
                choices=VOICE_CHOICES,
                value="af_heart",
                label="Voice",
            )
            speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Speed")

            with gr.Row():
                start_page = gr.Number(value=1, minimum=1, label="Start page", precision=0)
                end_page = gr.Number(value=999, minimum=1, label="End page (999 = all)", precision=0)

            with gr.Row():
                skip_refs = gr.Checkbox(value=True, label="Skip references")
                skip_eqs = gr.Checkbox(value=False, label="Skip equations")
            with gr.Row():
                skip_caps = gr.Checkbox(value=False, label="Skip captions")
                keep_fn = gr.Checkbox(value=False, label="Keep footnotes")

            fmt = gr.Radio(["mp3", "wav"], value="mp3", label="Format")
            convert_btn = gr.Button("Convert", variant="primary", size="lg")

        with gr.Column(scale=1):
            status_box = gr.Textbox(label="Status", interactive=False)
            audio_out = gr.Audio(label="Preview", type="filepath")
            file_out = gr.File(label="Download")
            with gr.Accordion("Cleaned text", open=False):
                text_out = gr.Textbox(lines=12, interactive=False, show_label=False)

    convert_btn.click(
        fn=convert,
        inputs=[pdf_input, voice, speed, start_page, end_page, skip_refs, skip_eqs, skip_caps, keep_fn, fmt],
        outputs=[audio_out, file_out, text_out, status_box],
    )

if __name__ == "__main__":
    demo.launch()
