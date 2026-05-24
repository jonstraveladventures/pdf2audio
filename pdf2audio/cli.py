#!/usr/bin/env python3
"""Convert PDF documents to audio narration using local TTS."""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from pdf2audio.extractor import extract_and_clean
from pdf2audio.synthesiser import Synthesiser, VOICES
from pdf2audio.audio import numpy_to_segment, concatenate_segments, export_audio

console = Console()


def build_parser():
    p = argparse.ArgumentParser(
        prog="pdf2audio",
        description="Convert PDF documents to audio narration using local TTS.",
    )
    p.add_argument("input", nargs="?", help="Input PDF file path")
    p.add_argument("-o", "--output", help="Output audio file (default: <input>.mp3)")
    p.add_argument("--voice", default="af_heart", help="Voice ID (default: af_heart)")
    p.add_argument("--speed", type=float, default=1.0, help="Speed multiplier (default: 1.0)")
    p.add_argument("--list-voices", action="store_true", help="List available voices and exit")
    p.add_argument("--keep-references", action="store_true", help="Keep bibliography section (stripped by default)")
    p.add_argument("--skip-equations", action="store_true", help="Remove equations entirely (default: replace with 'equation')")
    p.add_argument("--skip-captions", action="store_true", help="Remove figure/table captions")
    p.add_argument("--keep-footnotes", action="store_true", help="Include footnotes inline (skipped by default)")
    p.add_argument("--output-text", action="store_true", help="Save cleaned text to a .txt file")
    p.add_argument("--start-page", type=int, help="Start page (1-indexed)")
    p.add_argument("--end-page", type=int, help="End page (1-indexed, inclusive)")
    p.add_argument("--format", choices=["mp3", "wav"], default="mp3", help="Output format (default: mp3)")
    p.add_argument("--paragraph-pause", type=int, default=500, help="Pause between paragraphs in ms (default: 500)")
    p.add_argument("--section-pause", type=int, default=1500, help="Pause between sections in ms (default: 1500)")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list_voices:
        console.print("\n[bold]Available voices:[/bold]\n")
        for vid, desc in sorted(VOICES.items()):
            console.print(f"  {vid:20s} {desc}")
        console.print()
        return

    if args.input is None:
        parser.print_help()
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Error:[/red] file not found: {input_path}")
        sys.exit(1)
    if input_path.suffix.lower() != ".pdf":
        console.print("[red]Error:[/red] input must be a PDF file")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_suffix(f".{args.format}")

    # --- Extract ---
    console.print(f"\n[bold]Processing:[/bold] {input_path.name}")
    console.print("[dim]Extracting text from PDF...[/dim]")

    segments, cleaned_text = extract_and_clean(
        str(input_path),
        start_page=args.start_page,
        end_page=args.end_page,
        skip_references=not args.keep_references,
        skip_equations=args.skip_equations,
        skip_captions=args.skip_captions,
        keep_footnotes=args.keep_footnotes,
    )

    if not segments:
        console.print("[red]Error:[/red] no text extracted from PDF")
        sys.exit(1)

    console.print(f"  {len(segments)} text segments extracted")

    if args.output_text:
        txt_path = output_path.with_suffix(".txt")
        txt_path.write_text(cleaned_text, encoding="utf-8")
        console.print(f"  Saved cleaned text to {txt_path}")

    # --- Synthesise ---
    console.print(f"[dim]Loading TTS model (voice: {args.voice})...[/dim]")
    synth = Synthesiser(voice=args.voice, speed=args.speed)

    audio_parts = []
    seg_types = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Synthesising", total=len(segments))
        for seg in segments:
            try:
                arr = synth.synthesise(seg.text)
                if arr.size > 0:
                    audio_parts.append(numpy_to_segment(arr))
                    seg_types.append(seg.segment_type)
            except Exception as e:
                console.print(f"  [yellow]Warning:[/yellow] skipped segment: {e}")
            progress.update(task, advance=1)

    if not audio_parts:
        console.print("[red]Error:[/red] no audio generated")
        sys.exit(1)

    # --- Export ---
    console.print("[dim]Concatenating audio...[/dim]")
    final = concatenate_segments(
        audio_parts,
        seg_types,
        paragraph_pause_ms=args.paragraph_pause,
        section_pause_ms=args.section_pause,
    )

    duration = len(final) / 1000
    console.print(f"[dim]Exporting {args.format.upper()} ({int(duration // 60)}m {int(duration % 60)}s)...[/dim]")
    export_audio(final, str(output_path), fmt=args.format)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    console.print(f"\n[bold green]Done![/bold green] {output_path} ({size_mb:.1f} MB)\n")


if __name__ == "__main__":
    main()
