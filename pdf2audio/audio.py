"""Audio concatenation and export."""

import numpy as np
from pydub import AudioSegment

SAMPLE_RATE = 24000


def numpy_to_segment(audio_array, sample_rate=SAMPLE_RATE):
    """Convert a float32 numpy array to a pydub AudioSegment."""
    audio_int16 = np.clip(audio_array * 32767, -32768, 32767).astype(np.int16)
    return AudioSegment(
        data=audio_int16.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )


def concatenate_segments(audio_parts, segment_types, paragraph_pause_ms=500, section_pause_ms=1500):
    """Join AudioSegments with pauses that depend on segment type."""
    if not audio_parts:
        return AudioSegment.silent(duration=0, frame_rate=SAMPLE_RATE)

    silence_para = AudioSegment.silent(duration=paragraph_pause_ms, frame_rate=SAMPLE_RATE)
    silence_section = AudioSegment.silent(duration=section_pause_ms, frame_rate=SAMPLE_RATE)

    result = audio_parts[0]
    for i in range(1, len(audio_parts)):
        cur = segment_types[i] if i < len(segment_types) else "paragraph"
        prev = segment_types[i - 1] if (i - 1) < len(segment_types) else "paragraph"
        gap = silence_section if cur == "heading" or prev == "heading" else silence_para
        result = result + gap + audio_parts[i]

    return result


def export_audio(audio_segment, output_path, fmt="mp3", bitrate="192k"):
    params = {"format": fmt}
    if fmt == "mp3":
        params["bitrate"] = bitrate
    audio_segment.export(output_path, **params)
