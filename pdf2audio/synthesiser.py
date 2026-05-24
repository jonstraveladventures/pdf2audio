"""TTS wrapper using Kokoro."""

import os
import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from kokoro import KPipeline

SAMPLE_RATE = 24000

VOICES = {
    # American English
    "af_heart": "American Female - Heart (default)",
    "af_alloy": "American Female - Alloy",
    "af_aoede": "American Female - Aoede",
    "af_bella": "American Female - Bella",
    "af_jessica": "American Female - Jessica",
    "af_kore": "American Female - Kore",
    "af_nicole": "American Female - Nicole",
    "af_nova": "American Female - Nova",
    "af_river": "American Female - River",
    "af_sarah": "American Female - Sarah",
    "af_sky": "American Female - Sky",
    "am_adam": "American Male - Adam",
    "am_echo": "American Male - Echo",
    "am_eric": "American Male - Eric",
    "am_fenrir": "American Male - Fenrir",
    "am_liam": "American Male - Liam",
    "am_michael": "American Male - Michael",
    "am_onyx": "American Male - Onyx",
    "am_puck": "American Male - Puck",
    "am_santa": "American Male - Santa",
    # British English
    "bf_alice": "British Female - Alice",
    "bf_emma": "British Female - Emma",
    "bf_isabella": "British Female - Isabella",
    "bf_lily": "British Female - Lily",
    "bm_daniel": "British Male - Daniel",
    "bm_fable": "British Male - Fable",
    "bm_george": "British Male - George",
    "bm_lewis": "British Male - Lewis",
    # Spanish
    "ef_dora": "Spanish Female - Dora",
    "em_alex": "Spanish Male - Alex",
    "em_santa": "Spanish Male - Santa",
    # French
    "ff_siwis": "French Female - Siwis",
    # Hindi
    "hf_alpha": "Hindi Female - Alpha",
    "hf_beta": "Hindi Female - Beta",
    "hm_omega": "Hindi Male - Omega",
    "hm_psi": "Hindi Male - Psi",
    # Italian
    "if_sara": "Italian Female - Sara",
    "im_nicola": "Italian Male - Nicola",
    # Brazilian Portuguese
    "pf_dora": "Portuguese Female - Dora",
    "pm_alex": "Portuguese Male - Alex",
    "pm_santa": "Portuguese Male - Santa",
}

_LANG_CODES = {"a": "a", "b": "b", "e": "e", "f": "f", "h": "h", "i": "i", "j": "j", "p": "p", "z": "z"}


def _lang_code_for(voice):
    return _LANG_CODES.get(voice[0], "a")


class Synthesiser:
    def __init__(self, voice="af_heart", speed=1.0):
        self.voice = voice
        self.speed = speed
        self.pipeline = KPipeline(lang_code=_lang_code_for(voice))

    def synthesise(self, text):
        """Synthesise text to audio. Returns a float32 numpy array at 24 kHz."""
        if not text or not text.strip():
            return np.array([], dtype=np.float32)

        chunks = []
        for _gs, _ps, audio in self.pipeline(text, voice=self.voice, speed=self.speed, split_pattern=r"\n+"):
            if audio is None:
                continue
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            elif hasattr(audio, "numpy"):
                audio = audio.numpy()
            chunks.append(np.asarray(audio, dtype=np.float32).flatten())

        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)

    @staticmethod
    def list_voices():
        return VOICES
