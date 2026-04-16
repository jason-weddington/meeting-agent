"""Text-to-speech via Kokoro.

Kokoro outputs 24 kHz mono float32. Streaming synthesis yields sentence-sized
audio chunks so callers can start playback before the full response renders.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

TTSAudioArray = npt.NDArray[np.float32]

SAMPLE_RATE: int = 24_000
DEFAULT_VOICE: str = "af_heart"
DEFAULT_LANG_CODE: str = "a"  # American English


class TTS:
    """Kokoro TTS wrapper.

    Holds a warmed KPipeline. Instantiation is expensive (model load); reuse
    the instance across synthesis calls.
    """

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        lang_code: str = DEFAULT_LANG_CODE,
    ) -> None:
        """Load the Kokoro pipeline and warm it with a short sentence."""
        self.voice = voice
        self.lang_code = lang_code

    def synthesize(self, text: str) -> TTSAudioArray:
        """Synthesize ``text`` and return a 24 kHz mono float32 array."""
        raise NotImplementedError

    def stream_synthesize(self, text: str) -> Iterator[TTSAudioArray]:
        """Yield audio chunks as the synthesizer completes each sentence.

        Used by the pipeline to pipeline Claude's streamed text deltas into
        playback — we hear the first sentence while later sentences are still
        being generated. Each yielded array is 24 kHz mono float32.
        """
        raise NotImplementedError
