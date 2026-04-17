"""Text-to-speech via Kokoro.

Kokoro outputs 24 kHz mono float32. Streaming synthesis yields sentence-sized
audio chunks so callers can start playback before the full response renders.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
from kokoro import KPipeline

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
        self._pipeline: Any = KPipeline(lang_code=lang_code)
        # Warm the pipeline so the first real call doesn't pay model-load cost.
        for _ in self._pipeline("warmup.", voice=self.voice):
            break

    def synthesize(self, text: str) -> TTSAudioArray:
        """Synthesize ``text`` and return a 24 kHz mono float32 array."""
        chunks: list[TTSAudioArray] = []
        for result in self._pipeline(text, voice=self.voice):
            if result.audio is None:
                continue
            chunks.append(np.asarray(result.audio, dtype=np.float32))
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)

    def stream_synthesize(self, text: str) -> Iterator[TTSAudioArray]:
        """Yield audio chunks as the synthesizer completes each sentence.

        Used by the pipeline to pipeline Claude's streamed text deltas into
        playback — we hear the first sentence while later sentences are still
        being generated. Each yielded array is 24 kHz mono float32.
        """
        for result in self._pipeline(text, voice=self.voice):
            if result.audio is None:
                continue
            chunk = np.asarray(result.audio, dtype=np.float32)
            if chunk.size == 0:
                continue
            yield chunk
