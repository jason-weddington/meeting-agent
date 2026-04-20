"""Text-to-speech via Kokoro (mlx-audio MLX-native backend).

Kokoro outputs 24 kHz mono float32. Streaming synthesis yields sentence-sized
audio chunks so callers can start playback before the full response renders.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from mlx_audio.tts.utils import load_model

TTSAudioArray = npt.NDArray[np.float32]

SAMPLE_RATE: int = 24_000
DEFAULT_VOICE: str = "af_heart"
DEFAULT_LANG_CODE: str = "a"  # American English
_MODEL_REPO: str = "mlx-community/Kokoro-82M-bf16"

logger = logging.getLogger(__name__)


def _load_lexicon(path: Path) -> dict[str, str]:
    """Load a pronunciation lexicon JSON file and expand case variants."""
    with open(path) as f:
        raw: dict[str, str] = json.load(f)
    expanded: dict[str, str] = {}
    for word, phonemes in raw.items():
        if word.startswith("_"):
            continue
        expanded[word] = phonemes
        expanded[word.lower()] = phonemes
        expanded[word.capitalize()] = phonemes
    return expanded


class TTS:
    """Kokoro TTS wrapper (mlx-audio MLX-native backend).

    Holds a warmed model. Instantiation is expensive (model load); reuse
    the instance across synthesis calls.
    """

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        lang_code: str = DEFAULT_LANG_CODE,
        pronunciation_lexicon: Path | None = None,
    ) -> None:
        """Load the mlx-audio Kokoro model and warm it with a short sentence."""
        self.voice = voice
        self.lang_code = lang_code
        self._model: Any = load_model(_MODEL_REPO)
        # Warm the model so the first real call doesn't pay model-load cost.
        for _ in self._model.generate(
            text="warmup.", voice=self.voice, lang_code=self.lang_code, speed=1.0
        ):
            break
        if pronunciation_lexicon is not None:
            self._inject_lexicon(pronunciation_lexicon)

    def _inject_lexicon(self, path: Path) -> None:
        """Inject custom pronunciations into Kokoro's G2P lexicon."""
        entries = _load_lexicon(path)
        pipeline = self._model._get_pipeline(self.lang_code)
        pipeline.g2p.lexicon.golds.update(entries)
        logger.info("Injected %d pronunciation entries from %s", len(entries), path)

    def synthesize(self, text: str) -> TTSAudioArray:
        """Synthesize ``text`` and return a 24 kHz mono float32 array."""
        chunks: list[TTSAudioArray] = []
        for result in self._model.generate(
            text=text, voice=self.voice, lang_code=self.lang_code, speed=1.0
        ):
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
        for result in self._model.generate(
            text=text, voice=self.voice, lang_code=self.lang_code, speed=1.0
        ):
            if result.audio is None:
                continue
            chunk = np.asarray(result.audio, dtype=np.float32)
            if chunk.size == 0:
                continue
            yield chunk
