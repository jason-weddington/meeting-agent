"""Wake-word detection via openwakeword.

Consumes 16 kHz mono float32 audio chunks and returns a boolean: did this chunk
contain the wake phrase? The V1 pipeline arms ASR and LLM only after a wake-word
trigger. V2 will replace this with an "am I being addressed" classifier driven
by the reasoning LLM.

Note on chunk size: openwakeword is optimized for 80 ms chunks (1280 samples at
16 kHz). The pipeline supplies 100 ms chunks (1600 samples), which is close enough
to still produce correct predictions in practice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from openwakeword import models as _bundled
from openwakeword.model import Model

from meeting_agent.audio import AudioArray

DEFAULT_WAKE_PHRASE: str = "hey_jarvis"
DETECTION_THRESHOLD: float = 0.5


class WakeDetector:
    """openwakeword model wrapper.

    Model load is expensive; construct once and reuse. ``detect`` is designed
    to be called on every incoming audio chunk.
    """

    def __init__(self, wake_phrase: str = DEFAULT_WAKE_PHRASE) -> None:
        """Load the openwakeword model for the given wake phrase.

        Args:
            wake_phrase: One of openwakeword's bundled model names
                (``alexa``, ``hey_jarvis``, ``hey_mycroft``, ``timer``,
                ``weather``) or a filesystem path to a custom ``.onnx`` model.
        """
        self.wake_phrase = wake_phrase
        if wake_phrase in _bundled:
            model_path = _bundled[wake_phrase]["model_path"]
        elif Path(wake_phrase).is_file():
            model_path = wake_phrase
        else:
            raise ValueError(
                f"wake_phrase {wake_phrase!r} is not a bundled model "
                f"({sorted(_bundled.keys())}) or an existing file path"
            )
        self._model = Model(wakeword_model_paths=[model_path])

    def detect(self, audio_chunk: AudioArray) -> bool:
        """Return True if the wake phrase was detected in this chunk.

        Converts float32 audio to int16 (openwakeword's expected format) before
        running inference.

        Args:
            audio_chunk: 16 kHz mono float32 audio samples.

        Returns:
            True if any loaded wake-word model score exceeds the detection
            threshold; False otherwise.
        """
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        scores: dict[str, float] = self._model.predict(audio_int16)
        return max(scores.values(), default=0.0) > DETECTION_THRESHOLD
