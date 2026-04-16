"""Wake-word detection via openwakeword.

Consumes 16 kHz mono float32 audio chunks and returns a boolean: did this chunk
contain the wake phrase? The V1 pipeline arms ASR and LLM only after a wake-word
trigger. V2 will replace this with an "am I being addressed" classifier driven
by the reasoning LLM.
"""

from __future__ import annotations

from meeting_agent.audio import AudioArray

DEFAULT_WAKE_PHRASE: str = "hey_jarvis"


class WakeDetector:
    """openwakeword model wrapper.

    Model load is expensive; construct once and reuse. ``detect`` is designed
    to be called on every incoming audio chunk.
    """

    def __init__(self, wake_phrase: str = DEFAULT_WAKE_PHRASE) -> None:
        """Load the openwakeword model for the given wake phrase.

        Args:
            wake_phrase: One of openwakeword's bundled models
                (e.g. ``hey_jarvis``, ``alexa``) or a path to a custom model.
        """
        self.wake_phrase = wake_phrase

    def detect(self, audio_chunk: AudioArray) -> bool:
        """Return True if the wake phrase was detected in this chunk."""
        raise NotImplementedError
