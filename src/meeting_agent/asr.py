"""Streaming automatic speech recognition.

Pipelines mic audio chunks through Silero VAD for endpointing, then passes
VAD-segmented utterances through mlx-whisper (large-v3) for transcription.
Yields completed ``Utterance`` objects as they finish.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from meeting_agent.audio import AudioArray

DEFAULT_MODEL_REPO: str = "mlx-community/whisper-large-v3-mlx"


@dataclass(frozen=True)
class Utterance:
    """A single transcribed utterance with wall-clock timing."""

    text: str
    start_s: float
    end_s: float


class StreamingASR:
    """VAD-gated streaming ASR over Whisper.

    The class is stateful: it buffers incoming chunks until Silero VAD signals
    end-of-speech, then transcribes the buffered segment and yields an
    ``Utterance``.
    """

    def __init__(
        self,
        model_repo: str = DEFAULT_MODEL_REPO,
        initial_prompt: str | None = None,
    ) -> None:
        """Initialize the ASR pipeline.

        Args:
            model_repo: HuggingFace repo id for the mlx-whisper weights.
            initial_prompt: Optional custom-vocabulary prompt to bias Whisper's
                decoder toward project-specific jargon (team/product names,
                acronyms, etc.). Whisper's closest analogue to AWS Transcribe
                custom vocabulary.
        """
        self.model_repo = model_repo
        self.initial_prompt = initial_prompt

    def transcribe_stream(
        self,
        chunks: Iterator[AudioArray],
    ) -> Iterator[Utterance]:
        """Consume a chunk stream and yield ``Utterance`` objects as completed.

        Args:
            chunks: Iterator of 16 kHz mono float32 arrays (see
                ``meeting_agent.audio.record_chunks``).

        Yields:
            ``Utterance`` for each VAD-segmented speech span.
        """
        raise NotImplementedError
