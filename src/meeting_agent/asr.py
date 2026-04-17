"""Streaming automatic speech recognition.

Pipelines mic audio chunks through Silero VAD for endpointing, then passes
VAD-segmented utterances through mlx-whisper (large-v3) for transcription.
Yields completed ``Utterance`` objects as they finish.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from meeting_agent.audio import SAMPLE_RATE, AudioArray

DEFAULT_MODEL_REPO: str = "mlx-community/whisper-large-v3-mlx"

# Speech probability threshold: chunks scoring at or above this are speech.
_SPEECH_THRESHOLD: float = 0.5

# Consecutive silent chunks required to close an utterance (end-of-speech
# timeout). At the default 100 ms/chunk from record_chunks, 5 chunks ≈ 500 ms.
_EOS_CHUNK_COUNT: int = 5

# Silero VAD requires exactly this many samples per call at 16 kHz (32 ms).
_VAD_WINDOW: int = 512


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

        Loads mlx-whisper and Silero VAD lazily on first call (not in
        ``__init__``) to keep construction cheap for tests. Each incoming
        chunk is scored by running the VAD model over ``_VAD_WINDOW``-sample
        sub-windows and taking the maximum probability. When
        ``_EOS_CHUNK_COUNT`` consecutive silent chunks follow a speech segment,
        the buffered audio is transcribed and the resulting ``Utterance`` is
        yielded.

        Args:
            chunks: Iterator of 16 kHz mono float32 arrays (see
                ``meeting_agent.audio.record_chunks``).

        Yields:
            ``Utterance`` for each VAD-segmented speech span.
        """
        # Lazy imports keep __init__ cheap and let tests inject mocks before
        # these modules are ever resolved.
        import mlx_whisper
        import torch
        from silero_vad import load_silero_vad

        vad_model = load_silero_vad()

        buffer: list[AudioArray] = []
        in_speech: bool = False
        silent_count: int = 0
        start_s: float = 0.0
        elapsed_s: float = 0.0

        for chunk in chunks:
            chunk_s: float = len(chunk) / SAMPLE_RATE

            # Score the chunk via VAD: slide a 512-sample window across the
            # chunk and take the maximum speech probability.  Silero VAD
            # requires exactly _VAD_WINDOW samples per call at 16 kHz.
            max_prob: float = 0.0
            for i in range(0, len(chunk) - _VAD_WINDOW + 1, _VAD_WINDOW):
                sub: torch.Tensor = torch.from_numpy(chunk[i : i + _VAD_WINDOW])
                prob: float = float(vad_model(sub, SAMPLE_RATE))
                if prob > max_prob:
                    max_prob = prob
            speech_prob: float = max_prob

            if speech_prob >= _SPEECH_THRESHOLD:
                if not in_speech:
                    # Silence → speech transition: record segment start time.
                    in_speech = True
                    start_s = elapsed_s
                    silent_count = 0
                buffer.append(chunk)
            else:
                if in_speech:
                    # Keep buffering silent chunks so trailing silence is
                    # included in the transcribed audio, which helps Whisper.
                    buffer.append(chunk)
                    silent_count += 1
                    if silent_count >= _EOS_CHUNK_COUNT:
                        # End-of-speech timeout reached: transcribe segment.
                        segment: AudioArray = np.concatenate(buffer)
                        result = mlx_whisper.transcribe(
                            segment,
                            path_or_hf_repo=self.model_repo,
                            initial_prompt=self.initial_prompt,
                        )
                        end_s: float = elapsed_s + chunk_s
                        yield Utterance(
                            text=result["text"],
                            start_s=start_s,
                            end_s=end_s,
                        )
                        buffer = []
                        in_speech = False
                        silent_count = 0

            elapsed_s += chunk_s
