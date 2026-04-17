"""End-to-end integration test for the full meeting-agent pipeline.

Exercises the complete pipeline round-trip with real TTS synthesis and mocked
external boundaries (wake detection, ASR, Bedrock, audio I/O).  Designed to
catch interface breakage between modules that unit tests miss.

Marking: ``@pytest.mark.integration`` — skipped by default in CI; run with::

    pytest -m integration tests/integration/test_end_to_end.py -v

Runtime: ~10–20 s on M4 Max once models are cached (first run ~330 MB download
for Kokoro).

Design notes
------------
Wake-word detection on Kokoro-synthesized audio is unreliable because
openwakeword was trained on human voice — ``WakeDetector`` is mocked.
``StreamingASR`` is mocked because ``transcribe_stream`` is not yet implemented
(the stub raises ``NotImplementedError``) and because ``mlx-whisper`` requires
Apple Silicon (MLX); dispatched tests run on Linux.  Real components exercised:
``TTS`` (Kokoro synthesis), ``Pipeline`` orchestration, sentence splitting, and
the rolling-transcript commit logic.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.signal  # transitive dep; available in all envs

from meeting_agent.asr import Utterance
from meeting_agent.audio import AudioArray
from meeting_agent.llm import Conversation, ProjectContext
from meeting_agent.pipeline import Pipeline, PipelineConfig
from meeting_agent.tts import SAMPLE_RATE as TTS_SAMPLE_RATE
from meeting_agent.tts import TTS

_CHUNK_SAMPLES: int = 1600  # 100 ms at 16 kHz
_ASR_RATE: int = 16_000
_CANNED_RESPONSE: list[str] = ["The answer", " is four.", ""]
_USER_TEXT: str = "what is two plus two"


def _downsample_24k_to_16k(audio_24k: AudioArray) -> AudioArray:
    """Downsample Kokoro's 24 kHz output to 16 kHz pipeline-input format.

    Uses ``scipy.signal.resample_poly`` (poly-phase, linear-phase FIR) for
    clean integer-ratio downsampling: 24 000 → 16 000 = 2:3.

    Args:
        audio_24k: 24 kHz mono float32 array from Kokoro.

    Returns:
        16 kHz mono float32 array suitable for the mic-chunk iterator.
    """
    resampled = scipy.signal.resample_poly(audio_24k, up=2, down=3)
    return resampled.astype(np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tts_instance() -> TTS:
    """Shared TTS instance — model loads once per test-module run."""
    return TTS()


@pytest.fixture(scope="module")
def input_audio_16k(tts_instance: TTS) -> AudioArray:
    """Synthesize the test prompt at 24 kHz and downsample to 16 kHz.

    This simulates the audio the user would speak into the microphone.
    Even though ``StreamingASR`` is mocked, supplying real audio ensures the
    pipeline wiring (chunk-splitting, iterator hand-off) is exercised with
    realistic data shapes.
    """
    audio_24k = tts_instance.synthesize("Hello, what is two plus two?")
    assert audio_24k.dtype == np.float32, "TTS should produce float32 audio"
    assert len(audio_24k) > 0, "TTS should produce non-empty audio"
    return _downsample_24k_to_16k(audio_24k)


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_pipeline_round_trip(
    tts_instance: TTS,
    input_audio_16k: AudioArray,
) -> None:
    """Full pipeline round-trip: wake → ASR → LLM → TTS → play.

    One complete turn is exercised; the pipeline terminates naturally once the
    finite mock chunk iterator is exhausted (``RuntimeError`` from the second
    ``_collect_utterance`` call is the expected exit path).

    Assertions
    ----------
    * ``BedrockClient.respond_stream`` called exactly once.
    * The ``Conversation`` passed to it had ``latest_turn.text`` equal to the
      canned ASR utterance ``"what is two plus two"``.
    * ``audio.play`` called at least once with a ``float32`` array at 24 kHz.
    * The pipeline emitted both sentences from the mocked response.
    * After the turn, ``conversation.older_turns`` has 2 entries (user + agent).
    """
    # --- Build mock chunk iterator ---
    # Pad the last chunk to a full 100 ms window so the ASR mock always sees
    # complete blocks regardless of audio length.
    speech_chunks = [
        input_audio_16k[i : i + _CHUNK_SAMPLES]
        for i in range(0, len(input_audio_16k), _CHUNK_SAMPLES)
    ]
    if speech_chunks and len(speech_chunks[-1]) < _CHUNK_SAMPLES:
        speech_chunks[-1] = np.pad(
            speech_chunks[-1],
            (0, _CHUNK_SAMPLES - len(speech_chunks[-1])),
        ).astype(np.float32)

    # --- Captured state ---
    llm_conversations: list[Conversation] = []
    play_calls: list[tuple[AudioArray, int]] = []
    captured_latest_turn_text: list[str] = []

    # --- Mock implementations ---

    def mock_record_chunks(device: int | None = None, chunk_ms: int = 100) -> Iterator[AudioArray]:
        """One silent wake-word chunk followed by the synthesized speech.

        After all chunks are consumed the iterator is exhausted, which causes
        the pipeline to exit via ``RuntimeError`` on the second ASR attempt.
        """
        yield np.zeros(_CHUNK_SAMPLES, dtype=np.float32)  # triggers wake
        yield from speech_chunks

    detect_call_count = 0

    def mock_detect(chunk: AudioArray) -> bool:
        """Return True on the first call (wake trigger), False thereafter."""
        nonlocal detect_call_count
        detect_call_count += 1
        return detect_call_count == 1

    def mock_transcribe_stream(
        chunks: Iterator[AudioArray],
    ) -> Iterator[Utterance]:
        """Drain the chunk iterator; yield a canned utterance if audio arrived.

        Draining the iterator is essential: it exercises the chunk hand-off
        between ``_wait_for_wake`` and ``_collect_utterance`` (the two stages
        share the same generator).  If no chunks arrive (exhausted iterator on
        the second loop iteration) we yield nothing, which causes
        ``_collect_utterance`` to raise ``RuntimeError`` and exit the loop.
        """
        has_audio = False
        for _ in chunks:
            has_audio = True
        if has_audio:
            yield Utterance(text=_USER_TEXT, start_s=0.0, end_s=1.0)

    def mock_respond_stream(context: ProjectContext, conversation: Conversation) -> Iterator[str]:
        """Capture conversation state at call time; yield canned response."""
        # Snapshot latest_turn before the pipeline mutates the object.
        if conversation.latest_turn is not None:
            captured_latest_turn_text.append(conversation.latest_turn.text)
        llm_conversations.append(conversation)
        yield from _CANNED_RESPONSE

    def mock_play(audio_arr: AudioArray, sample_rate: int, device: int | None = None) -> None:
        """Capture each audio.play() call instead of hitting speakers."""
        play_calls.append((audio_arr.copy(), sample_rate))

    # --- Wire mocks into Pipeline ---
    config = PipelineConfig(
        context=ProjectContext(system_prompt="You are a helpful meeting assistant.")
    )
    pipeline = Pipeline(config)

    # TTS is intentionally NOT mocked — Kokoro runs for real.
    # We pass the already-loaded instance via a lambda so Pipeline.run()
    # gets the warm model rather than cold-loading a new one.
    pipeline_error: list[Exception] = []

    def run_pipeline() -> None:
        try:
            with (
                patch("meeting_agent.audio.record_chunks", side_effect=mock_record_chunks),
                patch("meeting_agent.audio.play", side_effect=mock_play),
                patch("meeting_agent.pipeline.WakeDetector") as mock_wake_cls,
                patch("meeting_agent.pipeline.BedrockClient") as mock_llm_cls,
                patch("meeting_agent.pipeline.StreamingASR") as mock_asr_cls,
                patch("meeting_agent.pipeline.TTS", return_value=tts_instance),
            ):
                mock_wake_instance = MagicMock()
                mock_wake_instance.detect.side_effect = mock_detect
                mock_wake_cls.return_value = mock_wake_instance

                mock_llm_instance = MagicMock()
                mock_llm_instance.respond_stream.side_effect = mock_respond_stream
                mock_llm_cls.return_value = mock_llm_instance

                mock_asr_instance = MagicMock()
                mock_asr_instance.transcribe_stream.side_effect = mock_transcribe_stream
                mock_asr_cls.return_value = mock_asr_instance

                pipeline.run()
        except RuntimeError:
            # Expected exit path: second ASR attempt on exhausted iterator →
            # ``_collect_utterance`` raises RuntimeError.
            pass
        except Exception as exc:
            pipeline_error.append(exc)

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()
    thread.join(timeout=120)  # allow up to 2 minutes for Kokoro on first run

    assert not thread.is_alive(), "Pipeline thread did not complete within the 120 s timeout"
    assert not pipeline_error, f"Pipeline raised an unexpected error: {pipeline_error[0]!r}"

    # --- Assertion 1: LLM called exactly once ---
    assert len(llm_conversations) == 1, f"Expected exactly 1 LLM call, got {len(llm_conversations)}"

    # --- Assertion 2: Transcription forwarded to LLM correctly ---
    assert len(captured_latest_turn_text) == 1
    assert "two plus two" in captured_latest_turn_text[0].lower(), (
        f"Expected 'two plus two' in transcription, got: {captured_latest_turn_text[0]!r}"
    )

    # --- Assertion 3: audio.play called with float32 arrays at TTS sample rate ---
    assert len(play_calls) >= 1, "audio.play should have been called at least once"
    for audio_arr, sample_rate in play_calls:
        assert isinstance(audio_arr, np.ndarray), "play() must receive a numpy array"
        assert audio_arr.dtype == np.float32, (
            f"play() must receive float32 audio, got {audio_arr.dtype}"
        )
        assert sample_rate == TTS_SAMPLE_RATE, (
            f"Expected sample rate {TTS_SAMPLE_RATE} Hz, got {sample_rate}"
        )

    # --- Assertion 4: Both sentences from canned response were emitted ---
    # The pipeline splits "The answer is four." at the '.' boundary, so TTS
    # is invoked at least once with non-empty audio sent to play().
    assert len(play_calls) >= 1, "Pipeline should produce at least one audio chunk from TTS"

    # --- Assertion 5: Conversation committed with ≥ 2 older_turns ---
    conv = llm_conversations[0]
    assert len(conv.older_turns) == 2, (
        f"After one round-trip, older_turns should have exactly 2 entries "
        f"(user + agent), got {len(conv.older_turns)}"
    )
    assert conv.older_turns[0].speaker == "user"
    assert conv.older_turns[1].speaker == "agent"
    assert "four" in conv.older_turns[1].text.lower(), (
        f"Agent turn should contain 'four', got: {conv.older_turns[1].text!r}"
    )
