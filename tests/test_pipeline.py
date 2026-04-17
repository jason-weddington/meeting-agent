"""Tests for meeting_agent.pipeline — orchestrator and TTS pipelining."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np

from meeting_agent.asr import Utterance
from meeting_agent.llm import Conversation, Turn
from meeting_agent.pipeline import Pipeline, PipelineConfig, _split_at_sentence_boundaries

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_MIC_CHUNK = np.zeros(1600, dtype=np.float32)
_FAKE_AUDIO_CHUNK = np.zeros(24_000, dtype=np.float32)


def _infinite_chunks() -> Iterator[np.ndarray]:  # type: ignore[type-arg]
    """Yield the same silent chunk forever."""
    while True:
        yield _FAKE_MIC_CHUNK.copy()


def _make_mocks(
    *,
    lm_deltas: list[str],
    utterance_text: str = "Test utterance",
    tts_audio: np.ndarray | None = None,  # type: ignore[type-arg]
    detect_true_on: int = 0,
    raise_keyboard_interrupt_on: int = 1,
) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Build component mocks for a single-turn pipeline run.

    Returns:
        (mock_wake_instance, mock_asr_instance, mock_llm_instance,
         mock_tts_instance)
    """
    if tts_audio is None:
        tts_audio = _FAKE_AUDIO_CHUNK

    detect_call_count: list[int] = [0]

    def _fake_detect(chunk: np.ndarray) -> bool:  # type: ignore[type-arg]
        idx = detect_call_count[0]
        detect_call_count[0] += 1
        if idx == detect_true_on:
            return True
        if idx == raise_keyboard_interrupt_on:
            raise KeyboardInterrupt
        return False

    mock_wake = MagicMock()
    mock_wake.detect.side_effect = _fake_detect

    mock_asr = MagicMock()
    mock_asr.transcribe_stream.return_value = iter(
        [Utterance(text=utterance_text, start_s=0.0, end_s=2.0)]
    )

    mock_llm = MagicMock()
    mock_llm.respond_stream.return_value = iter(lm_deltas)

    mock_tts = MagicMock()
    mock_tts.stream_synthesize.return_value = iter([tts_audio])

    return mock_wake, mock_asr, mock_llm, mock_tts


def _run_pipeline(
    mock_wake: MagicMock,
    mock_asr: MagicMock,
    mock_llm: MagicMock,
    mock_tts: MagicMock,
    config: PipelineConfig | None = None,
) -> Pipeline:
    """Patch all five boundaries and run Pipeline.run() to completion."""
    if config is None:
        config = PipelineConfig()

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.WakeDetector", return_value=mock_wake),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline.BedrockClient", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play"),
    ):
        pipeline = Pipeline(config)
        pipeline.run()
    return pipeline


# ---------------------------------------------------------------------------
# _split_at_sentence_boundaries unit tests
# ---------------------------------------------------------------------------


def test_split_basic():
    """Splits text into complete sentences and an incomplete tail."""
    complete, tail = _split_at_sentence_boundaries("Hello. World! How are you")
    assert complete == ["Hello.", " World!"]
    assert tail == " How are you"


def test_split_no_punctuation():
    """Text with no punctuation is entirely the tail."""
    complete, tail = _split_at_sentence_boundaries("No punctuation here")
    assert complete == []
    assert tail == "No punctuation here"


def test_split_empty_string():
    """Empty text returns empty complete list and empty tail."""
    complete, tail = _split_at_sentence_boundaries("")
    assert complete == []
    assert tail == ""


def test_split_trailing_punctuation_empty_tail():
    """Text ending with punctuation produces empty tail."""
    complete, tail = _split_at_sentence_boundaries("Hello.")
    assert complete == ["Hello."]
    assert tail == ""


def test_split_question_mark():
    """Question marks are treated as sentence boundaries."""
    complete, tail = _split_at_sentence_boundaries("Ready? Yes.")
    assert complete == ["Ready?", " Yes."]
    assert tail == ""


def test_split_accumulation_pattern():
    """Simulates the delta-accumulation pattern used in the main loop."""
    buffer = ""
    all_complete: list[str] = []

    for delta in ["Hello", ".", " world", "!"]:
        buffer += delta
        complete, buffer = _split_at_sentence_boundaries(buffer)
        all_complete.extend(complete)

    assert all_complete == ["Hello.", " world!"]
    assert buffer == ""


# ---------------------------------------------------------------------------
# Main loop: call order and args
# ---------------------------------------------------------------------------


def test_main_loop_calls_stages_in_order():
    """Main loop: wake → ASR → LLM → TTS → play, in the right order."""
    mock_wake, mock_asr, mock_llm, mock_tts = _make_mocks(lm_deltas=["Hello."])

    call_order: list[str] = []

    detect_count: list[int] = [0]

    def ordered_detect(chunk: np.ndarray) -> bool:  # type: ignore[type-arg]
        n = detect_count[0]
        detect_count[0] += 1
        call_order.append("detect")
        if n == 0:
            return True
        raise KeyboardInterrupt

    def ordered_synth(text: str) -> Iterator[np.ndarray]:  # type: ignore[type-arg]
        call_order.append("synth")
        yield _FAKE_AUDIO_CHUNK

    def ordered_play(
        chunk: np.ndarray,  # type: ignore[type-arg]
        sample_rate: int,
        device: int | None = None,
    ) -> None:
        call_order.append("play")

    mock_wake.detect.side_effect = ordered_detect
    mock_tts.stream_synthesize.side_effect = ordered_synth

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.WakeDetector", return_value=mock_wake),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline.BedrockClient", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play", side_effect=ordered_play),
    ):
        Pipeline(PipelineConfig()).run()

    assert "detect" in call_order
    assert "synth" in call_order
    assert "play" in call_order
    # detect must precede synth and play
    first_detect = call_order.index("detect")
    first_synth = call_order.index("synth")
    first_play = call_order.index("play")
    assert first_detect < first_synth
    assert first_detect < first_play


def test_wake_detector_constructed_with_config_phrase():
    """WakeDetector is created with the phrase from PipelineConfig."""
    mock_wake, mock_asr, mock_llm, mock_tts = _make_mocks(lm_deltas=["Hi."])
    config = PipelineConfig(wake_phrase="hey_claude")

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.WakeDetector", return_value=mock_wake) as MockWake,
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline.BedrockClient", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play"),
    ):
        Pipeline(config).run()

    MockWake.assert_called_once_with("hey_claude")


def test_bedrock_client_constructed_with_model_id():
    """BedrockClient is created with model_id from PipelineConfig."""
    mock_wake, mock_asr, mock_llm, mock_tts = _make_mocks(lm_deltas=["Hi."])
    config = PipelineConfig(model_id="us.anthropic.claude-opus-4-5")

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.WakeDetector", return_value=mock_wake),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline.BedrockClient", return_value=mock_llm) as MockLLM,
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play"),
    ):
        Pipeline(config).run()

    MockLLM.assert_called_once_with(model_id="us.anthropic.claude-opus-4-5")


def test_llm_respond_stream_receives_correct_user_turn():
    """respond_stream is called with the transcribed utterance as latest_turn."""
    # Capture a snapshot of the Conversation object AT CALL TIME (the object is
    # mutated after respond_stream returns, so call_args would show stale state).
    captured_latest_turn: list[Turn | None] = []

    def capturing_respond_stream(context: object, conv: Conversation) -> Iterator[str]:
        captured_latest_turn.append(conv.latest_turn)
        yield "Got it."

    mock_wake, mock_asr, mock_llm, mock_tts = _make_mocks(
        lm_deltas=[],  # overridden by side_effect below
        utterance_text="What is the project status?",
    )
    mock_llm.respond_stream.side_effect = capturing_respond_stream

    _run_pipeline(mock_wake, mock_asr, mock_llm, mock_tts)

    assert len(captured_latest_turn) == 1
    turn = captured_latest_turn[0]
    assert turn is not None
    assert turn.text == "What is the project status?"
    assert turn.speaker == "user"


def test_tts_stream_synthesize_called_with_first_sentence():
    """stream_synthesize is called with the first complete sentence."""
    mock_wake, mock_asr, mock_llm, mock_tts = _make_mocks(lm_deltas=["Hello."])
    _run_pipeline(mock_wake, mock_asr, mock_llm, mock_tts)

    mock_tts.stream_synthesize.assert_called_once_with("Hello.")


def test_play_called_with_tts_audio_chunk():
    """audio.play is called with the TTS audio chunk and TTS sample rate."""
    from meeting_agent.tts import SAMPLE_RATE as TTS_SR

    mock_wake, mock_asr, mock_llm, mock_tts = _make_mocks(lm_deltas=["Hi."])

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.WakeDetector", return_value=mock_wake),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline.BedrockClient", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play") as mock_play,
    ):
        Pipeline(PipelineConfig()).run()

    assert mock_play.called
    play_kwargs = mock_play.call_args
    assert play_kwargs[1]["sample_rate"] == TTS_SR or play_kwargs[0][1] == TTS_SR


# ---------------------------------------------------------------------------
# Rolling transcript
# ---------------------------------------------------------------------------


def test_rolling_transcript_after_one_turn():
    """First LLM call sees empty older_turns; transcript is empty at turn start."""
    # Capture a snapshot of older_turns AT CALL TIME to verify the conversation
    # starts with no history on the first turn.
    captured_older_turns: list[list[Turn]] = []

    def capturing_respond_stream(context: object, conv: Conversation) -> Iterator[str]:
        captured_older_turns.append(list(conv.older_turns))
        yield "Agent reply."

    mock_wake, mock_asr, mock_llm, mock_tts = _make_mocks(
        lm_deltas=[],
        utterance_text="User said this",
    )
    mock_llm.respond_stream.side_effect = capturing_respond_stream

    _run_pipeline(mock_wake, mock_asr, mock_llm, mock_tts)

    assert len(captured_older_turns) == 1
    assert captured_older_turns[0] == []  # no history on first turn


def test_rolling_transcript_older_turns_populated_after_turn():
    """After one full exchange, the second LLM call sees both turns in older_turns."""
    # Run TWO turns. On the second turn, capture a snapshot of older_turns.
    # This verifies that run() correctly appended [user_turn, agent_turn] after
    # the first exchange.
    detect_count: list[int] = [0]

    def two_turn_detect(chunk: np.ndarray) -> bool:  # type: ignore[type-arg]
        n = detect_count[0]
        detect_count[0] += 1
        if n in (0, 2):  # wake on first chunk of each turn
            return True
        if n == 4:  # exit after second turn completes
            raise KeyboardInterrupt
        return False

    mock_wake = MagicMock()
    mock_wake.detect.side_effect = two_turn_detect

    mock_asr = MagicMock()
    mock_asr.transcribe_stream.side_effect = [
        iter([Utterance(text="First question", start_s=0.0, end_s=1.0)]),
        iter([Utterance(text="Second question", start_s=2.0, end_s=3.0)]),
    ]

    # Capture snapshots of older_turns at each LLM call time
    captured_older_turns: list[list[Turn]] = []

    def capturing_respond_stream(context: object, conv: Conversation) -> Iterator[str]:
        captured_older_turns.append(list(conv.older_turns))
        if len(captured_older_turns) == 1:
            yield "First answer."
        else:
            yield "Second answer."

    mock_llm = MagicMock()
    mock_llm.respond_stream.side_effect = capturing_respond_stream

    mock_tts = MagicMock()
    mock_tts.stream_synthesize.return_value = iter([_FAKE_AUDIO_CHUNK])

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.WakeDetector", return_value=mock_wake),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline.BedrockClient", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play"),
    ):
        Pipeline(PipelineConfig()).run()

    assert mock_llm.respond_stream.call_count == 2
    # First turn: no prior history
    assert captured_older_turns[0] == []
    # Second turn: first exchange must be in older_turns
    assert len(captured_older_turns[1]) == 2
    assert captured_older_turns[1][0] == Turn(speaker="user", text="First question")
    assert captured_older_turns[1][1] == Turn(speaker="agent", text="First answer.")


# ---------------------------------------------------------------------------
# First-sentence TTS pipelining
# ---------------------------------------------------------------------------


def test_first_sentence_pipelining():
    """stream_synthesize("Hello.") is called before LLM yields all deltas."""
    # Synchronisation: LLM blocks after yielding "." until TTS has been
    # called with "Hello.", proving the worker thread consumed the sentence
    # before the LLM finished streaming.
    first_sentence_tts_called = threading.Event()

    def pipelining_lm_gen(context: object, conv: object) -> Iterator[str]:
        yield "Hello"
        yield "."
        # Block until the worker thread calls stream_synthesize("Hello.")
        assert first_sentence_tts_called.wait(timeout=5.0), (
            "stream_synthesize was not called with 'Hello.' before LLM continued"
        )
        yield " world"
        yield "!"

    synth_call_args: list[str] = []

    def fake_stream_synthesize(text: str) -> Iterator[np.ndarray]:  # type: ignore[type-arg]
        synth_call_args.append(text)
        if text == "Hello.":
            first_sentence_tts_called.set()
        yield _FAKE_AUDIO_CHUNK

    detect_count: list[int] = [0]

    def single_turn_detect(chunk: np.ndarray) -> bool:  # type: ignore[type-arg]
        n = detect_count[0]
        detect_count[0] += 1
        if n == 0:
            return True
        raise KeyboardInterrupt

    mock_wake = MagicMock()
    mock_wake.detect.side_effect = single_turn_detect

    mock_asr = MagicMock()
    mock_asr.transcribe_stream.return_value = iter(
        [Utterance(text="Hello?", start_s=0.0, end_s=1.0)]
    )

    mock_llm = MagicMock()
    mock_llm.respond_stream.side_effect = pipelining_lm_gen

    mock_tts = MagicMock()
    mock_tts.stream_synthesize.side_effect = fake_stream_synthesize

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.WakeDetector", return_value=mock_wake),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline.BedrockClient", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play"),
    ):
        Pipeline(PipelineConfig()).run()

    assert synth_call_args == ["Hello.", " world!"], (
        f"Expected ['Hello.', ' world!'], got {synth_call_args}"
    )
    assert first_sentence_tts_called.is_set(), "stream_synthesize was never called with 'Hello.'"


def test_unpunctuated_response_still_synthesised():
    """A response with no sentence-ending punctuation is still synthesised."""
    mock_wake, mock_asr, mock_llm, mock_tts = _make_mocks(
        lm_deltas=["Sure", " thing"],
    )
    _run_pipeline(mock_wake, mock_asr, mock_llm, mock_tts)

    # Full text "Sure thing" has no terminal punctuation → flushed as tail
    mock_tts.stream_synthesize.assert_called_once_with("Sure thing")


def test_multi_sentence_response_synthesised_in_order():
    """Multiple complete sentences are sent to TTS in order."""
    mock_wake, mock_asr, mock_llm, mock_tts = _make_mocks(
        lm_deltas=["First.", " Second.", " Third."],
    )
    _run_pipeline(mock_wake, mock_asr, mock_llm, mock_tts)

    calls = [c[0][0] for c in mock_tts.stream_synthesize.call_args_list]
    assert calls == ["First.", " Second.", " Third."]


# ---------------------------------------------------------------------------
# Mic gating during agent speech
# ---------------------------------------------------------------------------


def test_mic_gating_during_speech():
    """Wake detector is not called while TTS audio is being played."""
    playing = threading.Event()
    detect_during_play: list[bool] = []

    detect_count: list[int] = [0]

    def gating_detect(chunk: np.ndarray) -> bool:  # type: ignore[type-arg]
        # Record whether we're currently inside a play() call
        detect_during_play.append(playing.is_set())
        n = detect_count[0]
        detect_count[0] += 1
        if n == 0:
            return True
        raise KeyboardInterrupt

    def gating_play(
        chunk: np.ndarray,  # type: ignore[type-arg]
        sample_rate: int,
        device: int | None = None,
    ) -> None:
        playing.set()
        time.sleep(0.005)  # give other threads a chance to run
        playing.clear()

    mock_wake = MagicMock()
    mock_wake.detect.side_effect = gating_detect

    mock_asr = MagicMock()
    mock_asr.transcribe_stream.return_value = iter([Utterance(text="Hi", start_s=0.0, end_s=0.5)])

    mock_llm = MagicMock()
    mock_llm.respond_stream.return_value = iter(["Answer."])

    mock_tts = MagicMock()
    mock_tts.stream_synthesize.return_value = iter([_FAKE_AUDIO_CHUNK])

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.WakeDetector", return_value=mock_wake),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline.BedrockClient", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play", side_effect=gating_play),
    ):
        Pipeline(PipelineConfig()).run()

    # No detect() call should have occurred while playing was set
    assert all(not was_playing for was_playing in detect_during_play), (
        "detect() was called while TTS audio was playing — mic gating failed"
    )


# ---------------------------------------------------------------------------
# KeyboardInterrupt exits cleanly
# ---------------------------------------------------------------------------


def test_drain_echo_consumes_expected_chunk_count():
    """_drain_echo consumes duration_s / chunk_ms chunks from the iterator."""
    pipeline = Pipeline(PipelineConfig())
    chunks = iter([np.zeros(1600, dtype=np.float32) for _ in range(50)])

    # 1.5 s at 100 ms/chunk → 15 chunks drained
    pipeline._drain_echo(chunks, 1.5)

    remaining = list(chunks)
    assert len(remaining) == 35


def test_drain_echo_minimum_one_chunk_for_short_durations():
    """Sub-chunk durations still drain at least one chunk (never zero)."""
    pipeline = Pipeline(PipelineConfig())
    chunks = iter([np.zeros(1600, dtype=np.float32) for _ in range(10)])

    pipeline._drain_echo(chunks, 0.02)  # 20 ms < one 100 ms chunk

    remaining = list(chunks)
    assert len(remaining) == 9


def test_drain_echo_stops_cleanly_when_iterator_exhausts():
    """_drain_echo using next(..., None) handles iterator exhaustion without raising."""
    pipeline = Pipeline(PipelineConfig())
    chunks = iter([np.zeros(1600, dtype=np.float32) for _ in range(2)])

    # Asks for 10 chunks but iterator only has 2 — must not raise StopIteration
    pipeline._drain_echo(chunks, 1.0)

    assert list(chunks) == []


def test_keyboard_interrupt_exits_cleanly():
    """run() returns normally (does not raise) when Ctrl-C is received."""
    detect_count: list[int] = [0]

    def immediate_interrupt(chunk: np.ndarray) -> bool:  # type: ignore[type-arg]
        detect_count[0] += 1
        raise KeyboardInterrupt

    mock_wake = MagicMock()
    mock_wake.detect.side_effect = immediate_interrupt

    mock_asr = MagicMock()
    mock_llm = MagicMock()
    mock_tts = MagicMock()

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.WakeDetector", return_value=mock_wake),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline.BedrockClient", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play"),
    ):
        # Must not raise
        Pipeline(PipelineConfig()).run()

    assert detect_count[0] >= 1
