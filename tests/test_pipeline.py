"""Tests for meeting_agent.pipeline — V2 always-on, classifier-gated pipeline."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meeting_agent.asr import Utterance
from meeting_agent.classifier import BedrockClassifier, Decision, OllamaClassifier
from meeting_agent.llm import BedrockClient, OllamaClient, Turn
from meeting_agent.pipeline import (
    AirtimeTracker,
    CircuitBreaker,
    CircuitBreakerOpen,
    DeafnessProbe,
    Pipeline,
    PipelineConfig,
    _build_classifier,
    _build_llm_client,
    _install_exception_log,
    _is_low_confidence,
    _split_at_sentence_boundaries,
    _strip_tts_markdown,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_MIC_CHUNK = np.zeros(1600, dtype=np.float32)
_FAKE_AUDIO_CHUNK = np.zeros(24_000, dtype=np.float32)


def _infinite_chunks() -> Iterator[np.ndarray]:  # type: ignore[type-arg]
    """Yield the same silent chunk forever."""
    while True:
        yield _FAKE_MIC_CHUNK.copy()


def _make_utterance(
    text: str = "Test utterance",
    start_s: float = 0.0,
    end_s: float = 2.0,
    avg_logprob: float = -0.3,
    no_speech_prob: float = 0.05,
    compression_ratio: float = 1.2,
) -> Utterance:
    """Build a high-confidence Utterance for testing."""
    return Utterance(
        text=text,
        start_s=start_s,
        end_s=end_s,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
        compression_ratio=compression_ratio,
    )


def _full_answer_decision(speaker: str = "Jason") -> Decision:
    return Decision(speaker=speaker, action="full_answer", confidence=0.9)


def _silent_decision(speaker: str = "Jason") -> Decision:
    return Decision(speaker=speaker, action="silent", confidence=0.9)


def _make_v2_mocks(
    *,
    utterances: list[Utterance],
    decisions: list[Decision] | None = None,
    lm_deltas: list[str] | None = None,
    tts_audio: np.ndarray | None = None,  # type: ignore[type-arg]
) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Build V2 component mocks (no wake detector).

    Returns:
        (mock_asr, mock_classifier, mock_llm, mock_tts)
    """
    if tts_audio is None:
        tts_audio = _FAKE_AUDIO_CHUNK
    if lm_deltas is None:
        lm_deltas = ["Agent reply."]

    mock_asr = MagicMock()
    mock_asr.transcribe_stream.return_value = iter(utterances)

    mock_classifier = MagicMock()
    if decisions is not None:
        mock_classifier.classify.side_effect = decisions
    else:
        mock_classifier.classify.return_value = _full_answer_decision()

    mock_llm = MagicMock()
    mock_llm.respond_stream.return_value = iter(lm_deltas)

    mock_tts = MagicMock()
    mock_tts.stream_synthesize.return_value = iter([tts_audio])

    return mock_asr, mock_classifier, mock_llm, mock_tts


def _run_v2_pipeline(
    mock_asr: MagicMock,
    mock_classifier: MagicMock,
    mock_llm: MagicMock,
    mock_tts: MagicMock,
    config: PipelineConfig | None = None,
) -> Pipeline:
    """Patch all V2 boundaries and run Pipeline.run() to completion."""
    if config is None:
        config = PipelineConfig()

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline._build_classifier", return_value=mock_classifier),
        patch("meeting_agent.pipeline._build_llm_client", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play"),
        patch("meeting_agent.pipeline._install_exception_log", return_value=MagicMock()),
    ):
        pipeline = Pipeline(config)
        pipeline.run()
    return pipeline


# ---------------------------------------------------------------------------
# _split_at_sentence_boundaries unit tests (unchanged helper)
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
# _is_low_confidence unit tests
# ---------------------------------------------------------------------------


def test_is_low_confidence_clean_utterance():
    """Normal transcription is not flagged as low-confidence."""
    u = _make_utterance(avg_logprob=-0.3, no_speech_prob=0.05, compression_ratio=1.2)
    assert not _is_low_confidence(u)


def test_is_low_confidence_bad_logprob():
    """avg_logprob below -1.0 triggers low-confidence."""
    u = _make_utterance(avg_logprob=-1.5, no_speech_prob=0.05, compression_ratio=1.2)
    assert _is_low_confidence(u)


def test_is_low_confidence_high_no_speech():
    """no_speech_prob above 0.6 triggers low-confidence."""
    u = _make_utterance(avg_logprob=-0.3, no_speech_prob=0.7, compression_ratio=1.2)
    assert _is_low_confidence(u)


def test_is_low_confidence_high_compression():
    """compression_ratio above 2.4 triggers low-confidence."""
    u = _make_utterance(avg_logprob=-0.3, no_speech_prob=0.05, compression_ratio=2.5)
    assert _is_low_confidence(u)


# ---------------------------------------------------------------------------
# _drain_echo unit tests (unchanged helper)
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


# ---------------------------------------------------------------------------
# AirtimeTracker unit tests
# ---------------------------------------------------------------------------


def test_airtime_tracker_empty():
    """Fresh tracker returns 0 for any window."""
    at = AirtimeTracker()
    assert at.count_last(30) == 0
    assert at.count_last(300) == 0


def test_airtime_tracker_records_emission():
    """After recording an emission, count_last returns 1 for a wide window."""
    at = AirtimeTracker()
    at.record_emission(time.monotonic())
    assert at.count_last(30) == 1


def test_airtime_tracker_prunes_old_emissions():
    """Emissions older than 5 minutes are pruned."""
    at = AirtimeTracker()
    old = time.monotonic() - 400.0  # 400s ago, outside 300s window
    at.record_emission(old)
    # record_emission only prunes relative to the last emission's time
    at.record_emission(time.monotonic())
    # The old one should be pruned; the recent one should count
    assert at.count_last(300) == 1


# ---------------------------------------------------------------------------
# CircuitBreaker unit tests
# ---------------------------------------------------------------------------


def test_circuit_breaker_starts_closed():
    """Fresh circuit breaker allows entry."""
    cb = CircuitBreaker()
    with cb:
        pass  # must not raise


def test_circuit_breaker_opens_after_threshold():
    """Circuit opens after fail_threshold failures within the window."""
    cb = CircuitBreaker(fail_threshold=3, fail_window_s=10.0, open_s=15.0)

    # Record 3 failures
    for _ in range(3):
        try:
            with cb:
                raise RuntimeError("fail")
        except RuntimeError:
            pass

    # Circuit should now be open
    with pytest.raises(CircuitBreakerOpen):
        with cb:
            pass


def test_circuit_breaker_half_open_probe():
    """After open_s, the circuit goes half-open and allows one probe."""
    cb = CircuitBreaker(fail_threshold=2, fail_window_s=10.0, open_s=0.05)

    # Trip the circuit
    for _ in range(2):
        try:
            with cb:
                raise RuntimeError("fail")
        except RuntimeError:
            pass

    # Wait for open_s to pass
    time.sleep(0.1)

    # Should allow one probe (half-open)
    with cb:
        pass  # succeeds — circuit closes again


def test_circuit_breaker_open_callback_fires():
    """on_circuit_open callback is invoked with the failure count when circuit opens."""
    opened: list[int] = []
    cb = CircuitBreaker(
        fail_threshold=2,
        fail_window_s=10.0,
        open_s=15.0,
        on_circuit_open=opened.append,
    )

    for _ in range(2):
        try:
            with cb:
                raise RuntimeError("fail")
        except RuntimeError:
            pass

    assert opened == [2]


def test_circuit_breaker_half_open_callback_fires():
    """on_circuit_half_open callback is invoked when the circuit transitions to half-open."""
    half_opens: list[str] = []
    cb = CircuitBreaker(
        fail_threshold=2,
        fail_window_s=10.0,
        open_s=0.05,
        on_circuit_half_open=lambda: half_opens.append("half_open"),
    )

    for _ in range(2):
        try:
            with cb:
                raise RuntimeError("fail")
        except RuntimeError:
            pass

    time.sleep(0.1)  # let open_s expire

    with cb:
        pass  # probe succeeds

    assert half_opens == ["half_open"]


def test_circuit_breaker_close_callback_fires():
    """on_circuit_close callback is invoked when a half-open probe succeeds."""
    closed: list[bool] = []
    cb = CircuitBreaker(
        fail_threshold=2,
        fail_window_s=10.0,
        open_s=0.05,
        on_circuit_close=lambda: closed.append(True),
    )

    for _ in range(2):
        try:
            with cb:
                raise RuntimeError("fail")
        except RuntimeError:
            pass

    time.sleep(0.1)  # let open_s expire

    with cb:
        pass  # probe succeeds → circuit closes

    assert closed == [True]


# ---------------------------------------------------------------------------
# DeafnessProbe unit tests
# ---------------------------------------------------------------------------


def test_deafness_probe_does_not_fire_below_threshold():
    """Probe does not suggest probing before threshold drops."""
    dp = DeafnessProbe(threshold=3)
    dp.record_drop()
    dp.record_drop()
    assert not dp.should_probe()


def test_deafness_probe_fires_at_threshold():
    """Probe fires exactly when threshold drops is reached."""
    dp = DeafnessProbe(threshold=3)
    dp.record_drop()
    dp.record_drop()
    dp.record_drop()
    assert dp.should_probe()


def test_deafness_probe_does_not_repeat():
    """After mark_used(), should_probe() always returns False."""
    dp = DeafnessProbe(threshold=3)
    for _ in range(5):
        dp.record_drop()
    assert dp.should_probe()
    dp.mark_used()
    assert not dp.should_probe()


# ---------------------------------------------------------------------------
# V2 pipeline: classifier integration
# ---------------------------------------------------------------------------


def test_classifier_called_per_utterance():
    """Classifier is called once per utterance that passes the confidence gate."""
    utterances = [
        _make_utterance("First"),
        _make_utterance("Second"),
        _make_utterance("Third"),
    ]
    decisions = [
        _silent_decision(),
        _silent_decision(),
        _silent_decision(),
    ]
    mock_asr, mock_classifier, mock_llm, mock_tts = _make_v2_mocks(
        utterances=utterances, decisions=decisions
    )
    _run_v2_pipeline(mock_asr, mock_classifier, mock_llm, mock_tts)

    assert mock_classifier.classify.call_count == 3


def test_silent_decision_appends_and_skips_response():
    """When classifier returns 'silent', utterance is appended and LLM is not called."""
    utterance = _make_utterance("Background noise")
    mock_asr, mock_classifier, mock_llm, mock_tts = _make_v2_mocks(
        utterances=[utterance],
        decisions=[_silent_decision(speaker="Jason")],
    )

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline._build_classifier", return_value=mock_classifier),
        patch("meeting_agent.pipeline._build_llm_client", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play"),
        patch("meeting_agent.pipeline._install_exception_log", return_value=MagicMock()),
        patch.object(Pipeline, "_stream_and_play") as mock_sap,
    ):
        pipeline = Pipeline(PipelineConfig())
        pipeline.run()

    mock_llm.respond_stream.assert_not_called()
    mock_sap.assert_not_called()


def test_hedged_and_full_decisions_trigger_response():
    """When classifier returns 'full_answer', the response pipeline runs."""
    utterance = _make_utterance("What is the status?")
    mock_asr, mock_classifier, mock_llm, mock_tts = _make_v2_mocks(
        utterances=[utterance],
        decisions=[_full_answer_decision(speaker="Jason")],
        lm_deltas=["The status is good."],
    )
    _run_v2_pipeline(mock_asr, mock_classifier, mock_llm, mock_tts)

    mock_llm.respond_stream.assert_called_once()


def test_llm_model_override_reaches_bedrock_backend_pipeline():
    """llm_model override is forwarded to BedrockClient via _build_llm_client."""
    config = PipelineConfig(llm_model="us.anthropic.claude-opus-4-5")
    llm = _build_llm_client(config)
    assert isinstance(llm, BedrockClient)
    assert llm.model_id == "us.anthropic.claude-opus-4-5"


# ---------------------------------------------------------------------------
# V2 pipeline: confidence gate and deafness probe
# ---------------------------------------------------------------------------


def test_low_confidence_gate_drops_utterance():
    """Utterance with low avg_logprob is dropped before classifier."""
    low_conf = _make_utterance(
        "garbled audio", avg_logprob=-1.5, no_speech_prob=0.05, compression_ratio=1.0
    )
    mock_asr, mock_classifier, mock_llm, mock_tts = _make_v2_mocks(
        utterances=[low_conf],
    )
    _run_v2_pipeline(mock_asr, mock_classifier, mock_llm, mock_tts)

    # Classifier must NOT be called for low-confidence utterances
    mock_classifier.classify.assert_not_called()
    mock_llm.respond_stream.assert_not_called()


def test_deafness_probe_fires_after_threshold_drops():
    """TTS probe fires exactly once after 3 low-confidence drops; 4th does not re-fire."""
    low_conf = _make_utterance(
        "garbled", avg_logprob=-1.5, no_speech_prob=0.05, compression_ratio=1.0
    )
    # 4 consecutive low-confidence utterances
    utterances = [low_conf, low_conf, low_conf, low_conf]

    mock_asr = MagicMock()
    mock_asr.transcribe_stream.return_value = iter(utterances)
    mock_classifier = MagicMock()
    mock_llm = MagicMock()
    mock_tts = MagicMock()
    probe_audio = np.zeros(24_000, dtype=np.float32)
    mock_tts.stream_synthesize.return_value = iter([probe_audio])

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline._build_classifier", return_value=mock_classifier),
        patch("meeting_agent.pipeline._build_llm_client", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play"),
        patch("meeting_agent.pipeline._install_exception_log", return_value=MagicMock()),
    ):
        Pipeline(PipelineConfig()).run()

    # stream_synthesize called exactly once (for the probe) with the probe text
    mock_tts.stream_synthesize.assert_called_once()
    args = mock_tts.stream_synthesize.call_args[0][0]
    assert "losing" in args or "dropping" in args  # probe text keywords


# ---------------------------------------------------------------------------
# V2 pipeline: staleness gate
# ---------------------------------------------------------------------------


def test_staleness_gate_downgrades_above_1_5s():
    """full_answer decision is downgraded to hedged_answer when utterance is 2s old."""
    utterance = _make_utterance("What is the plan?")
    mock_asr, mock_classifier, mock_llm, mock_tts = _make_v2_mocks(
        utterances=[utterance],
        decisions=[_full_answer_decision(speaker="Jason")],
        lm_deltas=["Here is the plan."],
    )

    # Simulate: arrival at t=0, staleness check at t=2.1 (age > 1.5s)
    # Call sequence:
    #   1. utterance_arrival_monotonic = time.monotonic()  → 0.0
    #   2. airtime.count_last(300) → time.monotonic()     → 0.0
    #   3. airtime.count_last(30) → time.monotonic()      → 0.0
    #   4. staleness check: time.monotonic()               → 2.1
    #   rest: any large value
    call_idx = [0]
    times_seq = [0.0, 0.0, 0.0, 2.1]

    def fake_mono() -> float:
        idx = call_idx[0]
        call_idx[0] += 1
        if idx < len(times_seq):
            return times_seq[idx]
        return 2.1

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline._build_classifier", return_value=mock_classifier),
        patch("meeting_agent.pipeline._build_llm_client", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play"),
        patch("meeting_agent.pipeline._install_exception_log", return_value=MagicMock()),
        patch("meeting_agent.pipeline.time.monotonic", side_effect=fake_mono),
    ):
        Pipeline(PipelineConfig()).run()

    # Response should still be called (just downgraded, not dropped)
    mock_llm.respond_stream.assert_called_once()


def test_staleness_gate_drops_above_5s():
    """Utterance older than 5s is dropped; LLM is not called."""
    utterance = _make_utterance("What is the plan?")
    mock_asr, mock_classifier, mock_llm, mock_tts = _make_v2_mocks(
        utterances=[utterance],
        decisions=[_full_answer_decision(speaker="Jason")],
    )

    # Simulate: arrival at t=0, staleness check at t=6.0 (age > 5s → drop)
    call_idx = [0]
    times_seq = [0.0, 0.0, 0.0, 6.0]

    def fake_mono() -> float:
        idx = call_idx[0]
        call_idx[0] += 1
        if idx < len(times_seq):
            return times_seq[idx]
        return 6.0

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline._build_classifier", return_value=mock_classifier),
        patch("meeting_agent.pipeline._build_llm_client", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play"),
        patch("meeting_agent.pipeline._install_exception_log", return_value=MagicMock()),
        patch("meeting_agent.pipeline.time.monotonic", side_effect=fake_mono),
    ):
        Pipeline(PipelineConfig()).run()

    mock_llm.respond_stream.assert_not_called()


# ---------------------------------------------------------------------------
# V2 pipeline: circuit breaker
# ---------------------------------------------------------------------------


def test_circuit_breaker_opens_after_3_failures():
    """After 3 respond_stream failures within 10s, 4th utterance's response is skipped."""
    utterances = [_make_utterance(f"Question {i}") for i in range(4)]
    decisions = [_full_answer_decision() for _ in range(4)]

    mock_asr = MagicMock()
    mock_asr.transcribe_stream.return_value = iter(utterances)

    mock_classifier = MagicMock()
    mock_classifier.classify.side_effect = decisions

    mock_llm = MagicMock()
    # First 3 calls raise; 4th would succeed but shouldn't be reached
    mock_llm.respond_stream.side_effect = [
        RuntimeError("bedrock error"),
        RuntimeError("bedrock error"),
        RuntimeError("bedrock error"),
        iter(["Fourth reply."]),
    ]

    mock_tts = MagicMock()
    mock_tts.stream_synthesize.return_value = iter([_FAKE_AUDIO_CHUNK])

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline._build_classifier", return_value=mock_classifier),
        patch("meeting_agent.pipeline._build_llm_client", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play"),
        patch("meeting_agent.pipeline._install_exception_log", return_value=MagicMock()),
    ):
        Pipeline(PipelineConfig()).run()

    # LLM was called 3 times (all failed); 4th was skipped by open circuit
    assert mock_llm.respond_stream.call_count == 3


# ---------------------------------------------------------------------------
# V2 pipeline: airtime tracking
# ---------------------------------------------------------------------------


def test_airtime_count_passed_to_classifier():
    """After one agent turn, next classifier call has agent_turns_last_30s == 1."""
    utterances = [
        _make_utterance("First question"),
        _make_utterance("Second question"),
    ]
    decisions = [
        _full_answer_decision(speaker="Jason"),
        _full_answer_decision(speaker="Jason"),
    ]

    mock_asr = MagicMock()
    # Need to return a fresh iterator that supports two utterances and then stops
    mock_asr.transcribe_stream.return_value = iter(utterances)

    mock_classifier = MagicMock()
    mock_classifier.classify.side_effect = decisions

    respond_call_count = [0]

    def _lm_gen(ctx, conv):  # type: ignore[no-untyped-def]
        respond_call_count[0] += 1
        yield "Reply."

    mock_llm = MagicMock()
    mock_llm.respond_stream.side_effect = _lm_gen

    mock_tts = MagicMock()
    mock_tts.stream_synthesize.return_value = iter([_FAKE_AUDIO_CHUNK])

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline._build_classifier", return_value=mock_classifier),
        patch("meeting_agent.pipeline._build_llm_client", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play"),
        patch("meeting_agent.pipeline._install_exception_log", return_value=MagicMock()),
    ):
        Pipeline(PipelineConfig()).run()

    assert mock_classifier.classify.call_count == 2
    # On the second call, session should have agent_turns_last_30s == 1
    second_call_args = mock_classifier.classify.call_args_list[1]
    session = second_call_args[0][3]  # positional arg 3 = session
    assert session.agent_turns_last_30s == 1


# ---------------------------------------------------------------------------
# V2 pipeline: multi-speaker transcript
# ---------------------------------------------------------------------------


def test_multi_speaker_transcript_appended():
    """Classifier-attributed speaker names appear correctly in older_turns."""
    utterances = [
        _make_utterance("Question from Jason"),
        _make_utterance("Question from Aziz"),
        _make_utterance("Jason asks again"),
    ]
    decisions = [
        Decision(speaker="Jason", action="silent", confidence=0.9),
        Decision(speaker="Aziz", action="silent", confidence=0.9),
        Decision(speaker="Jason", action="full_answer", confidence=0.9),
    ]

    mock_asr = MagicMock()
    mock_asr.transcribe_stream.return_value = iter(utterances)

    mock_classifier = MagicMock()
    mock_classifier.classify.side_effect = decisions

    captured_older_turns: list[list[Turn]] = []

    def _lm_gen(ctx, conv):  # type: ignore[no-untyped-def]
        captured_older_turns.append(list(conv.older_turns))
        yield "Agent answer."

    mock_llm = MagicMock()
    mock_llm.respond_stream.side_effect = _lm_gen

    mock_tts = MagicMock()
    mock_tts.stream_synthesize.return_value = iter([_FAKE_AUDIO_CHUNK])

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline._build_classifier", return_value=mock_classifier),
        patch("meeting_agent.pipeline._build_llm_client", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play"),
        patch("meeting_agent.pipeline._install_exception_log", return_value=MagicMock()),
    ):
        Pipeline(PipelineConfig()).run()

    # At time of LLM call, older_turns should have the 2 silent turns
    assert len(captured_older_turns) == 1
    older = captured_older_turns[0]
    assert len(older) == 2
    assert older[0] == Turn(speaker="Jason", text="Question from Jason")
    assert older[1] == Turn(speaker="Aziz", text="Question from Aziz")


# ---------------------------------------------------------------------------
# V2 pipeline: KeyboardInterrupt exits cleanly
# ---------------------------------------------------------------------------


def test_keyboard_interrupt_exits_cleanly():
    """run() returns normally (does not raise) when Ctrl-C is received."""
    mock_asr = MagicMock()
    mock_asr.transcribe_stream.side_effect = KeyboardInterrupt

    mock_classifier = MagicMock()
    mock_llm = MagicMock()
    mock_tts = MagicMock()

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline._build_classifier", return_value=mock_classifier),
        patch("meeting_agent.pipeline._build_llm_client", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play"),
        patch("meeting_agent.pipeline._install_exception_log", return_value=MagicMock()),
    ):
        # Must not raise
        Pipeline(PipelineConfig()).run()


# ---------------------------------------------------------------------------
# Coverage gap tests: helpers not exercised by pipeline integration tests
# ---------------------------------------------------------------------------


def test_install_exception_log_creates_logger(tmp_path, monkeypatch):
    """_install_exception_log creates the log dir and returns a configured logger."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    # Remove any pre-existing handlers to test the initialisation path.
    exc_logger = logging.getLogger("meeting_agent.exceptions")
    exc_logger.handlers.clear()

    logger = _install_exception_log()

    assert logger.name == "meeting_agent.exceptions"
    assert logger.level == logging.INFO
    assert len(logger.handlers) >= 1
    assert (tmp_path / "exceptions.jsonl").parent.is_dir()


def test_install_exception_log_no_duplicate_handlers(tmp_path, monkeypatch):
    """_install_exception_log does not add a second handler when called twice."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    exc_logger = logging.getLogger("meeting_agent.exceptions")
    exc_logger.handlers.clear()

    _install_exception_log()
    handler_count_after_first = len(exc_logger.handlers)

    # Second call must not add another handler.
    _install_exception_log()
    assert len(exc_logger.handlers) == handler_count_after_first


def test_circuit_breaker_prunes_old_failures():
    """Failures older than fail_window_s do not count toward the threshold."""
    cb = CircuitBreaker(fail_threshold=2, fail_window_s=1.0, open_s=15.0)

    with patch("meeting_agent.pipeline.time.monotonic") as mock_time:
        # First failure at t=0
        mock_time.return_value = 0.0
        try:
            with cb:
                raise RuntimeError("old failure")
        except RuntimeError:
            pass

        # Second failure at t=2.0 — beyond the 1.0s window; old failure is pruned
        mock_time.return_value = 2.0
        try:
            with cb:
                raise RuntimeError("recent failure")
        except RuntimeError:
            pass

    # Only 1 failure within the window (fail_threshold=2) → circuit stays closed
    with cb:
        pass  # must not raise CircuitBreakerOpen


def test_deafness_probe_prunes_old_drops():
    """Drops older than window_s are pruned and don't count toward threshold."""
    dp = DeafnessProbe(threshold=2, window_s=1.0)

    with patch("meeting_agent.pipeline.time.monotonic") as mock_time:
        mock_time.return_value = 0.0
        dp.record_drop()

        # Advance time past the window; the old drop is pruned on next record_drop
        mock_time.return_value = 2.0
        dp.record_drop()

    # Only 1 drop is within the window → should not probe
    assert len(dp._drops) == 1
    assert not dp.should_probe()


def test_unpunctuated_lm_response_still_synthesised():
    """LLM output without terminal punctuation is synthesised via the tail flush path."""
    utterance = _make_utterance("Tell me something")
    mock_asr, mock_classifier, mock_llm, mock_tts = _make_v2_mocks(
        utterances=[utterance],
        decisions=[_full_answer_decision(speaker="Jason")],
        lm_deltas=["Sure thing"],  # no terminal punctuation → tail flush path
    )
    _run_v2_pipeline(mock_asr, mock_classifier, mock_llm, mock_tts)

    mock_tts.stream_synthesize.assert_called_once_with("Sure thing")


# ---------------------------------------------------------------------------
# _strip_tts_markdown unit tests
# ---------------------------------------------------------------------------


def test_strip_tts_markdown_bold_italic():
    """Bold and italic markers are replaced with the wrapped text."""
    assert _strip_tts_markdown("This is **bold** and *italic*") == "This is bold and italic"


def test_strip_tts_markdown_bullets():
    """Bullet list prefixes are stripped, leaving only the item text."""
    assert _strip_tts_markdown("- first\n- second") == "first\nsecond"


def test_strip_tts_markdown_numbered_list():
    """Numbered list prefixes are stripped, leaving only the item text."""
    assert _strip_tts_markdown("1. first\n2. second") == "first\nsecond"


def test_strip_tts_markdown_headings():
    """Heading prefixes (# through ######) are stripped."""
    assert _strip_tts_markdown("# H1\n## H2\ntext") == "H1\nH2\ntext"


def test_strip_tts_markdown_inline_code():
    """Inline code backticks are removed, leaving the code text."""
    assert _strip_tts_markdown("use `foo()` here") == "use foo() here"


def test_strip_tts_markdown_fenced_code_removed():
    """Fenced code blocks are removed entirely."""
    result = _strip_tts_markdown("before\n```\ncode\n```\nafter")
    assert result == "before\n\nafter"


def test_strip_tts_markdown_idempotent():
    """Running the filter twice produces the same result as running it once."""
    text = "This is **bold** and *italic*\n- bullet\n## Heading"
    once = _strip_tts_markdown(text)
    twice = _strip_tts_markdown(once)
    assert once == twice


def test_strip_tts_markdown_preserves_plain_prose():
    """Plain prose without any markdown is returned unchanged."""
    text = "Just a normal sentence."
    assert _strip_tts_markdown(text) == text


def test_strip_tts_markdown_preserves_punctuation_in_clarifier():
    """Apostrophes, question marks, and quoted phrases are preserved."""
    text = "When you say 'skill', do you mean Y?"
    assert _strip_tts_markdown(text) == text


def test_stream_and_play_applies_filter_before_tts():
    """TTS receives the filtered (markdown-stripped) text, not the raw LLM output."""
    utterance = _make_utterance("Question")
    mock_asr, mock_classifier, mock_llm, mock_tts = _make_v2_mocks(
        utterances=[utterance],
        decisions=[_full_answer_decision(speaker="Jason")],
        lm_deltas=["Here's **what** I think."],
    )
    _run_v2_pipeline(mock_asr, mock_classifier, mock_llm, mock_tts)

    mock_tts.stream_synthesize.assert_called_once_with("Here's what I think.")


def test_stream_and_play_preserves_raw_text_in_transcript():
    """_stream_and_play returns the raw (unfiltered) LLM text for the transcript.

    The pipeline stores the return value of _stream_and_play verbatim in
    conversation.older_turns.  Confirming the return value is raw ensures that
    the transcript records what the LLM actually generated, not the TTS-cleaned
    version.
    """
    import logging

    from meeting_agent.llm import Conversation
    from meeting_agent.trace import Tracer

    mock_llm = MagicMock()
    mock_llm.respond_stream.return_value = iter(["Here's **what** I think."])

    mock_tts = MagicMock()
    mock_tts.stream_synthesize.return_value = iter([_FAKE_AUDIO_CHUNK])

    pipeline = Pipeline(PipelineConfig())
    conversation = Conversation()
    conversation.latest_turn = Turn(speaker="Jason", text="Question")
    noop_tracer = Tracer(
        enabled=False, verbose=False, logger=logging.getLogger("meeting_agent.trace")
    )

    with patch("meeting_agent.audio.play"):
        raw, tts_s = pipeline._stream_and_play(
            PipelineConfig(),
            conversation,
            mock_llm,
            mock_tts,
            0.0,
            noop_tracer,
            "Question",
        )

    # Return value must be the unfiltered LLM output.
    assert raw == "Here's **what** I think."
    # TTS must have received the filtered version.
    mock_tts.stream_synthesize.assert_called_once_with("Here's what I think.")
    # tts_s is the audio duration of the mocked chunk (24000 samples / 24000 Hz = 1.0 s)
    assert tts_s == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# mic-echo fix: silence flush + drain window tests
# ---------------------------------------------------------------------------


def test_stream_and_play_silence_flush_after_last_tts_chunk():
    """_stream_and_play plays a zero-filled silence flush after all TTS chunks.

    Root cause: PortAudio fires "stream finished" when its ring-buffer is
    exhausted, but the hardware DAC still has the last callback-block of audio
    in its output buffer.  Without a silence pad the very end of the TTS audio
    is cut off (scrambled cutoff artefact).  The silence pad forces those
    samples to play out cleanly.

    The final call to ``audio.play()`` after ``worker.join()`` must be a
    silence buffer (all zeros) at the TTS sample rate.
    """
    import logging

    from meeting_agent.llm import Conversation
    from meeting_agent.trace import Tracer

    mock_llm = MagicMock()
    mock_llm.respond_stream.return_value = iter(["Hello world."])

    tts_chunk = np.ones(24_000, dtype=np.float32) * 0.5  # non-zero audio
    mock_tts = MagicMock()
    mock_tts.stream_synthesize.return_value = iter([tts_chunk])

    pipeline = Pipeline(PipelineConfig())
    conversation = Conversation()
    conversation.latest_turn = Turn(speaker="Jason", text="Hi")
    noop_tracer = Tracer(
        enabled=False, verbose=False, logger=logging.getLogger("meeting_agent.trace")
    )

    play_calls: list[np.ndarray] = []

    def capture_play(audio_arr: np.ndarray, **_kwargs: object) -> None:  # type: ignore[override]
        play_calls.append(audio_arr.copy())

    with patch("meeting_agent.audio.play", side_effect=capture_play):
        pipeline._stream_and_play(
            PipelineConfig(),
            conversation,
            mock_llm,
            mock_tts,
            0.0,
            noop_tracer,
            "Hi",
        )

    assert len(play_calls) >= 2, "Expected at least one TTS chunk + one silence flush"
    # The LAST play() call must be an all-zeros silence flush
    last = play_calls[-1]
    assert np.all(last == 0.0), "Last play() call should be the silence flush (all zeros)"
    # Silence flush must be non-empty (covers _SPEAKER_FLUSH_S > 0)
    assert len(last) > 0, "Silence flush must not be empty"


def test_drain_duration_covers_tts_audio_duration():
    """Echo drain window is at least as long as the TTS audio that played.

    Root cause: if the drain window is shorter than the TTS audio duration,
    the mic queue still contains echo from the end of the agent's speech when
    ASR resumes — producing a spurious utterance with the tail of the response.

    The drain call uses ``max(speak_duration, tts_audio_s) + _ECHO_TAIL_S``.
    In production speak_duration ≥ tts_audio_s; in the mocked test environment
    speak_duration ≈ 0, so tts_audio_s becomes the binding floor.
    Either way, drain_duration ≥ tts_audio_s must hold.
    """
    # 2 seconds of TTS audio at 24 kHz
    tts_audio_2s = np.zeros(int(24_000 * 2), dtype=np.float32)
    utterance = _make_utterance("What is the plan?")
    mock_asr, mock_classifier, mock_llm, mock_tts = _make_v2_mocks(
        utterances=[utterance],
        decisions=[_full_answer_decision()],
        lm_deltas=["Here is the plan."],
        tts_audio=tts_audio_2s,
    )

    drain_durations: list[float] = []
    original_drain = Pipeline._drain_echo

    def capture_drain(self: Pipeline, chunks_iter: object, duration_s: float) -> None:
        drain_durations.append(duration_s)
        original_drain(self, chunks_iter, duration_s)  # type: ignore[arg-type]

    with patch.object(Pipeline, "_drain_echo", capture_drain):
        _run_v2_pipeline(mock_asr, mock_classifier, mock_llm, mock_tts)

    assert drain_durations, "_drain_echo should have been called for the response"
    tts_audio_s = 2.0  # 24000 * 2 / 24000
    assert drain_durations[0] >= tts_audio_s, (
        f"Drain duration {drain_durations[0]:.3f}s is shorter than TTS audio "
        f"duration {tts_audio_s:.3f}s — echo tail will not be fully consumed"
    )


# ---------------------------------------------------------------------------
# Classifier factory unit tests
# ---------------------------------------------------------------------------


def test_classifier_factory_selects_bedrock_by_default():
    """Default PipelineConfig produces a BedrockClassifier."""
    config = PipelineConfig()
    classifier = _build_classifier(config)
    assert isinstance(classifier, BedrockClassifier)


def test_classifier_factory_selects_ollama_when_configured():
    """classifier_backend='ollama' in PipelineConfig produces an OllamaClassifier."""
    config = PipelineConfig(classifier_backend="ollama")
    classifier = _build_classifier(config)
    assert isinstance(classifier, OllamaClassifier)


def test_classifier_model_override_reaches_bedrock_backend():
    """classifier_model override is forwarded to BedrockClassifier.model_id."""
    config = PipelineConfig(classifier_model="us.anthropic.claude-opus-4-5")
    classifier = _build_classifier(config)
    assert isinstance(classifier, BedrockClassifier)
    assert classifier.model_id == "us.anthropic.claude-opus-4-5"


def test_classifier_model_override_reaches_ollama_backend():
    """classifier_model override is forwarded to OllamaClassifier.model."""
    config = PipelineConfig(
        classifier_backend="ollama",
        classifier_model="qwen3.6:35b-a3b-mlx-bf16",
    )
    classifier = _build_classifier(config)
    assert isinstance(classifier, OllamaClassifier)
    assert classifier.model == "qwen3.6:35b-a3b-mlx-bf16"


def test_classifier_factory_bedrock_default_model():
    """BedrockClassifier uses its DEFAULT_MODEL_ID when classifier_model is None."""
    config = PipelineConfig()
    classifier = _build_classifier(config)
    assert isinstance(classifier, BedrockClassifier)
    assert classifier.model_id == BedrockClassifier.DEFAULT_MODEL_ID


def test_classifier_factory_ollama_default_model():
    """OllamaClassifier uses its DEFAULT_MODEL when classifier_model is None."""
    config = PipelineConfig(classifier_backend="ollama")
    classifier = _build_classifier(config)
    assert isinstance(classifier, OllamaClassifier)
    assert classifier.model == OllamaClassifier.DEFAULT_MODEL


def test_pipeline_warms_up_ollama_classifier_on_run():
    """Pipeline.run() calls warm_up() on an OllamaClassifier backend before the first utterance."""
    # A real OllamaClassifier instance so isinstance(classifier, OllamaClassifier) passes;
    # ollama.Client is patched so warm_up's chat call hits the mock.
    mock_ollama_client = MagicMock()
    mock_ollama_client.chat.return_value = {"message": {"content": "ok"}}

    utterance = _make_utterance("Hello agent")
    mock_asr, _, mock_llm, mock_tts = _make_v2_mocks(utterances=[utterance])

    with patch("meeting_agent.classifier.ollama.Client", return_value=mock_ollama_client):
        classifier = OllamaClassifier()
        # Replace classify so the pipeline's per-utterance call doesn't hit the mock
        # and muddy the warm_up call-count assertion.
        classifier.classify = MagicMock(return_value=_full_answer_decision(speaker="Jason"))

        with (
            patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
            patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
            patch("meeting_agent.pipeline._build_classifier", return_value=classifier),
            patch("meeting_agent.pipeline._build_llm_client", return_value=mock_llm),
            patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
            patch("meeting_agent.audio.play"),
            patch("meeting_agent.pipeline._install_exception_log", return_value=MagicMock()),
        ):
            Pipeline(PipelineConfig(classifier_backend="ollama")).run()

    # warm_up() uses a minimal 1-token request; exactly one such call expected.
    warmup_calls = [
        c for c in mock_ollama_client.chat.call_args_list if c[1]["options"]["num_predict"] == 1
    ]
    assert len(warmup_calls) == 1, "warm_up() should issue exactly one minimal chat call"


def test_pipeline_skips_warm_up_for_bedrock_classifier():
    """Pipeline.run() does not call warm_up on a Bedrock classifier (no-op path)."""
    utterance = _make_utterance("Hello agent")
    mock_asr, mock_classifier, mock_llm, mock_tts = _make_v2_mocks(
        utterances=[utterance],
        decisions=[_full_answer_decision(speaker="Jason")],
    )
    # Spec'd as BedrockClassifier so isinstance(classifier, OllamaClassifier) is False.
    bedrock_mock = MagicMock(spec=BedrockClassifier)
    bedrock_mock.classify.side_effect = mock_classifier.classify.side_effect

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline._build_classifier", return_value=bedrock_mock),
        patch("meeting_agent.pipeline._build_llm_client", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play"),
        patch("meeting_agent.pipeline._install_exception_log", return_value=MagicMock()),
    ):
        Pipeline(PipelineConfig()).run()

    # BedrockClassifier has no warm_up attribute → MagicMock(spec=...) would also
    # not expose one. Assert that no warm_up was attempted.
    assert not hasattr(bedrock_mock, "warm_up") or not bedrock_mock.warm_up.called


# ---------------------------------------------------------------------------
# Response-LLM factory unit tests
# ---------------------------------------------------------------------------


def test_llm_factory_selects_bedrock_by_default():
    """Default PipelineConfig produces a BedrockClient."""
    config = PipelineConfig()
    llm = _build_llm_client(config)
    assert isinstance(llm, BedrockClient)


def test_llm_factory_selects_ollama_when_configured():
    """llm_backend='ollama' in PipelineConfig produces an OllamaClient."""
    config = PipelineConfig(llm_backend="ollama")
    llm = _build_llm_client(config)
    assert isinstance(llm, OllamaClient)


def test_llm_model_override_reaches_bedrock_backend():
    """llm_model override is forwarded to BedrockClient.model_id."""
    config = PipelineConfig(llm_model="us.anthropic.claude-opus-4-5")
    llm = _build_llm_client(config)
    assert isinstance(llm, BedrockClient)
    assert llm.model_id == "us.anthropic.claude-opus-4-5"


def test_llm_model_override_reaches_ollama_backend():
    """llm_model override is forwarded to OllamaClient.model."""
    config = PipelineConfig(
        llm_backend="ollama",
        llm_model="qwen3.6:35b-a3b-mlx-bf16",
    )
    llm = _build_llm_client(config)
    assert isinstance(llm, OllamaClient)
    assert llm.model == "qwen3.6:35b-a3b-mlx-bf16"


def test_llm_factory_bedrock_default_model():
    """BedrockClient uses its DEFAULT_MODEL_ID when llm_model is None."""
    config = PipelineConfig()
    llm = _build_llm_client(config)
    assert isinstance(llm, BedrockClient)
    assert llm.model_id == BedrockClient.DEFAULT_MODEL_ID


def test_llm_factory_ollama_default_model():
    """OllamaClient uses its DEFAULT_MODEL when llm_model is None."""
    config = PipelineConfig(llm_backend="ollama")
    llm = _build_llm_client(config)
    assert isinstance(llm, OllamaClient)
    assert llm.model == OllamaClient.DEFAULT_MODEL


def test_pipeline_warms_up_ollama_llm_on_run():
    """Pipeline.run() calls warm_up() on an OllamaClient LLM backend before the first utterance."""
    mock_ollama_client = MagicMock()
    mock_ollama_client.chat.return_value = {"message": {"content": "ok"}}

    utterance = _make_utterance("Hello agent")
    mock_asr, mock_classifier, _, mock_tts = _make_v2_mocks(utterances=[utterance])

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama_client):
        llm = OllamaClient()
        # Patch respond_stream so the pipeline's per-utterance call doesn't muddy
        # the warm_up call-count assertion.
        llm.respond_stream = MagicMock(  # type: ignore[method-assign]
            return_value=iter(["Reply."])
        )

        with (
            patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
            patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
            patch("meeting_agent.pipeline._build_classifier", return_value=mock_classifier),
            patch("meeting_agent.pipeline._build_llm_client", return_value=llm),
            patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
            patch("meeting_agent.audio.play"),
            patch("meeting_agent.pipeline._install_exception_log", return_value=MagicMock()),
        ):
            Pipeline(PipelineConfig(llm_backend="ollama")).run()

    # warm_up() uses a minimal 1-token request; exactly one such call expected.
    warmup_calls = [
        c for c in mock_ollama_client.chat.call_args_list if c[1]["options"]["num_predict"] == 1
    ]
    assert len(warmup_calls) == 1, "warm_up() should issue exactly one minimal chat call"


def test_pipeline_skips_warm_up_for_bedrock_llm():
    """Pipeline.run() does not call warm_up on a Bedrock LLM (no-op path)."""
    utterance = _make_utterance("Hello agent")
    mock_asr, mock_classifier, mock_llm, mock_tts = _make_v2_mocks(
        utterances=[utterance],
        decisions=[_full_answer_decision(speaker="Jason")],
    )
    # Spec'd as BedrockClient so isinstance(llm, OllamaClient) is False.
    bedrock_llm_mock = MagicMock(spec=BedrockClient)
    bedrock_llm_mock.respond_stream.return_value = iter(["Reply."])

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline._build_classifier", return_value=mock_classifier),
        patch("meeting_agent.pipeline._build_llm_client", return_value=bedrock_llm_mock),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play"),
        patch("meeting_agent.pipeline._install_exception_log", return_value=MagicMock()),
    ):
        Pipeline(PipelineConfig()).run()

    # BedrockClient has no warm_up attribute in its spec.
    assert not hasattr(bedrock_llm_mock, "warm_up") or not bedrock_llm_mock.warm_up.called
