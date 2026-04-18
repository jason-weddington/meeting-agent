"""Tests for meeting_agent.trace — dev-mode structured trace log."""

from __future__ import annotations

import io
import json
import logging
import queue
import sys
import time
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meeting_agent.asr import Utterance
from meeting_agent.classifier import Decision
from meeting_agent.pipeline import Pipeline, PipelineConfig
from meeting_agent.trace import (
    Tracer,
    _DropOldestQueue,
    _emit_verbose_line,
    install,
)

# ---------------------------------------------------------------------------
# Helpers shared with pipeline tests
# ---------------------------------------------------------------------------

_FAKE_MIC_CHUNK = np.zeros(1600, dtype=np.float32)
_FAKE_AUDIO_CHUNK = np.zeros(24_000, dtype=np.float32)


def _infinite_chunks() -> Iterator[np.ndarray]:  # type: ignore[type-arg]
    """Yield the same silent chunk forever."""
    while True:
        yield _FAKE_MIC_CHUNK.copy()


def _make_utterance(
    text: str = "Test utterance",
    avg_logprob: float = -0.3,
    no_speech_prob: float = 0.05,
    compression_ratio: float = 1.2,
) -> Utterance:
    """Build a high-confidence Utterance for testing."""
    return Utterance(
        text=text,
        start_s=0.0,
        end_s=2.0,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
        compression_ratio=compression_ratio,
    )


def _make_low_confidence_utterance() -> Utterance:
    """Build a low-confidence utterance that will be dropped by the gate."""
    return _make_utterance(avg_logprob=-1.5, no_speech_prob=0.05, compression_ratio=1.0)


def _run_traced_pipeline(
    config: PipelineConfig,
    utterances: list[Utterance],
    decisions: list[Decision] | None = None,
    lm_deltas: list[str] | None = None,
) -> None:
    """Run the pipeline with trace enabled, mocking all I/O boundaries."""
    if lm_deltas is None:
        lm_deltas = ["Agent reply."]

    mock_asr = MagicMock()
    mock_asr.transcribe_stream.return_value = iter(utterances)

    mock_classifier = MagicMock()
    if decisions is not None:
        mock_classifier.classify.side_effect = decisions
    else:
        mock_classifier.classify.return_value = Decision(
            speaker="Jason", action="silent", confidence=0.9
        )

    mock_llm = MagicMock()
    mock_llm.respond_stream.return_value = iter(lm_deltas)

    mock_tts = MagicMock()
    mock_tts.stream_synthesize.return_value = iter([_FAKE_AUDIO_CHUNK])

    with (
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline._build_classifier", return_value=mock_classifier),
        patch("meeting_agent.pipeline.BedrockClient", return_value=mock_llm),
        patch("meeting_agent.pipeline.TTS", return_value=mock_tts),
        patch("meeting_agent.audio.play"),
        patch("meeting_agent.pipeline._install_exception_log", return_value=MagicMock()),
    ):
        Pipeline(config).run()


# ---------------------------------------------------------------------------
# Fixture: clean trace logger between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_trace_logger() -> Iterator[None]:
    """Clear the trace logger's handlers before and after each test."""
    logger = logging.getLogger("meeting_agent.trace")
    logger.handlers.clear()
    yield
    logger.handlers.clear()


# ---------------------------------------------------------------------------
# test_tracer_disabled_is_no_op
# ---------------------------------------------------------------------------


def test_tracer_disabled_is_no_op(tmp_path: pytest.MonkeyPatch) -> None:
    """Disabled tracer emits nothing and creates no files."""
    tracer, listener = install(enabled=False, verbose=False, log_dir=tmp_path)  # type: ignore[arg-type]
    assert listener is None
    assert not tracer.enabled

    for i in range(1000):
        tracer.emit("test_event", index=i)

    assert not (tmp_path / "trace.jsonl").exists()  # type: ignore[operator]


# ---------------------------------------------------------------------------
# test_tracer_enabled_writes_jsonl
# ---------------------------------------------------------------------------


def test_tracer_enabled_writes_jsonl(tmp_path: pytest.MonkeyPatch) -> None:
    """Enabled tracer writes JSON records to trace.jsonl after listener.stop()."""
    tracer, listener = install(enabled=True, verbose=False, log_dir=tmp_path)  # type: ignore[arg-type]
    assert listener is not None

    try:
        tracer.emit("my_event", foo="bar", baz=42)
    finally:
        listener.stop()

    trace_file = tmp_path / "trace.jsonl"  # type: ignore[operator]
    assert trace_file.exists()

    lines = [ln for ln in trace_file.read_text().strip().split("\n") if ln.strip()]
    assert len(lines) >= 1

    record = json.loads(lines[0])
    assert record["event"] == "my_event"
    assert record["foo"] == "bar"
    assert record["baz"] == 42
    assert "ts" in record
    assert isinstance(record["ts"], float)


# ---------------------------------------------------------------------------
# test_queue_drops_oldest_on_overflow
# ---------------------------------------------------------------------------


def test_queue_drops_oldest_on_overflow() -> None:
    """_DropOldestQueue drops the oldest item when full, never blocks."""
    q: _DropOldestQueue = _DropOldestQueue(maxsize=2)

    q.put("first")
    q.put("second")
    q.put("third")  # "first" must be evicted

    collected: list[object] = []
    try:
        while True:
            collected.append(q.get_nowait())
    except queue.Empty:
        pass

    assert "first" not in collected
    assert "second" in collected
    assert "third" in collected
    assert len(collected) == 2


def test_queue_drops_oldest_put_nowait() -> None:
    """_DropOldestQueue.put_nowait also honours the drop-oldest policy."""
    q: _DropOldestQueue = _DropOldestQueue(maxsize=2)

    q.put_nowait("a")
    q.put_nowait("b")
    q.put_nowait("c")  # "a" must be evicted

    collected: list[object] = []
    try:
        while True:
            collected.append(q.get_nowait())
    except queue.Empty:
        pass

    assert "a" not in collected
    assert "b" in collected
    assert "c" in collected


# ---------------------------------------------------------------------------
# test_verbose_mode_writes_to_stderr
# ---------------------------------------------------------------------------


def test_verbose_mode_writes_to_stderr() -> None:
    """Verbose tracer writes one-line summaries to stderr."""
    fake_stderr = io.StringIO()
    # Build a tracer directly (no file I/O needed for this test)
    null_logger = logging.getLogger("test.trace.verbose")
    null_logger.addHandler(logging.NullHandler())
    null_logger.propagate = False

    tracer = Tracer(enabled=True, verbose=True, logger=null_logger)

    with patch("sys.stderr", fake_stderr):
        tracer.emit(
            "classifier_decision",
            utterance_text="what is the plan?",
            utterance_age_s=0.3,
            asr_confidence={
                "avg_logprob": -0.12,
                "no_speech_prob": 0.02,
                "compression_ratio": 1.80,
            },
            recent_turns_snapshot=[],
            agent_turns_last_30s=0,
            agent_turns_last_5min=0,
            decision_speaker="Jason",
            decision_action="full_answer",
            decision_confidence=0.88,
        )

    output = fake_stderr.getvalue()
    assert "classifier" in output
    assert "full_answer" in output
    assert "Jason" in output
    assert "0.88" in output
    assert len(output.rstrip("\n")) <= 200


def test_verbose_mode_pre_gate_drop_to_stderr() -> None:
    """pre_gate_drop events produce a stderr line containing the reason."""
    fake_stderr = io.StringIO()
    null_logger = logging.getLogger("test.trace.verbose2")
    null_logger.addHandler(logging.NullHandler())
    null_logger.propagate = False

    tracer = Tracer(enabled=True, verbose=True, logger=null_logger)

    with patch("sys.stderr", fake_stderr):
        tracer.emit(
            "pre_gate_drop",
            utterance_text="mumble",
            reason="low_confidence",
            asr_confidence={
                "avg_logprob": -1.4,
                "no_speech_prob": 0.8,
                "compression_ratio": 2.9,
            },
        )

    output = fake_stderr.getvalue()
    assert "pre_gate_drop" in output
    assert "low_confidence" in output


def test_verbose_mode_response_emitted_to_stderr() -> None:
    """response_emitted events show ttft/total in stderr."""
    fake_stderr = io.StringIO()
    null_logger = logging.getLogger("test.trace.verbose3")
    null_logger.addHandler(logging.NullHandler())
    null_logger.propagate = False

    tracer = Tracer(enabled=True, verbose=True, logger=null_logger)

    with patch("sys.stderr", fake_stderr):
        tracer.emit(
            "response_emitted",
            triggering_utterance="what now?",
            response_text="Here is the plan.",
            bedrock_ttft_s=0.38,
            tts_pipeline_overhead_s=0.14,
            total_turn_latency_s=0.52,
        )

    output = fake_stderr.getvalue()
    assert "response_emitted" in output
    assert "380ms" in output
    assert "520ms" in output


# ---------------------------------------------------------------------------
# test_run_emits_classifier_decision_when_traced
# ---------------------------------------------------------------------------


def test_run_emits_classifier_decision_when_traced(tmp_path: pytest.MonkeyPatch) -> None:
    """Pipeline emits a classifier_decision record to trace.jsonl when tracing is on."""
    config = PipelineConfig(
        trace_enabled=True,
        trace_log_dir=tmp_path,  # type: ignore[arg-type]
    )

    utterance = _make_utterance("What is the Q2 plan?")
    decisions = [Decision(speaker="Jason", action="silent", confidence=0.9)]

    _run_traced_pipeline(config, utterances=[utterance], decisions=decisions)

    trace_file = tmp_path / "trace.jsonl"  # type: ignore[operator]
    assert trace_file.exists()

    records = [json.loads(ln) for ln in trace_file.read_text().strip().split("\n") if ln.strip()]
    classifier_records = [r for r in records if r["event"] == "classifier_decision"]
    assert len(classifier_records) >= 1

    rec = classifier_records[0]
    assert rec["utterance_text"] == "What is the Q2 plan?"
    assert rec["decision_speaker"] == "Jason"
    assert rec["decision_action"] == "silent"
    assert "asr_confidence" in rec
    assert "utterance_age_s" in rec


# ---------------------------------------------------------------------------
# test_run_emits_pre_gate_drop_when_low_confidence
# ---------------------------------------------------------------------------


def test_run_emits_pre_gate_drop_when_low_confidence(
    tmp_path: pytest.MonkeyPatch,
) -> None:
    """Pipeline emits a pre_gate_drop record for low-confidence utterances."""
    config = PipelineConfig(
        trace_enabled=True,
        trace_log_dir=tmp_path,  # type: ignore[arg-type]
    )

    low_conf = _make_low_confidence_utterance()

    _run_traced_pipeline(config, utterances=[low_conf])

    trace_file = tmp_path / "trace.jsonl"  # type: ignore[operator]
    assert trace_file.exists()

    records = [json.loads(ln) for ln in trace_file.read_text().strip().split("\n") if ln.strip()]
    drop_records = [r for r in records if r["event"] == "pre_gate_drop"]
    assert len(drop_records) >= 1
    assert drop_records[0]["reason"] == "low_confidence"
    assert "asr_confidence" in drop_records[0]


# ---------------------------------------------------------------------------
# test_run_emits_response_emitted_on_full_answer
# ---------------------------------------------------------------------------


def test_run_emits_response_emitted_on_full_answer(
    tmp_path: pytest.MonkeyPatch,
) -> None:
    """Pipeline emits response_emitted with timing fields after a full_answer turn."""
    config = PipelineConfig(
        trace_enabled=True,
        trace_log_dir=tmp_path,  # type: ignore[arg-type]
    )

    utterance = _make_utterance("Tell me the status.")
    decisions = [Decision(speaker="Jason", action="full_answer", confidence=0.9)]

    _run_traced_pipeline(
        config,
        utterances=[utterance],
        decisions=decisions,
        lm_deltas=["The status is green."],
    )

    trace_file = tmp_path / "trace.jsonl"  # type: ignore[operator]
    assert trace_file.exists()

    records = [json.loads(ln) for ln in trace_file.read_text().strip().split("\n") if ln.strip()]
    response_records = [r for r in records if r["event"] == "response_emitted"]
    assert len(response_records) >= 1

    rec = response_records[0]
    assert rec["triggering_utterance"] == "Tell me the status."
    assert "The status is green." in rec["response_text"]
    # bedrock_ttft_s may be None if mock LLM responds instantly without a
    # measurable gap, but the field must be present.
    assert "bedrock_ttft_s" in rec
    assert "total_turn_latency_s" in rec


# ---------------------------------------------------------------------------
# test_listener_stops_on_pipeline_exit
# ---------------------------------------------------------------------------


def test_listener_stops_on_pipeline_exit(tmp_path: pytest.MonkeyPatch) -> None:
    """Pipeline calls listener.stop() in its finally block so the last records flush."""
    config = PipelineConfig(
        trace_enabled=True,
        trace_log_dir=tmp_path,  # type: ignore[arg-type]
    )

    mock_listener = MagicMock()
    mock_tracer = Tracer(
        enabled=False,
        verbose=False,
        logger=logging.getLogger("test.listener_stop"),
    )

    mock_asr = MagicMock()
    mock_asr.transcribe_stream.side_effect = KeyboardInterrupt

    with (
        patch(
            "meeting_agent.pipeline._install_trace",
            return_value=(mock_tracer, mock_listener),
        ),
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline.Classifier", return_value=MagicMock()),
        patch("meeting_agent.pipeline.BedrockClient", return_value=MagicMock()),
        patch("meeting_agent.pipeline.TTS", return_value=MagicMock()),
        patch("meeting_agent.audio.play"),
        patch("meeting_agent.pipeline._install_exception_log", return_value=MagicMock()),
    ):
        Pipeline(config).run()

    mock_listener.stop.assert_called_once()


def test_listener_stops_on_normal_exit(tmp_path: pytest.MonkeyPatch) -> None:
    """listener.stop() is also called when the iterator exhausts normally."""
    config = PipelineConfig(
        trace_enabled=True,
        trace_log_dir=tmp_path,  # type: ignore[arg-type]
    )

    mock_listener = MagicMock()
    mock_tracer = Tracer(
        enabled=False,
        verbose=False,
        logger=logging.getLogger("test.listener_stop2"),
    )

    mock_asr = MagicMock()
    mock_asr.transcribe_stream.return_value = iter([])  # empty — exits immediately

    with (
        patch(
            "meeting_agent.pipeline._install_trace",
            return_value=(mock_tracer, mock_listener),
        ),
        patch("meeting_agent.audio.record_chunks", return_value=_infinite_chunks()),
        patch("meeting_agent.pipeline.StreamingASR", return_value=mock_asr),
        patch("meeting_agent.pipeline.Classifier", return_value=MagicMock()),
        patch("meeting_agent.pipeline.BedrockClient", return_value=MagicMock()),
        patch("meeting_agent.pipeline.TTS", return_value=MagicMock()),
        patch("meeting_agent.audio.play"),
        patch("meeting_agent.pipeline._install_exception_log", return_value=MagicMock()),
    ):
        Pipeline(config).run()

    mock_listener.stop.assert_called_once()


# ---------------------------------------------------------------------------
# Additional coverage: _emit_verbose_line edge cases
# ---------------------------------------------------------------------------


def test_emit_verbose_line_unknown_event_does_not_raise() -> None:
    """_emit_verbose_line handles unknown event types without raising."""
    fake_stderr = io.StringIO()
    with patch("sys.stderr", fake_stderr):
        _emit_verbose_line({"ts": time.time(), "event": "unknown_xyz", "x": 1})
    assert "unknown_xyz" in fake_stderr.getvalue()


def test_emit_verbose_line_truncates_long_lines() -> None:
    """Output lines are capped at 200 characters."""
    fake_stderr = io.StringIO()
    long_text = "x" * 300
    with patch("sys.stderr", fake_stderr):
        _emit_verbose_line(
            {"ts": time.time(), "event": "utterance_received", "utterance_text": long_text}
        )
    line = fake_stderr.getvalue().rstrip("\n")
    assert len(line) <= 200


def test_emit_verbose_line_circuit_events() -> None:
    """Circuit state change events produce non-empty stderr output."""
    for event in ("circuit_open", "circuit_close", "circuit_half_open_probe"):
        fake_stderr = io.StringIO()
        with patch("sys.stderr", fake_stderr):
            _emit_verbose_line({"ts": time.time(), "event": event})
        assert event in fake_stderr.getvalue()


def test_emit_verbose_line_deafness_probe() -> None:
    """deafness_probe_fired events show drops and window in stderr."""
    fake_stderr = io.StringIO()
    with patch("sys.stderr", fake_stderr):
        _emit_verbose_line(
            {
                "ts": time.time(),
                "event": "deafness_probe_fired",
                "consecutive_drops": 3,
                "window_s": 30.0,
            }
        )
    output = fake_stderr.getvalue()
    assert "deafness_probe_fired" in output
    assert "3" in output


def test_emit_verbose_line_decision_outcome() -> None:
    """decision_outcome events show the outcome string in stderr."""
    fake_stderr = io.StringIO()
    with patch("sys.stderr", fake_stderr):
        _emit_verbose_line(
            {"ts": time.time(), "event": "decision_outcome", "outcome": "stale_drop"}
        )
    assert "stale_drop" in fake_stderr.getvalue()


# ---------------------------------------------------------------------------
# MEETING_AGENT_TRACE env var acceptance (via CLI)
# ---------------------------------------------------------------------------


def test_cli_env_var_enables_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """MEETING_AGENT_TRACE=1 sets trace_enabled=True on the produced config."""
    from meeting_agent.cli import main

    monkeypatch.setenv("MEETING_AGENT_TRACE", "1")

    captured_config: list[PipelineConfig] = []

    def fake_run(self: Pipeline) -> None:
        captured_config.append(self.config)

    with patch.object(Pipeline, "run", fake_run):
        import sys as _sys

        old_argv = _sys.argv
        _sys.argv = ["meeting-agent"]
        try:
            main()
        except SystemExit:
            pass
        finally:
            _sys.argv = old_argv

    if captured_config:
        assert captured_config[0].trace_enabled is True


# ---------------------------------------------------------------------------
# Coverage gaps: ANSI colour branch, line truncation, install variants
# ---------------------------------------------------------------------------


def test_emit_verbose_line_uses_ansi_when_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """_emit_verbose_line uses ANSI escape codes when stderr is a TTY."""
    fake_stderr = io.StringIO()
    fake_stderr.isatty = lambda: True  # type: ignore[method-assign]  # pretend TTY
    monkeypatch.setattr(sys, "stderr", fake_stderr)

    _emit_verbose_line(
        {
            "ts": time.time(),
            "event": "pre_gate_drop",
            "reason": "low_confidence",
            "asr_confidence": {
                "avg_logprob": -1.4,
                "no_speech_prob": 0.8,
                "compression_ratio": 2.9,
            },
        }
    )

    output = fake_stderr.getvalue()
    assert "\033[" in output  # ANSI escape sequences present


def test_emit_verbose_line_truncates_at_200_chars() -> None:
    """Generic event with very long data produces a line capped at 200 characters."""
    fake_stderr = io.StringIO()
    with patch("sys.stderr", fake_stderr):
        _emit_verbose_line(
            {
                "ts": time.time(),
                "event": "some_long_generic_event",
                "data": "x" * 300,
            }
        )
    line = fake_stderr.getvalue().rstrip("\n")
    assert len(line) <= 200
    assert line.endswith("...")


def test_install_uses_xdg_state_home_when_no_log_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.MonkeyPatch
) -> None:
    """install() falls back to $XDG_STATE_HOME/meeting-agent when log_dir is None."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    tracer, listener = install(enabled=True, verbose=False)
    assert listener is not None
    try:
        tracer.emit("probe", value=1)
    finally:
        listener.stop()

    expected_dir = tmp_path / "meeting-agent"  # type: ignore[operator]
    assert expected_dir.is_dir()


def test_install_replaces_stale_handlers(tmp_path: pytest.MonkeyPatch) -> None:
    """Calling install() twice replaces the old QueueHandler, avoiding duplicate writes."""
    first_dir = tmp_path / "run1"  # type: ignore[operator]
    second_dir = tmp_path / "run2"  # type: ignore[operator]

    tracer1, listener1 = install(enabled=True, verbose=False, log_dir=first_dir)
    first_handler = tracer1.logger.handlers[0]

    tracer2, listener2 = install(enabled=True, verbose=False, log_dir=second_dir)

    try:
        # Old handler should have been removed.
        assert first_handler not in tracer2.logger.handlers
    finally:
        if listener1 is not None:
            listener1.stop()
        if listener2 is not None:
            listener2.stop()


def test_cli_verbose_implies_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """--verbose flag implies trace_enabled=True."""
    from meeting_agent.cli import main

    monkeypatch.delenv("MEETING_AGENT_TRACE", raising=False)

    captured_config: list[PipelineConfig] = []

    def fake_run(self: Pipeline) -> None:
        captured_config.append(self.config)

    with patch.object(Pipeline, "run", fake_run):
        import sys as _sys

        old_argv = _sys.argv
        _sys.argv = ["meeting-agent", "--verbose"]
        try:
            main()
        except SystemExit:
            pass
        finally:
            _sys.argv = old_argv

    if captured_config:
        assert captured_config[0].trace_enabled is True
        assert captured_config[0].trace_verbose is True
