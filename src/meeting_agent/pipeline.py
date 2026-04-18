"""End-to-end meeting-agent pipeline orchestrator.

Wires audio capture → always-on streaming ASR → Bedrock Haiku classifier →
Bedrock Claude response LLM → sentence-pipelined TTS → audio playback.
Maintains the rolling transcript; gates the mic while the agent is speaking
to avoid feedback.

V2 flow (always-on, classifier-gated):
  1. Open mic stream.
  2. Feed every chunk to StreamingASR (VAD-gated Whisper).
  3. For each utterance, run the pre-classifier confidence gate.
  4. Pass surviving utterances through the Haiku classifier.
  5. If classifier says "silent", append to transcript and continue.
  6. If "hedged_answer" or "full_answer" (and not stale), respond via Bedrock
     Claude + TTS, then drain mic echo.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from meeting_agent import audio
from meeting_agent.asr import StreamingASR, Utterance
from meeting_agent.audio import AudioArray
from meeting_agent.classifier import Classifier, Confidence, SessionState
from meeting_agent.llm import BedrockClient, Conversation, ProjectContext, Turn
from meeting_agent.tts import SAMPLE_RATE as TTS_SAMPLE_RATE
from meeting_agent.tts import TTS

_metrics = logging.getLogger("meeting_agent.metrics")

_CHUNK_MS = 100
_ECHO_TAIL_S = 0.5  # extra drain time past playback to flush speaker echo

# Staleness thresholds: age > DROP_S → discard entirely; age in (HEDGE_S, DROP_S]
# and action == "full_answer" → downgrade to "hedged_answer".
_STALE_DROP_S = 5.0
_STALE_HEDGE_S = 1.5


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class CircuitBreakerOpen(Exception):
    """Raised by :class:`CircuitBreaker.__enter__` when the circuit is open."""


class CircuitBreaker:
    """Hand-rolled circuit breaker for the Bedrock Sonnet + TTS path.

    Opens after ``fail_threshold`` failures within ``fail_window_s`` seconds.
    Stays open for ``open_s`` seconds, then goes half-open (one probe allowed).
    """

    def __init__(
        self,
        fail_threshold: int = 3,
        fail_window_s: float = 10.0,
        open_s: float = 15.0,
    ) -> None:
        """Initialise closed circuit."""
        self._failures: deque[float] = deque()
        self._opened_at: float | None = None
        self._fail_threshold = fail_threshold
        self._fail_window_s = fail_window_s
        self._open_s = open_s

    def __enter__(self) -> CircuitBreaker:
        """Check if circuit is open; raise :class:`CircuitBreakerOpen` if so."""
        if self._opened_at is not None:
            if time.monotonic() - self._opened_at < self._open_s:
                raise CircuitBreakerOpen()
            self._opened_at = None  # half-open probe
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        """Record any exception as a failure; open circuit if threshold reached."""
        if exc is not None:
            now = time.monotonic()
            self._failures.append(now)
            # Prune failures outside the window.
            while self._failures and self._failures[0] < now - self._fail_window_s:
                self._failures.popleft()
            if len(self._failures) >= self._fail_threshold:
                self._opened_at = now


# ---------------------------------------------------------------------------
# AirtimeTracker
# ---------------------------------------------------------------------------


class AirtimeTracker:
    """Rolling count of agent emission events for airtime budgeting."""

    def __init__(self) -> None:
        """Initialise empty emission log."""
        self._emissions: deque[float] = deque()

    def record_emission(self, t: float) -> None:
        """Record a new emission at monotonic timestamp ``t``."""
        self._emissions.append(t)
        cutoff = t - 300.0  # keep at most last 5 minutes
        while self._emissions and self._emissions[0] < cutoff:
            self._emissions.popleft()

    def count_last(self, window_s: float) -> int:
        """Return the number of emissions in the last ``window_s`` seconds."""
        now = time.monotonic()
        return sum(1 for t in self._emissions if t > now - window_s)


# ---------------------------------------------------------------------------
# DeafnessProbe
# ---------------------------------------------------------------------------


class DeafnessProbe:
    """One-shot deafness probe: speaks if too many utterances drop on confidence.

    After ``threshold`` low-confidence drops within ``window_s``, emits a
    single verbal acknowledgement that audio quality is degraded. Fires at
    most once per ``Pipeline.run()`` invocation.
    """

    PROBE_TEXT = (
        "I think I'm losing some audio on my end — I might be dropping parts of what you're saying."
    )

    def __init__(self, threshold: int = 3, window_s: float = 30.0) -> None:
        """Initialise with zero recorded drops."""
        self._drops: deque[float] = deque()
        self._used = False
        self._threshold = threshold
        self._window_s = window_s

    def record_drop(self) -> None:
        """Record a confidence-gate drop at the current time."""
        now = time.monotonic()
        self._drops.append(now)
        while self._drops and self._drops[0] < now - self._window_s:
            self._drops.popleft()

    def should_probe(self) -> bool:
        """Return True if threshold drops reached and probe has not fired yet."""
        return not self._used and len(self._drops) >= self._threshold

    def mark_used(self) -> None:
        """Mark the probe as fired; subsequent ``should_probe()`` calls return False."""
        self._used = True


# ---------------------------------------------------------------------------
# Pre-classifier confidence gate
# ---------------------------------------------------------------------------


def _is_low_confidence(u: Utterance) -> bool:
    """Return True when ASR confidence is too low to classify reliably.

    Thresholds:
    - ``avg_logprob < -1.0``: Whisper assigned low probability to the tokens.
    - ``no_speech_prob > 0.6``: segment is more likely silence/noise than speech.
    - ``compression_ratio > 2.4``: token sequence is suspiciously repetitive (hallucination signal).
    """
    return u.avg_logprob < -1.0 or u.no_speech_prob > 0.6 or u.compression_ratio > 2.4


# ---------------------------------------------------------------------------
# Exception log installer
# ---------------------------------------------------------------------------


def _install_exception_log() -> logging.Logger:
    """Create and configure the JSONL exception log for post-hoc debugging.

    Writes to ``$XDG_STATE_HOME/meeting-agent/exceptions.jsonl`` (or
    ``~/.meeting-agent/exceptions.jsonl`` if the env var is unset).
    Uses a rotating handler capped at 1 MB × 3 backups.

    Returns:
        The configured ``meeting_agent.exceptions`` logger.
    """
    log_dir = Path(os.environ.get("XDG_STATE_HOME", "~/.meeting-agent")).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_dir / "exceptions.jsonl",
        maxBytes=1_000_000,
        backupCount=3,
    )
    handler.setFormatter(
        logging.Formatter('{"ts": "%(asctime)s", "level": "%(levelname)s", "msg": %(message)s}')
    )
    logger = logging.getLogger("meeting_agent.exceptions")
    # Avoid adding duplicate handlers on repeated calls (e.g. in tests).
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Runtime configuration for one meeting session."""

    input_device: int | None = None
    output_device: int | None = None
    model_id: str = "us.anthropic.claude-sonnet-4-6"
    classifier_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    asr_initial_prompt: str | None = None
    context: ProjectContext = field(default_factory=lambda: ProjectContext(system_prompt=""))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Main event loop for a meeting session.

    Lifecycle (V2 always-on):
      1. Open mic stream.
      2. Feed all chunks to StreamingASR (VAD-gated Whisper).
      3. For each utterance: confidence gate → classify → (silent | respond).
      4. After responding: drain mic echo, update rolling transcript.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Store config; model loads happen lazily in :meth:`run`."""
        self.config = config

    def run(self) -> None:
        """Run the pipeline until interrupted (Ctrl-C)."""
        config = self.config

        # Build components — model loads happen here.
        chunks_iter = audio.record_chunks(device=config.input_device, chunk_ms=_CHUNK_MS)
        asr = StreamingASR(initial_prompt=config.asr_initial_prompt)
        tts = TTS()
        llm = BedrockClient(model_id=config.model_id)
        classifier = Classifier(model_id=config.classifier_model_id)

        # Rolling transcript initialised empty.
        conversation = Conversation(older_turns=[], latest_turn=None)

        # Robustness scaffolding.
        airtime = AirtimeTracker()
        circuit_breaker = CircuitBreaker()
        probe = DeafnessProbe()
        exc_log = _install_exception_log()

        try:
            for utterance in asr.transcribe_stream(chunks_iter):
                utterance_arrival_monotonic = time.monotonic()

                # ------------------------------------------------------------------
                # Pre-classifier confidence gate
                # ------------------------------------------------------------------
                if _is_low_confidence(utterance):
                    probe.record_drop()
                    if probe.should_probe():
                        for chunk in tts.stream_synthesize(DeafnessProbe.PROBE_TEXT):
                            audio.play(
                                chunk,
                                sample_rate=TTS_SAMPLE_RATE,
                                device=config.output_device,
                            )
                        probe.mark_used()
                    exc_log.info(
                        json.dumps(
                            {
                                "event": "low_confidence_drop",
                                "text": utterance.text[:80],
                                "avg_logprob": utterance.avg_logprob,
                                "no_speech_prob": utterance.no_speech_prob,
                                "compression_ratio": utterance.compression_ratio,
                            }
                        )
                    )
                    continue

                # ------------------------------------------------------------------
                # Classify
                # ------------------------------------------------------------------
                session = SessionState(
                    recent_turns=tuple(conversation.older_turns[-5:]),
                    agent_turns_last_5min=airtime.count_last(300),
                    agent_turns_last_30s=airtime.count_last(30),
                )
                confidence = Confidence(
                    avg_logprob=utterance.avg_logprob,
                    no_speech_prob=utterance.no_speech_prob,
                    compression_ratio=utterance.compression_ratio,
                )
                decision = classifier.classify(utterance, confidence, config.context, session)

                # ------------------------------------------------------------------
                # Silent? Append turn and continue listening.
                # ------------------------------------------------------------------
                if decision.action == "silent":
                    conversation.older_turns.append(
                        Turn(speaker=decision.speaker, text=utterance.text)
                    )
                    continue

                # ------------------------------------------------------------------
                # Staleness gate
                # ------------------------------------------------------------------
                age = time.monotonic() - utterance_arrival_monotonic
                if age > _STALE_DROP_S:
                    exc_log.info(
                        json.dumps(
                            {
                                "event": "stale_drop_5s",
                                "age_s": round(age, 3),
                                "text": utterance.text[:80],
                            }
                        )
                    )
                    continue
                action = decision.action
                if age > _STALE_HEDGE_S and action == "full_answer":
                    action = "hedged_answer"  # downgrade

                # ------------------------------------------------------------------
                # Respond (Bedrock Sonnet + TTS, circuit-breaker guarded)
                # ------------------------------------------------------------------
                conversation.latest_turn = Turn(speaker=decision.speaker, text=utterance.text)
                try:
                    with circuit_breaker:
                        t_speak_start = time.monotonic()
                        full_response = self._stream_and_play(
                            config, conversation, llm, tts, utterance_arrival_monotonic
                        )
                        speak_duration = time.monotonic() - t_speak_start
                except CircuitBreakerOpen:
                    exc_log.info(json.dumps({"event": "circuit_open"}))
                    conversation.latest_turn = None
                    continue
                except Exception as exc:
                    exc_log.warning(
                        json.dumps(
                            {
                                "event": "bedrock_timeout",
                                "error": str(exc),
                            }
                        )
                    )
                    conversation.latest_turn = None
                    continue

                # ------------------------------------------------------------------
                # Commit exchange to rolling transcript
                # ------------------------------------------------------------------
                conversation.older_turns.append(Turn(speaker=decision.speaker, text=utterance.text))
                conversation.older_turns.append(Turn(speaker="agent", text=full_response))
                conversation.latest_turn = None
                airtime.record_emission(time.monotonic())

                # ------------------------------------------------------------------
                # Drain echo — discard backlogged mic audio captured while
                # the agent was speaking to prevent feedback on next utterance.
                # ------------------------------------------------------------------
                self._drain_echo(chunks_iter, speak_duration + _ECHO_TAIL_S)

        except KeyboardInterrupt:
            pass

    # ------------------------------------------------------------------
    # Named pipeline stages — kept separate for testability.
    # ------------------------------------------------------------------

    def _drain_echo(self, chunks_iter: Iterator[AudioArray], duration_s: float) -> None:
        """Discard ``duration_s`` seconds of chunks from the iterator.

        The mic-capture queue inside :func:`audio.record_chunks` keeps filling
        while the agent speaks (the callback runs regardless of consumption).
        After agent playback ends, that backlog is full of the agent's own
        audio captured via the speakers. Reading it into the ASR pipeline
        would produce a feedback loop.

        This drains the backlog plus a small tail to cover residual echo.
        Assumes 100 ms chunks (as configured in :meth:`run`). The backlog
        chunks pop immediately; only the tail chunks block on the queue.
        """
        n_chunks = max(1, int(duration_s * 1000 / _CHUNK_MS))
        for _ in range(n_chunks):
            next(chunks_iter, None)

    def _stream_and_play(
        self,
        config: PipelineConfig,
        conversation: Conversation,
        llm: BedrockClient,
        tts: TTS,
        t_asr_done: float,
    ) -> str:
        """Stream Claude's response, pipeline TTS on sentence boundaries.

        Sentence boundaries (``.``, ``?``, ``!``) flush the accumulated buffer
        into a worker thread that handles TTS synthesis and speaker playback in
        parallel with the LLM continuing to stream subsequent sentences.

        The mic is implicitly gated during this method: the main loop is blocked
        inside this call, so ASR chunk consumption is suspended for the duration
        of agent speech.

        Latency events logged to ``meeting_agent.metrics``:

        * ``bedrock_ttft_s`` — time from end-of-utterance to first LLM delta.
        * ``tts_pipeline_overhead_s`` — time from first LLM delta to first
          TTS audio chunk.
        * ``total_turn_latency_s`` — utterance-to-first-speaker-audio latency.

        Args:
            config: Pipeline config (output device, project context, model id).
            conversation: Rolling transcript with ``latest_turn`` already set.
            llm: Bedrock client for streaming the response.
            tts: TTS instance for synthesis.
            t_asr_done: Monotonic timestamp when ASR finished (latency anchor).

        Returns:
            The full agent response as a single concatenated string.
        """
        sentence_q: queue.Queue[str | None] = queue.Queue()
        first_delta_t: list[float] = []
        first_audio_t: list[float] = []

        def _tts_worker() -> None:
            """Consume sentences, synthesise, and play until sentinel arrives."""
            while True:
                sentence = sentence_q.get()
                if sentence is None:
                    return
                for chunk in tts.stream_synthesize(sentence):
                    if not first_audio_t:
                        t = time.monotonic()
                        first_audio_t.append(t)
                        if first_delta_t:
                            _metrics.info(
                                "tts_pipeline_overhead_s=%.3f",
                                t - first_delta_t[0],
                            )
                        _metrics.info("total_turn_latency_s=%.3f", t - t_asr_done)
                    audio.play(
                        chunk,
                        sample_rate=TTS_SAMPLE_RATE,
                        device=config.output_device,
                    )

        worker = threading.Thread(target=_tts_worker, daemon=True)
        worker.start()

        full_response = ""
        buffer = ""
        first_delta_logged = False

        try:
            for delta in llm.respond_stream(config.context, conversation):
                if not first_delta_logged:
                    t = time.monotonic()
                    first_delta_t.append(t)
                    _metrics.info("bedrock_ttft_s=%.3f", t - t_asr_done)
                    first_delta_logged = True

                full_response += delta
                buffer += delta

                # Flush completed sentences into the TTS worker queue.
                complete, buffer = _split_at_sentence_boundaries(buffer)
                for sentence in complete:
                    cleaned = _strip_tts_markdown(sentence)
                    if cleaned.strip():
                        sentence_q.put(cleaned)
        finally:
            # Flush any remaining (unpunctuated) tail as the final sentence.
            tail_cleaned = _strip_tts_markdown(buffer)
            if tail_cleaned.strip():
                sentence_q.put(tail_cleaned)
            sentence_q.put(None)  # sentinel — stops the worker
            worker.join()

        return full_response


def _split_at_sentence_boundaries(text: str) -> tuple[list[str], str]:
    """Split *text* into complete sentences and an incomplete tail.

    Splits on ``.``, ``?``, and ``!`` using a capturing regex so each complete
    sentence retains its terminal punctuation. The final segment (which has no
    terminal punctuation) is returned as the tail.

    V1 edge-case policy: over-segmentation (e.g. "Dr.") is acceptable — it
    results in an extra small TTS call rather than a correctness failure.

    Args:
        text: Accumulated text from LLM token deltas.

    Returns:
        A ``(complete_sentences, incomplete_tail)`` 2-tuple where each element
        of ``complete_sentences`` ends with ``.``, ``?``, or ``!``.

    Examples:
        >>> _split_at_sentence_boundaries("Hello. World!")
        (['Hello.', ' World!'], '')
        >>> _split_at_sentence_boundaries("Hello")
        ([], 'Hello')
    """
    parts = re.split(r"([.?!])", text)
    # parts alternates text / delimiter, with a final remainder element:
    # "Hello. World!" → ["Hello", ".", " World", "!", ""]
    sentences: list[str] = []
    i = 0
    while i + 1 < len(parts):
        sentences.append(parts[i] + parts[i + 1])
        i += 2
    remainder = parts[i] if i < len(parts) else ""
    return sentences, remainder


def _strip_tts_markdown(text: str) -> str:
    r"""Strip markdown / TTS-hostile symbols from text before synthesis.

    Replaces bold/italic/code wrappers with their inner text. Drops line-prefix
    markers (bullets, numbered lists, headings). Removes fenced code blocks
    entirely (they're not something the agent should be speaking).

    Idempotent — running it twice produces the same output.

    The raw LLM response is *not* affected by this function; it is only applied
    at the TTS boundary so the conversation transcript retains the original text.

    Args:
        text: Text from the LLM response, potentially containing markdown.

    Returns:
        Text with markdown symbols stripped, suitable for TTS synthesis.

    Examples:
        >>> _strip_tts_markdown("This is **bold** and *italic*")
        'This is bold and italic'
        >>> _strip_tts_markdown("- first\n- second\n- third")
        'first\nsecond\nthird'
        >>> _strip_tts_markdown("## Heading\n\ntext")
        'Heading\n\ntext'
    """
    # Remove fenced code blocks entirely (they'd be read literally).
    text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
    # Unwrap **bold**, __bold__, *italic*, _italic_, `code`.
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*(.+?)\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"_(.+?)_", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Strip line-prefix markers: bullets, numbered lists, headings.
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.MULTILINE)
    return text
