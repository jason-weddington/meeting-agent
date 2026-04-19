"""Dev-mode structured trace log for classifier decisions and pipeline events.

Architecture — near-zero hot-path latency via the QueueHandler pattern::

    run-loop thread                 background worker thread
    ---------------                 ------------------------
    tracer.emit(event)              QueueListener.drain()
      └→ logger.info(json)            └→ RotatingFileHandler
           └→ QueueHandler                (JSON serialize + write)
                └→ queue.put_nowait
                  (sub-μs)

Hot-path cost: ~1 μs for ``put_nowait``.  Background thread absorbs all
JSON serialisation and file I/O.  A drop-oldest bounded queue means disk
stalls can never back-pressure the audio pipeline.

Gated behind ``--trace`` / ``MEETING_AGENT_TRACE=1`` — default-off.
"""

from __future__ import annotations

import datetime
import json
import logging
import logging.handlers
import os
import queue
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TRACE_LOG_NAME = "meeting_agent.trace"
_QUEUE_MAX = 1024  # drop-oldest beyond this


# ---------------------------------------------------------------------------
# Drop-oldest queue
# ---------------------------------------------------------------------------


class _DropOldestQueue(queue.Queue):  # type: ignore[type-arg]
    """Queue that drops the oldest record when full instead of blocking."""

    def _drop_and_insert(self, item: object) -> None:
        """Drop the oldest item and insert the new one.

        Calls ``queue.Queue.put`` directly (bypassing our override) to avoid
        the recursion that arises because ``queue.Queue.put_nowait`` delegates
        to ``self.put``, which would otherwise re-enter this method.
        """
        while True:
            try:
                # Call the parent put directly to avoid re-entering our override.
                queue.Queue.put(self, item, block=False)
                return
            except queue.Full:
                try:
                    self.get_nowait()
                except queue.Empty:  # pragma: no cover — defensive race guard
                    pass

    def put(
        self,
        item: object,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Put item; drop the oldest if the queue is full."""
        self._drop_and_insert(item)

    def put_nowait(self, item: object) -> None:
        """Put item without blocking; drop the oldest if the queue is full."""
        self._drop_and_insert(item)


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


@dataclass
class Tracer:
    """Structured trace emitter.  No-op when disabled."""

    enabled: bool
    verbose: bool
    logger: logging.Logger

    def emit(self, event: str, **fields: Any) -> None:
        """Emit a structured trace record.  Near-zero cost when disabled."""
        if not self.enabled:
            return
        record: dict[str, Any] = {"ts": time.time(), "event": event, **fields}
        self.logger.info(json.dumps(record, default=str))
        if self.verbose:
            _emit_verbose_line(record)


# ---------------------------------------------------------------------------
# Verbose stderr formatter
# ---------------------------------------------------------------------------


def _emit_verbose_line(record: Mapping[str, Any]) -> None:
    """Write a concise human-readable summary to stderr.

    Uses ANSI colour codes when stderr is a TTY.  Each line is capped at
    200 characters.
    """
    use_color = sys.stderr.isatty()
    if use_color:
        RESET = "\033[0m"
        CYAN = "\033[36m"
        YELLOW = "\033[33m"
        GREEN = "\033[32m"
        RED = "\033[31m"
        DIM = "\033[2m"
    else:
        RESET = CYAN = YELLOW = GREEN = RED = DIM = ""

    ts = record.get("ts", 0.0)
    dt_str = datetime.datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S.%f")[:12]
    prefix = f"{DIM}[{dt_str}]{RESET}"

    event = record.get("event", "unknown")

    if event == "classifier_decision":
        text = str(record.get("utterance_text", ""))[:40]
        conf: Mapping[str, Any] = record.get("asr_confidence", {})
        lp = float(conf.get("avg_logprob", 0.0))
        nsp = float(conf.get("no_speech_prob", 0.0))
        cr = float(conf.get("compression_ratio", 0.0))
        age = float(record.get("utterance_age_s", 0.0))
        action = record.get("decision_action", "?")
        speaker = record.get("decision_speaker", "?")
        confidence = float(record.get("decision_confidence", 0.0))
        line = (
            f"{prefix} {CYAN}classifier{RESET}: "
            f'"{text}" conf_asr=({lp:.2f},{nsp:.2f},{cr:.2f}) age={age:.1f}s '
            f"→ {YELLOW}{action}{RESET}({speaker}, {confidence:.2f})"
        )

    elif event == "pre_gate_drop":
        conf = record.get("asr_confidence", {})
        lp = float(conf.get("avg_logprob", 0.0))
        nsp = float(conf.get("no_speech_prob", 0.0))
        cr = float(conf.get("compression_ratio", 0.0))
        reason = record.get("reason", "?")
        line = (
            f"{prefix} {RED}pre_gate_drop{RESET}: "
            f"conf_asr=({lp:.2f},{nsp:.2f},{cr:.2f}) reason={reason}"
        )

    elif event == "decision_outcome":
        outcome = str(record.get("outcome", "?"))
        color = GREEN if outcome.startswith("responded") else DIM
        line = f"{prefix}   {color}outcome{RESET}: {outcome}"

    elif event == "response_emitted":
        ttft_s = record.get("bedrock_ttft_s")
        total_s = record.get("total_turn_latency_s")
        ttft_str = f"ttft={int(float(ttft_s) * 1000)}ms" if ttft_s is not None else "ttft=?"
        total_str = f"total={int(float(total_s) * 1000)}ms" if total_s is not None else "total=?"
        line = f"{prefix}   {GREEN}response_emitted{RESET}: {ttft_str}  {total_str}"

    elif event == "deafness_probe_fired":
        drops = record.get("consecutive_drops", 0)
        window = float(record.get("window_s", 30.0))
        line = f"{prefix} {RED}deafness_probe_fired{RESET}: drops={drops} window={window:.0f}s"

    elif event in ("circuit_open", "circuit_close", "circuit_half_open_probe"):
        line = f"{prefix} {RED}{event}{RESET}"

    elif event == "utterance_received":
        text = str(record.get("utterance_text", ""))[:60]
        line = f'{prefix} utterance_received: "{text}"'

    elif event == "mcp_ready":
        tool_count = record.get("tool_count", 0)
        tool_names = record.get("tool_names", [])
        duration_s = float(record.get("duration_s", 0.0))
        names_str = ", ".join(str(n) for n in tool_names[:5])
        if len(tool_names) > 5:
            names_str += f", +{len(tool_names) - 5} more"
        line = (
            f"{prefix} {GREEN}mcp_ready{RESET}: "
            f"tools={tool_count} [{names_str}] startup={duration_s:.2f}s"
        )

    elif event == "mcp_unavailable":
        error_msg = str(record.get("error_msg", ""))[:80]
        line = f"{prefix} {RED}mcp_unavailable{RESET}: {error_msg}"

    elif event == "mcp_stopped":
        line = f"{prefix} {DIM}mcp_stopped{RESET}"

    elif event == "mcp_disabled_after_failures":
        error_count = record.get("error_count", 0)
        window_s = float(record.get("window_s", 60.0))
        line = (
            f"{prefix} {RED}mcp_disabled_after_failures{RESET}: "
            f"errors={error_count} window={window_s:.0f}s"
        )

    elif event == "tool_invoked":
        tool_name = str(record.get("tool_name", "?"))
        duration_s = float(record.get("duration_s", 0.0))
        is_error = bool(record.get("is_error", False))
        result_bytes = record.get("result_bytes", 0)
        status = f"{RED}error{RESET}" if is_error else f"{GREEN}ok{RESET}"
        line = (
            f"{prefix}   {YELLOW}tool_invoked{RESET}: {tool_name} "
            f"→ {status} ({duration_s * 1000:.0f}ms, {result_bytes}B)"
        )

    elif event == "tool_iteration_cap_hit":
        iterations = record.get("iterations", 0)
        line = f"{prefix} {RED}tool_iteration_cap_hit{RESET}: iterations={iterations}"

    else:
        # Generic fallback — truncate aggressively
        line = f"{prefix} {event}: {dict(record)}"

    if len(line) > 200:
        line = line[:197] + "..."

    print(line, file=sys.stderr)


# ---------------------------------------------------------------------------
# Installer
# ---------------------------------------------------------------------------


def install(
    *,
    enabled: bool,
    verbose: bool,
    log_dir: Path | None = None,
) -> tuple[Tracer, logging.handlers.QueueListener | None]:
    """Install the trace logger and return ``(tracer, listener)``.

    If *enabled* is ``False``, returns a no-op :class:`Tracer` and ``None``.
    The caller **must** call ``listener.stop()`` on shutdown; without this,
    the last few records are lost on Ctrl-C.

    *log_dir* defaults to ``$XDG_STATE_HOME/meeting-agent/`` or
    ``~/.meeting-agent/`` when the env var is unset.

    Args:
        enabled: Enable the trace log.  When ``False`` the returned tracer
            is a cheap no-op and no background thread is started.
        verbose: Tee one-line human-readable summaries to stderr per event
            (implies *enabled*).
        log_dir: Directory to write ``trace.jsonl`` into.  Created if absent.

    Returns:
        A ``(tracer, listener)`` tuple.  *listener* is ``None`` when
        *enabled* is ``False``.
    """
    if not enabled:
        return (
            Tracer(
                enabled=False,
                verbose=False,
                logger=logging.getLogger(_TRACE_LOG_NAME),
            ),
            None,
        )

    if log_dir is None:
        state = os.environ.get("XDG_STATE_HOME")
        log_dir = Path(state) / "meeting-agent" if state else Path.home() / ".meeting-agent"
    log_dir.mkdir(parents=True, exist_ok=True)

    q: queue.Queue[logging.LogRecord] = _DropOldestQueue(maxsize=_QUEUE_MAX)
    queue_handler = logging.handlers.QueueHandler(q)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "trace.jsonl",
        maxBytes=10_000_000,
        backupCount=5,
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    listener = logging.handlers.QueueListener(q, file_handler, respect_handler_level=True)
    listener.start()

    logger = logging.getLogger(_TRACE_LOG_NAME)
    logger.setLevel(logging.INFO)
    # Replace any stale handlers (e.g. from a previous install call in tests).
    for old_handler in logger.handlers[:]:
        logger.removeHandler(old_handler)
        old_handler.close()
    logger.addHandler(queue_handler)
    logger.propagate = False

    return Tracer(enabled=True, verbose=verbose, logger=logger), listener
