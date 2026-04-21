"""Classifier backends — gatekeeper that decides silent / hedged_answer / full_answer.

For each completed ``Utterance`` from ``StreamingASR``, determines who spoke and
what the agent should do.  All exceptions fail closed to a silent ``Decision`` so
the pipeline never blocks on a classifier failure.

Two backends are provided:

* :class:`BedrockClassifier` — Bedrock Haiku 4.5 via the Converse API (default).
* :class:`OllamaClassifier` — local Ollama daemon; model name is configurable.

Both satisfy the :class:`Classifier` protocol and can be swapped without changing
the calling code.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import boto3
import ollama
from botocore.config import Config

from meeting_agent.asr import Utterance
from meeting_agent.llm import ProjectContext, Turn

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

# ---------------------------------------------------------------------------
# Module-level defaults (kept for backward compatibility with direct imports)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_REGION: str = "us-west-2"
DEFAULT_TIMEOUT_S: float = 2.0

Action = Literal["silent", "hedged_answer", "full_answer"]

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------


def _log_cache_hit(logger: logging.Logger, call: str, usage: dict[str, Any]) -> None:
    """Log a human-readable cache-hit summary when total input tokens > 0."""
    read = usage.get("cacheReadInputTokens", 0)
    write = usage.get("cacheWriteInputTokens", 0)
    total_input = usage.get("inputTokens", 0)
    if total_input == 0:
        return
    hit_rate = read / total_input
    logger.info(
        "%s cache: read=%d write=%d hit_rate=%.0f%%",
        call,
        read,
        write,
        hit_rate * 100,
    )


@dataclass(frozen=True)
class Confidence:
    """ASR-level confidence features surfaced to the classifier."""

    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float


@dataclass(frozen=True)
class SessionState:
    """Rolling signals for airtime budgeting and context.

    Attributes:
        recent_turns: Last 5 turns or last 30 s worth, whichever is smaller.
        agent_turns_last_5min: Count of agent turns in the last 5 minutes.
        agent_turns_last_30s: Count of agent turns in the last 30 seconds.
    """

    recent_turns: tuple[Turn, ...]
    agent_turns_last_5min: int
    agent_turns_last_30s: int


@dataclass(frozen=True)
class Decision:
    """Classifier output.

    Attributes:
        speaker: Person's name from the roster, or ``"unknown"``.
        action: What the agent pipeline should do next.
        confidence: Classifier's 0.0–1.0 confidence in the decision.
    """

    speaker: str
    action: Action
    confidence: float


# ---------------------------------------------------------------------------
# Shared JSON schema — single source of truth for both backends
# ---------------------------------------------------------------------------

_DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "speaker": {
            "type": "string",
            "description": "Attributed speaker name from the roster, or 'unknown'.",
        },
        "action": {
            "type": "string",
            "enum": ["silent", "hedged_answer", "full_answer"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["speaker", "action", "confidence"],
}

# Bedrock tool config — wraps the shared schema in the toolSpec wire format.
_TOOL_CONFIG: dict[str, Any] = {
    "tools": [
        {
            "toolSpec": {
                "name": "classify",
                "description": "Return the classification decision for the current utterance.",
                "inputSchema": {
                    "json": _DECISION_JSON_SCHEMA,
                },
            }
        }
    ],
    "toolChoice": {"tool": {"name": "classify"}},
}


# ---------------------------------------------------------------------------
# Classifier protocol
# ---------------------------------------------------------------------------


class Classifier(Protocol):
    """Gatekeeper interface — decides silent / hedged_answer / full_answer per utterance."""

    def classify(
        self,
        utterance: Utterance,
        confidence: Confidence,
        context: ProjectContext,
        session: SessionState,
    ) -> Decision:
        """Classify one utterance and return a Decision; fail closed to silent on error."""
        ...


# ---------------------------------------------------------------------------
# Shared prompt builders (used by both backends unchanged)
# ---------------------------------------------------------------------------

# Mode-specific rule blocks — rule #1 and rule #4 vary; #2, #3, #5 are identical.

_AMBIENT_RULES: str = """\
# Decision rules — follow strictly

1. Silence is a first-class output, not an error. Prefer it. The agent is an
   ambient participant, not a service. When in doubt, return "silent".

2. Never choose "silent" + attempt to indicate the agent should ask for
   clarification. Clarifying questions are the response LLM's job, not yours.
   Your choice is only: respond at all (hedged/full) or don't (silent).

3. Audio-failure repair is about direction. The agent must not ask humans to
   repeat themselves, and must not respond to its own ASR hallucinations.
   But humans asking the agent to repeat or clarify is a legitimate meeting
   contribution and should be handled like any other addressed question.

   - Transcript looks garbled OR ASR confidence is poor (avg_logprob below
     about -1.0, no_speech_prob above 0.6, or compression_ratio above 2.4):
     return "silent". The agent should never react to noise.
   - Agent is asking the human to repeat ("sorry, can you repeat?",
     "I didn't catch that"): the response LLM is never supposed to produce
     this, so it should not appear; if it somehow does, return "silent".
   - Human is asking the agent to repeat or rephrase ("can you say that
     again?", "what did you just say?", "I didn't hear you"): return
     "full_answer". The response LLM will re-state or rephrase its prior
     turn using the rolling transcript.

4. Airtime budget: if `agent_turns_last_30s > 0` or `agent_turns_last_5min > 3`,
   raise your bar for speaking by one level. Prefer "silent" over "hedged"; prefer
   "hedged" over "full".

5. Speaker attribution: per-person speech patterns + role + topic ownership
   form a near-unique cluster even when individual tics are shared. Use the
   whole cluster, not any single phrase. If the cluster doesn't match any
   known speaker with reasonable confidence, return "unknown".\
"""

_DUET_RULES: str = """\
# Decision rules — follow strictly

1. This is a 1:1 working session. Every non-garbled utterance from the user is
   addressed to you. Default to "full_answer". Return "silent" only when the
   utterance looks like ASR hallucination (covered by rule #3) or is an echo
   of the agent's own prior turn (covered by rule #5 via speaker=agent attribution).

2. Never choose "silent" + attempt to indicate the agent should ask for
   clarification. Clarifying questions are the response LLM's job, not yours.
   Your choice is only: respond at all (hedged/full) or don't (silent).

3. Audio-failure repair is about direction. The agent must not ask humans to
   repeat themselves, and must not respond to its own ASR hallucinations.
   But humans asking the agent to repeat or clarify is a legitimate meeting
   contribution and should be handled like any other addressed question.

   - Transcript looks garbled OR ASR confidence is poor (avg_logprob below
     about -1.0, no_speech_prob above 0.6, or compression_ratio above 2.4):
     return "silent". The agent should never react to noise.
   - Agent is asking the human to repeat ("sorry, can you repeat?",
     "I didn't catch that"): the response LLM is never supposed to produce
     this, so it should not appear; if it somehow does, return "silent".
   - Human is asking the agent to repeat or rephrase ("can you say that
     again?", "what did you just say?", "I didn't hear you"): return
     "full_answer". The response LLM will re-state or rephrase its prior
     turn using the rolling transcript.

4. No airtime cap in 1:1 mode. Responding every turn is expected. The user is
   here specifically to work with you.

5. Speaker attribution: per-person speech patterns + role + topic ownership
   form a near-unique cluster even when individual tics are shared. Use the
   whole cluster, not any single phrase. If the cluster doesn't match any
   known speaker with reasonable confidence, return "unknown".\
"""


def _build_system_prompt(
    context: ProjectContext,
    mode: Literal["ambient", "duet"] = "ambient",
) -> str:
    """Build the classifier system prompt, embedding the project context verbatim.

    Args:
        context: Stable per-meeting project context (speaker roster, etc.).
        mode: ``"ambient"`` (default multi-person behavior) or ``"duet"``
            (1:1 working session — loosened silence rules, no airtime cap).
    """
    rules = _DUET_RULES if mode == "duet" else _AMBIENT_RULES
    return f"""\
You are the gatekeeper for a real-time AI meeting participant. For each utterance
in the meeting, decide two things: (1) who said it, (2) what the agent should do.

# Project context

{context.system_prompt}

# Your output

Return a JSON object with three fields:

- speaker: The attributed speaker's name from the known roster above, or the
  string "unknown" if you cannot confidently attribute. Use the speech-pattern
  signatures + project context as your primary signal; each person's cluster of
  tics + role + topic ownership is near-unique even when individual phrases
  overlap.

- action: One of:
  - "silent"        — the default. Choose this when the agent is not being
                      addressed, when the transcript is garbled or unclear,
                      when the agent has spoken recently and another
                      contribution would be noisy, or when there is any doubt.
  - "hedged_answer" — the agent was addressed but the content is ambiguous.
                      The response LLM will produce a short substantive reply
                      with an embedded parenthetical check.
  - "full_answer"   — the agent was clearly addressed and the question is
                      unambiguous; the response LLM will produce a direct reply.

- confidence: Your own 0.0–1.0 confidence in the decision.

{rules}"""


def _build_user_prompt(
    utterance: Utterance,
    confidence: Confidence,
    session: SessionState,
) -> str:
    """Build the per-utterance user prompt with ASR features and airtime signals."""
    turns_text = "\n".join(f"{t.speaker}: {t.text}" for t in session.recent_turns)
    return f"""\
# Recent transcript (last turns)

{turns_text}

# Current utterance to classify

Text: "{utterance.text}"
ASR avg_logprob: {confidence.avg_logprob:.3f}
ASR no_speech_prob: {confidence.no_speech_prob:.3f}
ASR compression_ratio: {confidence.compression_ratio:.3f}

# Airtime

agent_turns_last_5min: {session.agent_turns_last_5min}
agent_turns_last_30s: {session.agent_turns_last_30s}\
"""


# ---------------------------------------------------------------------------
# Backend: Bedrock Haiku 4.5
# ---------------------------------------------------------------------------


class BedrockClassifier:
    """Bedrock Haiku 4.5 classifier with fail-closed-to-silent semantics.

    For each completed ``Utterance``, queries Haiku 4.5 via the Bedrock Converse
    API (synchronous, tool-forced) to decide who spoke and whether the agent
    should stay silent, produce a hedged answer, or produce a full answer.

    Any exception — network timeout, Bedrock error, parse failure, schema
    violation — is logged as a warning and the method returns a silent
    ``Decision`` so the pipeline never blocks.
    """

    DEFAULT_MODEL_ID: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    DEFAULT_REGION: str = "us-west-2"
    DEFAULT_TIMEOUT_S: float = 2.0

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        region: str = DEFAULT_REGION,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        mode: Literal["ambient", "duet"] = "ambient",
    ) -> None:
        """Store config; the boto3 client is lazy-created on first classify call."""
        self.model_id = model_id
        self.region = region
        self.timeout_s = timeout_s
        self.mode = mode
        self._client: BedrockRuntimeClient | None = None

    def _get_client(self) -> BedrockRuntimeClient:
        """Return the boto3 bedrock-runtime client, creating it if needed."""
        if self._client is None:
            config = Config(
                connect_timeout=self.timeout_s,
                read_timeout=self.timeout_s,
            )
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=config,
            )
        return self._client

    def classify(
        self,
        utterance: Utterance,
        confidence: Confidence,
        context: ProjectContext,
        session: SessionState,
    ) -> Decision:
        """Classify one utterance and return a Decision; fail closed to silent on any error.

        Args:
            utterance: The completed utterance from ``StreamingASR``.
            confidence: ASR confidence features for this utterance.
            context: Stable per-meeting project context (speaker roster, etc.).
            session: Rolling airtime and transcript signals.

        Returns:
            A ``Decision`` with speaker attribution, action, and confidence.
            Returns ``Decision(speaker="unknown", action="silent", confidence=0.0)``
            on any error.
        """
        try:
            client = self._get_client()
            system: list[dict[str, Any]] = [
                {"text": _build_system_prompt(context, mode=self.mode)},
                {"cachePoint": {"type": "default", "ttl": "1h"}},
            ]
            messages: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [{"text": _build_user_prompt(utterance, confidence, session)}],
                }
            ]
            response = client.converse(
                modelId=self.model_id,
                system=system,  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
                toolConfig=_TOOL_CONFIG,  # type: ignore[arg-type]
            )
            content: list[dict[str, Any]] = response["output"]["message"]["content"]  # type: ignore[assignment]
            usage: dict[str, Any] = response.get("usage", {})  # type: ignore[assignment]
            for block in content:
                if "toolUse" in block:
                    raw: dict[str, Any] = block["toolUse"]["input"]
                    decision = Decision(
                        speaker=str(raw["speaker"]),
                        action=raw["action"],
                        confidence=float(raw["confidence"]),
                    )
                    _logger.info(
                        "classifier_usage",
                        extra={
                            "input_tokens": usage.get("inputTokens", 0),
                            "output_tokens": usage.get("outputTokens", 0),
                            "cache_read_tokens": usage.get("cacheReadInputTokens", 0),
                            "cache_write_tokens": usage.get("cacheWriteInputTokens", 0),
                        },
                    )
                    _log_cache_hit(_logger, "classifier", usage)
                    return decision
            raise ValueError("classifier response contained no toolUse block")
        except Exception:
            _logger.warning("Classifier failed; returning silent decision.", exc_info=True)
            return Decision(speaker="unknown", action="silent", confidence=0.0)


# ---------------------------------------------------------------------------
# Backend: local Ollama
# ---------------------------------------------------------------------------

_VALID_ACTIONS: frozenset[str] = frozenset({"silent", "hedged_answer", "full_answer"})


class OllamaClassifier:
    """Local Ollama classifier; mirrors BedrockClassifier's behavior via JSON-schema output.

    Uses Ollama's structured-output ``format=`` parameter (requires Ollama 0.5+ or
    ``ollama`` Python client >= 0.4.0) to constrain the model response to the
    same speaker / action / confidence schema used by the Bedrock backend.

    Any exception — connection error, timeout, parse failure, schema violation —
    is logged as a warning and the method returns a silent ``Decision`` so the
    pipeline never blocks.
    """

    DEFAULT_MODEL: str = "qwen3.6:35b-a3b-mlx-bf16"
    DEFAULT_HOST: str = "http://localhost:11434"
    # 300s accommodates cold-load of large MoE models. qwen3.6-mlx-bf16 is
    # ~70 GB on disk and takes ~20 s to page in; the bf16 build runs ~2x
    # faster than qwen3.5 at steady state on Apple Silicon. Warm-call
    # latency is hundreds of ms; the timeout only matters for the very
    # first call before the daemon has the weights resident.
    DEFAULT_TIMEOUT_S: float = 300.0

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        mode: Literal["ambient", "duet"] = "ambient",
    ) -> None:
        """Store config; the Ollama client is lazy-created on first classify call.

        Args:
            model: Ollama model tag (e.g. ``"qwen3.6:35b-a3b-mlx-bf16"``).
            host: Ollama daemon URL.  Falls back to the ``OLLAMA_HOST`` environment
                variable, then ``http://localhost:11434``.
            timeout_s: HTTP timeout in seconds for each classify call.
            mode: ``"ambient"`` (default) or ``"duet"`` (1:1 working session).
        """
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST") or self.DEFAULT_HOST
        self.timeout_s = timeout_s
        self.mode = mode
        self._client: ollama.Client | None = None

    def _get_client(self) -> ollama.Client:
        """Return the Ollama client, creating it if needed."""
        if self._client is None:
            self._client = ollama.Client(host=self.host, timeout=self.timeout_s)
        return self._client

    def warm_up(self) -> bool:
        """Force-load the model into Ollama's memory with a minimal request.

        Large MoE models (qwen3.6-mlx-bf16 is ~70 GB) take 10–30 s to cold-load. Without
        a warm-up, the first real utterance either times out or gets dropped
        by the pipeline's staleness gate. Call this once at pipeline startup
        so the first user turn hits a resident model.

        Returns:
            True on success, False if the warm-up request failed. Failures
            are logged but never raise — the pipeline continues and the real
            classify calls will fail-closed to silent.
        """
        try:
            self._get_client().chat(
                model=self.model,
                messages=[{"role": "user", "content": "ok"}],
                think=False,
                options={"temperature": 0.0, "num_predict": 1},
            )
            return True
        except Exception:
            _logger.warning(
                "OllamaClassifier.warm_up failed; model may cold-load on first real call.",
                exc_info=True,
            )
            return False

    def classify(
        self,
        utterance: Utterance,
        confidence: Confidence,
        context: ProjectContext,
        session: SessionState,
    ) -> Decision:
        """Classify one utterance via local Ollama; fail closed to silent on any error.

        Args:
            utterance: The completed utterance from ``StreamingASR``.
            confidence: ASR confidence features for this utterance.
            context: Stable per-meeting project context (speaker roster, etc.).
            session: Rolling airtime and transcript signals.

        Returns:
            A ``Decision`` with speaker attribution, action, and confidence.
            Returns ``Decision(speaker="unknown", action="silent", confidence=0.0)``
            on any error.
        """
        try:
            system = _build_system_prompt(context, mode=self.mode)
            user = _build_user_prompt(utterance, confidence, session)

            response = self._get_client().chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                format=_DECISION_JSON_SCHEMA,
                # Qwen3-family models are "hybrid reasoners" — with thinking
                # enabled, the reasoning chain consumes the whole output budget
                # and `message.content` comes back empty. We only need the
                # structured decision, not the chain-of-thought.
                think=False,
                options={"temperature": 0.0, "num_predict": 256},
            )
            text = response["message"]["content"]
            raw: dict[str, Any] = json.loads(text)
            action = raw["action"]
            if action not in _VALID_ACTIONS:
                raise ValueError(f"Unknown action value: {action!r}")
            return Decision(
                speaker=str(raw["speaker"]),
                action=action,
                confidence=float(raw["confidence"]),
            )
        except Exception:
            _logger.warning(
                "OllamaClassifier failed; returning silent decision.",
                exc_info=True,
            )
            return Decision(speaker="unknown", action="silent", confidence=0.0)
