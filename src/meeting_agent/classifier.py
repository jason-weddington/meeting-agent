"""Haiku 4.5 classifier — gatekeeper that decides silent / hedged_answer / full_answer.

For each completed ``Utterance`` from ``StreamingASR``, determines who spoke and
what the agent should do.  All exceptions fail closed to a silent ``Decision`` so
the pipeline never blocks on a classifier failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import boto3
from botocore.config import Config

from meeting_agent.asr import Utterance
from meeting_agent.llm import ProjectContext, Turn

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

DEFAULT_MODEL_ID: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_REGION: str = "us-west-2"
DEFAULT_TIMEOUT_S: float = 2.0

Action = Literal["silent", "hedged_answer", "full_answer"]

_logger = logging.getLogger(__name__)


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


_TOOL_CONFIG: dict[str, Any] = {
    "tools": [
        {
            "toolSpec": {
                "name": "classify",
                "description": "Return the classification decision for the current utterance.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "speaker": {
                                "type": "string",
                                "description": (
                                    "Attributed speaker name from the roster, or 'unknown'."
                                ),
                            },
                            "action": {
                                "type": "string",
                                "enum": ["silent", "hedged_answer", "full_answer"],
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                        "required": ["speaker", "action", "confidence"],
                    }
                },
            }
        }
    ],
    "toolChoice": {"tool": {"name": "classify"}},
}


def _build_system_prompt(context: ProjectContext) -> str:
    """Build the classifier system prompt, embedding the project context verbatim."""
    return f"""\
You are the gatekeeper for a real-time AI meeting participant. For each utterance
in the meeting, decide two things: (1) who said it, (2) what the agent should do.

# Project context

{context.system_prompt}

# Your output

Return a `classify` tool call with three fields:

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

# Decision rules — follow strictly

1. Silence is a first-class output, not an error. Prefer it. The agent is an
   ambient participant, not a service. When in doubt, return "silent".

2. Never choose "silent" + attempt to indicate the agent should ask for
   clarification. Clarifying questions are the response LLM's job, not yours.
   Your choice is only: respond at all (hedged/full) or don't (silent).

3. Audio-failure repair is NOT the agent's job at any layer. If the transcript
   looks garbled or the ASR confidence is low (see avg_logprob / no_speech_prob
   in the user prompt), return "silent". Do NOT return "hedged_answer" with the
   intent of asking the human to repeat themselves.

4. Airtime budget: if `agent_turns_last_30s > 0` or `agent_turns_last_5min > 3`,
   raise your bar for speaking by one level. Prefer "silent" over "hedged"; prefer
   "hedged" over "full".

5. Speaker attribution: per-person speech patterns + role + topic ownership
   form a near-unique cluster even when individual tics are shared. Use the
   whole cluster, not any single phrase. If the cluster doesn't match any
   known speaker with reasonable confidence, return "unknown".\
"""


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


class Classifier:
    """Bedrock Haiku 4.5 classifier with fail-closed-to-silent semantics.

    For each completed ``Utterance``, queries Haiku 4.5 via the Bedrock Converse
    API (synchronous, tool-forced) to decide who spoke and whether the agent
    should stay silent, produce a hedged answer, or produce a full answer.

    Any exception — network timeout, Bedrock error, parse failure, schema
    violation — is logged as a warning and the method returns a silent
    ``Decision`` so the pipeline never blocks.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        region: str = DEFAULT_REGION,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        """Store config; the boto3 client is lazy-created on first classify call."""
        self.model_id = model_id
        self.region = region
        self.timeout_s = timeout_s
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
                {"text": _build_system_prompt(context)},
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
            for block in content:
                if "toolUse" in block:
                    raw: dict[str, Any] = block["toolUse"]["input"]
                    return Decision(
                        speaker=str(raw["speaker"]),
                        action=raw["action"],
                        confidence=float(raw["confidence"]),
                    )
            raise ValueError("classifier response contained no toolUse block")
        except Exception:
            _logger.warning("Classifier failed; returning silent decision.", exc_info=True)
            return Decision(speaker="unknown", action="silent", confidence=0.0)
