"""Bedrock Claude client for the meeting-agent reasoning layer.

Uses ``bedrock-runtime.converse_stream`` with ``cachePoint`` breakpoints to keep
time-to-first-token low across a long meeting. Only transcribed text crosses
the network boundary — never audio.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import boto3

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

_logger = logging.getLogger(__name__)


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


DEFAULT_MODEL_ID: str = "us.anthropic.claude-sonnet-4-6"
DEFAULT_REGION: str = "us-west-2"


@dataclass
class ProjectContext:
    """Stable per-meeting context — cached behind a 1-hour breakpoint.

    Future versions will pull ``agenda``, ``decision_log``, ``risks``, and
    ``stakeholders`` from a persistent project store. For V1, pass them as
    plain strings.
    """

    system_prompt: str
    agenda: str = ""
    project_docs: str = ""
    decision_log: str = ""
    stakeholders: str = ""


@dataclass
class Turn:
    """One exchange in the meeting's rolling transcript."""

    speaker: str
    text: str


@dataclass
class Conversation:
    """Rolling transcript for one meeting.

    ``older_turns`` holds completed exchanges; ``latest_turn`` is the current
    utterance triggering this response. The Bedrock Converse API sees these as
    native multi-turn messages — not flat text with speaker labels.

    In V2, speaker names are real participant names (e.g. "Jason", "Aziz")
    rather than the generic "user" used in V1. Consecutive non-agent turns are
    collapsed into a single Bedrock user message with speaker prefixes to
    satisfy the alternation constraint.
    """

    older_turns: list[Turn] = field(default_factory=list)
    latest_turn: Turn | None = None


def _build_messages(conversation: Conversation) -> list[dict[str, Any]]:
    """Build a Bedrock-compatible messages list from the conversation.

    Collapses consecutive non-agent turns into a single user message using
    ``speaker: text`` prefixes, satisfying Bedrock's strict user/assistant
    alternation requirement.

    Args:
        conversation: The current conversation state. ``latest_turn`` must
            already be set (caller is responsible).

    Returns:
        List of ``{"role": ..., "content": [...]}`` dicts ready for the
        Bedrock Converse API.
    """
    # Caller (respond_stream) validates latest_turn is not None before calling.
    if conversation.latest_turn is None:  # pragma: no cover
        raise ValueError("latest_turn must be set before calling _build_messages")

    messages: list[dict[str, Any]] = []
    buffer: list[str] = []

    def flush_user() -> None:
        if buffer:
            messages.append({"role": "user", "content": [{"text": "\n".join(buffer)}]})
            buffer.clear()

    for turn in conversation.older_turns:
        if turn.speaker == "agent":
            flush_user()
            messages.append({"role": "assistant", "content": [{"text": turn.text}]})
        else:
            buffer.append(f"{turn.speaker}: {turn.text}")

    # Append the triggering utterance to the trailing user block.
    buffer.append(f"{conversation.latest_turn.speaker}: {conversation.latest_turn.text}")
    flush_user()
    return messages


class BedrockClient:
    """Thin wrapper over ``boto3.client("bedrock-runtime").converse_stream``.

    Responsibilities:
      * Assemble the ``converse_stream`` request with correct ``cachePoint``
        placement (1h block on the system prompt).
      * Stream text deltas out as a ``str`` iterator for the TTS pipeline.
      * Expose cache-hit metrics from the response so the pipeline can log
        effective TTFT.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        region: str = DEFAULT_REGION,
    ) -> None:
        """Store config; the boto3 client is lazy-created on first request."""
        self.model_id = model_id
        self.region = region
        self._client: BedrockRuntimeClient | None = None

    def _get_client(self) -> BedrockRuntimeClient:
        """Return the boto3 bedrock-runtime client, creating it if needed."""
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def respond_stream(
        self,
        context: ProjectContext,
        conversation: Conversation,
    ) -> Iterator[str]:
        """Yield text deltas from Claude's streamed response.

        V2 speaker rules:
        - ``latest_turn`` must be set and must not have ``speaker == "agent"``.
          Any other speaker name is valid (real participant names).
        - ``older_turns`` can contain any speaker name. ``"agent"`` maps to
          ``role: "assistant"``; any other name maps to ``role: "user"`` (with
          speaker prefix). Consecutive non-agent turns are collapsed into a
          single user message.

        Callers should split deltas on sentence boundaries (``.``, ``?``,
        ``!``) before feeding them to ``tts.TTS.stream_synthesize`` for
        low-latency playback.

        Raises:
            ValueError: If ``conversation.latest_turn`` is ``None``.
            ValueError: If ``conversation.latest_turn.speaker`` is ``"agent"``.
        """
        if conversation.latest_turn is None:
            raise ValueError("Conversation.latest_turn must be set")
        if conversation.latest_turn.speaker == "agent":
            raise ValueError(
                "latest_turn must not be from the 'agent' speaker; "
                f"got {conversation.latest_turn.speaker!r}"
            )

        system_text = "\n\n".join(
            filter(
                None,
                [
                    context.system_prompt,
                    context.agenda,
                    context.project_docs,
                    context.decision_log,
                    context.stakeholders,
                ],
            )
        )

        messages = _build_messages(conversation)

        request = {
            "modelId": self.model_id,
            "system": [
                {"text": system_text},
                {"cachePoint": {"type": "default", "ttl": "1h"}},
            ],
            "messages": messages,
        }

        client = self._get_client()
        response = client.converse_stream(**request)  # type: ignore[arg-type]

        for event in response["stream"]:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    yield delta["text"]
            elif "metadata" in event:
                usage: dict[str, Any] = event["metadata"].get("usage", {})  # type: ignore[assignment]
                _logger.info(
                    "response_llm_usage",
                    extra={
                        "input_tokens": usage.get("inputTokens", 0),
                        "output_tokens": usage.get("outputTokens", 0),
                        "cache_read_tokens": usage.get("cacheReadInputTokens", 0),
                        "cache_write_tokens": usage.get("cacheWriteInputTokens", 0),
                    },
                )
                _log_cache_hit(_logger, "response_llm", usage)
