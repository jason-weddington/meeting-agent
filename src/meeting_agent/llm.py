"""Bedrock Claude client for the meeting-agent reasoning layer.

Uses ``bedrock-runtime.converse_stream`` with ``cachePoint`` breakpoints to keep
time-to-first-token low across a long meeting. Only transcribed text crosses
the network boundary — never audio.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import boto3

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

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

    ``older_turns`` is cached behind a 5-minute breakpoint; ``latest_turn``
    is uncached and rewritten each request. The split point should advance
    only when the older-turns window has grown enough to re-anchor the cache.
    """

    older_turns: list[Turn] = field(default_factory=list)
    latest_turn: Turn | None = None


class BedrockClient:
    """Thin wrapper over ``boto3.client("bedrock-runtime").converse_stream``.

    Responsibilities:
      * Assemble the ``converse_stream`` request with correct ``cachePoint``
        placement (1h block before 5m block, both ≥ the model's min tokens).
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

        Callers should split deltas on sentence boundaries (``.``, ``?``,
        ``!``) before feeding them to ``tts.TTS.stream_synthesize`` for
        low-latency playback.

        Raises:
            ValueError: If ``conversation.latest_turn`` is ``None``.
        """
        if conversation.latest_turn is None:
            raise ValueError("Conversation.latest_turn must be set")

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

        older_text = "\n".join(f"{t.speaker}: {t.text}" for t in conversation.older_turns)
        latest_text = f"{conversation.latest_turn.speaker}: {conversation.latest_turn.text}"

        request = {
            "modelId": self.model_id,
            "system": [
                {"text": system_text},
                {"cachePoint": {"type": "default", "ttl": "1h"}},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": older_text or " "},  # empty string rejected; use space
                        {"cachePoint": {"type": "default", "ttl": "5m"}},
                        {"text": latest_text},
                    ],
                },
            ],
        }

        client = self._get_client()
        response = client.converse_stream(**request)  # type: ignore[arg-type]

        for event in response["stream"]:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    yield delta["text"]
