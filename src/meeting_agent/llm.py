"""Response-LLM clients for the meeting-agent reasoning layer.

Two backends are supported:

* :class:`BedrockClient` — streams from Amazon Bedrock Claude via
  ``bedrock-runtime.converse_stream`` with ``cachePoint`` breakpoints.
* :class:`OllamaClient` — streams from a local Ollama daemon for fully-offline
  operation.

Both implement the :class:`LLMClient` protocol so the pipeline can swap them
without touching orchestration logic.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import boto3
import ollama

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


class LLMClient(Protocol):
    """Streaming response-LLM interface — turns a conversation into text deltas.

    Both :class:`BedrockClient` and :class:`OllamaClient` implement this
    protocol structurally.  The pipeline only calls :meth:`respond_stream`
    and does not depend on any backend-specific attributes.
    """

    def respond_stream(
        self,
        context: ProjectContext,
        conversation: Conversation,
    ) -> Iterator[str]:
        """Yield text deltas from the LLM's streamed response."""
        ...


class BedrockClient:
    """Thin wrapper over ``boto3.client("bedrock-runtime").converse_stream``.

    Responsibilities:
      * Assemble the ``converse_stream`` request with correct ``cachePoint``
        placement (1h block on the system prompt).
      * Stream text deltas out as a ``str`` iterator for the TTS pipeline.
      * Expose cache-hit metrics from the response so the pipeline can log
        effective TTFT.
    """

    DEFAULT_MODEL_ID: str = DEFAULT_MODEL_ID

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


def _build_ollama_messages(
    conversation: Conversation,
    context: ProjectContext,
) -> list[dict[str, Any]]:
    """Build OpenAI-style chat messages from the conversation for Ollama.

    Translates the :class:`Conversation` structure to the
    ``[{"role": "system"|"user"|"assistant", "content": str}]`` format that
    Ollama's ``/api/chat`` endpoint expects.

    Mirrors the collapse logic of :func:`_build_messages`: consecutive
    non-agent turns are collapsed into a single ``"user"`` message with
    ``speaker: text`` prefixes to preserve attribution.  Agent turns map to
    ``"assistant"`` messages.

    No ``cachePoint`` blocks are emitted — Ollama handles KV-cache server-side
    via ``keep_alive`` (default 5 min).

    Args:
        conversation: Rolling transcript.  ``latest_turn`` must already be set.
        context: Stable per-meeting context used to build the system message.

    Returns:
        A list of ``{"role": ..., "content": ...}`` dicts suitable for
        passing to ``ollama.Client.chat(messages=...)``.
    """
    # Caller (respond_stream) validates latest_turn is not None before calling.
    if conversation.latest_turn is None:  # pragma: no cover
        raise ValueError("latest_turn must be set before calling _build_ollama_messages")

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

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_text}]
    buffer: list[str] = []

    def flush_user() -> None:
        if buffer:
            messages.append({"role": "user", "content": "\n".join(buffer)})
            buffer.clear()

    for turn in conversation.older_turns:
        if turn.speaker == "agent":
            flush_user()
            messages.append({"role": "assistant", "content": turn.text})
        else:
            buffer.append(f"{turn.speaker}: {turn.text}")

    # Append the triggering utterance to the trailing user block.
    buffer.append(f"{conversation.latest_turn.speaker}: {conversation.latest_turn.text}")
    flush_user()
    return messages


class OllamaClient:
    """Local Ollama response-LLM; mirrors :class:`BedrockClient`'s streaming behavior.

    Streams chat completions from a running ``ollama serve`` daemon.  Use
    ``--ollama-host`` or the ``OLLAMA_HOST`` environment variable to point at
    a non-default daemon address.

    No ``cachePoint`` blocks are sent — Ollama manages its own KV cache
    server-side via ``keep_alive`` (default 5 min).  The ``think=False`` flag
    disables Qwen3's hybrid-reasoner thinking mode so the model returns plain
    text deltas rather than ``<think>`` blocks.
    """

    DEFAULT_MODEL: str = "qwen3.6:35b-a3b-mlx-bf16"
    DEFAULT_HOST: str = "http://localhost:11434"
    # 300 s accommodates cold-load of large MoE models.  qwen3.6-mlx-bf16 is
    # ~70 GB on disk and takes ~20 s to page in.  Warm-call latency is
    # hundreds of ms; the timeout only matters for the very first call before
    # the daemon has the weights resident.
    DEFAULT_TIMEOUT_S: float = 300.0

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        """Store config; the Ollama client is lazy-created on first request.

        Args:
            model: Ollama model tag (e.g. ``"qwen3.6:35b-a3b-mlx-bf16"``).
            host: Ollama daemon URL.  Falls back to the ``OLLAMA_HOST``
                environment variable, then ``http://localhost:11434``.
            timeout_s: HTTP timeout in seconds for each request.
        """
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST") or self.DEFAULT_HOST
        self.timeout_s = timeout_s
        self._client: ollama.Client | None = None

    def _get_client(self) -> ollama.Client:
        """Return the Ollama client, creating it if needed."""
        if self._client is None:
            self._client = ollama.Client(host=self.host, timeout=self.timeout_s)
        return self._client

    def warm_up(self) -> bool:
        """Force-load the model into Ollama's memory with a minimal request.

        Large MoE models (qwen3.6-mlx-bf16 is ~70 GB) take 10–30 s to
        cold-load.  Without a warm-up the first real utterance either times
        out or gets dropped by the pipeline's staleness gate.  Call this once
        at pipeline startup so the first user turn hits a resident model.

        Returns:
            ``True`` on success, ``False`` if the warm-up request failed.
            Failures are logged as warnings but never raise — the pipeline
            will proceed and the first real call will cold-load the model.
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
                "OllamaClient.warm_up failed; model may cold-load on first turn.",
                exc_info=True,
            )
            return False

    def respond_stream(
        self,
        context: ProjectContext,
        conversation: Conversation,
    ) -> Iterator[str]:
        """Stream response deltas from local Ollama.

        V2 speaker rules (same as :class:`BedrockClient`):
        - ``latest_turn`` must be set and must not have ``speaker == "agent"``.
        - Agent turns in ``older_turns`` map to ``role: "assistant"``; all
          other speaker names map to ``role: "user"`` with speaker prefixes.
          Consecutive non-agent turns are collapsed into a single user message.

        Raises:
            ValueError: If ``conversation.latest_turn`` is ``None``.
            ValueError: If ``conversation.latest_turn.speaker`` is ``"agent"``.
            Exception: Re-raises any Ollama client error so the pipeline's
                circuit breaker can observe the failure.
        """
        if conversation.latest_turn is None:
            raise ValueError("Conversation.latest_turn must be set")
        if conversation.latest_turn.speaker == "agent":
            raise ValueError(
                "latest_turn must not be from the 'agent' speaker; "
                f"got {conversation.latest_turn.speaker!r}"
            )

        messages = _build_ollama_messages(conversation, context)
        try:
            stream = self._get_client().chat(
                model=self.model,
                messages=messages,
                think=False,  # Qwen3 hybrid-reasoner guard — prevents <think> blocks
                stream=True,
                options={"temperature": 0.7, "num_predict": 1024},
            )
            for chunk in stream:
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    yield delta
        except Exception:
            _logger.warning("OllamaClient.respond_stream failed.", exc_info=True)
            raise
