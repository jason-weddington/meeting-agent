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

import json
import logging
import os
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import boto3
import ollama

from meeting_agent.mcp_client import MCPClientError

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

    from meeting_agent.mcp_client import MCPClientLike, ToolSpec
    from meeting_agent.trace import Tracer

_logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS: int = 5

# Appended to the response-LLM system prompt only when an MCPClient is wired
# up (i.e. the model can emit tool_use blocks).  Stops dead-air while tool
# calls run.  Kept short so it doesn't dilute the main persona prompt.
_TOOL_NARRATION_GUARDRAIL: str = """

When you use tools to look something up during your response, narrate briefly
so the listener doesn't hear dead air. Before you call a tool, say what you're
about to do in one short phrase — for example "let me check the knowledge
base, one second" or "I'll look that up". If you need another tool after the
first, a brief transition works — "found some entries, reading now". A short
lead-in before your final answer is fine — "okay, here's what I found". Think
of a good phone-support agent who narrates each step so the caller knows the
line hasn't dropped. Do not over-explain; one short sentence per step is plenty.

If you retrieve content from the KB, paraphrase it in your own words. Never
read raw KB entries verbatim — they contain markdown, bullet points, emoji,
and formatting that sounds terrible when spoken aloud. Summarize the key facts
conversationally as if you already knew them.
"""


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


def _mcp_tools_to_ollama_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    """Translate MCP ToolSpec list → Ollama-format tools list (OpenAI-style).

    Args:
        tools: List of :class:`~meeting_agent.mcp_client.ToolSpec` descriptors
            returned by ``MCPClient.list_tools()``.

    Returns:
        A list of tool dicts in the OpenAI function-calling format expected by
        ``ollama.Client.chat(tools=...)``.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def _mcp_tools_to_bedrock_toolconfig(tools: Sequence[ToolSpec]) -> dict[str, Any]:
    """Translate MCP ToolSpec list into Bedrock converse_stream toolConfig.

    Args:
        tools: List of :class:`~meeting_agent.mcp_client.ToolSpec` descriptors
            returned by ``MCPClient.list_tools()``.

    Returns:
        A ``toolConfig`` dict suitable for passing to ``converse_stream``.
        Uses ``toolChoice: auto`` so Claude decides whether to use any tool.
    """
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": {"json": t.input_schema},
                }
            }
            for t in tools
        ],
        # Not forcing a tool — let Claude decide whether to use any.
        "toolChoice": {"auto": {}},
    }


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
        mcp_client: MCPClientLike | None = None,
        tracer: Tracer | None = None,
    ) -> Iterator[str]:
        """Yield text deltas from the LLM's streamed response.

        Args:
            context: Stable per-meeting context (system prompt, agenda, etc.).
            conversation: Rolling transcript.  ``latest_turn`` must be set.
            mcp_client: Optional MCP client for KB grounding.  Backends that
                do not support tool-use accept but ignore this argument.
            tracer: Optional structured trace emitter.
        """
        ...


class BedrockClient:
    """Thin wrapper over ``boto3.client("bedrock-runtime").converse_stream``.

    Responsibilities:
      * Assemble the ``converse_stream`` request with correct ``cachePoint``
        placement (1h block on the system prompt).
      * Stream text deltas out as a ``str`` iterator for the TTS pipeline.
      * Expose cache-hit metrics from the response so the pipeline can log
        effective TTFT.
      * Optionally drive an MCP tool-use loop when ``mcp_client`` is supplied.
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
        mcp_client: MCPClientLike | None = None,
        tracer: Tracer | None = None,
    ) -> Iterator[str]:
        """Yield text deltas from Claude's streamed response.

        When ``mcp_client`` is ``None`` the method behaves exactly as it did
        before V3.0.2 — a single ``converse_stream`` call with no tools.

        When ``mcp_client`` is supplied the method:

        1. Fetches the server's tool list and builds a Bedrock ``toolConfig``.
        2. Streams text deltas, yielding them as they arrive.
        3. On ``stopReason=tool_use``, executes each requested tool via the
           MCP client, appends the results, and restarts the stream.
        4. Loops until ``stopReason=end_turn`` or the
           :data:`MAX_TOOL_ITERATIONS` cap is reached.

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

        Args:
            context: Stable per-meeting context (system prompt, agenda, etc.).
            conversation: Rolling transcript. ``latest_turn`` must be set.
            mcp_client: Optional MCP client. When ``None``, no tools are
                exposed to the model and the method behaves as in V3.0.1.
                Accepts any object satisfying :class:`MCPClientLike`.
            tracer: Optional structured trace emitter. Emits
                ``tool_invoked`` and ``tool_iteration_cap_hit`` events.

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
        if mcp_client is not None:
            system_text = system_text + _TOOL_NARRATION_GUARDRAIL

        messages = _build_messages(conversation)

        request: dict[str, Any] = {
            "modelId": self.model_id,
            "system": [
                {"text": system_text},
                {"cachePoint": {"type": "default", "ttl": "1h"}},
            ],
            "messages": messages,
        }

        bedrock = self._get_client()

        # ---------------------------------------------------------------
        # Fast path: no MCP client — single stream, no toolConfig.
        # Behaviour is identical to pre-V3.0.2.
        # ---------------------------------------------------------------
        if mcp_client is None:
            response = bedrock.converse_stream(**request)
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
            return

        # ---------------------------------------------------------------
        # MCP tool-use loop
        # ---------------------------------------------------------------
        tools = mcp_client.list_tools()
        request["toolConfig"] = _mcp_tools_to_bedrock_toolconfig(tools)

        for _iteration in range(MAX_TOOL_ITERATIONS):
            request["messages"] = messages

            response = bedrock.converse_stream(**request)

            # content_blocks: index → {type, ...accumulated data}
            content_blocks: dict[int, dict[str, Any]] = {}
            stop_reason = "end_turn"

            for event in response["stream"]:
                if "contentBlockStart" in event:
                    cbs = event["contentBlockStart"]
                    idx: int = cbs["contentBlockIndex"]
                    start = cbs.get("start", {})
                    if "toolUse" in start:
                        content_blocks[idx] = {
                            "type": "toolUse",
                            "toolUseId": start["toolUse"]["toolUseId"],
                            "name": start["toolUse"]["name"],
                            "input_json": "",
                        }
                    else:
                        content_blocks[idx] = {"type": "text", "text": ""}

                elif "contentBlockDelta" in event:
                    cbd = event["contentBlockDelta"]
                    idx = cbd.get("contentBlockIndex", 0)
                    delta = cbd["delta"]
                    if "text" in delta:
                        yield delta["text"]
                        if idx in content_blocks:
                            content_blocks[idx]["text"] = (
                                content_blocks[idx].get("text", "") + delta["text"]
                            )
                        else:
                            content_blocks[idx] = {"type": "text", "text": delta["text"]}
                    elif "toolUse" in delta and idx in content_blocks:
                        content_blocks[idx]["input_json"] += delta["toolUse"].get("input", "")

                elif "messageStop" in event:
                    stop_reason = event["messageStop"].get("stopReason", "end_turn")

                elif "metadata" in event:
                    usage = event["metadata"].get("usage", {})  # type: ignore[assignment]
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

            # Terminal stop reasons — text already yielded, nothing more to do.
            if stop_reason != "tool_use":
                return

            # ------------------------------------------------------------------
            # Build assistant content (text blocks + toolUse blocks in order).
            # ------------------------------------------------------------------
            assistant_content: list[dict[str, Any]] = []
            tool_uses_to_execute: list[dict[str, Any]] = []

            for idx in sorted(content_blocks.keys()):
                block = content_blocks[idx]
                if block["type"] == "text" and block.get("text"):
                    assistant_content.append({"text": block["text"]})
                elif block["type"] == "toolUse":
                    try:
                        input_data: dict[str, Any] = (
                            json.loads(block["input_json"]) if block["input_json"] else {}
                        )
                    except json.JSONDecodeError:
                        _logger.warning(
                            "Failed to parse tool input JSON for tool %r; using empty dict.",
                            block["name"],
                        )
                        input_data = {}
                    assistant_content.append(
                        {
                            "toolUse": {
                                "toolUseId": block["toolUseId"],
                                "name": block["name"],
                                "input": input_data,
                            }
                        }
                    )
                    tool_uses_to_execute.append(
                        {
                            "toolUseId": block["toolUseId"],
                            "name": block["name"],
                            "input": input_data,
                        }
                    )

            if not tool_uses_to_execute:
                # tool_use stopReason but no toolUse blocks — defensive guard.
                return

            messages.append({"role": "assistant", "content": assistant_content})

            # ------------------------------------------------------------------
            # Execute tool calls (possibly in parallel from Claude's perspective,
            # but serially here — sync client).
            # ------------------------------------------------------------------
            tool_result_content: list[dict[str, Any]] = []

            for tu in tool_uses_to_execute:
                t0 = time.monotonic()
                is_error = False
                result_text = ""
                status = "success"

                try:
                    tool_result = mcp_client.call_tool(tu["name"], tu["input"])
                    duration_s = time.monotonic() - t0
                    is_error = tool_result.is_error
                    result_text = tool_result.content
                    # Non-text content is not supported in V3.0.2.
                    if not result_text and tool_result.is_error:
                        result_text = "tool returned an error with no content"
                    status = "error" if is_error else "success"

                except MCPClientError as exc:
                    duration_s = time.monotonic() - t0
                    is_error = True
                    result_text = f"MCP client error: {type(exc).__name__}"
                    status = "error"
                    _logger.warning(
                        "MCPClientError calling tool %r: %s",
                        tu["name"],
                        exc,
                    )

                except Exception:
                    _logger.exception(
                        "Unexpected error calling tool %r; aborting tool-use loop.",
                        tu["name"],
                    )
                    raise

                if tracer is not None:
                    tracer.emit(
                        "tool_invoked",
                        tool_name=tu["name"],
                        arguments_preview=json.dumps(tu["input"])[:200],
                        duration_s=duration_s,
                        result_bytes=len(result_text.encode()),
                        is_error=is_error,
                    )

                tool_result_content.append(
                    {
                        "toolResult": {
                            "toolUseId": tu["toolUseId"],
                            "content": [{"text": result_text}],
                            "status": status,
                        }
                    }
                )

            messages.append({"role": "user", "content": tool_result_content})

        # ------------------------------------------------------------------
        # Iteration cap reached without end_turn.
        # ------------------------------------------------------------------
        _logger.warning(
            "Tool-use iteration cap hit after %d iterations; stopping.",
            MAX_TOOL_ITERATIONS,
        )
        if tracer is not None:
            tracer.emit("tool_iteration_cap_hit", iterations=MAX_TOOL_ITERATIONS)
        yield "(tool-use limit reached)"


def _build_ollama_messages(
    conversation: Conversation,
    context: ProjectContext,
    system_suffix: str = "",
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
        system_suffix: Optional text appended to the system message after
            the context fields.  Used by the caller to inject the tool-use
            narration guardrail when an MCP client is in play.

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
    if system_suffix:
        system_text = system_text + system_suffix

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

    Supports MCP tool-use grounding (V3.0.5): when ``mcp_client`` is supplied,
    the model receives the tool list via Ollama's native ``tools=`` parameter
    and the tool-use loop runs identically to :class:`BedrockClient`.
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
        mcp_client: MCPClientLike | None = None,
        tracer: Tracer | None = None,
    ) -> Iterator[str]:
        """Stream response deltas from local Ollama.

        V2 speaker rules (same as :class:`BedrockClient`):
        - ``latest_turn`` must be set and must not have ``speaker == "agent"``.
        - Agent turns in ``older_turns`` map to ``role: "assistant"``; all
          other speaker names map to ``role: "user"`` with speaker prefixes.
          Consecutive non-agent turns are collapsed into a single user message.

        When ``mcp_client`` is ``None`` the method behaves exactly as it did
        before V3.0.5 — a single ``chat`` call with no tools.

        When ``mcp_client`` is supplied the method runs the MCP tool-use loop:

        1. Fetches the server's tool list and builds an Ollama ``tools=`` payload.
        2. Streams text deltas, yielding them as they arrive.
        3. On any tool_calls in the stream, executes each via MCPClient, appends
           tool result messages, and restarts the stream.
        4. Loops until no tool_calls or the :data:`MAX_TOOL_ITERATIONS` cap.

        Args:
            context: Stable per-meeting context (system prompt, agenda, etc.).
            conversation: Rolling transcript.  ``latest_turn`` must be set.
            mcp_client: Optional MCP client for KB grounding.  When ``None``,
                the fast path is used (single stream, no tools).
            tracer: Optional structured trace emitter.  Emits ``tool_invoked``
                and ``tool_iteration_cap_hit`` events (same shape as
                :class:`BedrockClient`).

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

        system_suffix = _TOOL_NARRATION_GUARDRAIL if mcp_client is not None else ""
        messages = _build_ollama_messages(conversation, context, system_suffix=system_suffix)

        # ------------------------------------------------------------------
        # Fast path: no MCP client — single stream, no tools.
        # Behaviour is identical to pre-V3.0.5.
        # ------------------------------------------------------------------
        if mcp_client is None:
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
            return

        # ------------------------------------------------------------------
        # MCP tool-use loop
        # ------------------------------------------------------------------
        tools_payload = _mcp_tools_to_ollama_tools(mcp_client.list_tools())

        for _iteration in range(MAX_TOOL_ITERATIONS):
            accumulated_text = ""
            tool_calls: list[dict[str, Any]] = []

            try:
                stream = self._get_client().chat(
                    model=self.model,
                    messages=messages,
                    tools=tools_payload,
                    think=False,  # Qwen3 hybrid-reasoner guard
                    stream=True,
                    options={"temperature": 0.7, "num_predict": 1024},
                )
            except Exception:
                _logger.warning("OllamaClient.respond_stream failed.", exc_info=True)
                raise

            for chunk in stream:
                msg = chunk.get("message", {})
                delta = msg.get("content", "")
                if delta:
                    yield delta
                    accumulated_text += delta
                chunk_tool_calls = msg.get("tool_calls")
                if chunk_tool_calls:
                    # Ollama returns tool_calls as a list on whichever chunk the
                    # model emitted them.  Accumulate across chunks defensively.
                    tool_calls.extend(chunk_tool_calls)

            if not tool_calls:
                return  # model finished normally — no tool calls

            # Build assistant message reflecting what the model just produced.
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": accumulated_text,
                "tool_calls": tool_calls,
            }
            messages.append(assistant_msg)

            # Execute each tool call and append tool result messages.
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", {})
                # Ollama 0.6.x returns arguments as a dict; older versions may
                # return a JSON string.  Handle both defensively.
                if isinstance(raw_args, str):
                    try:
                        args: dict[str, Any] = json.loads(raw_args)
                    except (json.JSONDecodeError, ValueError):
                        _logger.warning(
                            "Failed to parse tool arguments JSON for tool %r; using empty dict.",
                            name,
                        )
                        args = {}
                else:
                    args = raw_args

                t0 = time.monotonic()
                is_error = False
                result_text = ""

                try:
                    tr = mcp_client.call_tool(name, args)
                    duration_s = time.monotonic() - t0
                    is_error = tr.is_error
                    result_text = tr.content or (
                        "tool returned an error with no content" if is_error else ""
                    )
                except MCPClientError as exc:
                    duration_s = time.monotonic() - t0
                    is_error = True
                    result_text = f"MCP client error: {type(exc).__name__}"
                    _logger.warning("MCPClientError calling tool %r: %s", name, exc)
                except Exception:
                    _logger.exception("Unexpected error calling tool %r; aborting.", name)
                    raise

                if tracer is not None:
                    tracer.emit(
                        "tool_invoked",
                        tool_name=name,
                        arguments_preview=json.dumps(args)[:200],
                        duration_s=duration_s,
                        result_bytes=len(result_text.encode()),
                        is_error=is_error,
                    )

                messages.append(
                    {
                        "role": "tool",
                        "name": name,
                        "content": result_text,
                    }
                )

        # ------------------------------------------------------------------
        # Iteration cap reached without the model finishing cleanly.
        # ------------------------------------------------------------------
        _logger.warning(
            "Tool-use iteration cap hit after %d iterations; stopping.",
            MAX_TOOL_ITERATIONS,
        )
        if tracer is not None:
            tracer.emit("tool_iteration_cap_hit", iterations=MAX_TOOL_ITERATIONS)
        yield "(tool-use limit reached)"
