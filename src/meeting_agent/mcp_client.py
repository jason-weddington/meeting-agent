"""MCP client wrapper for meeting-agent.

Provides a synchronous facade over FastMCP's async client, suitable for use
from the synchronous pipeline (audio threads, sync generators, TTS worker).
The async client runs in a dedicated background event-loop thread; callers
interact only through the sync ``MCPClient`` API.

Only stdio subprocess transports are supported.  HTTP transport is a future
story.

``MCPClientLike`` is a structural :class:`~typing.Protocol` satisfied by both
:class:`MCPClient` and any thin wrapper (e.g. the pipeline's failure-gate
wrapper).  Use it as the type annotation where you accept either.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import mcp.types
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class MCPClientError(RuntimeError):
    """Raised on MCP transport / protocol failures (not tool-level errors)."""


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCPServerConfig:
    """Describes how to launch / connect to an MCP server.

    Attributes:
        command: Executable to run, e.g. ``"uv"``.
        args: Command-line arguments, e.g. ``("run", "personal-kb-mcp")``.
        env: Optional environment-variable overrides for the child process.
    """

    command: str
    args: tuple[str, ...] = field(default_factory=tuple)
    env: Mapping[str, str] | None = None


# ---------------------------------------------------------------------------
# Wire-format dataclasses (mirror FastMCP types with only the fields we need)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """Minimal tool descriptor surfaced to the meeting-agent pipeline.

    Attributes:
        name: Tool name as advertised by the MCP server.
        description: Human-readable description (empty string if absent).
        input_schema: JSON Schema dict passed through to Bedrock in V3.0.2.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Result from a single MCP tool call.

    Attributes:
        content: Flattened text from all TextContent blocks in the response.
        is_error: True when the server signalled a tool-level error.
    """

    content: str
    is_error: bool = False


# ---------------------------------------------------------------------------
# Sync client wrapper
# ---------------------------------------------------------------------------


class MCPClient:
    """Sync wrapper around fastmcp.Client for use from the sync pipeline.

    FastMCP is async.  Our pipeline is sync (generators + threads).  This
    wrapper runs a dedicated event loop in a background thread and bridges
    sync calls via ``asyncio.run_coroutine_threadsafe`` so callers stay in
    sync land.

    Typical usage::

        cfg = MCPServerConfig(command="uv", args=("run", "personal-kb-mcp"))
        client = MCPClient(cfg)
        client.start()
        try:
            tools = client.list_tools()
            result = client.call_tool("kb_search", {"query": "..."})
        finally:
            client.stop()
    """

    def __init__(self, config: MCPServerConfig) -> None:
        """Store config; background loop and client are created on start().

        Args:
            config: Server launch parameters.
        """
        self.config = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: Client[StdioTransport] | None = None

    def start(self, connect_timeout: float = 30.0) -> None:
        """Spawn the background loop thread and connect to the MCP server.

        Blocks until the server is reachable or *connect_timeout* elapses.
        Must be called before :meth:`list_tools` or :meth:`call_tool`.

        Args:
            connect_timeout: Seconds to wait for the initial handshake.

        Raises:
            MCPClientError: If the connection attempt fails or times out.
        """
        loop = asyncio.new_event_loop()
        self._loop = loop
        ready = threading.Event()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=_run, daemon=True, name="mcp-client-loop")
        self._thread.start()
        ready.wait()

        transport = StdioTransport(
            command=self.config.command,
            args=list(self.config.args),
            env=dict(self.config.env) if self.config.env else None,
        )
        self._client = Client(transport)
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._client.__aenter__(),  # type: ignore[no-untyped-call]
                loop,
            )
            fut.result(timeout=connect_timeout)
        except Exception as exc:
            raise MCPClientError(f"Failed to connect to MCP server: {exc}") from exc

    def stop(self) -> None:
        """Close the connection and stop the background loop.

        Safe to call even if :meth:`start` was never called or has already
        been called and the thread has stopped.
        """
        if self._client is not None and self._loop is not None:
            fut = asyncio.run_coroutine_threadsafe(
                self._client.__aexit__(None, None, None),  # type: ignore[no-untyped-call]
                self._loop,
            )
            try:
                fut.result(timeout=5.0)
            except Exception:
                _logger.warning("Error during MCP client disconnect.", exc_info=True)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def list_tools(self) -> list[ToolSpec]:
        """Return the MCP server's advertised tools as a list of ToolSpec.

        Returns:
            List of :class:`ToolSpec` dataclasses describing each tool.

        Raises:
            MCPClientError: If not connected or on transport / protocol errors.
        """
        if self._client is None or self._loop is None:
            raise MCPClientError("MCPClient not started; call start() first.")
        try:
            fut = asyncio.run_coroutine_threadsafe(self._client.list_tools(), self._loop)
            raw: list[mcp.types.Tool] = fut.result(timeout=10.0)
        except Exception as exc:
            raise MCPClientError(f"list_tools failed: {exc}") from exc
        return [
            ToolSpec(
                name=t.name,
                description=t.description or "",
                input_schema=dict(t.inputSchema),
            )
            for t in raw
        ]

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        """Synchronously invoke one MCP tool by name.

        Tool-level errors (server signalled ``isError: true``) surface as
        ``ToolResult(is_error=True)``; the caller decides how to handle them.
        Transport / protocol failures raise :class:`MCPClientError`.

        Args:
            name: Name of the MCP tool to invoke.
            arguments: Key-value arguments forwarded to the tool.

        Returns:
            :class:`ToolResult` with flattened text content and error flag.

        Raises:
            MCPClientError: If not connected or on transport / protocol errors.
        """
        if self._client is None or self._loop is None:
            raise MCPClientError("MCPClient not started; call start() first.")
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._client.call_tool(name, dict(arguments), raise_on_error=False),
                self._loop,
            )
            raw = fut.result(timeout=30.0)
        except Exception as exc:
            raise MCPClientError(f"call_tool({name!r}) failed: {exc}") from exc
        text_parts = [c.text for c in raw.content if isinstance(c, mcp.types.TextContent)]
        return ToolResult(content="\n".join(text_parts), is_error=raw.is_error)


# ---------------------------------------------------------------------------
# Structural protocol — satisfied by MCPClient and thin wrappers
# ---------------------------------------------------------------------------


class MCPClientLike(Protocol):
    """Structural protocol for the LLM-visible subset of an MCP client.

    Satisfied structurally by :class:`MCPClient` and any thin wrapper that
    exposes the same two methods.  Use as the type annotation wherever the
    pipeline or LLM layer accepts either.

    Note: lifecycle methods (``start`` / ``stop``) are intentionally absent —
    they are managed by the pipeline directly on the concrete :class:`MCPClient`
    instance, not through this protocol.
    """

    def list_tools(self) -> list[ToolSpec]:
        """Return the MCP server's advertised tools."""
        ...

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        """Synchronously invoke one MCP tool by name."""
        ...
