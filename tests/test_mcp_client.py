"""Unit tests for meeting_agent.mcp_client.

All tests mock fastmcp.Client and fastmcp.client.transports.StdioTransport so
no real subprocess is spawned.  The MCPClient itself starts a real background
thread with a real asyncio event loop — that part is intentional; we want to
exercise the threading / run_coroutine_threadsafe bridge.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import mcp.types
import pytest

from meeting_agent.mcp_client import (
    MCPClient,
    MCPClientError,
    MCPServerConfig,
    ToolResult,
    ToolSpec,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_SIMPLE_CONFIG = MCPServerConfig(
    command="uv",
    args=("run", "personal-kb-mcp"),
    env={"HOME": "/home/test"},
)


def _make_mock_client() -> MagicMock:
    """Return a MagicMock that looks like a connected fastmcp.Client."""
    mock = MagicMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)
    mock.list_tools = AsyncMock(return_value=[])
    mock.call_tool = AsyncMock()
    return mock


def _make_tool(
    name: str = "search",
    description: str = "Search the KB",
    input_schema: dict[str, Any] | None = None,
) -> mcp.types.Tool:
    """Return a minimal mcp.types.Tool for use in list_tools mocks."""
    return mcp.types.Tool(
        name=name,
        description=description,
        inputSchema=input_schema or {"type": "object", "properties": {}},
    )


def _make_call_result(
    text: str = "result text",
    is_error: bool = False,
) -> MagicMock:
    """Return a mock that looks like fastmcp.client.client.CallToolResult."""
    mock = MagicMock()
    content_block = mcp.types.TextContent(type="text", text=text)
    mock.content = [content_block]
    mock.is_error = is_error
    return mock


# ---------------------------------------------------------------------------
# start() tests
# ---------------------------------------------------------------------------


def test_start_spawns_background_loop() -> None:
    """start() creates a thread named 'mcp-client-loop' and calls __aenter__."""
    mock_client = _make_mock_client()

    with (
        patch("meeting_agent.mcp_client.StdioTransport"),
        patch("meeting_agent.mcp_client.Client", return_value=mock_client),
    ):
        client = MCPClient(_SIMPLE_CONFIG)
        client.start()

    try:
        assert client._thread is not None
        assert client._thread.name == "mcp-client-loop"
        assert client._thread.is_alive()
        mock_client.__aenter__.assert_awaited_once()
    finally:
        client.stop()


def test_start_blocks_until_ready() -> None:
    """start() blocks until the loop is running; list_tools is callable immediately."""
    tools = [_make_tool("kb_search")]
    mock_client = _make_mock_client()
    mock_client.list_tools = AsyncMock(return_value=tools)

    with (
        patch("meeting_agent.mcp_client.StdioTransport"),
        patch("meeting_agent.mcp_client.Client", return_value=mock_client),
    ):
        client = MCPClient(_SIMPLE_CONFIG)
        client.start()

    try:
        # Immediately after start(), list_tools must not raise a "not started" error.
        result = client.list_tools()
        assert len(result) == 1
        assert result[0].name == "kb_search"
    finally:
        client.stop()


def test_start_passes_config_overrides_on_top_of_parent_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User-provided env values are layered on top of os.environ."""
    monkeypatch.setenv("KB_DATABASE_URL", "postgres://parent")
    monkeypatch.setenv("UNRELATED_VAR", "passthrough")
    mock_client = _make_mock_client()

    with (
        patch("meeting_agent.mcp_client.StdioTransport") as MockTransport,
        patch("meeting_agent.mcp_client.Client", return_value=mock_client),
    ):
        client = MCPClient(_SIMPLE_CONFIG)
        client.start()

    try:
        MockTransport.assert_called_once()
        call_kwargs = MockTransport.call_args.kwargs
        assert call_kwargs["command"] == "uv"
        assert call_kwargs["args"] == ["run", "personal-kb-mcp"]
        env_passed = call_kwargs["env"]
        # The _SIMPLE_CONFIG sets HOME=/home/test as an override — that wins
        # over whatever HOME is in os.environ.
        assert env_passed["HOME"] == "/home/test"
        # Unrelated parent-env vars pass through.
        assert env_passed["UNRELATED_VAR"] == "passthrough"
        # This is the bug fix: KB_DATABASE_URL reaches the MCP subprocess.
        assert env_passed["KB_DATABASE_URL"] == "postgres://parent"
    finally:
        client.stop()


def test_start_inherits_parent_env_when_no_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """StdioTransport receives os.environ when MCPServerConfig.env is None."""
    monkeypatch.setenv("KB_DATABASE_URL", "postgres://from-shell")
    cfg = MCPServerConfig(command="python", args=("-m", "server"))
    mock_client = _make_mock_client()

    with (
        patch("meeting_agent.mcp_client.StdioTransport") as MockTransport,
        patch("meeting_agent.mcp_client.Client", return_value=mock_client),
    ):
        client = MCPClient(cfg)
        client.start()

    try:
        MockTransport.assert_called_once()
        env_passed = MockTransport.call_args.kwargs["env"]
        # Parent env passes through by default.
        assert env_passed["KB_DATABASE_URL"] == "postgres://from-shell"
        # Should also carry whatever else was in os.environ — spot-check PATH.
        assert "PATH" in env_passed
    finally:
        client.stop()


def test_start_user_env_overrides_parent_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When user env and parent env share a key, user value wins."""
    monkeypatch.setenv("CONFLICTING_KEY", "from-parent")
    cfg = MCPServerConfig(
        command="python",
        args=("-m", "server"),
        env={"CONFLICTING_KEY": "from-user"},
    )
    mock_client = _make_mock_client()

    with (
        patch("meeting_agent.mcp_client.StdioTransport") as MockTransport,
        patch("meeting_agent.mcp_client.Client", return_value=mock_client),
    ):
        client = MCPClient(cfg)
        client.start()

    try:
        env_passed = MockTransport.call_args.kwargs["env"]
        assert env_passed["CONFLICTING_KEY"] == "from-user"
    finally:
        client.stop()


def test_start_timeout_raises() -> None:
    """start() raises MCPClientError when __aenter__ does not resolve in time."""

    async def _hang() -> None:
        await asyncio.sleep(9999)

    mock_client = MagicMock()
    mock_client.__aenter__ = MagicMock(side_effect=lambda: _hang())
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("meeting_agent.mcp_client.StdioTransport"),
        patch("meeting_agent.mcp_client.Client", return_value=mock_client),
    ):
        client = MCPClient(_SIMPLE_CONFIG)
        with pytest.raises(MCPClientError, match="Failed to connect"):
            client.start(connect_timeout=0.1)

    # Clean up the background thread even though start() raised.
    if client._loop is not None:
        client._loop.call_soon_threadsafe(client._loop.stop)
    if client._thread is not None:
        client._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# list_tools() tests
# ---------------------------------------------------------------------------


def test_list_tools_returns_toolspecs() -> None:
    """list_tools() wraps two MCP Tool objects into two ToolSpec dataclasses."""
    tools = [
        _make_tool("kb_search", "Search knowledge base", {"type": "object"}),
        _make_tool("kb_get", "Get KB entry", {"type": "object", "properties": {"id": {}}}),
    ]
    mock_client = _make_mock_client()
    mock_client.list_tools = AsyncMock(return_value=tools)

    with (
        patch("meeting_agent.mcp_client.StdioTransport"),
        patch("meeting_agent.mcp_client.Client", return_value=mock_client),
    ):
        client = MCPClient(_SIMPLE_CONFIG)
        client.start()

    try:
        specs = client.list_tools()
    finally:
        client.stop()

    assert len(specs) == 2
    assert specs[0] == ToolSpec(
        name="kb_search",
        description="Search knowledge base",
        input_schema={"type": "object"},
    )
    assert specs[1] == ToolSpec(
        name="kb_get",
        description="Get KB entry",
        input_schema={"type": "object", "properties": {"id": {}}},
    )


def test_list_tools_empty_description_becomes_empty_string() -> None:
    """list_tools() converts None description to an empty string."""
    tool = mcp.types.Tool(name="ping", description=None, inputSchema={"type": "object"})
    mock_client = _make_mock_client()
    mock_client.list_tools = AsyncMock(return_value=[tool])

    with (
        patch("meeting_agent.mcp_client.StdioTransport"),
        patch("meeting_agent.mcp_client.Client", return_value=mock_client),
    ):
        client = MCPClient(_SIMPLE_CONFIG)
        client.start()

    try:
        specs = client.list_tools()
    finally:
        client.stop()

    assert specs[0].description == ""


def test_list_tools_raises_when_not_started() -> None:
    """list_tools() raises MCPClientError if start() was never called."""
    client = MCPClient(_SIMPLE_CONFIG)
    with pytest.raises(MCPClientError, match="not started"):
        client.list_tools()


def test_list_tools_raises_mcp_client_error_on_transport_failure() -> None:
    """list_tools() wraps underlying exceptions in MCPClientError."""
    mock_client = _make_mock_client()
    mock_client.list_tools = AsyncMock(side_effect=RuntimeError("connection lost"))

    with (
        patch("meeting_agent.mcp_client.StdioTransport"),
        patch("meeting_agent.mcp_client.Client", return_value=mock_client),
    ):
        client = MCPClient(_SIMPLE_CONFIG)
        client.start()

    try:
        with pytest.raises(MCPClientError, match="list_tools failed"):
            client.list_tools()
    finally:
        client.stop()


# ---------------------------------------------------------------------------
# call_tool() tests
# ---------------------------------------------------------------------------


def test_call_tool_returns_toolresult() -> None:
    """call_tool() flattens TextContent into ToolResult(content=..., is_error=False)."""
    mock_client = _make_mock_client()
    mock_client.call_tool = AsyncMock(return_value=_make_call_result("hello world"))

    with (
        patch("meeting_agent.mcp_client.StdioTransport"),
        patch("meeting_agent.mcp_client.Client", return_value=mock_client),
    ):
        client = MCPClient(_SIMPLE_CONFIG)
        client.start()

    try:
        result = client.call_tool("kb_search", {"query": "test"})
    finally:
        client.stop()

    assert result == ToolResult(content="hello world", is_error=False)
    mock_client.call_tool.assert_awaited_once_with(
        "kb_search", {"query": "test"}, raise_on_error=False
    )


def test_call_tool_concatenates_multiple_text_blocks() -> None:
    """call_tool() joins multiple TextContent blocks with newlines."""
    mock_client = _make_mock_client()
    raw = MagicMock()
    raw.content = [
        mcp.types.TextContent(type="text", text="line 1"),
        mcp.types.TextContent(type="text", text="line 2"),
    ]
    raw.is_error = False
    mock_client.call_tool = AsyncMock(return_value=raw)

    with (
        patch("meeting_agent.mcp_client.StdioTransport"),
        patch("meeting_agent.mcp_client.Client", return_value=mock_client),
    ):
        client = MCPClient(_SIMPLE_CONFIG)
        client.start()

    try:
        result = client.call_tool("tool", {})
    finally:
        client.stop()

    assert result.content == "line 1\nline 2"


def test_call_tool_surfaces_is_error() -> None:
    """call_tool() returns ToolResult(is_error=True) when server signals error."""
    mock_client = _make_mock_client()
    mock_client.call_tool = AsyncMock(
        return_value=_make_call_result("something went wrong", is_error=True)
    )

    with (
        patch("meeting_agent.mcp_client.StdioTransport"),
        patch("meeting_agent.mcp_client.Client", return_value=mock_client),
    ):
        client = MCPClient(_SIMPLE_CONFIG)
        client.start()

    try:
        result = client.call_tool("kb_get", {"id": "kb-12345"})
    finally:
        client.stop()

    assert result.is_error is True
    assert result.content == "something went wrong"


def test_call_tool_raises_mcp_client_error_on_transport_failure() -> None:
    """call_tool() wraps underlying exceptions in MCPClientError."""
    mock_client = _make_mock_client()
    mock_client.call_tool = AsyncMock(side_effect=Exception("subprocess died"))

    with (
        patch("meeting_agent.mcp_client.StdioTransport"),
        patch("meeting_agent.mcp_client.Client", return_value=mock_client),
    ):
        client = MCPClient(_SIMPLE_CONFIG)
        client.start()

    try:
        with pytest.raises(MCPClientError, match="call_tool"):
            client.call_tool("kb_search", {"query": "x"})
    finally:
        client.stop()


def test_call_tool_raises_when_not_started() -> None:
    """call_tool() raises MCPClientError if start() was never called."""
    client = MCPClient(_SIMPLE_CONFIG)
    with pytest.raises(MCPClientError, match="not started"):
        client.call_tool("tool", {})


# ---------------------------------------------------------------------------
# stop() tests
# ---------------------------------------------------------------------------


def test_stop_cleans_up_cleanly() -> None:
    """stop() calls __aexit__, stops the loop, and joins the thread."""
    mock_client = _make_mock_client()

    with (
        patch("meeting_agent.mcp_client.StdioTransport"),
        patch("meeting_agent.mcp_client.Client", return_value=mock_client),
    ):
        client = MCPClient(_SIMPLE_CONFIG)
        client.start()

        thread = client._thread
        assert thread is not None
        assert thread.is_alive()

        client.stop()

    mock_client.__aexit__.assert_awaited_once_with(None, None, None)
    assert not thread.is_alive()


def test_stop_is_safe_when_not_started() -> None:
    """stop() does not raise if start() was never called."""
    client = MCPClient(_SIMPLE_CONFIG)
    client.stop()  # must not raise


def test_stop_handles_aexit_error(caplog: pytest.LogCaptureFixture) -> None:
    """stop() logs a warning but does not re-raise if __aexit__ fails."""
    import logging

    mock_client = _make_mock_client()
    mock_client.__aexit__ = AsyncMock(side_effect=RuntimeError("disconnect error"))

    with (
        patch("meeting_agent.mcp_client.StdioTransport"),
        patch("meeting_agent.mcp_client.Client", return_value=mock_client),
    ):
        client = MCPClient(_SIMPLE_CONFIG)
        client.start()

    with caplog.at_level(logging.WARNING, logger="meeting_agent.mcp_client"):
        client.stop()  # must not raise

    assert any("Error during MCP client disconnect" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# MCPServerConfig / dataclass tests
# ---------------------------------------------------------------------------


def test_server_config_is_frozen() -> None:
    """MCPServerConfig is immutable (frozen=True)."""
    cfg = MCPServerConfig(command="uv")
    with pytest.raises((AttributeError, TypeError)):
        cfg.command = "python"  # type: ignore[misc]


def test_server_config_default_args_and_env() -> None:
    """MCPServerConfig default args is empty tuple and env is None."""
    cfg = MCPServerConfig(command="uv")
    assert cfg.args == ()
    assert cfg.env is None


def test_tool_spec_is_frozen() -> None:
    """ToolSpec is immutable (frozen=True)."""
    spec = ToolSpec(name="t", description="d", input_schema={})
    with pytest.raises((AttributeError, TypeError)):
        spec.name = "other"  # type: ignore[misc]


def test_tool_result_default_is_not_error() -> None:
    """ToolResult defaults is_error to False."""
    result = ToolResult(content="ok")
    assert result.is_error is False


def test_mcp_client_error_is_runtime_error() -> None:
    """MCPClientError is a RuntimeError subclass."""
    err = MCPClientError("oops")
    assert isinstance(err, RuntimeError)
    assert str(err) == "oops"
