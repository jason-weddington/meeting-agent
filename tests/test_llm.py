"""Unit tests for meeting_agent.llm — BedrockClient and OllamaClient."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from meeting_agent.llm import (
    BedrockClient,
    Conversation,
    OllamaClient,
    ProjectContext,
    Turn,
    _build_ollama_messages,
    _mcp_tools_to_bedrock_toolconfig,
    _mcp_tools_to_ollama_tools,
)
from meeting_agent.mcp_client import MCPClientError, ToolResult, ToolSpec
from meeting_agent.trace import Tracer


def _make_stream(*text_deltas: str, extra_events=None):
    """Build a fake converse_stream response stream."""
    events = []
    for text in text_deltas:
        events.append({"contentBlockDelta": {"delta": {"text": text}}})
    if extra_events:
        events.extend(extra_events)
    return {"stream": iter(events)}


def _make_client_mock(stream_response):
    """Return a mock boto3 bedrock-runtime client whose converse_stream returns stream_response."""
    mock_client = MagicMock()
    mock_client.converse_stream.return_value = stream_response
    return mock_client


# ---------------------------------------------------------------------------
# System block tests
# ---------------------------------------------------------------------------


def test_respond_stream_system_block_has_1h_cachepoint():
    """system is a 2-element list: text block with all fields, then 1h cachePoint."""
    mock_client = _make_client_mock(_make_stream("ok"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(
            system_prompt="You are a helpful assistant.",
            agenda="1. Intro\n2. Discussion",
            project_docs="Design doc here.",
            decision_log="Decision: use Python.",
            stakeholders="Alice, Bob",
        )
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))
        list(client.respond_stream(context, conversation))

    system = mock_client.converse_stream.call_args[1]["system"]
    assert len(system) == 2, f"expected 2 system entries, got {len(system)}"

    # First entry: text containing all ProjectContext fields
    assert "text" in system[0]
    text = system[0]["text"]
    assert "You are a helpful assistant." in text
    assert "1. Intro" in text
    assert "Design doc here." in text
    assert "Decision: use Python." in text
    assert "Alice, Bob" in text

    # Second entry: cachePoint with 1h TTL
    assert system[1] == {"cachePoint": {"type": "default", "ttl": "1h"}}


def test_system_text_joins_non_empty_fields_only():
    """Empty ProjectContext fields are not included in system text."""
    mock_client = _make_client_mock(_make_stream("ok"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(
            system_prompt="Be concise.",
            # agenda, project_docs, decision_log, stakeholders all empty
        )
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))
        list(client.respond_stream(context, conversation))

    system = mock_client.converse_stream.call_args[1]["system"]
    assert system[0]["text"] == "Be concise."


# ---------------------------------------------------------------------------
# Messages shape tests
# ---------------------------------------------------------------------------


def test_respond_stream_no_5m_cachepoint_in_messages():
    """No content block anywhere in messages contains a cachePoint key."""
    mock_client = _make_client_mock(_make_stream("reply"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(
            older_turns=[
                Turn(speaker="Jason", text="Hello"),
                Turn(speaker="agent", text="Hi there"),
            ],
            latest_turn=Turn(speaker="Jason", text="What's the status?"),
        )
        list(client.respond_stream(context, conversation))

    messages = mock_client.converse_stream.call_args[1]["messages"]
    for msg in messages:
        for block in msg["content"]:
            assert "cachePoint" not in block, f"Unexpected cachePoint in message content: {block}"


def test_respond_stream_first_turn_messages_shape():
    """With older_turns=[] and a non-agent latest_turn, messages is one user message with prefix."""
    mock_client = _make_client_mock(_make_stream("reply"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(
            older_turns=[],
            latest_turn=Turn(speaker="Jason", text="First message"),
        )
        list(client.respond_stream(context, conversation))

    messages = mock_client.converse_stream.call_args[1]["messages"]
    assert messages == [{"role": "user", "content": [{"text": "Jason: First message"}]}]


def test_respond_stream_multi_turn_messages_shape():
    """Multi-turn older_turns map to alternating user/assistant messages with speaker prefixes."""
    mock_client = _make_client_mock(_make_stream("reply"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(
            older_turns=[
                Turn(speaker="Jason", text="q1"),
                Turn(speaker="agent", text="a1"),
                Turn(speaker="Jason", text="q2"),
                Turn(speaker="agent", text="a2"),
            ],
            latest_turn=Turn(speaker="Jason", text="q3"),
        )
        list(client.respond_stream(context, conversation))

    messages = mock_client.converse_stream.call_args[1]["messages"]
    assert messages == [
        {"role": "user", "content": [{"text": "Jason: q1"}]},
        {"role": "assistant", "content": [{"text": "a1"}]},
        {"role": "user", "content": [{"text": "Jason: q2"}]},
        {"role": "assistant", "content": [{"text": "a2"}]},
        {"role": "user", "content": [{"text": "Jason: q3"}]},
    ]


def test_respond_stream_speaker_agent_maps_to_assistant():
    """A Turn(speaker='agent', ...) in older_turns becomes role: 'assistant'."""
    mock_client = _make_client_mock(_make_stream("reply"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(
            older_turns=[Turn(speaker="agent", text="I am the agent.")],
            latest_turn=Turn(speaker="Jason", text="Tell me more"),
        )
        list(client.respond_stream(context, conversation))

    messages = mock_client.converse_stream.call_args[1]["messages"]
    # First message should be the agent turn mapped to assistant
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == [{"text": "I am the agent."}]


def test_respond_stream_default_model_id():
    """The assembled request uses the default model ID."""
    mock_client = _make_client_mock(_make_stream("hi"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="You are a helpful assistant.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hello"))
        list(client.respond_stream(context, conversation))

    call_kwargs = mock_client.converse_stream.call_args[1]
    assert call_kwargs["modelId"] == "us.anthropic.claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_respond_stream_raises_on_none_latest_turn():
    """respond_stream raises ValueError when conversation.latest_turn is None."""
    client = BedrockClient()
    context = ProjectContext(system_prompt="Be helpful.")
    conversation = Conversation(latest_turn=None)

    with pytest.raises(ValueError, match="latest_turn"):
        list(client.respond_stream(context, conversation))


def test_latest_turn_non_user_still_raises():
    """respond_stream raises ValueError when latest_turn speaker is 'agent'."""
    client = BedrockClient()
    context = ProjectContext(system_prompt="Be helpful.")
    conversation = Conversation(latest_turn=Turn(speaker="agent", text="I speak"))

    with pytest.raises(ValueError, match="agent"):
        list(client.respond_stream(context, conversation))


def test_unknown_older_speaker_no_longer_raises():
    """Any non-'agent' speaker in older_turns maps to user role without error."""
    mock_client = _make_client_mock(_make_stream("reply"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(
            older_turns=[Turn(speaker="Elena", text="Do something")],
            latest_turn=Turn(speaker="Elena", text="Hi"),
        )
        # Must not raise
        result = list(client.respond_stream(context, conversation))

    assert result == ["reply"]
    messages = mock_client.converse_stream.call_args[1]["messages"]
    # Elena: Do something and Elena: Hi should both be in the user message
    assert messages[0]["role"] == "user"
    assert "Elena: Do something" in messages[0]["content"][0]["text"]


# ---------------------------------------------------------------------------
# Multi-speaker collapse tests
# ---------------------------------------------------------------------------


def test_consecutive_non_agent_turns_collapse_into_one_user_message():
    """Consecutive non-agent turns collapse into one user message with speaker prefixes."""
    mock_client = _make_client_mock(_make_stream("reply"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(
            older_turns=[
                Turn(speaker="Jason", text="q1"),
                Turn(speaker="Aziz", text="q2"),
                Turn(speaker="agent", text="a1"),
                Turn(speaker="Marcus", text="q3"),
            ],
            latest_turn=Turn(speaker="Jason", text="q4"),
        )
        list(client.respond_stream(context, conversation))

    messages = mock_client.converse_stream.call_args[1]["messages"]
    assert messages == [
        {"role": "user", "content": [{"text": "Jason: q1\nAziz: q2"}]},
        {"role": "assistant", "content": [{"text": "a1"}]},
        {"role": "user", "content": [{"text": "Marcus: q3\nJason: q4"}]},
    ]


# ---------------------------------------------------------------------------
# Streaming output tests
# ---------------------------------------------------------------------------


def test_respond_stream_yields_text_deltas():
    """respond_stream yields text deltas in order."""
    mock_client = _make_client_mock(_make_stream("hello", " world"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))
        result = list(client.respond_stream(context, conversation))

    assert result == ["hello", " world"]


def test_respond_stream_skips_non_content_block_delta_events():
    """respond_stream skips events that aren't contentBlockDelta."""
    extra_events = [
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 100, "outputTokens": 20}}},
    ]
    stream = _make_stream("hello", " world", extra_events=extra_events)
    mock_client = _make_client_mock(stream)
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))
        result = list(client.respond_stream(context, conversation))

    assert result == ["hello", " world"]


def test_respond_stream_skips_non_text_delta():
    """respond_stream skips contentBlockDelta events that have no text key."""
    events = [
        {"contentBlockDelta": {"delta": {"toolUse": {"input": "something"}}}},
        {"contentBlockDelta": {"delta": {"text": "actual text"}}},
    ]
    mock_client = _make_client_mock({"stream": iter(events)})
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))
        result = list(client.respond_stream(context, conversation))

    assert result == ["actual text"]


# ---------------------------------------------------------------------------
# Lazy client creation
# ---------------------------------------------------------------------------


def test_lazy_client_creation():
    """BedrockClient construction does not create a boto3 client."""
    with patch("boto3.client") as mock_boto3_client:
        client = BedrockClient()
        mock_boto3_client.assert_not_called()

        # Client should be None until first respond_stream call
        assert client._client is None


def test_boto3_client_created_with_correct_region():
    """boto3 client is created with the configured region."""
    mock_bedrock = _make_client_mock(_make_stream("ok"))
    with patch("boto3.client", return_value=mock_bedrock) as mock_boto3_client:
        client = BedrockClient(region="eu-west-1")
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hello"))
        list(client.respond_stream(context, conversation))

    mock_boto3_client.assert_called_once_with("bedrock-runtime", region_name="eu-west-1")


def test_client_reused_across_calls():
    """The boto3 client is created only once across multiple respond_stream calls."""
    mock_bedrock = _make_client_mock(_make_stream("ok"))
    mock_bedrock.converse_stream.return_value = _make_stream("ok")
    with patch("boto3.client", return_value=mock_bedrock) as mock_boto3_client:
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")

        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="First"))
        list(client.respond_stream(context, conversation))

        # Reset the stream for second call
        mock_bedrock.converse_stream.return_value = _make_stream("ok2")
        conversation2 = Conversation(latest_turn=Turn(speaker="Jason", text="Second"))
        list(client.respond_stream(context, conversation2))

    # boto3.client should only be called once (lazy init)
    mock_boto3_client.assert_called_once()


# ---------------------------------------------------------------------------
# Cache telemetry tests
# ---------------------------------------------------------------------------


def test_respond_stream_logs_cache_usage_on_metadata_event(caplog):
    """A metadata event in the stream fires a response_llm_usage log record."""
    metadata_event = {
        "metadata": {
            "usage": {
                "inputTokens": 500,
                "outputTokens": 30,
                "totalTokens": 530,
                "cacheReadInputTokens": 480,
                "cacheWriteInputTokens": 0,
            }
        }
    }
    stream = _make_stream("hello", extra_events=[metadata_event])
    mock_client = _make_client_mock(stream)
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))
        with caplog.at_level(logging.INFO, logger="meeting_agent.llm"):
            result = list(client.respond_stream(context, conversation))

    assert result == ["hello"]
    usage_records = [r for r in caplog.records if r.getMessage() == "response_llm_usage"]
    assert len(usage_records) == 1, (
        f"Expected 1 response_llm_usage record, got {len(usage_records)}"
    )
    rec = usage_records[0]
    assert rec.input_tokens == 500
    assert rec.output_tokens == 30
    assert rec.cache_read_tokens == 480
    assert rec.cache_write_tokens == 0


def test_respond_stream_handles_missing_metadata_event(caplog):
    """Stream without a metadata event produces no usage log and no crash."""
    stream = _make_stream("hello", " world")
    mock_client = _make_client_mock(stream)
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))
        with caplog.at_level(logging.INFO, logger="meeting_agent.llm"):
            result = list(client.respond_stream(context, conversation))

    assert result == ["hello", " world"]
    usage_records = [r for r in caplog.records if r.getMessage() == "response_llm_usage"]
    assert len(usage_records) == 0


def test_respond_stream_cache_hit_skipped_when_zero_input_tokens(caplog):
    """metadata event with no inputTokens suppresses cache-hit summary log."""
    metadata_event = {"metadata": {"usage": {"inputTokens": 0, "outputTokens": 0}}}
    stream = _make_stream("hi", extra_events=[metadata_event])
    mock_client = _make_client_mock(stream)
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))
        with caplog.at_level(logging.INFO, logger="meeting_agent.llm"):
            result = list(client.respond_stream(context, conversation))

    assert result == ["hi"]
    # usage event still logged; no cache-hit summary (hit_rate skipped when total=0)
    usage_records = [r for r in caplog.records if r.getMessage() == "response_llm_usage"]
    assert len(usage_records) == 1
    cache_records = [r for r in caplog.records if "cache:" in (r.getMessage() or "")]
    assert len(cache_records) == 0


# ===========================================================================
# OllamaClient tests
# ===========================================================================


def _make_ollama_stream(*content_parts: str) -> list[dict]:
    """Build a fake ollama chat stream (list of chunk dicts)."""
    return [{"message": {"content": part}} for part in content_parts]


def _make_ollama_client_mock(stream_chunks: list[dict]) -> MagicMock:
    """Return a mock ollama.Client whose chat() returns a stream iterator."""
    mock_client = MagicMock()
    mock_client.chat.return_value = iter(stream_chunks)
    return mock_client


# ---------------------------------------------------------------------------
# Init / defaults
# ---------------------------------------------------------------------------


def test_ollama_llm_init_honors_defaults():
    """OllamaClient uses DEFAULT_MODEL and DEFAULT_HOST when no args are given."""
    client = OllamaClient()
    assert client.model == OllamaClient.DEFAULT_MODEL
    assert client.host == OllamaClient.DEFAULT_HOST
    assert client._client is None


def test_ollama_llm_init_honors_model_override():
    """Custom model is stored on the client."""
    client = OllamaClient(model="llama3.2:latest")
    assert client.model == "llama3.2:latest"


def test_ollama_llm_init_honors_host_override():
    """Explicit host wins over environment variable."""
    client = OllamaClient(host="http://myserver:11434")
    assert client.host == "http://myserver:11434"


def test_ollama_llm_init_honors_env_host(monkeypatch):
    """OLLAMA_HOST env var is used when no explicit host is given."""
    monkeypatch.setenv("OLLAMA_HOST", "http://envserver:11434")
    client = OllamaClient()
    assert client.host == "http://envserver:11434"


def test_ollama_llm_init_explicit_host_wins_over_env(monkeypatch):
    """Explicit host takes precedence over OLLAMA_HOST env var."""
    monkeypatch.setenv("OLLAMA_HOST", "http://envserver:11434")
    client = OllamaClient(host="http://explicit:11434")
    assert client.host == "http://explicit:11434"


# ---------------------------------------------------------------------------
# warm_up
# ---------------------------------------------------------------------------


def test_ollama_llm_warm_up_sends_minimal_chat():
    """warm_up sends a 1-token chat call with think=False."""
    mock_ollama = MagicMock()
    mock_ollama.chat.return_value = {"message": {"content": "ok"}}

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama):
        client = OllamaClient()
        result = client.warm_up()

    assert result is True
    mock_ollama.chat.assert_called_once()
    call_kwargs = mock_ollama.chat.call_args[1]
    assert call_kwargs["think"] is False
    assert call_kwargs["options"]["num_predict"] == 1


def test_ollama_llm_warm_up_returns_false_on_failure(caplog):
    """warm_up returns False and logs a warning if the call fails."""
    mock_ollama = MagicMock()
    mock_ollama.chat.side_effect = ConnectionError("daemon not running")

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama):
        client = OllamaClient()
        with caplog.at_level(logging.WARNING, logger="meeting_agent.llm"):
            result = client.warm_up()

    assert result is False
    assert any("warm_up" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# respond_stream — streaming behavior
# ---------------------------------------------------------------------------


def test_ollama_llm_respond_stream_yields_deltas():
    """respond_stream yields all content deltas from the stream in order."""
    chunks = _make_ollama_stream("Hello", " there", "!")
    mock_ollama = _make_ollama_client_mock(chunks)

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama):
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))
        result = list(client.respond_stream(context, conversation))

    assert result == ["Hello", " there", "!"]


def test_ollama_llm_respond_stream_passes_think_false():
    """respond_stream passes think=False to ollama.Client.chat."""
    chunks = _make_ollama_stream("ok")
    mock_ollama = _make_ollama_client_mock(chunks)

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama):
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))
        list(client.respond_stream(context, conversation))

    call_kwargs = mock_ollama.chat.call_args[1]
    assert call_kwargs["think"] is False


def test_ollama_llm_respond_stream_passes_stream_true():
    """respond_stream passes stream=True to ollama.Client.chat."""
    chunks = _make_ollama_stream("ok")
    mock_ollama = _make_ollama_client_mock(chunks)

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama):
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))
        list(client.respond_stream(context, conversation))

    call_kwargs = mock_ollama.chat.call_args[1]
    assert call_kwargs["stream"] is True


def test_ollama_llm_respond_stream_raises_on_failure():
    """respond_stream re-raises on failure so the circuit breaker can observe it."""
    mock_ollama = MagicMock()
    mock_ollama.chat.side_effect = ConnectionError("daemon down")

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama):
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))

        with pytest.raises(ConnectionError):
            list(client.respond_stream(context, conversation))


def test_ollama_llm_respond_stream_raises_on_none_latest_turn():
    """respond_stream raises ValueError when conversation.latest_turn is None."""
    client = OllamaClient()
    context = ProjectContext(system_prompt="Be helpful.")
    conversation = Conversation(latest_turn=None)

    with pytest.raises(ValueError, match="latest_turn"):
        list(client.respond_stream(context, conversation))


def test_ollama_llm_respond_stream_raises_on_agent_latest_turn():
    """respond_stream raises ValueError when latest_turn speaker is 'agent'."""
    client = OllamaClient()
    context = ProjectContext(system_prompt="Be helpful.")
    conversation = Conversation(latest_turn=Turn(speaker="agent", text="I speak"))

    with pytest.raises(ValueError, match="agent"):
        list(client.respond_stream(context, conversation))


def test_ollama_llm_respond_stream_skips_empty_deltas():
    """respond_stream skips chunks with empty content strings."""
    chunks = [
        {"message": {"content": "Hello"}},
        {"message": {"content": ""}},
        {"message": {"content": " world"}},
    ]
    mock_ollama = _make_ollama_client_mock(chunks)

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama):
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))
        result = list(client.respond_stream(context, conversation))

    assert result == ["Hello", " world"]


# ---------------------------------------------------------------------------
# _build_ollama_messages — message shape tests
# ---------------------------------------------------------------------------


def test_ollama_llm_system_prompt_is_first_message():
    """_build_ollama_messages puts the system prompt as the first message."""
    context = ProjectContext(system_prompt="You are a meeting bot.")
    conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))
    messages = _build_ollama_messages(conversation, context)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a meeting bot."


def test_ollama_llm_system_text_joins_non_empty_fields():
    """All non-empty ProjectContext fields are joined into the system message."""
    context = ProjectContext(
        system_prompt="You are a meeting bot.",
        agenda="1. Kickoff",
        project_docs="Design doc.",
        decision_log="Decision: use Python.",
        stakeholders="Alice, Bob",
    )
    conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))
    messages = _build_ollama_messages(conversation, context)

    system_content = messages[0]["content"]
    assert "You are a meeting bot." in system_content
    assert "1. Kickoff" in system_content
    assert "Design doc." in system_content
    assert "Decision: use Python." in system_content
    assert "Alice, Bob" in system_content


def test_ollama_llm_assistant_role_for_agent_turns():
    """Agent turns in older_turns map to role: 'assistant'."""
    context = ProjectContext(system_prompt="Be helpful.")
    conversation = Conversation(
        older_turns=[Turn(speaker="agent", text="I am the agent.")],
        latest_turn=Turn(speaker="Jason", text="Tell me more"),
    )
    messages = _build_ollama_messages(conversation, context)

    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["content"] == "I am the agent."


def test_ollama_llm_user_role_collapses_non_agent_speakers():
    """Consecutive non-agent turns collapse into one user message with speaker prefixes."""
    context = ProjectContext(system_prompt="Be helpful.")
    conversation = Conversation(
        older_turns=[
            Turn(speaker="Jason", text="q1"),
            Turn(speaker="Aziz", text="q2"),
            Turn(speaker="agent", text="a1"),
            Turn(speaker="Marcus", text="q3"),
        ],
        latest_turn=Turn(speaker="Jason", text="q4"),
    )
    messages = _build_ollama_messages(conversation, context)

    # Filter out the system message.
    non_system = [m for m in messages if m["role"] != "system"]
    assert non_system == [
        {"role": "user", "content": "Jason: q1\nAziz: q2"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "Marcus: q3\nJason: q4"},
    ]


def test_ollama_llm_first_turn_messages_shape():
    """With no older_turns, messages is system + one user message."""
    context = ProjectContext(system_prompt="Be helpful.")
    conversation = Conversation(
        older_turns=[],
        latest_turn=Turn(speaker="Jason", text="First message"),
    )
    messages = _build_ollama_messages(conversation, context)

    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Jason: First message"}
    assert len(messages) == 2


def test_ollama_llm_no_cache_point_in_messages():
    """_build_ollama_messages never includes 'cachePoint' in any message."""
    context = ProjectContext(system_prompt="Be helpful.", agenda="1. Items")
    conversation = Conversation(
        older_turns=[Turn(speaker="Jason", text="hello")],
        latest_turn=Turn(speaker="Jason", text="world"),
    )
    messages = _build_ollama_messages(conversation, context)

    for msg in messages:
        assert "cachePoint" not in msg


# ---------------------------------------------------------------------------
# OllamaClient lazy client creation
# ---------------------------------------------------------------------------


def test_ollama_llm_lazy_client_creation():
    """OllamaClient construction does not create an ollama.Client."""
    with patch("meeting_agent.llm.ollama.Client") as mock_cls:
        client = OllamaClient()
        mock_cls.assert_not_called()
        assert client._client is None


def test_ollama_llm_client_created_with_correct_host():
    """ollama.Client is created with the configured host on first call."""
    chunks = _make_ollama_stream("ok")
    mock_ollama = _make_ollama_client_mock(chunks)

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama) as MockCls:
        client = OllamaClient(host="http://testhost:11434")
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))
        list(client.respond_stream(context, conversation))

    MockCls.assert_called_once_with(
        host="http://testhost:11434", timeout=OllamaClient.DEFAULT_TIMEOUT_S
    )


def test_ollama_llm_client_reused_across_calls():
    """The ollama.Client is created only once across multiple respond_stream calls."""
    mock_ollama = MagicMock()
    mock_ollama.chat.side_effect = [
        iter(_make_ollama_stream("first")),
        iter(_make_ollama_stream("second")),
    ]

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama) as MockCls:
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")

        conversation1 = Conversation(latest_turn=Turn(speaker="Jason", text="First"))
        list(client.respond_stream(context, conversation1))

        conversation2 = Conversation(latest_turn=Turn(speaker="Jason", text="Second"))
        list(client.respond_stream(context, conversation2))

    MockCls.assert_called_once()


# ===========================================================================
# MCP tool-use loop tests (V3.0.2)
# ===========================================================================

# ---------------------------------------------------------------------------
# Helper builders for tool-use event streams
# ---------------------------------------------------------------------------


def _make_tool_use_stream(
    tool_use_id: str,
    tool_name: str,
    tool_input_json: str,
    text_before: str = "",
) -> dict:
    """Build a fake stream that emits an optional text delta then a toolUse block."""
    events = []
    block_idx = 0

    if text_before:
        events.append({"contentBlockStart": {"contentBlockIndex": block_idx, "start": {}}})
        events.append(
            {
                "contentBlockDelta": {
                    "contentBlockIndex": block_idx,
                    "delta": {"text": text_before},
                }
            }
        )
        block_idx += 1

    events.append(
        {
            "contentBlockStart": {
                "contentBlockIndex": block_idx,
                "start": {
                    "toolUse": {
                        "toolUseId": tool_use_id,
                        "name": tool_name,
                    }
                },
            }
        }
    )
    events.append(
        {
            "contentBlockDelta": {
                "contentBlockIndex": block_idx,
                "delta": {"toolUse": {"input": tool_input_json}},
            }
        }
    )
    events.append({"messageStop": {"stopReason": "tool_use"}})
    return {"stream": iter(events)}


def _make_end_turn_stream(*text_deltas: str) -> dict:
    """Build a stream that emits text deltas then end_turn."""
    events = []
    if text_deltas:
        events.append({"contentBlockStart": {"contentBlockIndex": 0, "start": {}}})
        for delta in text_deltas:
            events.append(
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"text": delta},
                    }
                }
            )
    events.append({"messageStop": {"stopReason": "end_turn"}})
    return {"stream": iter(events)}


def _make_multi_tool_use_stream(
    tools: list[tuple[str, str, str]],
) -> dict:
    """Build a stream emitting multiple toolUse blocks in one turn.

    Each element of *tools* is ``(tool_use_id, name, input_json)``.
    """
    events = []
    for idx, (tool_use_id, name, input_json) in enumerate(tools):
        events.append(
            {
                "contentBlockStart": {
                    "contentBlockIndex": idx,
                    "start": {
                        "toolUse": {
                            "toolUseId": tool_use_id,
                            "name": name,
                        }
                    },
                }
            }
        )
        events.append(
            {
                "contentBlockDelta": {
                    "contentBlockIndex": idx,
                    "delta": {"toolUse": {"input": input_json}},
                }
            }
        )
    events.append({"messageStop": {"stopReason": "tool_use"}})
    return {"stream": iter(events)}


def _make_mcp_client(
    tools: list[ToolSpec] | None = None,
    call_tool_return: ToolResult | None = None,
    call_tool_side_effect: Exception | None = None,
) -> MagicMock:
    """Build a MagicMock that looks like an MCPClient."""
    mock = MagicMock()
    mock.list_tools.return_value = tools or [
        ToolSpec(name="search", description="Search the KB", input_schema={"type": "object"})
    ]
    if call_tool_side_effect is not None:
        mock.call_tool.side_effect = call_tool_side_effect
    else:
        mock.call_tool.return_value = call_tool_return or ToolResult(
            content="result", is_error=False
        )
    return mock


# ---------------------------------------------------------------------------
# Test: mcp_client=None → identical behaviour to pre-V3.0.2
# ---------------------------------------------------------------------------


def test_respond_stream_without_mcp_client_unchanged():
    """Passing mcp_client=None is identical to pre-V3.0.2: no toolConfig, single stream."""
    mock_bedrock = _make_client_mock(_make_stream("hello", " world"))
    with patch("boto3.client", return_value=mock_bedrock):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))
        result = list(client.respond_stream(context, conversation, mcp_client=None))

    assert result == ["hello", " world"]
    call_kwargs = mock_bedrock.converse_stream.call_args[1]
    assert "toolConfig" not in call_kwargs
    mock_bedrock.converse_stream.assert_called_once()


# ---------------------------------------------------------------------------
# Test: toolConfig is passed when mcp_client is given
# ---------------------------------------------------------------------------


def test_respond_stream_passes_toolconfig_when_mcp_client_given():
    """converse_stream receives a toolConfig with the tools from list_tools()."""
    tool1 = ToolSpec(
        name="kb_search",
        description="Search KB",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    tool2 = ToolSpec(name="kb_get", description="Get KB entry", input_schema={"type": "object"})

    mock_bedrock = _make_client_mock(_make_end_turn_stream("done"))
    mock_mcp = _make_mcp_client(tools=[tool1, tool2])

    with patch("boto3.client", return_value=mock_bedrock):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Search for X"))
        list(client.respond_stream(context, conversation, mcp_client=mock_mcp))

    call_kwargs = mock_bedrock.converse_stream.call_args[1]
    assert "toolConfig" in call_kwargs
    tc = call_kwargs["toolConfig"]
    assert tc["toolChoice"] == {"auto": {}}
    assert len(tc["tools"]) == 2
    names = {t["toolSpec"]["name"] for t in tc["tools"]}
    assert names == {"kb_search", "kb_get"}
    # Verify inputSchema shape
    assert tc["tools"][0]["toolSpec"]["inputSchema"] == {"json": tool1.input_schema} or tc["tools"][
        1
    ]["toolSpec"]["inputSchema"] == {"json": tool1.input_schema}


# ---------------------------------------------------------------------------
# Test: tool_use stopReason → execute tool, append result, continue
# ---------------------------------------------------------------------------


def test_respond_stream_executes_tool_use_and_continues():
    """Turn-1 emits toolUse; call_tool is invoked; turn-2 yields final text."""
    turn1 = _make_tool_use_stream(
        tool_use_id="tu-1",
        tool_name="search",
        tool_input_json='{"query": "weather"}',
        text_before="Let me check.",
    )
    turn2 = _make_end_turn_stream("It is sunny.")

    mock_bedrock = MagicMock()
    mock_bedrock.converse_stream.side_effect = [turn1, turn2]

    mock_mcp = _make_mcp_client(
        call_tool_return=ToolResult(content="sunny in Seattle", is_error=False)
    )

    with patch("boto3.client", return_value=mock_bedrock):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Weather?"))
        result = list(client.respond_stream(context, conversation, mcp_client=mock_mcp))

    # Text before tool call + text from second turn
    assert "Let me check." in result
    assert "It is sunny." in result

    # call_tool invoked with correct args
    mock_mcp.call_tool.assert_called_once_with("search", {"query": "weather"})

    # Two converse_stream calls total
    assert mock_bedrock.converse_stream.call_count == 2

    # Second call's messages include assistant (toolUse) + user (toolResult)
    second_call_messages = mock_bedrock.converse_stream.call_args_list[1][1]["messages"]
    roles = [m["role"] for m in second_call_messages]
    assert "assistant" in roles
    assert roles[-1] == "user"
    # The user message has a toolResult block
    last_msg = second_call_messages[-1]
    assert last_msg["content"][0]["toolResult"]["toolUseId"] == "tu-1"
    assert last_msg["content"][0]["toolResult"]["content"][0]["text"] == "sunny in Seattle"
    assert last_msg["content"][0]["toolResult"]["status"] == "success"


# ---------------------------------------------------------------------------
# Test: multiple toolUse blocks in one turn (parallel execution)
# ---------------------------------------------------------------------------


def test_respond_stream_parallel_tool_use():
    """Model emits 2 toolUse blocks; both call_tool calls are made; one user message."""
    turn1 = _make_multi_tool_use_stream(
        [
            ("tu-1", "search", '{"query": "A"}'),
            ("tu-2", "get", '{"id": "B"}'),
        ]
    )
    turn2 = _make_end_turn_stream("Combined result.")

    mock_bedrock = MagicMock()
    mock_bedrock.converse_stream.side_effect = [turn1, turn2]

    mock_mcp = _make_mcp_client()
    mock_mcp.call_tool.side_effect = [
        ToolResult(content="result A", is_error=False),
        ToolResult(content="result B", is_error=False),
    ]

    with patch("boto3.client", return_value=mock_bedrock):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Query both"))
        result = list(client.respond_stream(context, conversation, mcp_client=mock_mcp))

    assert result == ["Combined result."]
    assert mock_mcp.call_tool.call_count == 2

    # The toolResult user message should have 2 toolResult blocks
    second_messages = mock_bedrock.converse_stream.call_args_list[1][1]["messages"]
    last_msg = second_messages[-1]
    assert last_msg["role"] == "user"
    assert len(last_msg["content"]) == 2
    tool_use_ids = {blk["toolResult"]["toolUseId"] for blk in last_msg["content"]}
    assert tool_use_ids == {"tu-1", "tu-2"}


# ---------------------------------------------------------------------------
# Test: MCPClientError → toolResult with status=error, loop continues
# ---------------------------------------------------------------------------


def test_respond_stream_handles_mcp_client_error_as_tool_error():
    """MCPClientError from call_tool → toolResult status=error; stream continues."""
    turn1 = _make_tool_use_stream("tu-1", "search", '{"query": "X"}')
    turn2 = _make_end_turn_stream("Sorry, I could not search.")

    mock_bedrock = MagicMock()
    mock_bedrock.converse_stream.side_effect = [turn1, turn2]

    mock_mcp = _make_mcp_client(call_tool_side_effect=MCPClientError("timeout"))

    with patch("boto3.client", return_value=mock_bedrock):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Search X"))
        result = list(client.respond_stream(context, conversation, mcp_client=mock_mcp))

    # Second turn text should still be yielded
    assert "Sorry, I could not search." in result

    # The toolResult should have status=error
    second_messages = mock_bedrock.converse_stream.call_args_list[1][1]["messages"]
    last_msg = second_messages[-1]
    tr = last_msg["content"][0]["toolResult"]
    assert tr["status"] == "error"
    assert "MCPClientError" in tr["content"][0]["text"]


# ---------------------------------------------------------------------------
# Test: iteration cap
# ---------------------------------------------------------------------------


def test_respond_stream_caps_tool_iterations():
    """Loop stops after MAX_TOOL_ITERATIONS and yields the sentinel."""
    from meeting_agent.llm import MAX_TOOL_ITERATIONS

    # Each call returns a tool_use stream
    streams = [
        _make_tool_use_stream(f"tu-{i}", "search", '{"query": "q"}')
        for i in range(MAX_TOOL_ITERATIONS + 2)
    ]

    mock_bedrock = MagicMock()
    mock_bedrock.converse_stream.side_effect = streams

    mock_mcp = _make_mcp_client()

    with patch("boto3.client", return_value=mock_bedrock):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Go"))
        result = list(client.respond_stream(context, conversation, mcp_client=mock_mcp))

    # Sentinel must be in the output
    assert "(tool-use limit reached)" in result

    # converse_stream called exactly MAX_TOOL_ITERATIONS times
    assert mock_bedrock.converse_stream.call_count == MAX_TOOL_ITERATIONS

    # call_tool called MAX_TOOL_ITERATIONS times (one per iteration)
    assert mock_mcp.call_tool.call_count == MAX_TOOL_ITERATIONS


# ---------------------------------------------------------------------------
# Test: _mcp_tools_to_bedrock_toolconfig pure function
# ---------------------------------------------------------------------------


def test_mcp_tools_to_bedrock_toolconfig_translates_correctly():
    """_mcp_tools_to_bedrock_toolconfig maps ToolSpec fields to Bedrock toolSpec."""
    tools = [
        ToolSpec(
            name="kb_search",
            description="Search the knowledge base",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        ToolSpec(
            name="kb_get",
            description="",
            input_schema={"type": "object"},
        ),
    ]
    config = _mcp_tools_to_bedrock_toolconfig(tools)

    assert config["toolChoice"] == {"auto": {}}
    assert len(config["tools"]) == 2

    spec0 = config["tools"][0]["toolSpec"]
    assert spec0["name"] == "kb_search"
    assert spec0["description"] == "Search the knowledge base"
    assert spec0["inputSchema"] == {
        "json": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    }

    spec1 = config["tools"][1]["toolSpec"]
    assert spec1["name"] == "kb_get"
    assert spec1["description"] == ""
    assert spec1["inputSchema"] == {"json": {"type": "object"}}


# ---------------------------------------------------------------------------
# Test: tracer.emit called with correct fields after a tool call
# ---------------------------------------------------------------------------


def test_respond_stream_emits_tool_invoked_trace():
    """tracer.emit is called with tool_invoked and the expected fields."""
    turn1 = _make_tool_use_stream("tu-1", "search", '{"query": "test"}')
    turn2 = _make_end_turn_stream("Here you go.")

    mock_bedrock = MagicMock()
    mock_bedrock.converse_stream.side_effect = [turn1, turn2]

    mock_mcp = _make_mcp_client(call_tool_return=ToolResult(content="found it", is_error=False))

    tracer = MagicMock(spec=Tracer)
    tracer.enabled = True

    with patch("boto3.client", return_value=mock_bedrock):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Search"))
        list(client.respond_stream(context, conversation, mcp_client=mock_mcp, tracer=tracer))

    # tracer.emit should have been called at least once with tool_invoked
    emit_calls = tracer.emit.call_args_list
    tool_invoked_calls = [c for c in emit_calls if c[0][0] == "tool_invoked"]
    assert len(tool_invoked_calls) == 1

    _, kwargs = tool_invoked_calls[0]
    assert kwargs["tool_name"] == "search"
    assert "query" in kwargs["arguments_preview"]
    assert kwargs["result_bytes"] == len(b"found it")
    assert kwargs["is_error"] is False
    assert "duration_s" in kwargs


# ---------------------------------------------------------------------------
# Coverage fill: defensive branches in the tool-use loop
# ---------------------------------------------------------------------------


def test_respond_stream_handles_text_delta_without_start_in_tool_loop():
    """In the tool-use loop, a text delta for an idx without a prior
    contentBlockStart still yields and registers the block (defensive path)."""
    events = [
        {"contentBlockDelta": {"contentBlockIndex": 5, "delta": {"text": "hello"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    turn = {"stream": iter(events)}
    mock_bedrock = MagicMock()
    mock_bedrock.converse_stream.return_value = turn
    mock_mcp = _make_mcp_client()
    with patch("boto3.client", return_value=mock_bedrock):
        client = BedrockClient()
        context = ProjectContext(system_prompt="x")
        conv = Conversation(latest_turn=Turn(speaker="Jason", text="hi"))
        result = list(client.respond_stream(context, conv, mcp_client=mock_mcp))
    assert result == ["hello"]


def test_respond_stream_logs_metadata_usage_in_tool_loop(caplog):
    """metadata event in the tool-use loop path emits response_llm_usage."""
    events = [
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}},
        {
            "metadata": {
                "usage": {
                    "inputTokens": 100,
                    "outputTokens": 20,
                    "cacheReadInputTokens": 80,
                    "cacheWriteInputTokens": 0,
                }
            }
        },
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    turn = {"stream": iter(events)}
    mock_bedrock = MagicMock()
    mock_bedrock.converse_stream.return_value = turn
    mock_mcp = _make_mcp_client()
    with patch("boto3.client", return_value=mock_bedrock):
        client = BedrockClient()
        context = ProjectContext(system_prompt="x")
        conv = Conversation(latest_turn=Turn(speaker="Jason", text="hi"))
        with caplog.at_level(logging.INFO, logger="meeting_agent.llm"):
            list(client.respond_stream(context, conv, mcp_client=mock_mcp))
    records = [r for r in caplog.records if r.getMessage() == "response_llm_usage"]
    assert len(records) == 1
    assert records[0].input_tokens == 100
    assert records[0].cache_read_tokens == 80


def test_respond_stream_handles_malformed_tool_input_json():
    """Invalid JSON in toolUse input falls back to empty dict in call_tool."""
    events = [
        {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": "tu-1", "name": "search"}},
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": "not-json{"}},
            }
        },
        {"messageStop": {"stopReason": "tool_use"}},
    ]
    turn1 = {"stream": iter(events)}
    turn2 = _make_end_turn_stream("ok")
    mock_bedrock = MagicMock()
    mock_bedrock.converse_stream.side_effect = [turn1, turn2]
    mock_mcp = _make_mcp_client(call_tool_return=ToolResult(content="result", is_error=False))
    with patch("boto3.client", return_value=mock_bedrock):
        client = BedrockClient()
        context = ProjectContext(system_prompt="x")
        conv = Conversation(latest_turn=Turn(speaker="Jason", text="hi"))
        list(client.respond_stream(context, conv, mcp_client=mock_mcp))
    mock_mcp.call_tool.assert_called_once_with("search", {})


def test_respond_stream_tool_use_stop_with_no_tool_blocks():
    """stopReason=tool_use but no toolUse blocks — defensive guard returns cleanly."""
    events = [
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]
    turn = {"stream": iter(events)}
    mock_bedrock = MagicMock()
    mock_bedrock.converse_stream.return_value = turn
    mock_mcp = _make_mcp_client()
    with patch("boto3.client", return_value=mock_bedrock):
        client = BedrockClient()
        context = ProjectContext(system_prompt="x")
        conv = Conversation(latest_turn=Turn(speaker="Jason", text="hi"))
        result = list(client.respond_stream(context, conv, mcp_client=mock_mcp))
    assert result == ["hi"]
    mock_mcp.call_tool.assert_not_called()


def test_respond_stream_tool_error_with_empty_content():
    """ToolResult with is_error=True and empty content gets default error message."""
    turn1 = _make_tool_use_stream("tu-1", "search", "{}")
    turn2 = _make_end_turn_stream("done")
    mock_bedrock = MagicMock()
    mock_bedrock.converse_stream.side_effect = [turn1, turn2]
    mock_mcp = _make_mcp_client(call_tool_return=ToolResult(content="", is_error=True))
    with patch("boto3.client", return_value=mock_bedrock):
        client = BedrockClient()
        context = ProjectContext(system_prompt="x")
        conv = Conversation(latest_turn=Turn(speaker="Jason", text="hi"))
        list(client.respond_stream(context, conv, mcp_client=mock_mcp))
    second_msgs = mock_bedrock.converse_stream.call_args_list[1][1]["messages"]
    tool_result_text = second_msgs[-1]["content"][0]["toolResult"]["content"][0]["text"]
    assert tool_result_text == "tool returned an error with no content"


def test_respond_stream_reraises_unexpected_tool_exception():
    """Non-MCPClientError exception in call_tool propagates (for circuit breaker)."""
    turn = _make_tool_use_stream("tu-1", "search", "{}")
    mock_bedrock = MagicMock()
    mock_bedrock.converse_stream.return_value = turn
    mock_mcp = _make_mcp_client(call_tool_side_effect=RuntimeError("unexpected"))
    with patch("boto3.client", return_value=mock_bedrock):
        client = BedrockClient()
        context = ProjectContext(system_prompt="x")
        conv = Conversation(latest_turn=Turn(speaker="Jason", text="hi"))
        with pytest.raises(RuntimeError, match="unexpected"):
            list(client.respond_stream(context, conv, mcp_client=mock_mcp))


def test_respond_stream_tool_iteration_cap_emits_trace():
    """When iteration cap hits with a tracer, tool_iteration_cap_hit event fires."""
    turns = [_make_tool_use_stream(f"tu-{i}", "search", '{"q": "x"}') for i in range(10)]
    mock_bedrock = MagicMock()
    mock_bedrock.converse_stream.side_effect = turns
    mock_mcp = _make_mcp_client(call_tool_return=ToolResult(content="r", is_error=False))
    mock_tracer = MagicMock()
    with patch("boto3.client", return_value=mock_bedrock):
        client = BedrockClient()
        context = ProjectContext(system_prompt="x")
        conv = Conversation(latest_turn=Turn(speaker="Jason", text="hi"))
        list(client.respond_stream(context, conv, mcp_client=mock_mcp, tracer=mock_tracer))
    event_names = [call.args[0] for call in mock_tracer.emit.call_args_list]
    assert "tool_iteration_cap_hit" in event_names


# ===========================================================================
# OllamaClient MCP tool-use loop tests (V3.0.5)
# ===========================================================================

# ---------------------------------------------------------------------------
# Helper builders for Ollama tool-use streams
# ---------------------------------------------------------------------------


def _make_ollama_tool_call(name: str, arguments: dict) -> dict:
    """Build a single Ollama tool call dict (as returned in message.tool_calls)."""
    return {"function": {"name": name, "arguments": arguments}}


def _make_ollama_tool_use_stream(
    tool_calls: list[dict],
    text_before: str = "",
) -> list[dict]:
    """Build an Ollama stream that yields optional text then tool_calls.

    Ollama emits tool_calls in the final summary chunk.  The text chunks
    come before it.
    """
    chunks: list[dict] = []
    if text_before:
        chunks.append({"message": {"content": text_before}})
    # Summary chunk: content may be empty, tool_calls populated.
    chunks.append({"message": {"content": "", "tool_calls": tool_calls}})
    return chunks


def _make_ollama_end_stream(*text_parts: str) -> list[dict]:
    """Build an Ollama stream that yields text and no tool_calls."""
    return [{"message": {"content": part}} for part in text_parts]


def _make_ollama_mcp_client(
    tools: list[ToolSpec] | None = None,
    call_tool_return: ToolResult | None = None,
    call_tool_side_effect: Exception | None = None,
) -> MagicMock:
    """Build a MagicMock that looks like an MCPClient for Ollama tests."""
    mock = MagicMock()
    mock.list_tools.return_value = tools or [
        ToolSpec(name="kb_search", description="Search KB", input_schema={"type": "object"})
    ]
    if call_tool_side_effect is not None:
        mock.call_tool.side_effect = call_tool_side_effect
    else:
        mock.call_tool.return_value = call_tool_return or ToolResult(
            content="result", is_error=False
        )
    return mock


# ---------------------------------------------------------------------------
# Test: mcp_client=None → V2.10 fast path unchanged
# ---------------------------------------------------------------------------


def test_ollama_respond_stream_without_mcp_client_unchanged():
    """mcp_client=None → fast path: single chat call, no tools= arg."""
    chunks = _make_ollama_end_stream("hello", " world")
    mock_ollama = _make_ollama_client_mock(chunks)

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama):
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))
        result = list(client.respond_stream(context, conversation, mcp_client=None))

    assert result == ["hello", " world"]
    call_kwargs = mock_ollama.chat.call_args[1]
    assert "tools" not in call_kwargs
    mock_ollama.chat.assert_called_once()


# ---------------------------------------------------------------------------
# Test: tools= payload is passed when mcp_client is given
# ---------------------------------------------------------------------------


def test_ollama_respond_stream_passes_tools_when_mcp_client_given():
    """chat(..., tools=...) payload includes the translated tools from list_tools()."""
    tool1 = ToolSpec(
        name="kb_search",
        description="Search KB",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    tool2 = ToolSpec(name="kb_get", description="Get entry", input_schema={"type": "object"})

    # End-of-turn stream (no tool_calls → loop returns after first iteration)
    chunks = _make_ollama_end_stream("done")
    mock_ollama = _make_ollama_client_mock(chunks)
    mock_mcp = _make_ollama_mcp_client(tools=[tool1, tool2])

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama):
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Search for X"))
        list(client.respond_stream(context, conversation, mcp_client=mock_mcp))

    call_kwargs = mock_ollama.chat.call_args[1]
    assert "tools" in call_kwargs
    tools = call_kwargs["tools"]
    assert len(tools) == 2
    names = {t["function"]["name"] for t in tools}
    assert names == {"kb_search", "kb_get"}
    # Verify full shape of first tool
    assert tools[0]["type"] == "function"
    fn = tools[0]["function"]
    assert fn["description"] == tool1.description or fn["description"] == tool2.description


# ---------------------------------------------------------------------------
# Test: tool_calls → execute tool, append result, continue
# ---------------------------------------------------------------------------


def test_ollama_respond_stream_executes_tool_use_and_continues():
    """First stream emits tool_calls; call_tool is invoked; second stream yields text."""
    tool_call = _make_ollama_tool_call("kb_search", {"query": "weather"})
    stream1 = _make_ollama_tool_use_stream([tool_call], text_before="Let me check.")
    stream2 = _make_ollama_end_stream("It is sunny.")

    mock_ollama = MagicMock()
    mock_ollama.chat.side_effect = [iter(stream1), iter(stream2)]

    mock_mcp = _make_ollama_mcp_client(
        call_tool_return=ToolResult(content="sunny in Seattle", is_error=False)
    )

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama):
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Weather?"))
        result = list(client.respond_stream(context, conversation, mcp_client=mock_mcp))

    # Text before tool call + final text
    assert "Let me check." in result
    assert "It is sunny." in result

    # call_tool invoked with correct name and args
    mock_mcp.call_tool.assert_called_once_with("kb_search", {"query": "weather"})

    # Two chat calls total
    assert mock_ollama.chat.call_count == 2

    # Second call's messages include: original + assistant (with tool_calls) + tool result
    second_messages = mock_ollama.chat.call_args_list[1][1]["messages"]
    roles = [m["role"] for m in second_messages]
    assert "assistant" in roles
    assert "tool" in roles
    # Tool result message
    tool_msg = next(m for m in second_messages if m["role"] == "tool")
    assert tool_msg["name"] == "kb_search"
    assert tool_msg["content"] == "sunny in Seattle"


# ---------------------------------------------------------------------------
# Test: parallel tool_calls (2 in one stream)
# ---------------------------------------------------------------------------


def test_ollama_respond_stream_parallel_tool_use():
    """Model emits 2 tool_calls in one stream; both call_tools fire; both tool messages appended."""
    tc1 = _make_ollama_tool_call("kb_search", {"query": "A"})
    tc2 = _make_ollama_tool_call("kb_get", {"id": "B"})
    stream1 = _make_ollama_tool_use_stream([tc1, tc2])
    stream2 = _make_ollama_end_stream("Combined result.")

    mock_ollama = MagicMock()
    mock_ollama.chat.side_effect = [iter(stream1), iter(stream2)]

    mock_mcp = _make_ollama_mcp_client()
    mock_mcp.call_tool.side_effect = [
        ToolResult(content="result A", is_error=False),
        ToolResult(content="result B", is_error=False),
    ]

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama):
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Query both"))
        result = list(client.respond_stream(context, conversation, mcp_client=mock_mcp))

    assert result == ["Combined result."]
    assert mock_mcp.call_tool.call_count == 2

    # Second call messages should include 2 tool result messages
    second_messages = mock_ollama.chat.call_args_list[1][1]["messages"]
    tool_msgs = [m for m in second_messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    tool_names = {m["name"] for m in tool_msgs}
    assert tool_names == {"kb_search", "kb_get"}


# ---------------------------------------------------------------------------
# Test: MCPClientError → tool message with error content; loop continues
# ---------------------------------------------------------------------------


def test_ollama_respond_stream_handles_mcp_client_error_as_tool_error():
    """MCPClientError from call_tool → tool message with error text; stream continues."""
    tc = _make_ollama_tool_call("kb_search", {"query": "X"})
    stream1 = _make_ollama_tool_use_stream([tc])
    stream2 = _make_ollama_end_stream("Sorry, I could not search.")

    mock_ollama = MagicMock()
    mock_ollama.chat.side_effect = [iter(stream1), iter(stream2)]

    mock_mcp = _make_ollama_mcp_client(call_tool_side_effect=MCPClientError("timeout"))

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama):
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Search X"))
        result = list(client.respond_stream(context, conversation, mcp_client=mock_mcp))

    # Second turn text still yielded
    assert "Sorry, I could not search." in result

    # Tool message should contain the MCPClientError class name
    second_messages = mock_ollama.chat.call_args_list[1][1]["messages"]
    tool_msg = next(m for m in second_messages if m["role"] == "tool")
    assert "MCPClientError" in tool_msg["content"]


# ---------------------------------------------------------------------------
# Test: iteration cap
# ---------------------------------------------------------------------------


def test_ollama_respond_stream_caps_tool_iterations():
    """Loop stops after MAX_TOOL_ITERATIONS and yields the sentinel string."""
    from meeting_agent.llm import MAX_TOOL_ITERATIONS

    tc = _make_ollama_tool_call("kb_search", {"query": "q"})
    # Always return tool_calls — force the loop to cap.
    streams = [iter(_make_ollama_tool_use_stream([tc])) for _ in range(MAX_TOOL_ITERATIONS + 2)]

    mock_ollama = MagicMock()
    mock_ollama.chat.side_effect = streams

    mock_mcp = _make_ollama_mcp_client()

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama):
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Go"))
        result = list(client.respond_stream(context, conversation, mcp_client=mock_mcp))

    # Sentinel must be in the output
    assert "(tool-use limit reached)" in result

    # chat called exactly MAX_TOOL_ITERATIONS times
    assert mock_ollama.chat.call_count == MAX_TOOL_ITERATIONS

    # call_tool called once per iteration
    assert mock_mcp.call_tool.call_count == MAX_TOOL_ITERATIONS


# ---------------------------------------------------------------------------
# Test: tracer.emit called with tool_invoked fields
# ---------------------------------------------------------------------------


def test_ollama_respond_stream_emits_tool_invoked_trace():
    """tracer.emit is called with tool_invoked and the expected fields."""
    tc = _make_ollama_tool_call("kb_search", {"query": "test"})
    stream1 = _make_ollama_tool_use_stream([tc])
    stream2 = _make_ollama_end_stream("Here you go.")

    mock_ollama = MagicMock()
    mock_ollama.chat.side_effect = [iter(stream1), iter(stream2)]

    mock_mcp = _make_ollama_mcp_client(
        call_tool_return=ToolResult(content="found it", is_error=False)
    )
    mock_tracer = MagicMock(spec=Tracer)
    mock_tracer.enabled = True

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama):
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Search"))
        list(client.respond_stream(context, conversation, mcp_client=mock_mcp, tracer=mock_tracer))

    emit_calls = mock_tracer.emit.call_args_list
    tool_invoked_calls = [c for c in emit_calls if c[0][0] == "tool_invoked"]
    assert len(tool_invoked_calls) == 1

    _, kwargs = tool_invoked_calls[0]
    assert kwargs["tool_name"] == "kb_search"
    assert "query" in kwargs["arguments_preview"]
    assert kwargs["result_bytes"] == len(b"found it")
    assert kwargs["is_error"] is False
    assert "duration_s" in kwargs


# ---------------------------------------------------------------------------
# Test: non-MCPClientError re-raises (circuit breaker)
# ---------------------------------------------------------------------------


def test_ollama_respond_stream_reraises_unexpected_tool_exception():
    """Non-MCPClientError exception in call_tool propagates (for circuit breaker)."""
    tc = _make_ollama_tool_call("kb_search", {})
    stream1 = _make_ollama_tool_use_stream([tc])

    mock_ollama = MagicMock()
    mock_ollama.chat.return_value = iter(stream1)

    mock_mcp = _make_ollama_mcp_client(call_tool_side_effect=RuntimeError("unexpected"))

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama):
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="hi"))
        with pytest.raises(RuntimeError, match="unexpected"):
            list(client.respond_stream(context, conversation, mcp_client=mock_mcp))


# ---------------------------------------------------------------------------
# Test: _mcp_tools_to_ollama_tools pure function
# ---------------------------------------------------------------------------


def test_mcp_tools_to_ollama_tools_translates_correctly():
    """_mcp_tools_to_ollama_tools maps ToolSpec fields to Ollama function format."""
    tools = [
        ToolSpec(
            name="kb_search",
            description="Search the knowledge base",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        ToolSpec(
            name="kb_get",
            description="Get an entry",
            input_schema={"type": "object"},
        ),
    ]
    result = _mcp_tools_to_ollama_tools(tools)

    assert len(result) == 2

    t0 = result[0]
    assert t0["type"] == "function"
    assert t0["function"]["name"] == "kb_search"
    assert t0["function"]["description"] == "Search the knowledge base"
    assert t0["function"]["parameters"] == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    t1 = result[1]
    assert t1["type"] == "function"
    assert t1["function"]["name"] == "kb_get"
    assert t1["function"]["description"] == "Get an entry"
    assert t1["function"]["parameters"] == {"type": "object"}


# ---------------------------------------------------------------------------
# Test: no warning when mcp_client is given (V3.0.3 stub warning removed)
# ---------------------------------------------------------------------------


def test_ollama_respond_stream_no_warning_when_mcp_client_given(caplog):
    """The V3.0.3 'ignoring MCP client' warning is gone — no log on the Ollama path."""
    tc = _make_ollama_tool_call("kb_search", {"query": "x"})
    stream1 = _make_ollama_tool_use_stream([tc])
    stream2 = _make_ollama_end_stream("done")

    mock_ollama = MagicMock()
    mock_ollama.chat.side_effect = [iter(stream1), iter(stream2)]

    mock_mcp = _make_ollama_mcp_client()

    with (
        patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama),
        caplog.at_level(logging.WARNING, logger="meeting_agent.llm"),
    ):
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="hello"))
        list(client.respond_stream(context, conversation, mcp_client=mock_mcp))

    # No warning mentioning "not supported" or "ignoring"
    warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("not supported" in m.lower() or "ignoring" in m.lower() for m in warning_msgs), (
        f"Unexpected warning found: {warning_msgs}"
    )


# ---------------------------------------------------------------------------
# Test: string arguments fallback (Ollama older version compatibility)
# ---------------------------------------------------------------------------


def test_ollama_respond_stream_handles_string_tool_arguments():
    """When tool arguments come as a JSON string, they are parsed to a dict."""
    # Simulate an Ollama version that returns arguments as a JSON string
    tc_with_string_args = {"function": {"name": "kb_search", "arguments": '{"query": "x"}'}}
    stream1 = [{"message": {"content": "", "tool_calls": [tc_with_string_args]}}]
    stream2 = _make_ollama_end_stream("found it")

    mock_ollama = MagicMock()
    mock_ollama.chat.side_effect = [iter(stream1), iter(stream2)]

    mock_mcp = _make_ollama_mcp_client()

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama):
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="hi"))
        result = list(client.respond_stream(context, conversation, mcp_client=mock_mcp))

    assert "found it" in result
    # call_tool should receive a dict, not a string
    mock_mcp.call_tool.assert_called_once_with("kb_search", {"query": "x"})


def test_ollama_respond_stream_handles_malformed_string_tool_arguments(caplog):
    """Malformed JSON string in tool arguments falls back to empty dict."""
    tc_bad = {"function": {"name": "kb_search", "arguments": "not-json{"}}
    stream1 = [{"message": {"content": "", "tool_calls": [tc_bad]}}]
    stream2 = _make_ollama_end_stream("ok")

    mock_ollama = MagicMock()
    mock_ollama.chat.side_effect = [iter(stream1), iter(stream2)]

    mock_mcp = _make_ollama_mcp_client()

    with (
        patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama),
        caplog.at_level(logging.WARNING, logger="meeting_agent.llm"),
    ):
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="hi"))
        list(client.respond_stream(context, conversation, mcp_client=mock_mcp))

    # call_tool invoked with empty dict fallback
    mock_mcp.call_tool.assert_called_once_with("kb_search", {})
    # Warning should be logged
    assert any("parse" in r.message.lower() for r in caplog.records)


def test_ollama_respond_stream_tool_error_with_empty_content():
    """ToolResult with is_error=True and empty content gets default error message."""
    tc = _make_ollama_tool_call("kb_search", {})
    stream1 = _make_ollama_tool_use_stream([tc])
    stream2 = _make_ollama_end_stream("done")

    mock_ollama = MagicMock()
    mock_ollama.chat.side_effect = [iter(stream1), iter(stream2)]

    mock_mcp = _make_ollama_mcp_client(call_tool_return=ToolResult(content="", is_error=True))

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama):
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="hi"))
        list(client.respond_stream(context, conversation, mcp_client=mock_mcp))

    second_messages = mock_ollama.chat.call_args_list[1][1]["messages"]
    tool_msg = next(m for m in second_messages if m["role"] == "tool")
    assert tool_msg["content"] == "tool returned an error with no content"


def test_ollama_respond_stream_tool_iteration_cap_emits_trace():
    """When iteration cap hits with a tracer, tool_iteration_cap_hit event fires."""
    from meeting_agent.llm import MAX_TOOL_ITERATIONS

    tc = _make_ollama_tool_call("kb_search", {})
    streams = [iter(_make_ollama_tool_use_stream([tc])) for _ in range(MAX_TOOL_ITERATIONS + 2)]

    mock_ollama = MagicMock()
    mock_ollama.chat.side_effect = streams

    mock_mcp = _make_ollama_mcp_client(call_tool_return=ToolResult(content="r", is_error=False))
    mock_tracer = MagicMock()

    with patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama):
        client = OllamaClient()
        context = ProjectContext(system_prompt="x")
        conv = Conversation(latest_turn=Turn(speaker="Jason", text="hi"))
        list(client.respond_stream(context, conv, mcp_client=mock_mcp, tracer=mock_tracer))

    event_names = [call.args[0] for call in mock_tracer.emit.call_args_list]
    assert "tool_iteration_cap_hit" in event_names


def test_ollama_respond_stream_reraises_chat_exception_in_mcp_loop(caplog):
    """chat() raising in the MCP loop path logs a warning and re-raises."""
    mock_ollama = MagicMock()
    mock_ollama.chat.side_effect = ConnectionError("daemon down")

    mock_mcp = _make_ollama_mcp_client()

    with (
        patch("meeting_agent.llm.ollama.Client", return_value=mock_ollama),
        caplog.at_level(logging.WARNING, logger="meeting_agent.llm"),
    ):
        client = OllamaClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="Jason", text="Hi"))
        with pytest.raises(ConnectionError):
            list(client.respond_stream(context, conversation, mcp_client=mock_mcp))

    assert any("respond_stream failed" in r.message for r in caplog.records)
