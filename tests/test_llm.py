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
)


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
