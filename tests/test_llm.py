"""Unit tests for meeting_agent.llm — BedrockClient with native multi-turn messages."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from meeting_agent.llm import (
    BedrockClient,
    Conversation,
    ProjectContext,
    Turn,
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
        conversation = Conversation(latest_turn=Turn(speaker="user", text="Hi"))
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
        conversation = Conversation(latest_turn=Turn(speaker="user", text="Hi"))
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
                Turn(speaker="user", text="Hello"),
                Turn(speaker="agent", text="Hi there"),
            ],
            latest_turn=Turn(speaker="user", text="What's the status?"),
        )
        list(client.respond_stream(context, conversation))

    messages = mock_client.converse_stream.call_args[1]["messages"]
    for msg in messages:
        for block in msg["content"]:
            assert "cachePoint" not in block, f"Unexpected cachePoint in message content: {block}"


def test_respond_stream_first_turn_messages_shape():
    """With older_turns=[] and a user latest_turn, messages is exactly one user message."""
    mock_client = _make_client_mock(_make_stream("reply"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(
            older_turns=[],
            latest_turn=Turn(speaker="user", text="First message"),
        )
        list(client.respond_stream(context, conversation))

    messages = mock_client.converse_stream.call_args[1]["messages"]
    assert messages == [{"role": "user", "content": [{"text": "First message"}]}]


def test_respond_stream_multi_turn_messages_shape():
    """Multi-turn older_turns map to alternating user/assistant messages."""
    mock_client = _make_client_mock(_make_stream("reply"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(
            older_turns=[
                Turn(speaker="user", text="q1"),
                Turn(speaker="agent", text="a1"),
                Turn(speaker="user", text="q2"),
                Turn(speaker="agent", text="a2"),
            ],
            latest_turn=Turn(speaker="user", text="q3"),
        )
        list(client.respond_stream(context, conversation))

    messages = mock_client.converse_stream.call_args[1]["messages"]
    assert messages == [
        {"role": "user", "content": [{"text": "q1"}]},
        {"role": "assistant", "content": [{"text": "a1"}]},
        {"role": "user", "content": [{"text": "q2"}]},
        {"role": "assistant", "content": [{"text": "a2"}]},
        {"role": "user", "content": [{"text": "q3"}]},
    ]


def test_respond_stream_speaker_agent_maps_to_assistant():
    """A Turn(speaker='agent', ...) in older_turns becomes role: 'assistant'."""
    mock_client = _make_client_mock(_make_stream("reply"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(
            older_turns=[Turn(speaker="agent", text="I am the agent.")],
            latest_turn=Turn(speaker="user", text="Tell me more"),
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
        conversation = Conversation(latest_turn=Turn(speaker="user", text="Hello"))
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


def test_respond_stream_raises_on_non_user_latest_turn():
    """respond_stream raises ValueError when latest_turn speaker is not 'user'."""
    client = BedrockClient()
    context = ProjectContext(system_prompt="Be helpful.")
    conversation = Conversation(latest_turn=Turn(speaker="agent", text="I speak"))

    with pytest.raises(ValueError, match="latest_turn"):
        list(client.respond_stream(context, conversation))


def test_respond_stream_raises_on_unknown_older_speaker():
    """respond_stream raises ValueError when an older_turn has an unknown speaker."""
    mock_client = _make_client_mock(_make_stream("reply"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(
            older_turns=[Turn(speaker="system", text="Do something")],
            latest_turn=Turn(speaker="user", text="Hi"),
        )

        with pytest.raises(ValueError, match="system"):
            list(client.respond_stream(context, conversation))


# ---------------------------------------------------------------------------
# Streaming output tests
# ---------------------------------------------------------------------------


def test_respond_stream_yields_text_deltas():
    """respond_stream yields text deltas in order."""
    mock_client = _make_client_mock(_make_stream("hello", " world"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="user", text="Hi"))
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
        conversation = Conversation(latest_turn=Turn(speaker="user", text="Hi"))
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
        conversation = Conversation(latest_turn=Turn(speaker="user", text="Hi"))
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
        conversation = Conversation(latest_turn=Turn(speaker="user", text="Hello"))
        list(client.respond_stream(context, conversation))

    mock_boto3_client.assert_called_once_with("bedrock-runtime", region_name="eu-west-1")


def test_client_reused_across_calls():
    """The boto3 client is created only once across multiple respond_stream calls."""
    mock_bedrock = _make_client_mock(_make_stream("ok"))
    mock_bedrock.converse_stream.return_value = _make_stream("ok")
    with patch("boto3.client", return_value=mock_bedrock) as mock_boto3_client:
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")

        conversation = Conversation(latest_turn=Turn(speaker="user", text="First"))
        list(client.respond_stream(context, conversation))

        # Reset the stream for second call
        mock_bedrock.converse_stream.return_value = _make_stream("ok2")
        conversation2 = Conversation(latest_turn=Turn(speaker="user", text="Second"))
        list(client.respond_stream(context, conversation2))

    # boto3.client should only be called once (lazy init)
    mock_boto3_client.assert_called_once()
