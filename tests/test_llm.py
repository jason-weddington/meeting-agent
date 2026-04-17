"""Unit tests for meeting_agent.llm — BedrockClient with prompt caching."""

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
# Request shape tests
# ---------------------------------------------------------------------------


def test_request_model_id():
    """converse_stream is called with the correct modelId."""
    mock_client = _make_client_mock(_make_stream("hi"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient(model_id="us.anthropic.claude-sonnet-4-6")
        context = ProjectContext(system_prompt="You are a helpful assistant.")
        conversation = Conversation(latest_turn=Turn(speaker="User", text="Hello"))
        list(client.respond_stream(context, conversation))

    call_kwargs = mock_client.converse_stream.call_args[1]
    assert call_kwargs["modelId"] == "us.anthropic.claude-sonnet-4-6"


def test_system_block_structure():
    """system has exactly 2 entries: text block then 1h cachePoint."""
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
        conversation = Conversation(latest_turn=Turn(speaker="User", text="Hi"))
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
        conversation = Conversation(latest_turn=Turn(speaker="User", text="Hi"))
        list(client.respond_stream(context, conversation))

    system = mock_client.converse_stream.call_args[1]["system"]
    assert system[0]["text"] == "Be concise."


def test_messages_content_order():
    """messages[0].content is: text(older_turns) -> cachePoint(5m) -> text(latest_turn)."""
    mock_client = _make_client_mock(_make_stream("reply"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(
            older_turns=[
                Turn(speaker="Alice", text="Hello"),
                Turn(speaker="Bob", text="Hi there"),
            ],
            latest_turn=Turn(speaker="Alice", text="What's the status?"),
        )
        list(client.respond_stream(context, conversation))

    content = mock_client.converse_stream.call_args[1]["messages"][0]["content"]
    assert len(content) == 3

    # First: text with older turns
    assert "text" in content[0]
    assert "Alice: Hello" in content[0]["text"]
    assert "Bob: Hi there" in content[0]["text"]

    # Second: cachePoint with 5m TTL
    assert content[1] == {"cachePoint": {"type": "default", "ttl": "5m"}}

    # Third: text with latest turn
    assert "text" in content[2]
    assert "Alice: What's the status?" in content[2]["text"]


def test_messages_older_turns_empty_uses_space():
    """When older_turns is empty, the text block uses a single space (not empty string)."""
    mock_client = _make_client_mock(_make_stream("reply"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(
            older_turns=[],
            latest_turn=Turn(speaker="User", text="First message"),
        )
        list(client.respond_stream(context, conversation))

    content = mock_client.converse_stream.call_args[1]["messages"][0]["content"]
    assert content[0]["text"] == " "


# ---------------------------------------------------------------------------
# Streaming output tests
# ---------------------------------------------------------------------------


def test_respond_stream_yields_text_deltas():
    """respond_stream yields text deltas in order."""
    mock_client = _make_client_mock(_make_stream("hello", " world"))
    with patch("boto3.client", return_value=mock_client):
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")
        conversation = Conversation(latest_turn=Turn(speaker="User", text="Hi"))
        result = list(client.respond_stream(context, conversation))

    assert result == ["hello", " world"]


def test_respond_stream_skips_non_delta_events():
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
        conversation = Conversation(latest_turn=Turn(speaker="User", text="Hi"))
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
        conversation = Conversation(latest_turn=Turn(speaker="User", text="Hi"))
        result = list(client.respond_stream(context, conversation))

    assert result == ["actual text"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_respond_stream_raises_if_latest_turn_is_none():
    """respond_stream raises ValueError when conversation.latest_turn is None."""
    client = BedrockClient()
    context = ProjectContext(system_prompt="Be helpful.")
    conversation = Conversation(latest_turn=None)

    with pytest.raises(ValueError, match="latest_turn"):
        list(client.respond_stream(context, conversation))


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
        conversation = Conversation(latest_turn=Turn(speaker="User", text="Hello"))
        list(client.respond_stream(context, conversation))

    mock_boto3_client.assert_called_once_with("bedrock-runtime", region_name="eu-west-1")


def test_client_reused_across_calls():
    """The boto3 client is created only once across multiple respond_stream calls."""
    mock_bedrock = _make_client_mock(_make_stream("ok"))
    mock_bedrock.converse_stream.return_value = _make_stream("ok")
    with patch("boto3.client", return_value=mock_bedrock) as mock_boto3_client:
        client = BedrockClient()
        context = ProjectContext(system_prompt="Be helpful.")

        conversation = Conversation(latest_turn=Turn(speaker="User", text="First"))
        list(client.respond_stream(context, conversation))

        # Reset the stream for second call
        mock_bedrock.converse_stream.return_value = _make_stream("ok2")
        conversation2 = Conversation(latest_turn=Turn(speaker="User", text="Second"))
        list(client.respond_stream(context, conversation2))

    # boto3.client should only be called once (lazy init)
    mock_boto3_client.assert_called_once()
