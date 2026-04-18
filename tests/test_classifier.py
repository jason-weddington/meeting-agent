"""Unit tests for meeting_agent.classifier — Haiku decision gate."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import botocore.exceptions

from meeting_agent.asr import Utterance
from meeting_agent.classifier import (
    Classifier,
    Confidence,
    Decision,
    SessionState,
)
from meeting_agent.llm import ProjectContext, Turn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_classify_response(speaker: str, action: str, confidence: float) -> dict:
    """Build a fake Bedrock converse response with a toolUse output block."""
    return {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tool-id-1",
                            "name": "classify",
                            "input": {
                                "speaker": speaker,
                                "action": action,
                                "confidence": confidence,
                            },
                        }
                    }
                ]
            }
        }
    }


def _make_utterance(text: str = "Hello agent") -> Utterance:
    """Return a minimal Utterance for use in tests."""
    return Utterance(text=text, start_s=0.0, end_s=1.0)


def _make_confidence(
    avg_logprob: float = -0.3,
    no_speech_prob: float = 0.1,
    compression_ratio: float = 1.5,
) -> Confidence:
    """Return a Confidence with the given values."""
    return Confidence(
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
        compression_ratio=compression_ratio,
    )


def _make_session(
    recent_turns: tuple[Turn, ...] = (),
    agent_turns_last_5min: int = 0,
    agent_turns_last_30s: int = 0,
) -> SessionState:
    """Return a SessionState with the given values."""
    return SessionState(
        recent_turns=recent_turns,
        agent_turns_last_5min=agent_turns_last_5min,
        agent_turns_last_30s=agent_turns_last_30s,
    )


def _make_context(system_prompt: str = "You are a meeting agent.") -> ProjectContext:
    """Return a ProjectContext with the given system prompt."""
    return ProjectContext(system_prompt=system_prompt)


_SILENT_DECISION = Decision(speaker="unknown", action="silent", confidence=0.0)


# ---------------------------------------------------------------------------
# Lazy init tests
# ---------------------------------------------------------------------------


def test_init_is_cheap():
    """Classifier() construction does not call boto3; _client is None."""
    with patch("boto3.client") as mock_boto3:
        classifier = Classifier()
        mock_boto3.assert_not_called()
        assert classifier._client is None


# ---------------------------------------------------------------------------
# Request shape tests
# ---------------------------------------------------------------------------


def test_classify_request_shape():
    """classify() assembles a converse request with correct structure."""
    mock_client = MagicMock()
    mock_client.converse.return_value = _make_classify_response("Alice", "full_answer", 0.9)
    with patch("boto3.client", return_value=mock_client):
        classifier = Classifier()
        context = _make_context()
        utterance = _make_utterance("Alice, what's the status?")
        confidence = _make_confidence()
        session = _make_session(agent_turns_last_5min=1, agent_turns_last_30s=0)
        classifier.classify(utterance, confidence, context, session)

    call_kwargs = mock_client.converse.call_args[1]

    # Model ID
    assert call_kwargs["modelId"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    # System: 2 blocks — text then 1h cachePoint
    system = call_kwargs["system"]
    assert len(system) == 2
    assert "text" in system[0]
    assert system[1] == {"cachePoint": {"type": "default", "ttl": "1h"}}

    # Tool config forces the classify tool
    tool_config = call_kwargs["toolConfig"]
    assert tool_config["toolChoice"] == {"tool": {"name": "classify"}}
    tool_names = [t["toolSpec"]["name"] for t in tool_config["tools"]]
    assert "classify" in tool_names

    # User prompt contains utterance text, confidence field names, and airtime values
    user_text = call_kwargs["messages"][0]["content"][0]["text"]
    assert "Alice, what's the status?" in user_text
    assert "avg_logprob" in user_text
    assert "agent_turns_last_5min: 1" in user_text
    assert "agent_turns_last_30s: 0" in user_text


def test_system_prompt_embeds_project_context():
    """System block text contains the full context.system_prompt verbatim."""
    mock_client = MagicMock()
    mock_client.converse.return_value = _make_classify_response("Bob", "silent", 0.5)
    system_prompt = "Meeting context: Q4 roadmap review. Attendees: Jason, Alice, Bob."
    with patch("boto3.client", return_value=mock_client):
        classifier = Classifier()
        context = _make_context(system_prompt=system_prompt)
        classifier.classify(_make_utterance(), _make_confidence(), context, _make_session())

    system = mock_client.converse.call_args[1]["system"]
    assert system_prompt in system[0]["text"]


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------


def test_parses_tool_use_response():
    """A valid toolUse response is parsed into the correct Decision."""
    mock_client = MagicMock()
    mock_client.converse.return_value = _make_classify_response("Aziz", "full_answer", 0.88)
    with patch("boto3.client", return_value=mock_client):
        classifier = Classifier()
        decision = classifier.classify(
            _make_utterance(), _make_confidence(), _make_context(), _make_session()
        )

    assert decision == Decision(speaker="Aziz", action="full_answer", confidence=0.88)


# ---------------------------------------------------------------------------
# Fail-closed tests
# ---------------------------------------------------------------------------


def test_fail_closed_on_timeout(caplog):
    """ReadTimeoutError returns a silent Decision and logs a warning."""
    mock_client = MagicMock()
    mock_client.converse.side_effect = botocore.exceptions.ReadTimeoutError(
        endpoint_url="https://bedrock.test"
    )
    with patch("boto3.client", return_value=mock_client):
        classifier = Classifier()
        with caplog.at_level(logging.WARNING, logger="meeting_agent.classifier"):
            decision = classifier.classify(
                _make_utterance(), _make_confidence(), _make_context(), _make_session()
            )

    assert decision == _SILENT_DECISION
    assert caplog.records, "Expected at least one warning log entry"


def test_fail_closed_on_boto_error(caplog):
    """BotoCoreError returns a silent Decision and logs a warning."""
    mock_client = MagicMock()
    mock_client.converse.side_effect = botocore.exceptions.BotoCoreError()
    with patch("boto3.client", return_value=mock_client):
        classifier = Classifier()
        with caplog.at_level(logging.WARNING, logger="meeting_agent.classifier"):
            decision = classifier.classify(
                _make_utterance(), _make_confidence(), _make_context(), _make_session()
            )

    assert decision == _SILENT_DECISION
    assert caplog.records, "Expected at least one warning log entry"


def test_fail_closed_on_missing_tool_use(caplog):
    """Response with no toolUse block returns a silent Decision."""
    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "output": {
            "message": {
                "content": [{"text": "I should not be here"}],
            }
        }
    }
    with patch("boto3.client", return_value=mock_client):
        classifier = Classifier()
        with caplog.at_level(logging.WARNING, logger="meeting_agent.classifier"):
            decision = classifier.classify(
                _make_utterance(), _make_confidence(), _make_context(), _make_session()
            )

    assert decision == _SILENT_DECISION
    assert caplog.records, "Expected at least one warning log entry"


def test_fail_closed_on_schema_violation(caplog):
    """toolUse input missing required fields returns a silent Decision."""
    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tool-id-1",
                            "name": "classify",
                            "input": {
                                # Missing "action" and "confidence"
                                "speaker": "Alice",
                            },
                        }
                    }
                ],
            }
        }
    }
    with patch("boto3.client", return_value=mock_client):
        classifier = Classifier()
        with caplog.at_level(logging.WARNING, logger="meeting_agent.classifier"):
            decision = classifier.classify(
                _make_utterance(), _make_confidence(), _make_context(), _make_session()
            )

    assert decision == _SILENT_DECISION
    assert caplog.records, "Expected at least one warning log entry"


# ---------------------------------------------------------------------------
# Prompt content tests
# ---------------------------------------------------------------------------


def test_airtime_features_in_prompt():
    """SessionState airtime values appear in the user prompt."""
    mock_client = MagicMock()
    mock_client.converse.return_value = _make_classify_response("Bob", "silent", 0.5)
    session = _make_session(agent_turns_last_5min=3, agent_turns_last_30s=2)
    with patch("boto3.client", return_value=mock_client):
        classifier = Classifier()
        classifier.classify(_make_utterance(), _make_confidence(), _make_context(), session)

    user_text = mock_client.converse.call_args[1]["messages"][0]["content"][0]["text"]
    assert "agent_turns_last_30s: 2" in user_text
    assert "agent_turns_last_5min: 3" in user_text


def test_recent_turns_in_prompt():
    """Recent turns in SessionState appear formatted as 'speaker: text' in the user prompt."""
    mock_client = MagicMock()
    mock_client.converse.return_value = _make_classify_response("Jason", "silent", 0.5)
    session = SessionState(
        recent_turns=(Turn("Jason", "hello"), Turn("agent", "hi")),
        agent_turns_last_5min=0,
        agent_turns_last_30s=0,
    )
    with patch("boto3.client", return_value=mock_client):
        classifier = Classifier()
        classifier.classify(_make_utterance(), _make_confidence(), _make_context(), session)

    user_text = mock_client.converse.call_args[1]["messages"][0]["content"][0]["text"]
    assert "Jason: hello" in user_text
    assert "agent: hi" in user_text


def test_confidence_features_in_prompt():
    """Confidence feature values appear formatted to 3 decimal places in the user prompt."""
    mock_client = MagicMock()
    mock_client.converse.return_value = _make_classify_response("Alice", "silent", 0.5)
    conf = Confidence(avg_logprob=-0.5, no_speech_prob=0.2, compression_ratio=1.8)
    with patch("boto3.client", return_value=mock_client):
        classifier = Classifier()
        classifier.classify(_make_utterance(), conf, _make_context(), _make_session())

    user_text = mock_client.converse.call_args[1]["messages"][0]["content"][0]["text"]
    assert "-0.500" in user_text
    assert "0.200" in user_text
    assert "1.800" in user_text
