"""Unit tests for meeting_agent.classifier — Haiku decision gate + Ollama backend."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import botocore.exceptions

from meeting_agent.asr import Utterance
from meeting_agent.classifier import (
    _DECISION_JSON_SCHEMA,
    BedrockClassifier,
    Confidence,
    Decision,
    OllamaClassifier,
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


def _make_ollama_response(speaker: str, action: str, confidence: float) -> dict:
    """Build a fake Ollama chat response dict."""
    import json

    return {
        "message": {
            "content": json.dumps({"speaker": speaker, "action": action, "confidence": confidence})
        }
    }


_SILENT_DECISION = Decision(speaker="unknown", action="silent", confidence=0.0)


# ---------------------------------------------------------------------------
# BedrockClassifier: lazy init tests
# ---------------------------------------------------------------------------


def test_init_is_cheap():
    """BedrockClassifier() construction does not call boto3; _client is None."""
    with patch("boto3.client") as mock_boto3:
        classifier = BedrockClassifier()
        mock_boto3.assert_not_called()
        assert classifier._client is None


# ---------------------------------------------------------------------------
# BedrockClassifier: request shape tests
# ---------------------------------------------------------------------------


def test_classify_request_shape():
    """classify() assembles a converse request with correct structure."""
    mock_client = MagicMock()
    mock_client.converse.return_value = _make_classify_response("Alice", "full_answer", 0.9)
    with patch("boto3.client", return_value=mock_client):
        classifier = BedrockClassifier()
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
        classifier = BedrockClassifier()
        context = _make_context(system_prompt=system_prompt)
        classifier.classify(_make_utterance(), _make_confidence(), context, _make_session())

    system = mock_client.converse.call_args[1]["system"]
    assert system_prompt in system[0]["text"]


# ---------------------------------------------------------------------------
# BedrockClassifier: response parsing tests
# ---------------------------------------------------------------------------


def test_parses_tool_use_response():
    """A valid toolUse response is parsed into the correct Decision."""
    mock_client = MagicMock()
    mock_client.converse.return_value = _make_classify_response("Aziz", "full_answer", 0.88)
    with patch("boto3.client", return_value=mock_client):
        classifier = BedrockClassifier()
        decision = classifier.classify(
            _make_utterance(), _make_confidence(), _make_context(), _make_session()
        )

    assert decision == Decision(speaker="Aziz", action="full_answer", confidence=0.88)


# ---------------------------------------------------------------------------
# BedrockClassifier: fail-closed tests
# ---------------------------------------------------------------------------


def test_fail_closed_on_timeout(caplog):
    """ReadTimeoutError returns a silent Decision and logs a warning."""
    mock_client = MagicMock()
    mock_client.converse.side_effect = botocore.exceptions.ReadTimeoutError(
        endpoint_url="https://bedrock.test"
    )
    with patch("boto3.client", return_value=mock_client):
        classifier = BedrockClassifier()
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
        classifier = BedrockClassifier()
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
        classifier = BedrockClassifier()
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
        classifier = BedrockClassifier()
        with caplog.at_level(logging.WARNING, logger="meeting_agent.classifier"):
            decision = classifier.classify(
                _make_utterance(), _make_confidence(), _make_context(), _make_session()
            )

    assert decision == _SILENT_DECISION
    assert caplog.records, "Expected at least one warning log entry"


# ---------------------------------------------------------------------------
# BedrockClassifier: prompt content tests
# ---------------------------------------------------------------------------


def test_airtime_features_in_prompt():
    """SessionState airtime values appear in the user prompt."""
    mock_client = MagicMock()
    mock_client.converse.return_value = _make_classify_response("Bob", "silent", 0.5)
    session = _make_session(agent_turns_last_5min=3, agent_turns_last_30s=2)
    with patch("boto3.client", return_value=mock_client):
        classifier = BedrockClassifier()
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
        classifier = BedrockClassifier()
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
        classifier = BedrockClassifier()
        classifier.classify(_make_utterance(), conf, _make_context(), _make_session())

    user_text = mock_client.converse.call_args[1]["messages"][0]["content"][0]["text"]
    assert "-0.500" in user_text
    assert "0.200" in user_text
    assert "1.800" in user_text


# ---------------------------------------------------------------------------
# Cache telemetry tests
# ---------------------------------------------------------------------------


def test_classifier_logs_cache_usage(caplog):
    """classify() logs a classifier_usage record with cache token fields."""
    mock_client = MagicMock()
    response = _make_classify_response("Alice", "full_answer", 0.9)
    response["usage"] = {
        "inputTokens": 1234,
        "outputTokens": 56,
        "totalTokens": 1290,
        "cacheReadInputTokens": 1200,
        "cacheWriteInputTokens": 0,
    }
    mock_client.converse.return_value = response
    with patch("boto3.client", return_value=mock_client):
        classifier = BedrockClassifier()
        with caplog.at_level(logging.INFO, logger="meeting_agent.classifier"):
            classifier.classify(
                _make_utterance(), _make_confidence(), _make_context(), _make_session()
            )

    usage_records = [r for r in caplog.records if r.getMessage() == "classifier_usage"]
    assert len(usage_records) == 1, f"Expected 1 classifier_usage record, got {len(usage_records)}"
    rec = usage_records[0]
    assert rec.input_tokens == 1234
    assert rec.output_tokens == 56
    assert rec.cache_read_tokens == 1200
    assert rec.cache_write_tokens == 0


# ---------------------------------------------------------------------------
# OllamaClassifier: init / defaults tests
# ---------------------------------------------------------------------------


def test_ollama_init_honors_defaults():
    """OllamaClassifier() uses DEFAULT_MODEL and DEFAULT_HOST when no args given."""
    classifier = OllamaClassifier()
    assert classifier.model == OllamaClassifier.DEFAULT_MODEL
    assert classifier.host == OllamaClassifier.DEFAULT_HOST
    assert classifier._client is None


def test_ollama_init_honors_model_override():
    """Custom model string is forwarded to chat(model=...)."""
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_ollama_response("x", "silent", 0.0)
    with patch("meeting_agent.classifier.ollama.Client", return_value=mock_client):
        classifier = OllamaClassifier(model="qwen3.6:35b-a3b-mlx-bf16")
        classifier.classify(_make_utterance(), _make_confidence(), _make_context(), _make_session())

    assert mock_client.chat.call_args[1]["model"] == "qwen3.6:35b-a3b-mlx-bf16"


def test_ollama_init_honors_host_override(monkeypatch):
    """Explicit host wins over OLLAMA_HOST env var."""
    monkeypatch.setenv("OLLAMA_HOST", "http://env-host:11434")
    classifier = OllamaClassifier(host="http://explicit:11434")
    assert classifier.host == "http://explicit:11434"


def test_ollama_init_honors_env_host(monkeypatch):
    """OLLAMA_HOST environment variable is used when no explicit host is given."""
    monkeypatch.setenv("OLLAMA_HOST", "http://env-host:11434")
    classifier = OllamaClassifier()
    assert classifier.host == "http://env-host:11434"


# ---------------------------------------------------------------------------
# OllamaClassifier: request shape tests
# ---------------------------------------------------------------------------


def test_ollama_chat_receives_full_prompts():
    """classify() sends system and user messages built from prompt helpers."""
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_ollama_response("Alice", "full_answer", 0.9)
    context = _make_context("Q4 roadmap attendees: Jason, Alice.")
    utterance = _make_utterance("Alice, what's the status?")

    with patch("meeting_agent.classifier.ollama.Client", return_value=mock_client):
        classifier = OllamaClassifier()
        classifier.classify(utterance, _make_confidence(), context, _make_session())

    call_kwargs = mock_client.chat.call_args[1]
    messages = call_kwargs["messages"]
    system_msg = next(m for m in messages if m["role"] == "system")
    user_msg = next(m for m in messages if m["role"] == "user")

    assert "Q4 roadmap attendees: Jason, Alice." in system_msg["content"]
    assert "Alice, what's the status?" in user_msg["content"]


def test_ollama_format_schema_passed():
    """classify() passes _DECISION_JSON_SCHEMA as the format= argument."""
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_ollama_response("x", "silent", 0.0)
    with patch("meeting_agent.classifier.ollama.Client", return_value=mock_client):
        classifier = OllamaClassifier()
        classifier.classify(_make_utterance(), _make_confidence(), _make_context(), _make_session())

    call_kwargs = mock_client.chat.call_args[1]
    assert call_kwargs["format"] == _DECISION_JSON_SCHEMA
    assert "speaker" in call_kwargs["format"]["properties"]
    assert "action" in call_kwargs["format"]["properties"]
    assert "confidence" in call_kwargs["format"]["properties"]


def test_ollama_temperature_zero_and_num_predict():
    """classify() passes temperature=0.0 and num_predict=256 in options."""
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_ollama_response("x", "silent", 0.0)
    with patch("meeting_agent.classifier.ollama.Client", return_value=mock_client):
        classifier = OllamaClassifier()
        classifier.classify(_make_utterance(), _make_confidence(), _make_context(), _make_session())

    options = mock_client.chat.call_args[1]["options"]
    assert options["temperature"] == 0.0
    assert options["num_predict"] == 256


# ---------------------------------------------------------------------------
# OllamaClassifier: response parsing tests
# ---------------------------------------------------------------------------


def test_ollama_parses_decision_from_response():
    """Valid JSON response is parsed into the correct Decision."""
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_ollama_response("Alice", "full_answer", 0.92)
    with patch("meeting_agent.classifier.ollama.Client", return_value=mock_client):
        classifier = OllamaClassifier()
        decision = classifier.classify(
            _make_utterance(), _make_confidence(), _make_context(), _make_session()
        )

    assert decision == Decision(speaker="Alice", action="full_answer", confidence=0.92)


# ---------------------------------------------------------------------------
# OllamaClassifier: fail-closed tests
# ---------------------------------------------------------------------------


def test_ollama_fail_closed_on_connection_error(caplog):
    """Connection error returns a silent Decision and logs a warning."""
    mock_client = MagicMock()
    mock_client.chat.side_effect = OSError("Connection refused")
    with patch("meeting_agent.classifier.ollama.Client", return_value=mock_client):
        classifier = OllamaClassifier()
        with caplog.at_level(logging.WARNING, logger="meeting_agent.classifier"):
            decision = classifier.classify(
                _make_utterance(), _make_confidence(), _make_context(), _make_session()
            )

    assert decision == _SILENT_DECISION
    assert caplog.records, "Expected at least one warning log entry"


def test_ollama_fail_closed_on_timeout(caplog):
    """Timeout error returns a silent Decision and logs a warning."""
    mock_client = MagicMock()
    mock_client.chat.side_effect = TimeoutError("Request timed out")
    with patch("meeting_agent.classifier.ollama.Client", return_value=mock_client):
        classifier = OllamaClassifier()
        with caplog.at_level(logging.WARNING, logger="meeting_agent.classifier"):
            decision = classifier.classify(
                _make_utterance(), _make_confidence(), _make_context(), _make_session()
            )

    assert decision == _SILENT_DECISION
    assert caplog.records, "Expected at least one warning log entry"


def test_ollama_fail_closed_on_malformed_json(caplog):
    """Non-JSON response content returns a silent Decision and logs a warning."""
    mock_client = MagicMock()
    mock_client.chat.return_value = {"message": {"content": "not valid json {{{"}}
    with patch("meeting_agent.classifier.ollama.Client", return_value=mock_client):
        classifier = OllamaClassifier()
        with caplog.at_level(logging.WARNING, logger="meeting_agent.classifier"):
            decision = classifier.classify(
                _make_utterance(), _make_confidence(), _make_context(), _make_session()
            )

    assert decision == _SILENT_DECISION
    assert caplog.records, "Expected at least one warning log entry"


def test_ollama_fail_closed_on_schema_violation(caplog):
    """JSON response missing required 'action' field returns a silent Decision."""
    import json

    mock_client = MagicMock()
    # Missing "action" field
    mock_client.chat.return_value = {
        "message": {"content": json.dumps({"speaker": "Alice", "confidence": 0.9})}
    }
    with patch("meeting_agent.classifier.ollama.Client", return_value=mock_client):
        classifier = OllamaClassifier()
        with caplog.at_level(logging.WARNING, logger="meeting_agent.classifier"):
            decision = classifier.classify(
                _make_utterance(), _make_confidence(), _make_context(), _make_session()
            )

    assert decision == _SILENT_DECISION
    assert caplog.records, "Expected at least one warning log entry"


def test_ollama_fail_closed_on_unknown_action_value(caplog):
    """JSON with an unknown action enum value returns a silent Decision."""
    import json

    mock_client = MagicMock()
    mock_client.chat.return_value = {
        "message": {
            "content": json.dumps({"speaker": "Alice", "action": "wiggle", "confidence": 0.9})
        }
    }
    with patch("meeting_agent.classifier.ollama.Client", return_value=mock_client):
        classifier = OllamaClassifier()
        with caplog.at_level(logging.WARNING, logger="meeting_agent.classifier"):
            decision = classifier.classify(
                _make_utterance(), _make_confidence(), _make_context(), _make_session()
            )

    assert decision == _SILENT_DECISION
    assert caplog.records, "Expected at least one warning log entry"


# ---------------------------------------------------------------------------
# OllamaClassifier: warm_up tests
# ---------------------------------------------------------------------------


def test_ollama_warm_up_sends_minimal_chat():
    """warm_up() sends a minimal chat request to force-load the model."""
    mock_client = MagicMock()
    mock_client.chat.return_value = {"message": {"content": "ok"}}
    with patch("meeting_agent.classifier.ollama.Client", return_value=mock_client):
        classifier = OllamaClassifier(model="qwen3.5:35b-a3b")
        ok = classifier.warm_up()

    assert ok is True
    call_kwargs = mock_client.chat.call_args[1]
    assert call_kwargs["model"] == "qwen3.5:35b-a3b"
    assert call_kwargs["options"]["num_predict"] == 1
    assert call_kwargs["options"]["temperature"] == 0.0


def test_ollama_warm_up_returns_false_on_failure(caplog):
    """warm_up() returns False and logs a warning on connection failure."""
    mock_client = MagicMock()
    mock_client.chat.side_effect = OSError("Connection refused")
    with patch("meeting_agent.classifier.ollama.Client", return_value=mock_client):
        classifier = OllamaClassifier()
        with caplog.at_level(logging.WARNING, logger="meeting_agent.classifier"):
            ok = classifier.warm_up()

    assert ok is False
    assert caplog.records, "Expected at least one warning log entry"


def test_ollama_default_timeout_accommodates_cold_load():
    """DEFAULT_TIMEOUT_S is at least 30s so large-model cold-load doesn't abort."""
    assert OllamaClassifier.DEFAULT_TIMEOUT_S >= 30.0
