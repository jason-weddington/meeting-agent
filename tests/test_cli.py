"""Tests for meeting_agent.cli."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from meeting_agent.cli import DEFAULT_MODEL_ID, DEFAULT_SYSTEM_PROMPT, main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_main(argv: list[str]) -> None:
    """Set sys.argv and call main()."""
    with patch.object(sys, "argv", argv):
        main()


def _run_main_with_pipeline(argv: list[str]) -> MagicMock:
    """Run main() with Pipeline mocked; return the mock class for assertions."""
    with patch.object(sys, "argv", argv), patch("meeting_agent.cli.Pipeline") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        main()
    return mock_cls


# ---------------------------------------------------------------------------
# --list-devices
# ---------------------------------------------------------------------------


def test_list_devices_prints_correct_format_and_exits_zero(monkeypatch, capsys):
    """--list-devices prints device table and exits 0 without calling Pipeline."""
    input_devs = [
        {"index": 0, "name": "MacBook Pro Microphone", "channels": 1},
        {"index": 1, "name": "BlackHole 2ch", "channels": 2},
    ]
    output_devs = [
        {"index": 0, "name": "MacBook Pro Speakers", "channels": 2},
        {"index": 1, "name": "BlackHole 2ch", "channels": 2},
    ]

    monkeypatch.setattr(sys, "argv", ["meeting-agent", "--list-devices"])

    with (
        patch("meeting_agent.audio.list_input_devices", return_value=input_devs),
        patch("meeting_agent.audio.list_output_devices", return_value=output_devs),
        patch("meeting_agent.cli.Pipeline") as mock_pipeline,
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 0
    mock_pipeline.assert_not_called()

    out = capsys.readouterr().out
    assert "Input devices:" in out
    assert "  [0] MacBook Pro Microphone (1ch)" in out
    assert "  [1] BlackHole 2ch (2ch)" in out
    assert "Output devices:" in out
    assert "  [0] MacBook Pro Speakers (2ch)" in out


# ---------------------------------------------------------------------------
# Default flags → PipelineConfig
# ---------------------------------------------------------------------------


def test_default_flags_produce_expected_pipeline_config(monkeypatch):
    """No flags → PipelineConfig defaults and built-in system prompt."""
    monkeypatch.setattr(sys, "argv", ["meeting-agent"])

    mock_cls = _run_main_with_pipeline(["meeting-agent"])
    config = mock_cls.call_args[0][0]

    assert config.input_device is None
    assert config.output_device is None
    assert config.model_id == DEFAULT_MODEL_ID
    assert config.asr_initial_prompt is None
    assert config.context.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert config.context.project_docs == ""

    # run() must be called exactly once
    mock_cls.return_value.run.assert_called_once()


def test_custom_flags_produce_expected_pipeline_config():
    """Explicit flags override all defaults in PipelineConfig."""
    argv = [
        "meeting-agent",
        "--input-device",
        "3",
        "--output-device",
        "5",
        "--model-id",
        "us.anthropic.claude-opus-4-5",
        "--initial-prompt",
        "AWS EKS S3 Lambda",
        "--system-prompt",
        "Be terse.",
    ]

    mock_cls = _run_main_with_pipeline(argv)
    config = mock_cls.call_args[0][0]

    assert config.input_device == 3
    assert config.output_device == 5
    assert config.model_id == "us.anthropic.claude-opus-4-5"
    assert config.asr_initial_prompt == "AWS EKS S3 Lambda"
    assert config.context.system_prompt == "Be terse."


# ---------------------------------------------------------------------------
# File-based prompt flags
# ---------------------------------------------------------------------------


def test_initial_prompt_file_reads_file_contents(tmp_path):
    """--initial-prompt-file reads text and passes it as asr_initial_prompt."""
    prompt_file = tmp_path / "vocab.txt"
    prompt_file.write_text("Kubernetes Helm ArgoCD")

    argv = ["meeting-agent", "--initial-prompt-file", str(prompt_file)]
    mock_cls = _run_main_with_pipeline(argv)
    config = mock_cls.call_args[0][0]

    assert config.asr_initial_prompt == "Kubernetes Helm ArgoCD"


def test_system_prompt_file_reads_file_contents(tmp_path):
    """--system-prompt-file reads text and passes it as context.system_prompt."""
    prompt_file = tmp_path / "system.txt"
    prompt_file.write_text("You are a terse assistant.")

    argv = ["meeting-agent", "--system-prompt-file", str(prompt_file)]
    mock_cls = _run_main_with_pipeline(argv)
    config = mock_cls.call_args[0][0]

    assert config.context.system_prompt == "You are a terse assistant."


# ---------------------------------------------------------------------------
# --context-dir / --context-uri
# ---------------------------------------------------------------------------


def test_context_dir_populates_system_prompt(tmp_path):
    """--context-dir with .md files sets context.system_prompt to loaded content."""
    (tmp_path / "001-role.md").write_text("You are a terse meeting bot.")
    (tmp_path / "002-rules.md").write_text("No bullet lists.")

    argv = ["meeting-agent", "--context-dir", str(tmp_path)]
    mock_cls = _run_main_with_pipeline(argv)
    config = mock_cls.call_args[0][0]

    assert "001-role" in config.context.system_prompt
    assert "You are a terse meeting bot." in config.context.system_prompt
    assert "002-rules" in config.context.system_prompt
    assert "No bullet lists." in config.context.system_prompt


def test_context_uri_populates_system_prompt():
    """--context-uri calls load_context and sets context.system_prompt."""
    with patch("meeting_agent.cli.load_context", return_value="S3 context loaded") as mock_lc:
        mock_cls = _run_main_with_pipeline(
            ["meeting-agent", "--context-uri", "s3://bucket/prefix/"]
        )
    config = mock_cls.call_args[0][0]
    mock_lc.assert_called_once_with("s3://bucket/prefix/")
    assert config.context.system_prompt == "S3 context loaded"


def test_context_dir_and_system_prompt_mutually_exclusive(monkeypatch, tmp_path):
    """--context-dir and --system-prompt together raise SystemExit(2)."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["meeting-agent", "--context-dir", str(tmp_path), "--system-prompt", "Be terse."],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_context_dir_and_system_prompt_file_mutually_exclusive(monkeypatch, tmp_path):
    """--context-dir and --system-prompt-file together raise SystemExit(2)."""
    prompt_file = tmp_path / "system.txt"
    prompt_file.write_text("A prompt.")
    monkeypatch.setattr(
        sys,
        "argv",
        ["meeting-agent", "--context-dir", str(tmp_path), "--system-prompt-file", str(prompt_file)],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_context_uri_and_system_prompt_mutually_exclusive(monkeypatch):
    """--context-uri and --system-prompt together raise SystemExit(2)."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["meeting-agent", "--context-uri", "s3://b/p/", "--system-prompt", "Be terse."],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_context_dir_and_context_uri_mutually_exclusive(monkeypatch, tmp_path):
    """--context-dir and --context-uri together raise SystemExit(2)."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["meeting-agent", "--context-dir", str(tmp_path), "--context-uri", "s3://b/p/"],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Mutually exclusive flag pairs
# ---------------------------------------------------------------------------


def test_initial_prompt_and_initial_prompt_file_are_mutually_exclusive(monkeypatch):
    """--initial-prompt + --initial-prompt-file raises SystemExit(2)."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["meeting-agent", "--initial-prompt", "hello", "--initial-prompt-file", "/tmp/x.txt"],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_system_prompt_and_system_prompt_file_are_mutually_exclusive(monkeypatch):
    """--system-prompt + --system-prompt-file raises SystemExit(2)."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["meeting-agent", "--system-prompt", "hi", "--system-prompt-file", "/tmp/x.txt"],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# KeyboardInterrupt handling
# ---------------------------------------------------------------------------


def test_keyboard_interrupt_is_caught_cleanly(monkeypatch, capsys):
    """Ctrl-C while Pipeline.run() is running prints a message instead of traceback."""
    monkeypatch.setattr(sys, "argv", ["meeting-agent"])

    with (
        patch("meeting_agent.cli.Pipeline") as mock_cls,
    ):
        mock_instance = MagicMock()
        mock_instance.run.side_effect = KeyboardInterrupt
        mock_cls.return_value = mock_instance
        main()  # must not raise

    out = capsys.readouterr().out
    assert "Exiting" in out


# ---------------------------------------------------------------------------
# Classifier backend / model / host flags
# ---------------------------------------------------------------------------


def test_cli_classifier_backend_defaults_to_bedrock():
    """No --classifier-backend flag → PipelineConfig.classifier_backend == 'bedrock'."""
    mock_cls = _run_main_with_pipeline(["meeting-agent"])
    config = mock_cls.call_args[0][0]
    assert config.classifier_backend == "bedrock"


def test_cli_classifier_backend_ollama():
    """--classifier-backend ollama sets PipelineConfig.classifier_backend to 'ollama'."""
    mock_cls = _run_main_with_pipeline(["meeting-agent", "--classifier-backend", "ollama"])
    config = mock_cls.call_args[0][0]
    assert config.classifier_backend == "ollama"


def test_cli_classifier_model_forwarded():
    """--classifier-model <name> populates PipelineConfig.classifier_model."""
    mock_cls = _run_main_with_pipeline(
        ["meeting-agent", "--classifier-model", "qwen3.6:35b-a3b-mlx-bf16"]
    )
    config = mock_cls.call_args[0][0]
    assert config.classifier_model == "qwen3.6:35b-a3b-mlx-bf16"


def test_cli_classifier_model_defaults_to_none():
    """No --classifier-model flag → PipelineConfig.classifier_model is None."""
    mock_cls = _run_main_with_pipeline(["meeting-agent"])
    config = mock_cls.call_args[0][0]
    assert config.classifier_model is None


def test_cli_ollama_host_forwarded():
    """--ollama-host <url> populates PipelineConfig.ollama_host."""
    mock_cls = _run_main_with_pipeline(
        ["meeting-agent", "--ollama-host", "http://example.local:11434"]
    )
    config = mock_cls.call_args[0][0]
    assert config.ollama_host == "http://example.local:11434"


def test_cli_ollama_host_defaults_to_none():
    """No --ollama-host flag → PipelineConfig.ollama_host is None."""
    mock_cls = _run_main_with_pipeline(["meeting-agent"])
    config = mock_cls.call_args[0][0]
    assert config.ollama_host is None
