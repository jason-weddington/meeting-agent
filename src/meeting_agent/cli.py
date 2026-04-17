"""Entry point for the ``meeting-agent`` CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from meeting_agent.llm import ProjectContext
from meeting_agent.pipeline import Pipeline, PipelineConfig

DEFAULT_MODEL_ID: str = "us.anthropic.claude-sonnet-4-6"

DEFAULT_SYSTEM_PROMPT: str = """You are an AI meeting participant. You were activated
because the user said your wake phrase. Keep responses concise — under 3 sentences
unless asked for detail. If the question is ambiguous, ask one clarifying question
rather than guessing."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meeting-agent",
        description="Real-time AI participant for live meetings.",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        metavar="INT",
        help="PortAudio input device index (default: OS default)",
    )
    parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        metavar="INT",
        help="PortAudio output device index (default: OS default)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available input + output devices and exit",
    )
    parser.add_argument(
        "--wake-phrase",
        default="hey_jarvis",
        metavar="NAME",
        help="openwakeword model name or custom path (default: hey_jarvis)",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        metavar="ID",
        help=f"Bedrock model id (default: {DEFAULT_MODEL_ID})",
    )

    initial_prompt_group = parser.add_mutually_exclusive_group()
    initial_prompt_group.add_argument(
        "--initial-prompt",
        default=None,
        metavar="TEXT",
        help="Whisper custom-vocab prompt (default: none)",
    )
    initial_prompt_group.add_argument(
        "--initial-prompt-file",
        default=None,
        metavar="PATH",
        type=Path,
        help="Read custom-vocab prompt from a file (default: none)",
    )

    system_prompt_group = parser.add_mutually_exclusive_group()
    system_prompt_group.add_argument(
        "--system-prompt",
        default=None,
        metavar="TEXT",
        help="System prompt for Claude (default: built-in default)",
    )
    system_prompt_group.add_argument(
        "--system-prompt-file",
        default=None,
        metavar="PATH",
        type=Path,
        help="Read system prompt from a file",
    )

    parser.add_argument(
        "--project-docs",
        default=None,
        metavar="PATH",
        type=Path,
        help="Read project docs / decision log / stakeholders (default: none)",
    )

    return parser


def main() -> None:
    """Parse CLI arguments and run the meeting-agent pipeline."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_devices:
        from meeting_agent.audio import list_input_devices, list_output_devices

        print("Input devices:")
        for dev in list_input_devices():
            print(f"  [{dev['index']}] {dev['name']} ({dev['channels']}ch)")
        print("Output devices:")
        for dev in list_output_devices():
            print(f"  [{dev['index']}] {dev['name']} ({dev['channels']}ch)")
        sys.exit(0)

    # Resolve initial (Whisper) prompt
    initial_prompt: str | None = args.initial_prompt
    if args.initial_prompt_file is not None:
        initial_prompt = args.initial_prompt_file.read_text()

    # Resolve system prompt
    if args.system_prompt is not None:
        system_prompt: str = args.system_prompt
    elif args.system_prompt_file is not None:
        system_prompt = args.system_prompt_file.read_text()
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    # Resolve project docs
    project_docs: str = ""
    if args.project_docs is not None:
        project_docs = args.project_docs.read_text()

    context = ProjectContext(
        system_prompt=system_prompt,
        project_docs=project_docs,
    )

    config = PipelineConfig(
        input_device=args.input_device,
        output_device=args.output_device,
        wake_phrase=args.wake_phrase,
        model_id=args.model_id,
        asr_initial_prompt=initial_prompt,
        context=context,
    )

    try:
        Pipeline(config).run()
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
