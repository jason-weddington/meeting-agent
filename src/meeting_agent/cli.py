"""Entry point for the ``meeting-agent`` CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from meeting_agent.context import load_context
from meeting_agent.llm import ProjectContext
from meeting_agent.pipeline import Pipeline, PipelineConfig

DEFAULT_MODEL_ID: str = "us.anthropic.claude-sonnet-4-6"

DEFAULT_SYSTEM_PROMPT: str = """You are a real-time participant in a meeting, not
an assistant. Silence is a valid contribution — prefer it when you don't have
something specifically useful to add. When you do speak, write as you would
speak: natural prose, no markdown, no bulleted or numbered lists, no symbols
that would pronounce as words when read aloud by a text-to-speech engine. Keep
responses concise (under three sentences) unless asked for more detail.

The meeting transcript you see is imperfect — audio drops, speaker overlap, and
garbled segments are normal. Do not ask humans to repeat themselves or
acknowledge audio failures; that's not your job. You MAY ask clarifying
questions about the topic being discussed — "when you say X, do you mean Y?",
"let me play that back: the decision is Z, right?" — these are legitimate
meeting contributions. You MAY NOT ask about the audio itself — "sorry, can
you repeat?", "I didn't catch that", "you cut out". Treat audio difficulties
as your own problem to handle silently.

When the context is thin or the question is ambiguous and you must respond,
produce a short substantive answer with an embedded parenthetical check
rather than asking for clarification — "Sounds like Thursday the 23rd —
is that what you meant?" is better than "Which date?"."""


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

    context_source_group = parser.add_mutually_exclusive_group()
    context_source_group.add_argument(
        "--system-prompt",
        default=None,
        metavar="TEXT",
        help="System prompt for Claude (default: built-in default)",
    )
    context_source_group.add_argument(
        "--system-prompt-file",
        default=None,
        metavar="PATH",
        type=Path,
        help="Read system prompt from a file",
    )
    context_source_group.add_argument(
        "--context-dir",
        default=None,
        metavar="PATH",
        type=Path,
        help="Local directory of *.md context files (sorted and concatenated into system prompt)",
    )
    context_source_group.add_argument(
        "--context-uri",
        default=None,
        metavar="URI",
        help="s3://bucket/prefix/ of *.md context files (sorted + concatenated into system prompt)",
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

    # Resolve system prompt (one source wins; all four flags are mutually exclusive)
    if args.context_dir is not None:
        system_prompt: str = load_context(args.context_dir)
    elif args.context_uri is not None:
        system_prompt = load_context(args.context_uri)
    elif args.system_prompt is not None:
        system_prompt = args.system_prompt
    elif args.system_prompt_file is not None:
        system_prompt = args.system_prompt_file.read_text()
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    context = ProjectContext(system_prompt=system_prompt)

    config = PipelineConfig(
        input_device=args.input_device,
        output_device=args.output_device,
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
