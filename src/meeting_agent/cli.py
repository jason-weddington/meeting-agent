"""Entry point for the ``meeting-agent`` CLI."""

from __future__ import annotations


def main() -> None:
    """Parse args and run the pipeline.

    Flags to support:
      --input-device / --output-device (PortAudio indices; see
      ``audio.list_input_devices()``).
      --wake-phrase (openwakeword model name).
      --model-id (Bedrock model ID; default Sonnet 4.6 US CRIS).
      --initial-prompt (custom-vocab jargon to bias Whisper).
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
