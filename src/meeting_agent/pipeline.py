"""End-to-end meeting-agent pipeline orchestrator.

Wires audio capture → wake detection → streaming ASR → Bedrock Claude →
sentence-pipelined TTS → audio playback. Maintains the rolling transcript and
gates the mic while the agent is speaking to avoid feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from meeting_agent.llm import ProjectContext


@dataclass
class PipelineConfig:
    """Runtime configuration for one meeting session."""

    input_device: int | None = None
    output_device: int | None = None
    wake_phrase: str = "hey_jarvis"
    model_id: str = "us.anthropic.claude-sonnet-4-6"
    asr_initial_prompt: str | None = None
    context: ProjectContext = field(default_factory=lambda: ProjectContext(system_prompt=""))


class Pipeline:
    """Main event loop for a meeting session.

    Lifecycle:
      1. Open mic stream.
      2. Feed every chunk to wake-word detector.
      3. On wake, start streaming ASR from the mic until VAD end-of-speech.
      4. Send the transcribed turn + rolling context to Bedrock Claude.
      5. Pipeline Claude's streamed sentences into Kokoro TTS.
      6. Play synthesized sentences while gating the mic.
      7. Append the exchange to the rolling transcript; return to step 2.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Store config; model loads happen lazily in :meth:`run`."""
        self.config = config

    def run(self) -> None:
        """Run the pipeline until interrupted (Ctrl-C)."""
        raise NotImplementedError
