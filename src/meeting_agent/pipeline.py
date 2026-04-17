"""End-to-end meeting-agent pipeline orchestrator.

Wires audio capture → wake detection → streaming ASR → Bedrock Claude →
sentence-pipelined TTS → audio playback. Maintains the rolling transcript and
gates the mic while the agent is speaking to avoid feedback.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from meeting_agent import audio
from meeting_agent.asr import StreamingASR, Utterance
from meeting_agent.audio import AudioArray
from meeting_agent.llm import BedrockClient, Conversation, ProjectContext, Turn
from meeting_agent.tts import SAMPLE_RATE as TTS_SAMPLE_RATE
from meeting_agent.tts import TTS
from meeting_agent.wake import WakeDetector

_metrics = logging.getLogger("meeting_agent.metrics")

_CHUNK_MS = 100
_ECHO_TAIL_S = 0.5  # extra drain time past playback to flush speaker echo


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
        config = self.config

        # Build components — model loads happen here.
        chunks_iter = audio.record_chunks(device=config.input_device, chunk_ms=100)
        wake = WakeDetector(config.wake_phrase)
        asr = StreamingASR(initial_prompt=config.asr_initial_prompt)
        tts = TTS()
        llm = BedrockClient(model_id=config.model_id)

        # Rolling transcript initialised empty.
        conversation = Conversation(older_turns=[], latest_turn=None)

        try:
            while True:
                # Stage 1: wait for wake-word (mic is active; chunks consumed here).
                t_wake = self._wait_for_wake(chunks_iter, wake)

                # Stage 2: transcribe the user's utterance.
                utterance = self._collect_utterance(chunks_iter, asr)
                t_asr_done = time.monotonic()
                _metrics.info("asr_latency_s=%.3f", t_asr_done - t_wake)

                # Update conversation with the new user turn.
                conversation.latest_turn = Turn(speaker="user", text=utterance.text)

                # Stage 3: stream LLM response, pipeline TTS, play audio.
                # Mic is implicitly gated here: chunks_iter is not read while
                # _stream_and_play is executing, so wake detection is suspended.
                t_speak_start = time.monotonic()
                full_response = self._stream_and_play(config, conversation, llm, tts, t_asr_done)
                speak_duration = time.monotonic() - t_speak_start

                # Stage 4: flush the mic-queue backlog of the agent's own voice
                # before returning to wake-word listening. Without this, the
                # next _wait_for_wake call processes buffered echo and triggers
                # on the agent's previous response — a feedback loop.
                self._drain_echo(chunks_iter, speak_duration + _ECHO_TAIL_S)

                # Commit the completed exchange to the rolling transcript.
                conversation.older_turns.append(Turn(speaker="user", text=utterance.text))
                conversation.older_turns.append(Turn(speaker="agent", text=full_response))
                conversation.latest_turn = None

        except KeyboardInterrupt:
            pass

    # ------------------------------------------------------------------
    # Named pipeline stages — kept separate for testability.
    # ------------------------------------------------------------------

    def _wait_for_wake(
        self,
        chunks_iter: Iterator[AudioArray],
        wake: WakeDetector,
    ) -> float:
        """Block until a wake-word trigger is detected.

        Args:
            chunks_iter: Continuous 16 kHz audio chunk iterator.
            wake: Configured wake-word detector.

        Returns:
            Monotonic timestamp of the wake detection event.
        """
        print(f"Listening for wake phrase ({self.config.wake_phrase!r})...", flush=True)
        for chunk in chunks_iter:
            if wake.detect(chunk):
                print("Wake detected. Listening for your question...", flush=True)
                return time.monotonic()
        return time.monotonic()  # iterator exhausted (edge case / tests)

    def _collect_utterance(
        self,
        chunks_iter: Iterator[AudioArray],
        asr: StreamingASR,
    ) -> Utterance:
        """Feed audio chunks to ASR and return the first completed utterance.

        Args:
            chunks_iter: Same continuous chunk iterator used by
                :meth:`_wait_for_wake`; ASR consumes subsequent chunks.
            asr: Streaming ASR instance with Silero VAD endpointing.

        Returns:
            The first :class:`~meeting_agent.asr.Utterance` produced.

        Raises:
            RuntimeError: If the ASR stream ends without yielding an utterance.
        """
        for utterance in asr.transcribe_stream(chunks_iter):
            return utterance
        raise RuntimeError("ASR stream ended without producing an utterance")

    def _drain_echo(self, chunks_iter: Iterator[AudioArray], duration_s: float) -> None:
        """Discard ``duration_s`` seconds of chunks from the iterator.

        The mic-capture queue inside :func:`audio.record_chunks` keeps filling
        while the agent speaks (the callback runs regardless of consumption).
        After agent playback ends, that backlog is full of the agent's own
        audio captured via the speakers. Reading it into the wake detector
        produces a feedback loop: the agent hears itself, triggers wake, and
        transcribes its prior response as the user's turn.

        This drains the backlog plus a small tail to cover residual echo.
        Assumes 100 ms chunks (as configured in :meth:`run`). The backlog
        chunks pop immediately; only the tail chunks block on the queue.
        """
        n_chunks = max(1, int(duration_s * 1000 / _CHUNK_MS))
        for _ in range(n_chunks):
            next(chunks_iter, None)

    def _stream_and_play(
        self,
        config: PipelineConfig,
        conversation: Conversation,
        llm: BedrockClient,
        tts: TTS,
        t_asr_done: float,
    ) -> str:
        """Stream Claude's response, pipeline TTS on sentence boundaries.

        Sentence boundaries (``.``, ``?``, ``!``) flush the accumulated buffer
        into a worker thread that handles TTS synthesis and speaker playback in
        parallel with the LLM continuing to stream subsequent sentences.

        The mic is implicitly gated during this method: :meth:`_wait_for_wake`
        is not invoked, so wake-word detection is suspended for the duration of
        agent speech.

        Latency events logged to ``meeting_agent.metrics``:

        * ``bedrock_ttft_s`` — time from end-of-utterance to first LLM delta.
        * ``tts_pipeline_overhead_s`` — time from first LLM delta to first
          TTS audio chunk.
        * ``total_turn_latency_s`` — wake-to-first-speaker-audio latency.

        Args:
            config: Pipeline config (output device, project context, model id).
            conversation: Rolling transcript with ``latest_turn`` already set.
            llm: Bedrock client for streaming the response.
            tts: TTS instance for synthesis.
            t_asr_done: Monotonic timestamp when ASR finished (latency anchor).

        Returns:
            The full agent response as a single concatenated string.
        """
        sentence_q: queue.Queue[str | None] = queue.Queue()
        first_delta_t: list[float] = []
        first_audio_t: list[float] = []

        def _tts_worker() -> None:
            """Consume sentences, synthesise, and play until sentinel arrives."""
            while True:
                sentence = sentence_q.get()
                if sentence is None:
                    return
                for chunk in tts.stream_synthesize(sentence):
                    if not first_audio_t:
                        t = time.monotonic()
                        first_audio_t.append(t)
                        if first_delta_t:
                            _metrics.info(
                                "tts_pipeline_overhead_s=%.3f",
                                t - first_delta_t[0],
                            )
                        _metrics.info("total_turn_latency_s=%.3f", t - t_asr_done)
                    audio.play(
                        chunk,
                        sample_rate=TTS_SAMPLE_RATE,
                        device=config.output_device,
                    )

        worker = threading.Thread(target=_tts_worker, daemon=True)
        worker.start()

        full_response = ""
        buffer = ""
        first_delta_logged = False

        try:
            for delta in llm.respond_stream(config.context, conversation):
                if not first_delta_logged:
                    t = time.monotonic()
                    first_delta_t.append(t)
                    _metrics.info("bedrock_ttft_s=%.3f", t - t_asr_done)
                    first_delta_logged = True

                full_response += delta
                buffer += delta

                # Flush completed sentences into the TTS worker queue.
                complete, buffer = _split_at_sentence_boundaries(buffer)
                for sentence in complete:
                    if sentence.strip():
                        sentence_q.put(sentence)
        finally:
            # Flush any remaining (unpunctuated) tail as the final sentence.
            if buffer.strip():
                sentence_q.put(buffer)
            sentence_q.put(None)  # sentinel — stops the worker
            worker.join()

        return full_response


def _split_at_sentence_boundaries(text: str) -> tuple[list[str], str]:
    """Split *text* into complete sentences and an incomplete tail.

    Splits on ``.``, ``?``, and ``!`` using a capturing regex so each complete
    sentence retains its terminal punctuation. The final segment (which has no
    terminal punctuation) is returned as the tail.

    V1 edge-case policy: over-segmentation (e.g. "Dr.") is acceptable — it
    results in an extra small TTS call rather than a correctness failure.

    Args:
        text: Accumulated text from LLM token deltas.

    Returns:
        A ``(complete_sentences, incomplete_tail)`` 2-tuple where each element
        of ``complete_sentences`` ends with ``.``, ``?``, or ``!``.

    Examples:
        >>> _split_at_sentence_boundaries("Hello. World!")
        (['Hello.', ' World!'], '')
        >>> _split_at_sentence_boundaries("Hello")
        ([], 'Hello')
    """
    parts = re.split(r"([.?!])", text)
    # parts alternates text / delimiter, with a final remainder element:
    # "Hello. World!" → ["Hello", ".", " World", "!", ""]
    sentences: list[str] = []
    i = 0
    while i + 1 < len(parts):
        sentences.append(parts[i] + parts[i + 1])
        i += 2
    remainder = parts[i] if i < len(parts) else ""
    return sentences, remainder
