"""End-to-end V2 classifier-gated pipeline integration test.

Exercises the complete V2 always-on pipeline with real TTS + ASR and mocked
Bedrock / Classifier — designed to catch interface drift between modules that
unit tests miss.

Real components:
    - ``TTS`` (Kokoro via mlx-audio) synthesises the input utterances.
    - ``StreamingASR`` (mlx-whisper + Silero VAD) transcribes them back.
    - ``Pipeline`` orchestration (circuit breaker, airtime, staleness gate,
      deafness probe, exception log).

Mocked components:
    - ``BedrockClient.respond_stream`` — yields a canned text response.
    - ``Classifier.classify`` — returns canned ``Decision`` objects.
    - ``audio.record_chunks`` — replays TTS-synthesised audio as 16 kHz chunks.
    - ``audio.play`` — captures audio arrays rather than hitting speakers.

Scenario
--------
Two utterances are synthesised via real TTS and fed through real ASR:

* **utterance_a** — "Hey Jarvis, what's the weather?"  (addressed → full_answer)
* **utterance_b** — "Just a side comment to the team."  (non-addressed → silent)

A block of pure silence is also injected between the two utterances to probe
whether VAD + Whisper produce a low-confidence utterance.  In practice silence
never makes it past the ASR silence-peak guard (``_SILENCE_PEAK_THRESHOLD``),
so no utterance is yielded and the low-confidence gate is not exercised here.
That is acceptable — the assertion on ``classify call_count >= 2`` still holds
for the two real utterances, and a note below explains the limitation.

Marking
-------
``@pytest.mark.integration`` — skipped by default; opt-in on Apple Silicon::

    uv run pytest -m integration tests/integration/test_v2_pipeline.py -v

Runtime
-------
* First run: downloads mlx-whisper large-v3 (~3 GB) and Kokoro (~330 MB).
* Subsequent runs: ~10–20 s on M-series hardware.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.signal

from meeting_agent.asr import Utterance
from meeting_agent.audio import AudioArray
from meeting_agent.classifier import Decision
from meeting_agent.llm import Conversation, ProjectContext
from meeting_agent.pipeline import Pipeline, PipelineConfig
from meeting_agent.tts import SAMPLE_RATE as TTS_SAMPLE_RATE
from meeting_agent.tts import TTS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHUNK_SAMPLES: int = 1600  # 100 ms × 16 000 Hz = 1 600 samples/chunk

# Number of consecutive silent chunks required for ASR end-of-speech timeout
# (mirrors StreamingASR._EOS_CHUNK_COUNT = 5); add 2 extra for headroom.
_EOS_CHUNKS: int = 7

# Echo-drain buffer between utterances.  After the agent responds to
# utterance_a, Pipeline._drain_echo() pulls (speak_duration + 0.5 s) chunks
# directly from the raw iterator.  TTS synthesis of the short canned response
# takes ≤ 2 s on M-series → drain ≤ 25 chunks.  We pad to 40 for safety.
_GAP_CHUNKS: int = 40

# Pure-silence block injected to try triggering the pre-classifier confidence
# gate.  In practice np.zeros never exceeds _SILENCE_PEAK_THRESHOLD (0.01), so
# the ASR skips transcription entirely — no utterance is produced and the gate
# is not exercised.  This is noted in the assertion comment below.
_SILENCE_BLOCK_CHUNKS: int = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _downsample_24k_to_16k(audio_24k: AudioArray) -> AudioArray:
    """Downsample 24 kHz Kokoro output to 16 kHz pipeline-input format.

    Uses ``scipy.signal.resample_poly`` (poly-phase FIR) for clean 3:2
    integer-ratio downsampling (24 000 → 16 000 samples/s).
    """
    resampled = scipy.signal.resample_poly(audio_24k, up=2, down=3)
    return resampled.astype(np.float32)


def _split_into_chunks(audio: AudioArray, chunk_size: int = _CHUNK_SAMPLES) -> list[AudioArray]:
    """Split *audio* into fixed-size chunks, zero-padding the last one."""
    chunks: list[AudioArray] = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk))).astype(np.float32)
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Module-scoped fixtures (model loads once per test-module run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tts_instance() -> TTS:
    """Load and warm the Kokoro TTS model once for the entire module."""
    return TTS()


@pytest.fixture(scope="module")
def utterance_a_16k(tts_instance: TTS) -> AudioArray:
    """Synthesise the addressed utterance and downsample to 16 kHz."""
    audio_24k = tts_instance.synthesize("Hey Jarvis, what's the weather?")
    assert audio_24k.dtype == np.float32
    assert len(audio_24k) > 0
    return _downsample_24k_to_16k(audio_24k)


@pytest.fixture(scope="module")
def utterance_b_16k(tts_instance: TTS) -> AudioArray:
    """Synthesise the non-addressed utterance and downsample to 16 kHz."""
    audio_24k = tts_instance.synthesize("Just a side comment to the team.")
    assert audio_24k.dtype == np.float32
    assert len(audio_24k) > 0
    return _downsample_24k_to_16k(audio_24k)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_v2_classifier_gated_pipeline(
    tts_instance: TTS,
    utterance_a_16k: AudioArray,
    utterance_b_16k: AudioArray,
) -> None:
    """End-to-end V2 pipeline round-trip with real ASR + TTS, mocked Bedrock.

    Scenario
    --------
    1. Real TTS synthesises two utterances at 24 kHz → downsampled to 16 kHz.
    2. A finite ``record_chunks`` mock replays them as 100 ms chunks.
    3. Real ``StreamingASR`` (VAD + Whisper) transcribes the chunks.
    4. Mocked ``Classifier`` returns canned decisions.
    5. Mocked ``BedrockClient`` yields a canned response for the full_answer.
    6. Real TTS synthesises the agent response; mocked ``audio.play`` captures it.
    7. The finite iterator exhausts → ``Pipeline.run()`` returns naturally.

    Assertions
    ----------
    * Classifier called at least twice (once per real utterance).
    * ``BedrockClient.respond_stream`` called exactly once.
    * ``audio.play`` called ≥ 1 time with 24 kHz mono float32 audio.
    * ``conversation.older_turns`` has entries for Jason, agent, and Aziz.
    * First transcribed utterance text contains "weather".
    * Thread finishes within 60 s.

    Note on the silence block
    -------------------------
    The injected ``np.zeros`` silence block does NOT reliably produce an ASR
    utterance because pure silence is filtered by the ``_SILENCE_PEAK_THRESHOLD``
    guard before Whisper is invoked.  The deafness-probe low-confidence path
    therefore cannot be exercised with synthetic silence; it is already covered
    deterministically by V2.3 unit tests.
    """
    # ------------------------------------------------------------------
    # Build the finite chunk stream
    # ------------------------------------------------------------------
    silence_chunk = np.zeros(_CHUNK_SAMPLES, dtype=np.float32)
    a_chunks = _split_into_chunks(utterance_a_16k)
    b_chunks = _split_into_chunks(utterance_b_16k)

    def mock_record_chunks(
        device: int | None = None,
        chunk_ms: int = 100,  # noqa: ARG001
    ) -> Iterator[AudioArray]:
        """Finite chunk stream: lead-in → utterance_a → silence → utterance_b."""
        # Lead-in: let VAD initialise before speech starts.
        for _ in range(5):
            yield silence_chunk.copy()

        # Utterance A (addressed) — real Kokoro audio at 16 kHz.
        yield from a_chunks

        # End-of-speech silence for utterance_a (VAD EOS timeout = 5 chunks).
        for _ in range(_EOS_CHUNKS):
            yield silence_chunk.copy()

        # Near-silence block: probes the confidence gate.  Pure zeros never
        # exceed _SILENCE_PEAK_THRESHOLD (0.01), so no utterance is produced.
        for _ in range(_SILENCE_BLOCK_CHUNKS):
            yield silence_chunk.copy()

        # Echo-drain buffer: absorbs the chunks consumed by _drain_echo() after
        # the agent speaks, so utterance_b audio is not eaten by the drain.
        for _ in range(_GAP_CHUNKS):
            yield silence_chunk.copy()

        # Utterance B (non-addressed) — real Kokoro audio at 16 kHz.
        yield from b_chunks

        # End-of-speech silence for utterance_b.
        for _ in range(_EOS_CHUNKS):
            yield silence_chunk.copy()

        # Trailing silence: ASR drains these to close the final EOS window
        # before the generator exhausts and Pipeline.run() returns naturally.
        for _ in range(20):
            yield silence_chunk.copy()

    # ------------------------------------------------------------------
    # State captured by mocks
    # ------------------------------------------------------------------
    play_calls: list[tuple[np.ndarray, int]] = []
    classify_utterance_texts: list[str] = []
    captured_conversation: list[Conversation] = []
    pipeline_errors: list[Exception] = []

    # Canned classifier decisions.  Any call beyond the 2nd (e.g. if silence
    # somehow produces an utterance) returns the last decision (silent — benign).
    _decisions = [
        Decision("Jason", "full_answer", 0.9),
        Decision("Aziz", "silent", 0.8),
    ]
    _classify_idx = 0

    def mock_classify(
        utterance: Utterance,
        confidence: object,
        context: object,
        session: object,
    ) -> Decision:
        nonlocal _classify_idx
        classify_utterance_texts.append(utterance.text)
        decision = _decisions[min(_classify_idx, len(_decisions) - 1)]
        _classify_idx += 1
        return decision

    def mock_respond_stream(
        context: ProjectContext,
        conversation: Conversation,
    ) -> Iterator[str]:
        """Capture the conversation reference; yield a canned response."""
        # Short response → fast TTS synthesis → small echo drain.
        captured_conversation.append(conversation)
        yield from ["The weather is sunny!", ""]

    def mock_play(
        audio_arr: AudioArray,
        sample_rate: int,
        device: int | None = None,  # noqa: ARG001
    ) -> None:
        """Capture play() calls instead of emitting audio to speakers."""
        play_calls.append((audio_arr.copy(), sample_rate))

    # ------------------------------------------------------------------
    # Wire mocks and run the pipeline in a daemon thread
    # ------------------------------------------------------------------
    config = PipelineConfig(
        context=ProjectContext(system_prompt="You are a helpful meeting assistant.")
    )
    pipeline = Pipeline(config)

    def run_pipeline() -> None:
        try:
            with (
                patch("meeting_agent.audio.record_chunks", side_effect=mock_record_chunks),
                patch("meeting_agent.audio.play", side_effect=mock_play),
                # Pass the pre-warmed TTS instance so Pipeline.run() doesn't
                # cold-load a second Kokoro model.
                patch("meeting_agent.pipeline.TTS", return_value=tts_instance),
                patch("meeting_agent.pipeline.BedrockClient") as mock_llm_cls,
                patch("meeting_agent.pipeline.Classifier") as mock_classifier_cls,
            ):
                mock_llm_instance = MagicMock()
                mock_llm_instance.respond_stream.side_effect = mock_respond_stream
                mock_llm_cls.return_value = mock_llm_instance

                mock_classifier_instance = MagicMock()
                mock_classifier_instance.classify.side_effect = mock_classify
                mock_classifier_cls.return_value = mock_classifier_instance

                # StreamingASR is NOT mocked — the real VAD + Whisper run here.
                pipeline.run()
        except Exception as exc:  # noqa: BLE001
            pipeline_errors.append(exc)

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()
    # 60 s budget: subsequent runs finish in ~20 s with cached models;
    # first run may need extra time for mlx-whisper weight download (~3 GB).
    thread.join(timeout=60)

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    assert not thread.is_alive(), (
        "Pipeline thread did not finish within the 60 s timeout — "
        "verify that mlx-whisper and Kokoro models are cached on this machine."
    )
    assert not pipeline_errors, f"Pipeline raised an unexpected error: {pipeline_errors[0]!r}"

    # 1. Classifier called ≥ 2 times (once per real utterance; 3rd call
    #    possible if Whisper produces an utterance from the silence block).
    assert _classify_idx >= 2, (
        f"Classifier.classify called {_classify_idx} time(s); expected >= 2. "
        "Both synthesised utterances should have passed ASR and the confidence gate."
    )

    # 2. BedrockClient.respond_stream called exactly once (Jason's full_answer).
    assert len(captured_conversation) == 1, (
        f"Expected exactly 1 Bedrock call, got {len(captured_conversation)}. "
        "Only Jason's full_answer decision should trigger a response."
    )

    # 3. audio.play called ≥ 1 time with 24 kHz mono float32.
    assert len(play_calls) >= 1, (
        "audio.play was never called — TTS synthesis or pipeline routing may be broken."
    )
    for audio_arr, sample_rate in play_calls:
        assert isinstance(audio_arr, np.ndarray), (
            f"audio.play received non-ndarray argument: {type(audio_arr)}"
        )
        assert audio_arr.dtype == np.float32, (
            f"audio.play expected float32 audio, got dtype={audio_arr.dtype}"
        )
        assert sample_rate == TTS_SAMPLE_RATE, (
            f"audio.play expected {TTS_SAMPLE_RATE} Hz (24 kHz Kokoro output), got {sample_rate} Hz"
        )

    # 4. Conversation transcript has Jason, agent, and Aziz entries.
    conv = captured_conversation[0]
    speakers = {t.speaker for t in conv.older_turns}
    assert "Jason" in speakers, (
        f"Expected a 'Jason' turn in older_turns; found speakers: {speakers!r}"
    )
    assert "agent" in speakers, (
        f"Expected an 'agent' turn in older_turns; found speakers: {speakers!r}"
    )
    assert "Aziz" in speakers, (
        f"Expected an 'Aziz' turn in older_turns; found speakers: {speakers!r}. "
        "utterance_b may have been consumed by the echo drain — increase _GAP_CHUNKS."
    )

    # 5. First transcribed utterance text contains "weather".
    #    Whisper transcribes TTS-synthesised speech accurately; exact wording
    #    may vary slightly (e.g. capitalisation, leading spaces).
    assert classify_utterance_texts, (
        "Classifier was never called — no utterances reached the classify step."
    )
    assert "weather" in classify_utterance_texts[0].lower(), (
        f"Expected 'weather' in first transcription, got: {classify_utterance_texts[0]!r}. "
        "Whisper may have produced an unexpected transcript for the TTS audio."
    )
