"""Unit tests for the streaming ASR module.

Mocks out mlx_whisper and silero_vad so no model downloads occur.
The conftest.py injects MagicMock stubs for both packages into sys.modules
before any test imports meeting_agent.asr.

Test chunks are sized to 512 samples (one Silero VAD window at 16 kHz) so
that each chunk produces exactly one VAD call, keeping side_effect lists
simple and timing assertions easy to reason about.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meeting_agent.asr import (
    _EOS_CHUNK_COUNT,
    _SILENCE_PEAK_THRESHOLD,
    _VAD_WINDOW,
    DEFAULT_MODEL_REPO,
    StreamingASR,
    Utterance,
)
from meeting_agent.audio import SAMPLE_RATE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Use exactly one VAD window per chunk so that every chunk produces a single
# VAD call, making side_effect lists and timing arithmetic straightforward.
CHUNK_SAMPLES = _VAD_WINDOW  # 512 samples = 32 ms at 16 kHz
CHUNK_S = CHUNK_SAMPLES / SAMPLE_RATE  # ≈ 0.032 s


def _silent_chunk() -> np.ndarray:
    return np.zeros(CHUNK_SAMPLES, dtype=np.float32)


def _speech_chunk() -> np.ndarray:
    # Non-zero content — doesn't matter for mocked VAD.
    return np.ones(CHUNK_SAMPLES, dtype=np.float32) * 0.1


def _make_chunk_iter(
    silent_before: int,
    speech: int,
    silent_after: int,
) -> Iterator[np.ndarray]:
    """Yield *silent_before* silent chunks, then *speech* speech chunks, then
    *silent_after* silent chunks."""
    for _ in range(silent_before):
        yield _silent_chunk()
    for _ in range(speech):
        yield _speech_chunk()
    for _ in range(silent_after):
        yield _silent_chunk()


def _vad_probs(
    silent_before: int,
    speech: int,
    silent_after: int,
) -> list[float]:
    """Return per-chunk VAD probabilities matching ``_make_chunk_iter``."""
    return [0.0] * silent_before + [1.0] * speech + [0.0] * silent_after


# ---------------------------------------------------------------------------
# Dataclass smoke tests
# ---------------------------------------------------------------------------


def test_utterance_dataclass() -> None:
    """Utterance is a frozen dataclass with expected fields."""
    u = Utterance(text="hello", start_s=0.5, end_s=2.0)
    assert u.text == "hello"
    assert u.start_s == 0.5
    assert u.end_s == 2.0

    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        u.text = "bye"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


def test_init_stores_args() -> None:
    """__init__ stores model_repo and initial_prompt."""
    asr = StreamingASR(model_repo="custom/repo", initial_prompt="ACME Corp")
    assert asr.model_repo == "custom/repo"
    assert asr.initial_prompt == "ACME Corp"


def test_init_defaults() -> None:
    """__init__ uses DEFAULT_MODEL_REPO and None initial_prompt by default."""
    asr = StreamingASR()
    assert asr.model_repo == DEFAULT_MODEL_REPO
    assert asr.initial_prompt is None


def test_lazy_load_no_model_calls_at_init() -> None:
    """__init__ must NOT call mlx_whisper.transcribe or load_silero_vad."""
    with (
        patch("mlx_whisper.transcribe") as mock_transcribe,
        patch("silero_vad.load_silero_vad") as mock_load_vad,
    ):
        StreamingASR()
        mock_transcribe.assert_not_called()
        mock_load_vad.assert_not_called()


# ---------------------------------------------------------------------------
# transcribe_stream behaviour
# ---------------------------------------------------------------------------


@patch("silero_vad.load_silero_vad")
@patch("mlx_whisper.transcribe")
def test_silence_only_yields_nothing(
    mock_transcribe: MagicMock,
    mock_load_vad: MagicMock,
) -> None:
    """Five silent chunks with no speech → no Utterance yielded."""
    mock_vad = MagicMock(side_effect=[0.0] * 5)
    mock_load_vad.return_value = mock_vad

    asr = StreamingASR()
    results = list(asr.transcribe_stream(_make_chunk_iter(5, 0, 0)))

    assert results == []
    mock_transcribe.assert_not_called()


@patch("silero_vad.load_silero_vad")
@patch("mlx_whisper.transcribe")
def test_speech_without_trailing_silence_yields_nothing(
    mock_transcribe: MagicMock,
    mock_load_vad: MagicMock,
) -> None:
    """Ongoing speech with no trailing silence → segment not closed → nothing yielded."""
    probs = _vad_probs(5, 10, 0)
    mock_vad = MagicMock(side_effect=probs)
    mock_load_vad.return_value = mock_vad

    asr = StreamingASR()
    results = list(asr.transcribe_stream(_make_chunk_iter(5, 10, 0)))

    assert results == []
    mock_transcribe.assert_not_called()


@patch("silero_vad.load_silero_vad")
@patch("mlx_whisper.transcribe")
def test_fewer_than_eos_silent_chunks_yields_nothing(
    mock_transcribe: MagicMock,
    mock_load_vad: MagicMock,
) -> None:
    """Fewer than _EOS_CHUNK_COUNT trailing silent chunks → segment not closed."""
    probs = _vad_probs(5, 10, _EOS_CHUNK_COUNT - 1)
    mock_vad = MagicMock(side_effect=probs)
    mock_load_vad.return_value = mock_vad

    asr = StreamingASR()
    results = list(asr.transcribe_stream(_make_chunk_iter(5, 10, _EOS_CHUNK_COUNT - 1)))

    assert results == []
    mock_transcribe.assert_not_called()


@patch("silero_vad.load_silero_vad")
@patch("mlx_whisper.transcribe")
def test_speech_then_silence_yields_utterance(
    mock_transcribe: MagicMock,
    mock_load_vad: MagicMock,
) -> None:
    """5 silent → 10 speech → 5 silent yields exactly one Utterance."""
    probs = _vad_probs(5, 10, 5)
    mock_vad = MagicMock(side_effect=probs)
    mock_load_vad.return_value = mock_vad
    mock_transcribe.return_value = {"text": "hello world"}

    asr = StreamingASR()
    results = list(asr.transcribe_stream(_make_chunk_iter(5, 10, 5)))

    assert len(results) == 1
    assert isinstance(results[0], Utterance)
    assert results[0].text == "hello world"


@patch("silero_vad.load_silero_vad")
@patch("mlx_whisper.transcribe")
def test_timing_start_and_end(
    mock_transcribe: MagicMock,
    mock_load_vad: MagicMock,
) -> None:
    """start_s and end_s match expected offsets from chunk count × CHUNK_S.

    Layout: 5 silent | 10 speech | 5 silent  (20 chunks total)

    start_s: elapsed after the 5 leading silent chunks = 5 × CHUNK_S
    end_s:   elapsed at start of the 20th chunk + CHUNK_S = 20 × CHUNK_S
    """
    probs = _vad_probs(5, 10, 5)
    mock_vad = MagicMock(side_effect=probs)
    mock_load_vad.return_value = mock_vad
    mock_transcribe.return_value = {"text": "timed"}

    asr = StreamingASR()
    results = list(asr.transcribe_stream(_make_chunk_iter(5, 10, 5)))

    assert len(results) == 1
    expected_start_s = 5 * CHUNK_S
    expected_end_s = 20 * CHUNK_S
    assert results[0].start_s == pytest.approx(expected_start_s)
    assert results[0].end_s == pytest.approx(expected_end_s)


@patch("silero_vad.load_silero_vad")
@patch("mlx_whisper.transcribe")
def test_initial_prompt_forwarded(
    mock_transcribe: MagicMock,
    mock_load_vad: MagicMock,
) -> None:
    """initial_prompt is passed through to mlx_whisper.transcribe."""
    probs = _vad_probs(5, 10, 5)
    mock_vad = MagicMock(side_effect=probs)
    mock_load_vad.return_value = mock_vad
    mock_transcribe.return_value = {"text": "aws rekognition"}

    asr = StreamingASR(initial_prompt="AWS Rekognition Bedrock SageMaker")
    list(asr.transcribe_stream(_make_chunk_iter(5, 10, 5)))

    assert mock_transcribe.call_count == 1
    _, kwargs = mock_transcribe.call_args
    assert kwargs.get("initial_prompt") == "AWS Rekognition Bedrock SageMaker"


@patch("silero_vad.load_silero_vad")
@patch("mlx_whisper.transcribe")
def test_no_initial_prompt_passes_none(
    mock_transcribe: MagicMock,
    mock_load_vad: MagicMock,
) -> None:
    """When no initial_prompt is set, None is passed to mlx_whisper.transcribe."""
    probs = _vad_probs(5, 10, 5)
    mock_vad = MagicMock(side_effect=probs)
    mock_load_vad.return_value = mock_vad
    mock_transcribe.return_value = {"text": "hello"}

    asr = StreamingASR()
    list(asr.transcribe_stream(_make_chunk_iter(5, 10, 5)))

    _, kwargs = mock_transcribe.call_args
    assert kwargs.get("initial_prompt") is None


@patch("silero_vad.load_silero_vad")
@patch("mlx_whisper.transcribe")
def test_model_repo_forwarded(
    mock_transcribe: MagicMock,
    mock_load_vad: MagicMock,
) -> None:
    """model_repo is passed as path_or_hf_repo to mlx_whisper.transcribe."""
    probs = _vad_probs(5, 10, 5)
    mock_vad = MagicMock(side_effect=probs)
    mock_load_vad.return_value = mock_vad
    mock_transcribe.return_value = {"text": "ok"}

    asr = StreamingASR(model_repo="custom/whisper-repo")
    list(asr.transcribe_stream(_make_chunk_iter(5, 10, 5)))

    _, kwargs = mock_transcribe.call_args
    assert kwargs.get("path_or_hf_repo") == "custom/whisper-repo"


@patch("silero_vad.load_silero_vad")
@patch("mlx_whisper.transcribe")
def test_vad_loaded_once_per_stream(
    mock_transcribe: MagicMock,
    mock_load_vad: MagicMock,
) -> None:
    """load_silero_vad() is called exactly once per transcribe_stream call."""
    probs = _vad_probs(5, 10, 5)
    mock_vad = MagicMock(side_effect=probs)
    mock_load_vad.return_value = mock_vad
    mock_transcribe.return_value = {"text": ""}

    asr = StreamingASR()
    list(asr.transcribe_stream(_make_chunk_iter(5, 10, 5)))

    mock_load_vad.assert_called_once()


@patch("silero_vad.load_silero_vad")
@patch("mlx_whisper.transcribe")
def test_vad_called_with_each_chunk(
    mock_transcribe: MagicMock,
    mock_load_vad: MagicMock,
) -> None:
    """VAD model is called once for every 512-sample chunk (1 window/chunk)."""
    n_chunks = 7
    probs = [0.0] * n_chunks  # all silent
    mock_vad = MagicMock(side_effect=probs)
    mock_load_vad.return_value = mock_vad

    asr = StreamingASR()
    list(asr.transcribe_stream(_make_chunk_iter(n_chunks, 0, 0)))

    assert mock_vad.call_count == n_chunks


@patch("silero_vad.load_silero_vad")
@patch("mlx_whisper.transcribe")
def test_two_speech_segments_two_utterances(
    mock_transcribe: MagicMock,
    mock_load_vad: MagicMock,
) -> None:
    """Two speech segments separated by enough silence produce two Utterances."""
    # Layout: 2 silent | 5 speech | 5 silent | 5 speech | 5 silent
    probs = [0.0] * 2 + [1.0] * 5 + [0.0] * 5 + [1.0] * 5 + [0.0] * 5
    mock_vad = MagicMock(side_effect=probs)
    mock_load_vad.return_value = mock_vad
    mock_transcribe.side_effect = [{"text": "first"}, {"text": "second"}]

    def _chunks() -> Iterator[np.ndarray]:
        for _ in range(2):
            yield _silent_chunk()
        for _ in range(5):
            yield _speech_chunk()
        for _ in range(5):
            yield _silent_chunk()
        for _ in range(5):
            yield _speech_chunk()
        for _ in range(5):
            yield _silent_chunk()

    asr = StreamingASR()
    results = list(asr.transcribe_stream(_chunks()))

    assert len(results) == 2
    assert results[0].text == "first"
    assert results[1].text == "second"
    # Second segment must start no earlier than the first one ends.
    assert results[1].start_s >= results[0].end_s


# ---------------------------------------------------------------------------
# V1.5: Silence peak gate
# ---------------------------------------------------------------------------


@patch("silero_vad.load_silero_vad")
@patch("mlx_whisper.transcribe")
def test_silence_peak_gate_skips_transcription(
    mock_transcribe: MagicMock,
    mock_load_vad: MagicMock,
) -> None:
    """VAD triggers but audio peak is below _SILENCE_PEAK_THRESHOLD → transcribe NOT called."""
    # VAD reports speech for 10 chunks, then silence for 5 (enough to close EOS).
    probs = _vad_probs(0, 10, 5)
    mock_vad = MagicMock(side_effect=probs)
    mock_load_vad.return_value = mock_vad

    # Amplitude well below _SILENCE_PEAK_THRESHOLD (0.01) — near-silent buffer.
    near_silent_amplitude = _SILENCE_PEAK_THRESHOLD * 0.5

    def _near_silent_chunk() -> np.ndarray:
        return np.ones(CHUNK_SAMPLES, dtype=np.float32) * near_silent_amplitude

    def _chunks() -> Iterator[np.ndarray]:
        for _ in range(10):
            yield _near_silent_chunk()
        for _ in range(5):
            yield _silent_chunk()

    asr = StreamingASR()
    results = list(asr.transcribe_stream(_chunks()))

    assert results == []
    mock_transcribe.assert_not_called()


@patch("silero_vad.load_silero_vad")
@patch("mlx_whisper.transcribe")
def test_peak_gate_allows_real_speech(
    mock_transcribe: MagicMock,
    mock_load_vad: MagicMock,
) -> None:
    """When peak is above _SILENCE_PEAK_THRESHOLD, transcribe is called and utterance yielded."""
    probs = _vad_probs(0, 10, 5)
    mock_vad = MagicMock(side_effect=probs)
    mock_load_vad.return_value = mock_vad
    mock_transcribe.return_value = {"text": "real speech here"}

    # _speech_chunk() amplitude is 0.1, well above _SILENCE_PEAK_THRESHOLD (0.01).
    asr = StreamingASR()
    results = list(asr.transcribe_stream(_make_chunk_iter(0, 10, 5)))

    assert len(results) == 1
    assert results[0].text == "real speech here"
    mock_transcribe.assert_called_once()


# ---------------------------------------------------------------------------
# V1.5: Hallucination blocklist
# ---------------------------------------------------------------------------


@patch("silero_vad.load_silero_vad")
@patch("mlx_whisper.transcribe")
def test_hallucination_blocklist_drops_transcript(
    mock_transcribe: MagicMock,
    mock_load_vad: MagicMock,
) -> None:
    """Whisper hallucinations listed in _WHISPER_HALLUCINATIONS are not yielded."""
    # Test a representative sample: empty string, "thank you", and a variant
    # that requires normalization ("Thanks for watching." → "thanks for watching").
    hallucination_cases = (
        "",
        "thank you",
        "Thanks for watching.",
        "SUBTITLES BY THE AMARA.ORG COMMUNITY",
    )
    for text in hallucination_cases:
        probs = _vad_probs(0, 10, 5)
        mock_vad = MagicMock(side_effect=probs)
        mock_load_vad.return_value = mock_vad
        mock_transcribe.return_value = {"text": text}

        asr = StreamingASR()
        results = list(asr.transcribe_stream(_make_chunk_iter(0, 10, 5)))

        assert results == [], f"Expected no utterance for hallucination text: {text!r}"


@patch("silero_vad.load_silero_vad")
@patch("mlx_whisper.transcribe")
def test_hallucination_blocklist_case_insensitive(
    mock_transcribe: MagicMock,
    mock_load_vad: MagicMock,
) -> None:
    """Hallucination detection is case-insensitive and ignores trailing punctuation."""
    probs = _vad_probs(0, 10, 5)
    mock_vad = MagicMock(side_effect=probs)
    mock_load_vad.return_value = mock_vad
    # "THANK YOU." → strip → "THANK YOU." → lower → "thank you." → rstrip → "thank you"
    mock_transcribe.return_value = {"text": "THANK YOU."}

    asr = StreamingASR()
    results = list(asr.transcribe_stream(_make_chunk_iter(0, 10, 5)))

    assert results == []


@patch("silero_vad.load_silero_vad")
@patch("mlx_whisper.transcribe")
def test_non_hallucination_transcript_is_yielded(
    mock_transcribe: MagicMock,
    mock_load_vad: MagicMock,
) -> None:
    """Legitimate transcripts that are not in the blocklist are yielded normally."""
    probs = _vad_probs(0, 10, 5)
    mock_vad = MagicMock(side_effect=probs)
    mock_load_vad.return_value = mock_vad
    mock_transcribe.return_value = {"text": "what is the project status"}

    asr = StreamingASR()
    results = list(asr.transcribe_stream(_make_chunk_iter(0, 10, 5)))

    assert len(results) == 1
    assert results[0].text == "what is the project status"


# ---------------------------------------------------------------------------
# V1.5: VAD threshold constant guard
# ---------------------------------------------------------------------------


def test_vad_threshold_constant_exists() -> None:
    """_VAD_SPEECH_THRESHOLD must be >= 0.6 (regression guard against accidental downgrades)."""
    from meeting_agent.asr import _VAD_SPEECH_THRESHOLD

    assert _VAD_SPEECH_THRESHOLD >= 0.6


# ---------------------------------------------------------------------------
# Integration test (skipped unless -m integration)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_transcribe_stream_integration() -> None:
    """Real clip through the full VAD → Whisper pipeline.

    Requires silero-vad and mlx-whisper to be installed and ~3 GB weights
    available locally (downloaded automatically on first run).
    """
    # Sine wave at 440 Hz approximates a tonal speech-like signal for VAD.
    duration_s = 2.0
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), dtype=np.float32)
    sine_wave = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    n_chunks = len(sine_wave) // _VAD_WINDOW

    def _chunks() -> Iterator[np.ndarray]:
        for i in range(n_chunks):
            yield sine_wave[i * _VAD_WINDOW : (i + 1) * _VAD_WINDOW]

    asr = StreamingASR()
    results = list(asr.transcribe_stream(_chunks()))
    for u in results:
        assert isinstance(u, Utterance)
        assert isinstance(u.text, str)
        assert u.end_s > u.start_s
