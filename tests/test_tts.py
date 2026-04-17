"""Unit and integration tests for the TTS module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meeting_agent.tts import _MODEL_REPO, DEFAULT_LANG_CODE, DEFAULT_VOICE, TTS


def _make_result(audio: np.ndarray | None) -> MagicMock:
    """Create a mock mlx-audio GenerationResult with an audio attribute."""
    result = MagicMock()
    result.audio = audio
    return result


@patch("meeting_agent.tts.load_model")
def test_synthesize_returns_float32_array(mock_load_model: MagicMock) -> None:
    """synthesize() concatenates chunks and returns a float32 ndarray."""
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    warmup = np.array([0.0], dtype=np.float32)
    chunk1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    chunk2 = np.array([0.4, 0.5], dtype=np.float32)

    mock_model.generate.side_effect = [
        iter([_make_result(warmup)]),  # warmup call in __init__
        iter([_make_result(chunk1), _make_result(chunk2)]),  # synthesize call
    ]

    tts = TTS()
    result = tts.synthesize("Hello world")

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(result) == 5  # 3 + 2 samples
    mock_load_model.assert_called_once_with(_MODEL_REPO)


@patch("meeting_agent.tts.load_model")
def test_synthesize_skips_none_audio(mock_load_model: MagicMock) -> None:
    """synthesize() skips results where audio is None."""
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    warmup = np.array([0.0], dtype=np.float32)
    chunk = np.array([0.1, 0.2], dtype=np.float32)

    mock_model.generate.side_effect = [
        iter([_make_result(warmup)]),
        iter([_make_result(None), _make_result(chunk), _make_result(None)]),
    ]

    tts = TTS()
    result = tts.synthesize("Hello world")

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(result) == 2


@patch("meeting_agent.tts.load_model")
def test_synthesize_no_audio_returns_empty_array(mock_load_model: MagicMock) -> None:
    """synthesize() returns an empty float32 array when no audio chunks are produced."""
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    warmup = np.array([0.0], dtype=np.float32)

    mock_model.generate.side_effect = [
        iter([_make_result(warmup)]),
        iter([_make_result(None)]),
    ]

    tts = TTS()
    result = tts.synthesize("silence")

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(result) == 0


@patch("meeting_agent.tts.load_model")
def test_stream_synthesize_yields_multiple_chunks(mock_load_model: MagicMock) -> None:
    """stream_synthesize() yields multiple float32 chunks as the model produces them."""
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    warmup = np.array([0.0], dtype=np.float32)
    chunk1 = np.array([0.1, 0.2], dtype=np.float32)
    chunk2 = np.array([0.3, 0.4, 0.5], dtype=np.float32)
    chunk3 = np.array([0.6], dtype=np.float32)

    mock_model.generate.side_effect = [
        iter([_make_result(warmup)]),
        iter(
            [
                _make_result(chunk1),
                _make_result(None),  # skipped
                _make_result(chunk2),
                _make_result(chunk3),
            ]
        ),
    ]

    tts = TTS()
    chunks = list(tts.stream_synthesize("Hello world"))

    assert len(chunks) == 3
    for chunk in chunks:
        assert isinstance(chunk, np.ndarray)
        assert chunk.dtype == np.float32
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 3
    assert len(chunks[2]) == 1


@patch("meeting_agent.tts.load_model")
def test_stream_synthesize_skips_empty_chunks(mock_load_model: MagicMock) -> None:
    """stream_synthesize() skips audio chunks with zero samples."""
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    warmup = np.array([0.0], dtype=np.float32)
    empty = np.array([], dtype=np.float32)
    chunk = np.array([0.1, 0.2], dtype=np.float32)

    mock_model.generate.side_effect = [
        iter([_make_result(warmup)]),
        iter([_make_result(empty), _make_result(chunk)]),
    ]

    tts = TTS()
    chunks = list(tts.stream_synthesize("Hello world"))

    assert len(chunks) == 1
    assert chunks[0].dtype == np.float32


@patch("meeting_agent.tts.load_model")
def test_tts_stores_voice_and_lang_code(mock_load_model: MagicMock) -> None:
    """TTS.__init__ stores voice and lang_code as instance attributes."""
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    mock_model.generate.return_value = iter([])  # warmup yields nothing

    tts = TTS(voice="test_voice", lang_code="b")

    assert tts.voice == "test_voice"
    assert tts.lang_code == "b"
    mock_load_model.assert_called_once_with(_MODEL_REPO)


@patch("meeting_agent.tts.load_model")
def test_tts_uses_defaults(mock_load_model: MagicMock) -> None:
    """TTS defaults to DEFAULT_VOICE and DEFAULT_LANG_CODE."""
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    mock_model.generate.return_value = iter([])

    tts = TTS()

    assert tts.voice == DEFAULT_VOICE
    assert tts.lang_code == DEFAULT_LANG_CODE


@patch("meeting_agent.tts.load_model")
def test_generate_called_with_correct_args(mock_load_model: MagicMock) -> None:
    """synthesize() calls model.generate with expected keyword arguments."""
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    warmup = np.array([0.0], dtype=np.float32)
    chunk = np.array([0.1, 0.2], dtype=np.float32)

    mock_model.generate.side_effect = [
        iter([_make_result(warmup)]),
        iter([_make_result(chunk)]),
    ]

    tts = TTS(voice="af_heart", lang_code="a")
    tts.synthesize("Test sentence.")

    # Verify the synthesis call uses the right kwargs
    _, synth_call = mock_model.generate.call_args_list
    assert synth_call.kwargs["text"] == "Test sentence."
    assert synth_call.kwargs["voice"] == "af_heart"
    assert synth_call.kwargs["lang_code"] == "a"
    assert synth_call.kwargs["speed"] == 1.0


@pytest.mark.integration
def test_synthesize_integration() -> None:
    """Real end-to-end synthesis with mlx-audio Kokoro. Requires model download (~330MB)."""
    tts = TTS()
    audio = tts.synthesize("Hello, world.")

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) > 0


@pytest.mark.integration
def test_stream_synthesize_integration() -> None:
    """Real streaming synthesis with mlx-audio Kokoro. Requires model download on first run."""
    tts = TTS()
    chunks = list(tts.stream_synthesize("Hello, world."))

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, np.ndarray)
        assert chunk.dtype == np.float32
