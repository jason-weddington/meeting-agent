"""Tests for the WakeDetector in meeting_agent.wake."""

from unittest.mock import MagicMock, patch

import numpy as np

from meeting_agent.wake import DETECTION_THRESHOLD, WakeDetector


@patch("meeting_agent.wake.Model")
def test_init_passes_wake_phrase_to_model(mock_model_cls):
    """WakeDetector.__init__ passes wakeword_models=[wake_phrase] to Model."""
    WakeDetector("hey_jarvis")
    mock_model_cls.assert_called_once_with(wakeword_models=["hey_jarvis"])


@patch("meeting_agent.wake.Model")
def test_detect_returns_true_when_score_above_threshold(mock_model_cls):
    """detect() returns True when model score exceeds DETECTION_THRESHOLD."""
    mock_instance = MagicMock()
    mock_instance.predict.return_value = {"hey_jarvis": 0.9}
    mock_model_cls.return_value = mock_instance

    detector = WakeDetector("hey_jarvis")
    audio = np.zeros(1600, dtype=np.float32)
    assert detector.detect(audio) is True


@patch("meeting_agent.wake.Model")
def test_detect_returns_false_when_score_below_threshold(mock_model_cls):
    """detect() returns False when model score is below DETECTION_THRESHOLD."""
    mock_instance = MagicMock()
    mock_instance.predict.return_value = {"hey_jarvis": 0.1}
    mock_model_cls.return_value = mock_instance

    detector = WakeDetector("hey_jarvis")
    audio = np.zeros(1600, dtype=np.float32)
    assert detector.detect(audio) is False


@patch("meeting_agent.wake.Model")
def test_detect_at_threshold_is_false(mock_model_cls):
    """detect() returns False when score exactly equals DETECTION_THRESHOLD (strict >)."""
    mock_instance = MagicMock()
    mock_instance.predict.return_value = {"hey_jarvis": DETECTION_THRESHOLD}
    mock_model_cls.return_value = mock_instance

    detector = WakeDetector("hey_jarvis")
    audio = np.zeros(1600, dtype=np.float32)
    assert detector.detect(audio) is False


@patch("meeting_agent.wake.Model")
def test_detect_converts_float32_to_int16(mock_model_cls):
    """detect() converts float32 audio to int16 before passing to the model."""
    mock_instance = MagicMock()
    mock_instance.predict.return_value = {"hey_jarvis": 0.0}
    mock_model_cls.return_value = mock_instance

    detector = WakeDetector("hey_jarvis")
    audio = np.array([0.5, -0.5, 1.0], dtype=np.float32)
    detector.detect(audio)

    call_args = mock_instance.predict.call_args
    passed_audio = call_args[0][0]
    assert passed_audio.dtype == np.int16
    np.testing.assert_array_equal(passed_audio, (audio * 32767).astype(np.int16))


@patch("meeting_agent.wake.Model")
def test_detect_empty_scores_returns_false(mock_model_cls):
    """detect() returns False when model returns no scores (empty dict)."""
    mock_instance = MagicMock()
    mock_instance.predict.return_value = {}
    mock_model_cls.return_value = mock_instance

    detector = WakeDetector("hey_jarvis")
    audio = np.zeros(1600, dtype=np.float32)
    assert detector.detect(audio) is False
