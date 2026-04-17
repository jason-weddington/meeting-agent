"""Tests for meeting_agent.audio — mic + speaker I/O."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from meeting_agent.audio import (
    SAMPLE_RATE,
    list_input_devices,
    list_output_devices,
    play,
    record_chunks,
)

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

FAKE_DEVICES = [
    {"name": "Built-in Mic", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "Built-in Output", "max_input_channels": 0, "max_output_channels": 2},
    {"name": "USB Audio", "max_input_channels": 2, "max_output_channels": 2},
    {"name": "HDMI Out", "max_input_channels": 0, "max_output_channels": 8},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_stream(fake_indata: np.ndarray):
    """Return a FakeInputStream class that calls its callback on __enter__."""

    class FakeInputStream:
        def __init__(self, **kwargs):
            self._callback = kwargs["callback"]

        def __enter__(self):
            # Simulate one PortAudio callback firing before the generator yields
            self._callback(fake_indata, fake_indata.shape[0], None, None)
            return self

        def __exit__(self, *args):
            return False

    return FakeInputStream


# ---------------------------------------------------------------------------
# list_input_devices
# ---------------------------------------------------------------------------


def test_list_input_devices_filters_correctly():
    with patch("sounddevice.query_devices", return_value=FAKE_DEVICES):
        devices = list_input_devices()

    # Only Built-in Mic (idx 0) and USB Audio (idx 2) have input channels
    assert len(devices) == 2
    assert devices[0] == {"index": 0, "name": "Built-in Mic", "channels": 1}
    assert devices[1] == {"index": 2, "name": "USB Audio", "channels": 2}


def test_list_input_devices_excludes_output_only():
    with patch("sounddevice.query_devices", return_value=FAKE_DEVICES):
        devices = list_input_devices()

    names = [d["name"] for d in devices]
    assert "Built-in Output" not in names
    assert "HDMI Out" not in names


def test_list_input_devices_empty():
    with patch("sounddevice.query_devices", return_value=[]):
        devices = list_input_devices()
    assert devices == []


# ---------------------------------------------------------------------------
# list_output_devices
# ---------------------------------------------------------------------------


def test_list_output_devices_filters_correctly():
    with patch("sounddevice.query_devices", return_value=FAKE_DEVICES):
        devices = list_output_devices()

    # Built-in Output (idx 1), USB Audio (idx 2), HDMI Out (idx 3)
    assert len(devices) == 3
    assert devices[0] == {"index": 1, "name": "Built-in Output", "channels": 2}
    assert devices[1] == {"index": 2, "name": "USB Audio", "channels": 2}
    assert devices[2] == {"index": 3, "name": "HDMI Out", "channels": 8}


def test_list_output_devices_excludes_input_only():
    with patch("sounddevice.query_devices", return_value=FAKE_DEVICES):
        devices = list_output_devices()

    names = [d["name"] for d in devices]
    assert "Built-in Mic" not in names


def test_list_output_devices_empty():
    with patch("sounddevice.query_devices", return_value=[]):
        devices = list_output_devices()
    assert devices == []


# ---------------------------------------------------------------------------
# record_chunks
# ---------------------------------------------------------------------------


def test_record_chunks_yields_correct_shape():
    blocksize = int(SAMPLE_RATE * 100 / 1000)  # 1600 samples for 100 ms
    fake_indata = np.ones((blocksize, 1), dtype=np.float32) * 0.5
    FakeInputStream = _make_fake_stream(fake_indata)

    with patch("sounddevice.InputStream", FakeInputStream):
        gen = record_chunks(device=None, chunk_ms=100)
        chunk = next(gen)
        gen.close()

    assert chunk.shape == (blocksize,)
    assert chunk.dtype == np.float32


def test_record_chunks_mono_flatten():
    """Chunks must be 1-D (mono), not 2-D."""
    blocksize = int(SAMPLE_RATE * 50 / 1000)  # 800 samples for 50 ms
    fake_indata = np.ones((blocksize, 1), dtype=np.float32)
    FakeInputStream = _make_fake_stream(fake_indata)

    with patch("sounddevice.InputStream", FakeInputStream):
        gen = record_chunks(device=None, chunk_ms=50)
        chunk = next(gen)
        gen.close()

    assert chunk.ndim == 1
    assert chunk.shape == (blocksize,)


def test_record_chunks_data_values():
    """The yielded values should match indata[:, 0] from the callback."""
    blocksize = int(SAMPLE_RATE * 100 / 1000)
    fake_indata = np.linspace(-1.0, 1.0, blocksize, dtype=np.float32).reshape(-1, 1)
    FakeInputStream = _make_fake_stream(fake_indata)

    with patch("sounddevice.InputStream", FakeInputStream):
        gen = record_chunks(device=None, chunk_ms=100)
        chunk = next(gen)
        gen.close()

    np.testing.assert_array_equal(chunk, fake_indata[:, 0])


def test_record_chunks_passes_correct_kwargs_to_inputstream():
    """InputStream must receive the right samplerate, channels, dtype, device."""
    blocksize = int(SAMPLE_RATE * 100 / 1000)
    fake_indata = np.zeros((blocksize, 1), dtype=np.float32)

    captured = {}

    class CapturingInputStream:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            kwargs["callback"](fake_indata, blocksize, None, None)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with patch("sounddevice.InputStream", CapturingInputStream):
        gen = record_chunks(device=3, chunk_ms=100)
        next(gen)
        gen.close()

    assert captured["device"] == 3
    assert captured["samplerate"] == SAMPLE_RATE
    assert captured["channels"] == 1
    assert captured["dtype"] == "float32"
    assert captured["blocksize"] == blocksize


# ---------------------------------------------------------------------------
# play
# ---------------------------------------------------------------------------


def test_play_calls_sounddevice_play_and_wait():
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    play_calls = []
    wait_calls = []

    def fake_play(arr, sr, device=None):
        play_calls.append((arr, sr, device))

    def fake_wait():
        wait_calls.append(True)

    with (
        patch("sounddevice.play", fake_play),
        patch("sounddevice.wait", fake_wait),
    ):
        play(audio, SAMPLE_RATE, device=None)

    assert len(play_calls) == 1
    assert len(wait_calls) == 1
    _, sr, dev = play_calls[0]
    assert sr == SAMPLE_RATE
    assert dev is None


def test_play_forwards_device_arg():
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    play_calls = []

    def fake_play(arr, sr, device=None):
        play_calls.append(device)

    with (
        patch("sounddevice.play", fake_play),
        patch("sounddevice.wait", lambda: None),
    ):
        play(audio, 24000, device=2)

    assert play_calls[0] == 2


def test_play_forwards_sample_rate():
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    play_calls = []

    def fake_play(arr, sr, device=None):
        play_calls.append(sr)

    with (
        patch("sounddevice.play", fake_play),
        patch("sounddevice.wait", lambda: None),
    ):
        play(audio, 24000, device=None)

    assert play_calls[0] == 24000
