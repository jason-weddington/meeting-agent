"""Mic capture and speaker playback via sounddevice.

All in-pipeline audio is 16 kHz mono float32. Device IDs are PortAudio indices
from ``sounddevice.query_devices()``; pass ``None`` to use the OS default.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypedDict

import numpy as np
import numpy.typing as npt

AudioArray = npt.NDArray[np.float32]

SAMPLE_RATE: int = 16_000


class DeviceInfo(TypedDict):
    """Subset of sounddevice's device dict that callers care about."""

    index: int
    name: str
    channels: int


def list_input_devices() -> list[DeviceInfo]:
    """Return all input devices with at least one input channel."""
    raise NotImplementedError


def list_output_devices() -> list[DeviceInfo]:
    """Return all output devices with at least one output channel."""
    raise NotImplementedError


def record_chunks(
    device: int | None = None,
    chunk_ms: int = 100,
) -> Iterator[AudioArray]:
    """Yield 16 kHz mono float32 chunks from the given input device.

    Each yielded array contains ``chunk_ms`` of audio. Runs indefinitely until
    the consumer stops iterating. Implementation should use
    ``sounddevice.InputStream`` with a callback that enqueues chunks, and a
    generator that drains the queue.

    Args:
        device: PortAudio device index, or None for the OS default input.
        chunk_ms: Chunk size in milliseconds.

    Yields:
        ``numpy.ndarray`` of shape ``(chunk_ms * SAMPLE_RATE // 1000,)`` dtype float32.
    """
    raise NotImplementedError


def play(audio: AudioArray, sample_rate: int, device: int | None = None) -> None:
    """Play ``audio`` through the given output device and block until finished.

    Args:
        audio: 1-D mono float32 samples at ``sample_rate``.
        sample_rate: Samples per second of the ``audio`` array.
        device: PortAudio device index, or None for the OS default output.
    """
    raise NotImplementedError
