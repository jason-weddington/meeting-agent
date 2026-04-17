"""Mic capture and speaker playback via sounddevice.

All in-pipeline audio is 16 kHz mono float32. Device IDs are PortAudio indices
from ``sounddevice.query_devices()``; pass ``None`` to use the OS default.
"""

from __future__ import annotations

import queue
from collections.abc import Iterator
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import sounddevice

AudioArray = npt.NDArray[np.float32]

SAMPLE_RATE: int = 16_000


class DeviceInfo(TypedDict):
    """Subset of sounddevice's device dict that callers care about."""

    index: int
    name: str
    channels: int


def list_input_devices() -> list[DeviceInfo]:
    """Return all input devices with at least one input channel."""
    devices = sounddevice.query_devices()
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


def list_output_devices() -> list[DeviceInfo]:
    """Return all output devices with at least one output channel."""
    devices = sounddevice.query_devices()
    return [
        {"index": i, "name": d["name"], "channels": d["max_output_channels"]}
        for i, d in enumerate(devices)
        if d["max_output_channels"] > 0
    ]


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
    blocksize = int(SAMPLE_RATE * chunk_ms / 1000)
    q: queue.Queue[AudioArray] = queue.Queue()

    def callback(
        indata: npt.NDArray[np.float32],
        frames: int,  # noqa: ARG001
        time: object,  # noqa: ARG001
        status: object,  # noqa: ARG001
    ) -> None:
        q.put(indata[:, 0].copy())

    with sounddevice.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        device=device,
        callback=callback,
    ):
        while True:
            chunk = q.get()
            if chunk is None:
                break
            yield chunk


def play(audio: AudioArray, sample_rate: int, device: int | None = None) -> None:
    """Play ``audio`` through the given output device and block until finished.

    Args:
        audio: 1-D mono float32 samples at ``sample_rate``.
        sample_rate: Samples per second of the ``audio`` array.
        device: PortAudio device index, or None for the OS default output.
    """
    sounddevice.play(audio, sample_rate, device=device)
    sounddevice.wait()
