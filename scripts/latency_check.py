"""Latency smoke-test for the voice->text and text->voice pipeline stages.

Run on Apple Silicon. Downloads large-v3 Whisper (~3GB) and Kokoro (~330MB)
on first use.

Usage:
    uv run python scripts/latency_check.py asr [--seconds 5]
    uv run python scripts/latency_check.py tts ["sentence to speak"]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import sounddevice as sd

WHISPER_REPO = "mlx-community/whisper-large-v3-mlx"
WHISPER_SAMPLE_RATE = 16_000
KOKORO_SAMPLE_RATE = 24_000
KOKORO_VOICE = "af_heart"


def record(seconds: float) -> np.ndarray:
    """Record `seconds` of mono audio from the default input device at 16 kHz."""
    device_info = sd.query_devices(kind="input")
    print(f"input device   : {device_info['name']}")
    print(f"recording {seconds:.1f}s... speak now")
    audio = sd.rec(
        int(seconds * WHISPER_SAMPLE_RATE),
        samplerate=WHISPER_SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    print(f"captured peak  : {peak:.4f}  rms: {rms:.4f}")
    if peak < 0.001:
        print("!! captured audio is effectively silent — check mic routing/permissions")
    return audio


def asr_check(seconds: float) -> None:
    """Record a clip and measure end-of-speech -> transcript latency."""
    import mlx_whisper

    print(f"loading {WHISPER_REPO} (first run downloads ~3GB)...")
    # Warmup forces model load + graph compile out of the measured window.
    _ = mlx_whisper.transcribe(
        np.zeros(WHISPER_SAMPLE_RATE, dtype=np.float32),
        path_or_hf_repo=WHISPER_REPO,
    )

    audio = record(seconds)

    t0 = time.perf_counter()
    result = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_REPO)
    elapsed = time.perf_counter() - t0

    print()
    print(f"audio duration : {seconds:.2f}s")
    print(f"inference time : {elapsed:.2f}s")
    print(f"realtime factor: {seconds / elapsed:.1f}x")
    print(f"transcript     : {result['text'].strip()!r}")


def tts_check(sentence: str) -> None:
    """Synthesize a sentence and measure time-to-first-audio and total time."""
    from kokoro import KPipeline

    print("loading kokoro (first run downloads ~330MB)...")
    pipeline = KPipeline(lang_code="a")  # 'a' = American English

    # Warmup to exclude model load from measurements.
    for _ in pipeline("warmup.", voice=KOKORO_VOICE):
        break

    t0 = time.perf_counter()
    first_chunk_time: float | None = None
    chunks: list[np.ndarray] = []
    for result in pipeline(sentence, voice=KOKORO_VOICE):
        if result.audio is None:
            continue
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter() - t0
        chunks.append(np.asarray(result.audio, dtype=np.float32))
    total_time = time.perf_counter() - t0

    if not chunks:
        print("no audio produced")
        return

    audio = np.concatenate(chunks)
    audio_duration = len(audio) / KOKORO_SAMPLE_RATE
    ttfa = first_chunk_time if first_chunk_time is not None else float("nan")

    print()
    print(f"sentence        : {sentence!r}")
    print(f"time to first   : {ttfa:.2f}s")
    print(f"total gen time  : {total_time:.2f}s")
    print(f"audio duration  : {audio_duration:.2f}s")
    print(f"realtime factor : {audio_duration / total_time:.1f}x")

    print("playing back...")
    sd.play(audio, KOKORO_SAMPLE_RATE)
    sd.wait()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("asr", help="Record and transcribe; measure ASR latency.")
    pa.add_argument("--seconds", type=float, default=5.0)

    pt = sub.add_parser("tts", help="Synthesize a sentence; measure TTS latency.")
    pt.add_argument(
        "text",
        nargs="?",
        default=(
            "Hello, this is a test of text to speech latency. "
            "We're just testing how well this speach library works. "
            "Pretty great, huh?!"
        ),
    )

    args = parser.parse_args()
    if args.cmd == "asr":
        asr_check(args.seconds)
    elif args.cmd == "tts":
        tts_check(args.text)


if __name__ == "__main__":
    main()
