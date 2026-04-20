"""Speak every name in a pronunciation lexicon file for auditory review."""

import json
import sys
import time
from pathlib import Path

import sounddevice as sd

from meeting_agent.tts import TTS

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pronunciations.json")
if not path.exists():
    print(f"File not found: {path}")
    sys.exit(1)

with open(path) as f:
    names = [k for k in json.load(f) if not k.startswith("_")]

print(f"Speaking {len(names)} names from {path}\n")
tts = TTS(pronunciation_lexicon=path)

for name in names:
    print(f"  {name}...", flush=True)
    audio = tts.synthesize(name)
    sd.play(audio, 24000)
    sd.wait()
    time.sleep(0.5)

print("\nDone.")
