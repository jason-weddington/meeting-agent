# CLAUDE.md — meeting-agent

Real-time AI meeting participant. Read [README.md](README.md) first for the project
vision and the hard "audio stays on-device" constraint.

## Living roadmap

Project-level roadmap (V1 → V1.5 → V2 → V3, with stage scope, dependencies, and
current status) lives as a GTD note on the meeting-agent project. Read it before
starting new work — and update it before (not after) starting a new stage.

- **GTD note id:** `a3f8eecb-7871-4c6e-8d9a-c838182d6007` — *"meeting-agent — living roadmap"*
- Retrieve with `mcp__agent-gtd__get_note` (or `list_notes` for the meeting-agent project).

## Quick orientation

- Python 3.13, uv-managed, src-layout (`src/meeting_agent/`).
- macOS Apple Silicon target — uses MLX (mlx-whisper) and PyTorch-MPS (Kokoro).
- Strict typing (mypy), ruff lint + format, conventional commits on main (hook enforced).
- Durable context lives in the personal-kb MCP. Anchor entries:
  - `kb-01305` — project architecture and constraints
  - `kb-01322` — measured latency baseline on M4 Max (Python stack is viable)
  - `kb-01324` — Parakeet-MLX (V1.5 candidate ASR swap)
  - `kb-01325` — Pipecat smart-turn-v2 (V1.5 turn-detection upgrade)
  - `kb-01329` — Bedrock Claude streaming + prompt-caching patterns
  - `kb-01332` — Meeting bot frameworks and audio routing

## Build / test commands

```bash
uv sync                                     # install deps
uv run pytest                               # run tests
uv run pre-commit run --all-files           # lint + typecheck
uv run python scripts/latency_check.py tts  # TTS latency smoke test
uv run python scripts/latency_check.py asr  # ASR latency smoke test (needs mic)
```

## Module layout

```
src/meeting_agent/
  audio.py     # mic capture + speaker playback via sounddevice (16 kHz mono)
  asr.py       # streaming Whisper (mlx-whisper) + Silero VAD → Utterance stream
  tts.py       # Kokoro TTS (sentence-by-sentence streaming)
  llm.py       # Bedrock Claude via converse_stream, with cachePoint breakpoints
  wake.py      # openwakeword wake-word detector
  pipeline.py  # orchestrator: wires modules; maintains rolling transcript
  cli.py       # `meeting-agent run` entry point
```

Each module is currently a stub with signatures only. Fill in the body for the
module you're working on; do not change signatures without updating callers.

## Key patterns

- **Audio format in-pipeline**: 16 kHz mono float32 everywhere except TTS output
  (Kokoro produces 24 kHz mono float32). Use `numpy.typing.NDArray[np.float32]`.
- **Streaming-first**: prefer generator interfaces (`Iterator[np.ndarray]`,
  `Iterator[str]`, `Iterator[Utterance]`) over batch calls. The LLM → TTS boundary
  should pipeline sentence-by-sentence (split Claude's streamed deltas on `.?!`)
  so the user hears audio while later sentences are still generating.
- **Bedrock**: use `bedrock-runtime.converse_stream`, not `InvokeModelWithResponseStream`.
  Use `cachePoint` breakpoints — 1h TTL for {system prompt + project context},
  5m TTL for rolling transcript. Do NOT set `performanceConfig.latency=optimized`
  (preview, 3.5 Haiku only). Do NOT send Anthropic's `scope` beta header — Bedrock
  rejects it.
- **Local-only audio**: no module may send raw or encoded audio to any network
  destination. Transcribed text only crosses a network boundary.
- **No cross-module imports between siblings** except what `pipeline.py` orchestrates.
  `audio.py`, `asr.py`, `tts.py`, `llm.py`, `wake.py` should stand alone.

## Testing conventions

- Unit tests under `tests/`, mirror the module name (`tests/test_audio.py` etc.).
- Mock external boundaries (sounddevice, boto3, model loads) — don't hit Bedrock
  or load large models in unit tests.
- Integration tests that need real models/audio go under `tests/integration/`
  and are opt-in (marked `@pytest.mark.integration`, skipped by default in CI).

## Git workflow

Branch-based (`feat/...`, `fix/...`, `chore/...`). Squash-merge to main with a
conventional commit message (hook enforced). Semantic-release runs post-commit
on main and bumps the version automatically.
