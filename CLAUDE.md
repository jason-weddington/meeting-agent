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

## Inference backends

`meeting-agent` supports two backends for both the **classifier** and the
**response LLM**.  Bedrock is the default for both; Ollama enables fully-local
operation for sensitive meetings.

### Classifier

- **Bedrock Haiku** (default) — `--classifier-backend bedrock`
- **Local Ollama** — `--classifier-backend ollama` (defaults to `qwen3.6:35b-a3b-mlx-bf16`)

```bash
uv run meeting-agent --classifier-backend ollama                                   # qwen3.6 default
uv run meeting-agent --classifier-backend ollama --classifier-model qwen3.5:35b-a3b  # fallback
```

### Response LLM

- **Bedrock Claude Sonnet** (default) — `--llm-backend bedrock`
- **Local Ollama** — `--llm-backend ollama` (defaults to `qwen3.6:35b-a3b-mlx-bf16`)

```bash
uv run meeting-agent --llm-backend ollama                                          # qwen3.6 default
uv run meeting-agent --llm-backend ollama --llm-model llama3.2:latest             # custom model
```

### Fully-local mode

Run both backends on Ollama to avoid all Bedrock calls:

```bash
# Fully local (classifier + response both on Ollama)
uv run meeting-agent --classifier-backend ollama --llm-backend ollama

# Local classifier + Bedrock response (current V2.9 behavior, still default)
uv run meeting-agent --classifier-backend ollama
```

The Ollama backends both hit the same `ollama serve` daemon. Use `--ollama-host`
or the `OLLAMA_HOST` env var to override the default `http://localhost:11434`
(shared by both classifier and response-LLM paths).

### KB grounding (MCP)

Enable real-time KB grounding via a local stdio MCP server using `--kb-mcp`:

```bash
# Enable KB grounding via a local stdio MCP server
uv run meeting-agent --kb-mcp "uv run personal-kb-mcp" --trace --verbose

# With a path and custom argument
uv run meeting-agent --kb-mcp "/path/to/personal-kb-mcp --some-arg"
```

`SERVER_SPEC` is parsed with `shlex.split`: first token is the command,
remaining tokens are arguments.  No env-var passthrough yet — use a wrapper
script if you need to inject env vars.

**Note:** Ollama tool-use is not yet supported.  The `--kb-mcp` flag is silently
ignored (with a one-time log warning) when `--llm-backend ollama` is used.
Grounding currently only applies when the response LLM is Bedrock (the default).

## Module layout

```
src/meeting_agent/
  audio.py       # mic capture + speaker playback via sounddevice (16 kHz mono)
  asr.py         # streaming Whisper (mlx-whisper) + Silero VAD → Utterance stream
  classifier.py  # classifier Protocol + BedrockClassifier + OllamaClassifier
  tts.py         # Kokoro TTS (sentence-by-sentence streaming)
  llm.py         # LLMClient Protocol + BedrockClient + OllamaClient response-LLM backends
  wake.py        # openwakeword wake-word detector
  pipeline.py    # orchestrator: wires modules; maintains rolling transcript
  cli.py         # `meeting-agent run` entry point
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
