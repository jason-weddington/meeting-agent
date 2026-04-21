# meeting-agent

A real-time AI participant for live meetings. Listens, reasons, and speaks — with the same rich project context a good TPM would carry in their head.

## The problem this solves

Today, the best pattern for using AI in meetings is **post-hoc**: record the meeting, transcribe it, feed the transcript to a reasoning agent, get a digest back. That works — but the agent is always one meeting behind. It can't point out a conflicting decision **while** the team is making it, can't flag a missing stakeholder **before** the room moves on, can't surface a prior risk **in the moment** it would actually change the outcome.

This project puts the agent **in the room, in real time**.

The north-star agent has:
- A carefully curated project context: org chart, workstreams, decision log, active risks, goals, stakeholders.
- A reasoning model strong enough to actually be useful (Claude Sonnet/Opus class, not a chat-bot tier model).
- A transcription layer that handles team/product/project jargon without garbling it.
- A voice it speaks with — so it participates, not just transcribes.
- A feedback loop: every meeting updates the context for the next meeting.

## Architecture

```
┌─────────┐   ┌─────────┐   ┌────────────────┐   ┌─────────┐   ┌─────────┐
│  mic    │──▶│  ASR    │──▶│    reasoning   │──▶│   TTS   │──▶│ speaker │
│ BlackHole│   │ Whisper │   │ Claude/Bedrock │   │ Kokoro  │   │BlackHole│
└─────────┘   └─────────┘   └────────────────┘   └─────────┘   └─────────┘
                   ▲                  ▲
                   │                  │
           ┌───────┴────────┐   ┌─────┴──────────┐
           │ Silero VAD +   │   │ cached project │
           │ openwakeword   │   │    context     │
           └────────────────┘   └────────────────┘
```

**Hard constraint: all audio stays on-device.** Raw meeting audio never leaves the laptop. Only transcribed text crosses a network boundary (to AWS Bedrock for reasoning).

**Platform**: macOS on Apple Silicon, M-series (M4/M5 Max recommended, 128GB helps). Cross-platform later.

## Roadmap

### V1 — the pipeline works end-to-end
- Built-in mic, wake-word activation, single user
- Streaming Whisper (mlx-whisper, large-v3) + Silero VAD
- Claude Sonnet on Bedrock with `converse_stream` and prompt caching
- Kokoro TTS, first-sentence streaming (speak as the LLM generates)
- `meeting-agent run` CLI

### V1.5 — latency, quality, and ergonomics
- Custom-vocabulary biasing on ASR (Whisper `initial_prompt` + post-ASR jargon repair)
- Explore Parakeet-MLX as ASR alternative (3-6× faster on Apple Silicon, per [kb-01324](.))
- MLX-native Kokoro via `mlx-audio` (drop PyTorch-on-MPS)
- Pipecat smart-turn-v2 for semantic endpointing (over plain silence-timer VAD)
- Core Audio taps helper (remove BlackHole install friction on macOS 14.2+)

### V2 — real meeting integration + better conversation
- BlackHole/Aggregate-device routing for Zoom / Meet / Teams
- Speaker diarization (diart, streaming on macOS)
- "Am I being addressed" detection — LLM classifier at turn boundaries (no wake word needed)
- Gating: mic-mute while the agent speaks; interruption handling
- Per-participant audio where available (`py-zoom-meeting-sdk` for Zoom)

### V3 — the richly-contextualized agent
- First-class project context: org chart, workstreams, decision log, risks, goals
- Post-meeting digest → context-update loop (what did we decide, what changed, what's new)
- Mid-meeting tool use: look up docs, check prior decisions, query the decision log
- Proactive contribution mode (speak up when confidence is high, stay quiet otherwise)

## Quick start

Requires macOS on Apple Silicon, Python 3.13, and microphone/speaker permission for your terminal.

```bash
# Clone and install
git clone git@github.com:jason-weddington/meeting-agent.git
cd meeting-agent
uv sync

# Optional: latency smoke tests to verify your machine is fast enough
uv run python scripts/latency_check.py tts
uv run python scripts/latency_check.py asr --seconds 5
```

### Running the agent

The default configuration is Bedrock Claude Sonnet for responses, Bedrock Claude Haiku for the classifier. AWS credentials need to be active (the `bedrock-runtime` service).

```bash
# Minimal: default models, no KB grounding
uv run meeting-agent --trace --verbose

# Grounded: give the agent an MCP server it can query mid-response
uv run meeting-agent --kb-mcp ~/scripts/personal-kb.sh --trace --verbose
```

`--trace` writes JSONL events to `~/.meeting-agent/trace.jsonl` (decisions, tool calls, latencies). `--verbose` also tees a one-line summary per event to stderr while the agent runs.

### Fully-local mode

Swap Bedrock out for a local Ollama daemon on both tiers. No AWS, no network egress beyond the MCP server you point at. Requires an Ollama install with a tool-calling-capable model pulled (default: `qwen3.6:35b-a3b-mlx-bf16`).

```bash
# Fully local: classifier + response + KB grounding all on-device
uv run meeting-agent \
    --classifier-backend ollama \
    --llm-backend ollama \
    --kb-mcp ~/scripts/personal-kb.sh \
    --trace --verbose
```

First invocation cold-loads the model (20–30s for a 35B MoE). Subsequent runs re-use Ollama's resident model via `keep_alive` (default 5 min).

The `--kb-mcp` flag takes the full command string to launch a stdio MCP server. A wrapper script like `~/scripts/personal-kb.sh` is the ergonomic way to keep the command out of the flag. The meeting-agent spawns the server as a subprocess, speaks JSON-RPC over stdin/stdout, and terminates it on shutdown. Shell-exported environment variables (e.g. `KB_DATABASE_URL`) are inherited by the MCP subprocess.

### Just-us mode (1:1 working session)

The default mode (`ambient`) is designed for a multi-person meeting where the agent is a passive participant that speaks only when clearly addressed. `--just-us` switches to a focused 1:1 mode (`duet`) optimized for working directly with the agent:

- **Classifier loosens silence**: every non-garbled utterance is treated as addressed to the agent — default action is `full_answer`. No airtime budget applies.
- **Proactive KB capture**: when the conversation produces a decision, insight, or framework, the agent offers to save it to the knowledge base (`"want me to save that as a decision?"`). Writes happen only if you say yes.

```bash
# 1:1 working session with KB grounding
uv run meeting-agent --just-us --kb-mcp ~/scripts/personal-kb.sh --trace --verbose

# Fully local 1:1 session
uv run meeting-agent \
    --just-us \
    --classifier-backend ollama \
    --llm-backend ollama \
    --kb-mcp ~/scripts/personal-kb.sh \
    --trace --verbose
```

Use `ambient` (default) when you are in a multi-person meeting and want the agent to listen quietly and speak selectively. Use `--just-us` when you want a focused working session — like pair-programming or brainstorming — with the agent as your primary collaborator.

## Design notes and prior art

Key decisions and rationale are captured in the knowledge base:

- [kb-01305](.) — Project architecture and constraints
- [kb-01322](.) — Python stack viability and measured latency baseline
- [kb-01324](.) — Parakeet-MLX as a future ASR alternative
- [kb-01325](.) — Turn detection (Pipecat smart-turn-v2)
- [kb-01328](.) — Why we're not using an end-to-end speech model
- [kb-01329](.) — Bedrock Claude streaming + prompt caching patterns
- [kb-01332](.) — Meeting bot frameworks and audio routing

## License

MIT — see [LICENSE](LICENSE).
