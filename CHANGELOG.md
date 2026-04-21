# CHANGELOG

<!-- version list -->

## v0.12.0 (2026-04-21)

### Features

- Inject current timestamp into each LLM turn
  ([`14a2bc5`](https://github.com/jason-weddington/meeting-agent/commit/14a2bc5b265f1f9d09511c23afa071f0ea6e8de7))


## v0.11.0 (2026-04-20)

### Features

- Custom pronunciation lexicon for TTS
  ([`8014058`](https://github.com/jason-weddington/meeting-agent/commit/8014058eed1c98db886285e91986755a3b01ca7c))


## v0.10.3 (2026-04-20)

### Bug Fixes

- **llm**: Narrate tool calls to avoid dead air
  ([`78ea1ae`](https://github.com/repos/meeting-agent/commit/78ea1aeba67c1585fba95bb86ec5e4300f5d7e88))


## v0.10.2 (2026-04-20)

### Bug Fixes

- **classifier**: Distinguish repeat-request direction; route user asks through
  ([`8e2a7b4`](https://github.com/repos/meeting-agent/commit/8e2a7b42adfea147fd238498b7cf2ab05af1c3d0))


## v0.10.1 (2026-04-20)

### Bug Fixes

- **mcp**: Inherit parent env for subprocess by default
  ([`afd0fef`](https://github.com/repos/meeting-agent/commit/afd0fef4e075c5da187c3b4bc3a70b8a376dcef5))


## v0.10.0 (2026-04-19)

### Bug Fixes

- **ollama**: Disable thinking + make shared prompt backend-neutral
  ([`1cf3179`](https://github.com/repos/meeting-agent/commit/1cf31796a9f250239501a9b5d516e0ac3e4d9929))

- **ollama**: Pre-warm model and bump timeout to 60s to survive cold-load
  ([`9210699`](https://github.com/repos/meeting-agent/commit/921069968285df2914d7ec0431e24fcba68c596d))

- **pipeline**: Silence-flush TTS output + extend echo drain window
  ([`6729c1e`](https://github.com/repos/meeting-agent/commit/6729c1eb7c026aa663c5963161b9420f058c2c4d))

### Chores

- Retire wake.py and openwakeword dependency
  ([`a83fb4e`](https://github.com/repos/meeting-agent/commit/a83fb4e1270895b7489467cdf1087f5fdaca450d))

- **ollama**: Default to qwen3.6:35b-a3b-mlx-bf16
  ([`07aea8a`](https://github.com/repos/meeting-agent/commit/07aea8a35a35dfd1262d4f712c7c73081f89301b))

### Features

- **classifier**: Haiku 4.5 gatekeeper — silent / hedged_answer / full_answer
  ([`ae21b3c`](https://github.com/repos/meeting-agent/commit/ae21b3cdae14228525d72a5ef9376a169b85f138))

- **classifier**: Local Ollama backend with configurable model (V2.9)
  ([`5fe2cb9`](https://github.com/repos/meeting-agent/commit/5fe2cb9c2320ffc7830b9649b30595ee9a907bc0))

- **context**: Markdown directory loader for Living Context Pattern
  ([`89100e4`](https://github.com/repos/meeting-agent/commit/89100e4e1cec770c8182543443d168678a0b0721))

- **llm**: Bedrock tool-use loop via MCP client (V3.0.2)
  ([`f883750`](https://github.com/repos/meeting-agent/commit/f8837504c165ccc64d2ec02626033ab087928a25))

- **llm**: Local Ollama response-LLM backend (configurable model, Bedrock default)
  ([`1eaee66`](https://github.com/repos/meeting-agent/commit/1eaee6655020786531947a92ee8c40063323c6a0))

- **llm**: Ollama tool-use loop via MCP client (V3.0.5)
  ([`8a9308d`](https://github.com/repos/meeting-agent/commit/8a9308d2fae2e53f61d791483e37c568adce466d))

- **mcp**: Add MCPClient stdio foundation (V3.0.1)
  ([`60a7223`](https://github.com/repos/meeting-agent/commit/60a72237d0d5ffd7338d96b590c211a3ad889e1d))

- **mcp**: End-to-end integration test + smoke-test docs (V3.0.4)
  ([`21b80b0`](https://github.com/repos/meeting-agent/commit/21b80b0685e0eddce38baae9786151a37fa7f5ed))

- **observability**: Dev-mode trace log — classifier decisions, pipeline events, latency
  ([`75ca09d`](https://github.com/repos/meeting-agent/commit/75ca09df620ae9a867f5319061f9c84330700504))

- **observability**: Surface Bedrock cache-hit telemetry for classifier + response LLM
  ([`0080740`](https://github.com/repos/meeting-agent/commit/0080740b5ac77d8dd51e8738cd604ea7e3b9bdcf))

- **pipeline**: Always-on classifier-gated V2 pipeline with lossy-aware scaffolding
  ([`51e20b5`](https://github.com/repos/meeting-agent/commit/51e20b591a8c2c7e75da56c746917b959adb54e9))

- **pipeline**: TTS post-filter strips markdown symbols before synthesis
  ([`c35e4de`](https://github.com/repos/meeting-agent/commit/c35e4dec8f08bd51bdbdd4a7281d30806b3ce938))

- **pipeline**: Wire MCP client + --kb-mcp CLI flag (V3.0.3)
  ([`7281f81`](https://github.com/repos/meeting-agent/commit/7281f810808a94ae526a9a1387c704bf3c99da8a))

### Refactoring

- **llm**: Switch from flat-text transcript to native multi-turn messages
  ([`299dcd1`](https://github.com/repos/meeting-agent/commit/299dcd1670addee7983038774c4e92e4e5446c89))

### Testing

- **integration**: End-to-end test for V2 classifier-gated pipeline
  ([`0378c12`](https://github.com/repos/meeting-agent/commit/0378c126083ab8fa2dba360d7bdae6ff218ff262))


## v0.9.4 (2026-04-17)

### Bug Fixes

- **asr, pipeline**: V1.5 robustness bundle — silence guard + VAD + timeout
  ([`a2f26ac`](https://github.com/repos/meeting-agent/commit/a2f26ac45636b45997047d47f0a0f97f2e0736a4))

- **cli**: Tune default system prompt for text-to-speech output
  ([`dba3ea9`](https://github.com/repos/meeting-agent/commit/dba3ea9e53a80865ad1cf9be29326e8a43f9de06))

- **pipeline**: Drain mic-queue backlog of agent audio to stop feedback loop
  ([`6211aad`](https://github.com/repos/meeting-agent/commit/6211aadeff945debc5da06cfb722a6a8d325f200))

### Chores

- Decouple release from deploy
  ([`af52d6c`](https://github.com/repos/meeting-agent/commit/af52d6cfd8716d7740d1e718a63321e32b6dadf5))

### Documentation

- **claude**: Point to GTD roadmap note
  ([`59f8b4a`](https://github.com/repos/meeting-agent/commit/59f8b4aa79f8eeb44466818b2ab6cfbc46514215))


## v0.9.3 (2026-04-17)

### Bug Fixes

- **deps**: Add awscrt for new aws login CRT-based cred provider
  ([`072dc0d`](https://github.com/repos/meeting-agent/commit/072dc0d1a7c219a16ff5121fcbea6dfadcedf051))

### Chores

- **pipeline**: Print status when listening for wake and after trigger
  ([`b8ff607`](https://github.com/repos/meeting-agent/commit/b8ff6070a28fefa473404b3a787bcc61b7e3cab9))


## v0.9.2 (2026-04-17)

### Bug Fixes

- **wake**: Use wakeword_model_paths and resolve bundled names to paths
  ([`5ec4f61`](https://github.com/repos/meeting-agent/commit/5ec4f61927fb475f4d700583b61f5f86183e0602))


## v0.9.1 (2026-04-17)

### Bug Fixes

- **scripts**: Update latency_check.py to mlx-audio TTS API
  ([`cdbeb2c`](https://github.com/repos/meeting-agent/commit/cdbeb2c74581ca2553bdbd343843418a7f72f53b))

### Testing

- Add end-to-end smoke test for full pipeline round-trip
  ([`c9e241c`](https://github.com/repos/meeting-agent/commit/c9e241cd1743db013488b62371ab206a0cb7edec))


## v0.9.0 (2026-04-17)

### Features

- **pipeline**: Implement Pipeline.run with first-sentence TTS pipelining
  ([`7211683`](https://github.com/repos/meeting-agent/commit/7211683f60b4ab4fcf6ba4d0187d359e08bc41e3))


## v0.8.0 (2026-04-17)

### Features

- **cli**: Implement meeting-agent CLI entry point
  ([`7415064`](https://github.com/repos/meeting-agent/commit/7415064bfbb503294cc618be1177d28e16411702))


## v0.7.0 (2026-04-17)

### Features

- **asr**: Implement streaming Whisper + Silero VAD + custom vocab
  ([`03b3aa2`](https://github.com/repos/meeting-agent/commit/03b3aa22024c8afdf69e1406e1be074e9ae27b9d))


## v0.6.0 (2026-04-17)

### Features

- **tts**: Swap PyTorch-MPS Kokoro backend to mlx-audio MLX-native
  ([`ef5e9ae`](https://github.com/repos/meeting-agent/commit/ef5e9ae0c38660a33732baafed3260e57478a6d6))


## v0.5.0 (2026-04-16)

### Features

- **wake**: Implement openwakeword wake-word detector
  ([`540b7c2`](https://github.com/repos/meeting-agent/commit/540b7c2660550605674b73f547d21cb2ffa148b5))


## v0.4.0 (2026-04-16)

### Features

- **llm**: Implement BedrockClient.respond_stream with prompt caching
  ([`4cf4e23`](https://github.com/repos/meeting-agent/commit/4cf4e23eb5e934721f5c57dc79f6153bffd04f2b))


## v0.3.0 (2026-04-16)

### Features

- **audio**: Implement mic capture and speaker playback via sounddevice
  ([`8a595ed`](https://github.com/repos/meeting-agent/commit/8a595ed25532f46f36076ae482d77b5551c8fd66))


## v0.2.0 (2026-04-16)

### Chores

- Repo bootstrap — README north star, CLAUDE.md, module stubs, MIT license
  ([`0534027`](https://github.com/repos/meeting-agent/commit/053402767bf8b534a87d5df1b90b556a6cc102e8))

### Features

- **tts**: Implement Kokoro TTS wrapper with streaming and tests
  ([`39a7012`](https://github.com/repos/meeting-agent/commit/39a70124c221871c7771d3a922790012df338adc))


## v0.1.0 (2026-04-16)

- Initial Release
