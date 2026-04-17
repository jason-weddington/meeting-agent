# CHANGELOG

<!-- version list -->

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
