# tts.nim

Native text-to-speech, speech-to-text, and video rendering engine for Nim. Uses [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) for TTS and [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) for STT. Apple Silicon uses [MLX](https://github.com/ml-explore/mlx) with 4-bit quantization; other platforms use [ggml](https://github.com/ggerganov/ggml). No Python, no ONNX — just C and Nim.

- 54+ voices across 9 languages (English, Chinese, Spanish, French, Hindi, Italian, Japanese, Portuguese, + more via espeak-ng)
- ~130MB quantized TTS models (4-bit MLX on Apple Silicon, Q5 GGML elsewhere)
- 8-12x realtime on Apple M2
- Metal acceleration on macOS, CPU on Linux
- Direct audio I/O via [miniaudio](https://miniaud.io/) (playback + mic capture)
- Voice activity detection (neural Silero VAD on MLX, energy-based on GGML)
- Qwen3-ASR 4-bit speech recognition (30+ languages, ~92ms first-token latency)
- Script-to-video rendering with [Remotion](https://www.remotion.dev/) (radio studio theme, timed subtitles)
- Vendored espeak-ng for phonemization (statically linked, zero system deps)
- Native Bopomofo phonemizer for Chinese

## Demo

https://github.com/JK8769/tts.nim/releases/download/v0.2.0/test_video.mp4

A short script rendered with the `script` MCP tool — TTS audio + Remotion video with radio studio visuals, speaker panels, mood-reactive orb, and timed subtitles.

## Quick start

### Install

```bash
nimble install https://github.com/JK8769/tts.nim
```

Builds native deps, downloads models, and installs `tts_cli`. Requires: Nim 2.0+, CMake, a C compiler. Platform is auto-detected — Apple Silicon builds the MLX backend, everything else builds GGML.

### CLI

```bash
tts_cli synth "Hello world" -v af_heart -o output.wav
tts_cli synth "This is fast" -v am_adam -s 1.2
tts_cli voices                    # list all voices
tts_cli voices --en --female      # filter by language and gender
tts_cli models                    # list downloaded models
tts_cli download kokoro-en        # download English model
```

### MCP server

Set up the MCP config for any AI agent (Claude Desktop, Claude Code, etc.):

```bash
tts_cli mcp              # writes .mcp.json to current directory
tts_cli mcp --print      # print config to stdout
```

Or configure manually:

```json
{
  "mcpServers": {
    "tts": { "command": "tts_cli", "args": ["serve"] }
  }
}
```

MCP tools:

| Tool | Description |
|------|-------------|
| **synth** | Synthesize text to WAV file (with chunk timing) |
| **speak** | Play text through speakers (non-blocking, auto-interrupts previous) |
| **stop** | Silence any playing speech |
| **listen** | Record from mic until silence, return transcribed text (Qwen3-ASR) |
| **voices** | List available voices with language/gender metadata |
| **models** | List downloaded models |
| **converse_start/send/recv/stop** | Managed conversation sessions |
| **script** | Multi-voice script rendering with audio and video output |
| **calibrate** | Calibrate speech timing for a voice |

### Script rendering

The `script` MCP tool renders multi-voice JSONL scripts to audio and video:

```jsonl
{"type":"header","title":"My Story","format":"show","cast":{"Alice":"af_heart","Bob":"am_adam"}}
{"type":"chapter","text":"Chapter 1"}
{"type":"scene","text":"A cozy room.","mood":"warm","narrate":true}
{"type":"line","name":"Alice","text":"Hello Bob!"}
{"type":"line","name":"Bob","text":"Hey Alice!"}
```

Actions:
- **render** — synthesize all lines to a single audio file with timeline
- **video** — generate video with radio studio visuals (requires Node.js/Bun)

### Voice mixing

Blend two voices with a weight (0.0 = pure first, 1.0 = pure second):

```bash
tts_cli synth "Hello" -v "af_heart+am_adam:0.3"     # 70% af_heart, 30% am_adam
tts_cli synth "Hello" -v "af_heart+bf_emma:0.5"     # 50/50 blend
```

### Streaming

Stream raw PCM (s16le, 24kHz, mono) to stdout per sentence:

```bash
tts_cli synth "First sentence. Second sentence." --stream | ffplay -f s16le -ar 24000 -ac 1 -nodisp -
```

### Batch synthesis

```bash
tts_cli batch chapters.txt -d output/          # one WAV per line
tts_cli batch script.txt -v am_adam --json      # JSON output for automation
```

### Conversation mode

Live voice conversation — mic capture, VAD, STT, and TTS in a loop:

```bash
tts_cli converse                                     # default: English, af_heart voice
tts_cli converse -v am_adam --lang en                 # male voice
tts_cli converse --greeting "Hi, how can I help?"     # agent greeting
```

### Agent / programmatic use

Every command supports `--json` for structured output:

```bash
tts_cli synth "Hello" --json
# {"output":"output.wav","duration":1.62,"sample_rate":24000,"voice":"af_heart",...}

tts_cli voices --en --female --json
tts_cli models --json

tts_cli schema                   # flat JSON schema
tts_cli schema --per-command     # per-command schemas (5-20x fewer tokens per LLM request)
```

## Performance

Benchmarked on Apple M2 (8-core GPU, 24GB) with 4-bit quantized MLX models:

| Input | Audio length | Wall time | Realtime factor |
|-------|-------------|-----------|-----------------|
| 1 sentence (12 words) | 3.4s | 0.4s | **8.5x** |
| 2 sentences (19 words) | 6.2s | 0.6s | **10x** |
| 1 paragraph (73 words) | 22.6s | 1.9s | **12x** |

## Models

Platform is auto-detected: Apple Silicon gets MLX models, everything else gets GGML.

**TTS — Kokoro:**

| Model | Size | Backend | Languages | Voices |
|-------|------|---------|-----------|--------|
| `kokoro-mlx-q4` | 131 MB | MLX 4-bit | en | 28 |
| `kokoro-zh-mlx-q4` | 134 MB | MLX 4-bit | zh, en | 103 |
| `kokoro-en-q5.gguf` | 191 MB | GGML Q5 | en | 28 |
| `kokoro-v1.1-zh-q5.gguf` | 198 MB | GGML Q5 | zh, en | 103 |

**STT — Qwen3-ASR (Apple Silicon):**

| Model | Size | Quantization | WER impact | Speed |
|-------|------|-------------|------------|-------|
| `qwen3-asr-0.6b-4bit` | 675 MB | 4-bit | +0.43 WER | fastest (default) |
| `qwen3-asr-0.6b-8bit` | 960 MB | 8-bit | +0.04 WER | baseline |

**VAD — Silero v5 (Apple Silicon):**

| Model | Size | Description |
|-------|------|-------------|
| `silero-vad` | 1.6 MB | Neural voice activity detection |

Models are downloaded automatically during `nimble install`. Set `TTS_MODEL_DIR` to override the search path.

## Voices

Kokoro voices follow the pattern `{lang}{gender}_{name}`:

| Prefix | Language | Example voices |
|--------|----------|---------------|
| `af_*` | English (US, female) | af_heart, af_maple, af_sky |
| `am_*` | English (US, male) | am_adam, am_michael |
| `bf_*` | English (UK, female) | bf_emma, bf_isabella |
| `bm_*` | English (UK, male) | bm_george, bm_lewis |
| `zf_*` | Chinese (female) | zf_001, zf_002 |
| `zm_*` | Chinese (male) | zm_001, zm_002 |
| `ef_*` | Spanish (female) | ef_dora |
| `ff_*` | French (female) | ff_siwis |
| `hf_*` | Hindi (female) | hf_alpha |
| `if_*` | Italian (female) | if_sara |
| `jf_*` | Japanese (female) | jf_alpha |
| `pf_*` | Portuguese (female) | pf_dora |

Run `tts_cli voices` for the full list, or filter with `--male`, `--female`, `--en`, `--zh`.

## Languages

English is installed by default. Chinese uses a native Bopomofo phonemizer.

To add more languages for espeak-ng phonemization:

```bash
nimble lang                       # list installed + available
nimble lang add es fr ja          # add Spanish, French, Japanese
```

## Upgrade

```bash
nimble install https://github.com/JK8769/tts.nim@#head
```

Nimble caches by version tag, so `@#head` forces it to fetch the latest commit.

### From source

```bash
git clone --recurse-submodules https://github.com/JK8769/tts.nim
cd tts.nim
nimble install
```

## Project structure

```
src/
  tts.nim                         # library entry point
  tts_cli.nim                     # CLI binary (docopt + MCP server + converse)
  config.nims                     # auto-detect platform → -d:useMlx on Apple Silicon
  tts/
    common.nim                    # AudioOutput, WAV writer, path utils
    engine.nim                    # TTSEngine API (dispatches to MLX or GGML)
    converse.nim                  # conversation loop (mic → VAD → STT → TTS → speaker)
    tokenizer.nim                 # phoneme string → token IDs
    audio/
      device.nim                  # miniaudio playback + capture (ring buffer)
      silero_vad.nim              # neural Silero VAD v5 (MLX)
      smart_turn.nim              # smart turn detection for conversation
      ma_bridge.{c,h}            # C bridge for miniaudio
    stt/
      qwen3_asr.nim               # Qwen3-ASR speech-to-text (MLX, 4/8-bit)
      whisper_mlx.nim             # Whisper STT via MLX
      sensevoice_mlx.nim          # SenseVoice STT (MLX)
    mlx/
      mlx.nim                     # high-level MLX tensor API
      mlx_capi.nim                # mlx-c FFI bindings
    models/
      kokoro_mlx.nim              # Kokoro TTS via MLX (quantization-aware)
    phonem/
      phonemizer.nim              # unified phonemizer (routes to espeak/bopomofo)
      espeak.nim                  # espeak-ng static bindings
      bopomofo.nim                # native Chinese phonemizer
  res/
    video-template/               # Remotion video template (shipped with package)
vendor/
  mlx-c-src/                      # mlx-c submodule (Apple Silicon)
  espeak-ng/                      # espeak-ng submodule
  miniaudio/                      # miniaudio single-header audio library
scripts/
  audio/                          # demo scripts, voice downloader
  benchmark/                      # performance benchmarks
  debug/                          # debugging tools
  model/                          # model quantization
  video/                          # video rendering workspace
```

## License

MIT
