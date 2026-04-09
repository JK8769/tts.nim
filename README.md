# tts.nim

Native text-to-speech and speech-to-text engine for Nim. Uses [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) for TTS and [Whisper](https://github.com/openai/whisper) for STT. Apple Silicon uses [MLX](https://github.com/ml-explore/mlx) with 4-bit quantization; other platforms use [ggml](https://github.com/ggerganov/ggml). No Python, no ONNX -- just C and Nim.

- 54+ voices across 9 languages (English, Chinese, Spanish, French, Hindi, Italian, Japanese, Portuguese, + more via espeak-ng)
- ~130MB quantized TTS models (4-bit MLX on Apple Silicon, Q5 GGML elsewhere)
- 8-12x realtime on Apple M2
- Metal acceleration on macOS, CPU on Linux
- Direct audio I/O via [miniaudio](https://miniaud.io/) (playback + mic capture)
- Voice activity detection for hands-free conversation (neural Silero VAD on MLX, energy-based on GGML)
- Vendored espeak-ng for phonemization (statically linked, zero system deps)
- Native Bopomofo phonemizer for Chinese

## Quick start

### As a library

```nim
import tts

var engine = newTTSEngine()
engine.loadModel("kokoro-en-q5.gguf")
let audio = engine.synthesize("Hello world!", voice = "af_heart")
audio.writeWav("output.wav")
engine.close()
```

### As a CLI

```bash
nimble install tts
tts_cli synth "Hello world" -v af_heart -o output.wav
tts_cli synth "This is fast" -v am_adam -s 1.2
tts_cli voices                    # list all voices
tts_cli voices --en --female      # filter by language and gender
tts_cli voices -m kokoro-zh       # voices from a specific model
tts_cli models                    # list downloaded models
tts_cli download kokoro-en        # download English model
tts_cli download kokoro-zh        # download Chinese model
```

### Voice mixing

Blend two voices with a weight (0.0 = pure first, 1.0 = pure second):

```bash
tts_cli synth "Hello" -v "af_heart+am_adam:0.3"     # 70% af_heart, 30% am_adam
tts_cli synth "Hello" -v "af_heart+bf_emma:0.5"     # 50/50 blend
tts_cli synth "Hello" -v "af_heart+am_adam"          # default 50/50
```

### Streaming

Stream raw PCM (s16le, 24kHz, mono) to stdout per sentence for low-latency playback:

```bash
tts_cli synth "First sentence. Second sentence." --stream | ffplay -f s16le -ar 24000 -ac 1 -nodisp -
tts_cli synth "Long text..." --stream --json 2>meta.json | sox -t raw -r 24000 -b 16 -e signed -c 1 - out.wav
```

### Batch synthesis

```bash
tts_cli batch chapters.txt -d output/          # one WAV per line
tts_cli batch script.txt -v am_adam --json      # JSON output for automation
tts_cli batch book.txt -v "af_heart+bf_emma:0.4" -d audiobook/  # batch with voice mix
```

### Piping

```bash
echo "Hello from stdin" | tts_cli synth -                     # read text from stdin
tts_cli synth "pipe me" --output - | ffplay -nodisp -         # WAV to stdout
```

### Long text handling

Long sentences are automatically split on clause boundaries (commas, semicolons, colons) when they exceed ~400 phoneme tokens. This prevents OOM errors and quality degradation on long inputs — just pass your text and the engine handles chunking.

### Agent / programmatic use

Every command supports `--json` for structured output:

```bash
tts_cli synth "Hello" --json
# {"output":"output.wav","duration":1.62,"sample_rate":24000,"voice":"af_heart",...}

tts_cli voices --en --female --json
# [{"model":"kokoro-en-q5.gguf","voices":[{"name":"af_heart","language":"en","gender":"female"},...]}]

tts_cli models --json
# [{"model":"kokoro-en-q5.gguf","path":"...","size_bytes":200180704,"voices":28},...]
```

Errors are also JSON when `--json` is set:
```bash
tts_cli synth "test" -v bogus --json
# {"error":"voice 'bogus' not found in model 'kokoro-en-q5.gguf'"}  (exit 1)
```

### Schema introspection

```bash
tts_cli schema                   # flat JSON schema (all commands + options)
tts_cli schema --per-command     # per-command schemas (each subcommand isolated)
```

Per-command schema gives each usage pattern its own tool definition with only relevant parameters — **5-20x fewer tokens** per LLM request compared to the flat schema.

### Conversation mode

Talk to your computer with voice — mic capture, VAD, Whisper STT, and TTS in a live loop:

```bash
tts_cli converse                                     # default: English, af_heart voice
tts_cli converse -v am_adam --lang en                 # male voice
tts_cli converse --greeting "Hi, how can I help?"     # agent greeting
tts_cli converse --whisper ggml-base.en.bin           # specify whisper model
```

The conversation loop handles turn-taking automatically:
- **Barge-in**: start talking while the agent is speaking and it stops immediately
- **VAD**: energy-based voice activity detection with holdoff to avoid false triggers
- **Echo mode**: by default, the agent echoes back what you said (plug in an LLM for real conversation)

### MCP server

`tts_cli serve` runs an [MCP](https://modelcontextprotocol.io/) server over stdio, exposing `synth`, `speak`, `stop`, `listen`, `voices`, and `models` as tools:

```json
// claude_desktop_config.json or any MCP client config
{
  "mcpServers": {
    "tts": { "command": "tts_cli", "args": ["serve"] }
  }
}
```

MCP tools:
- **synth** — synthesize to WAV file (with chunk timing)
- **speak** — play text through speakers (non-blocking, auto-interrupts previous)
- **stop** — silence any playing speech
- **listen** — record from mic until user stops talking, return transcribed text
- **voices** — list available voices with language/gender metadata
- **models** — list downloaded models

## Performance

Benchmarked on Apple M2 (8-core GPU, 24GB) with 4-bit quantized MLX models:

| Input | Audio length | Wall time | Realtime factor |
|-------|-------------|-----------|-----------------|
| 1 sentence (12 words) | 3.4s | 0.4s | **8.5x** |
| 2 sentences (19 words) | 6.2s | 0.6s | **10x** |
| 1 paragraph (73 words) | 22.6s | 1.9s | **12x** |

Wall time includes model loading, phonemization, and WAV encoding. Longer inputs are more efficient due to amortized startup cost. Streaming mode (`--stream`) delivers the first audio chunk in ~200ms.

## Install

```bash
nimble install https://github.com/JK8769/tts.nim
```

Builds native deps, downloads models, and installs `tts_cli`. Requires: Nim 2.0+, CMake, a C compiler. Platform is auto-detected — Apple Silicon builds the MLX backend, everything else builds GGML.

### From source

```bash
git clone --recurse-submodules https://github.com/JK8769/tts.nim
cd tts.nim
nimble install        # or: nimble build_deps && nimble download && nimble build
```

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

English is installed by default. Chinese uses a native Bopomofo phonemizer (no espeak-ng dict needed).

To add more languages for espeak-ng phonemization:

```bash
nimble build_deps                 # build espeak-ng first
nimble lang                       # list installed + available
nimble lang add es fr ja          # add Spanish, French, Japanese
nimble lang remove es             # remove Spanish
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
    tokenizer.nim                 # phoneme string -> token IDs
    audio/
      device.nim                  # miniaudio playback + capture (ring buffer)
      vad.nim                     # energy-based VAD (GGML backend)
      silero_vad.nim              # neural Silero VAD v5 (MLX backend)
      ma_bridge.{c,h}            # C bridge for miniaudio
    stt/
      whisper.nim                 # Whisper STT via whisper.cpp (GGML backend)
      whisper_mlx.nim             # Whisper STT via MLX (Apple Silicon)
      whisper_bridge.{c,h}       # C bridge for whisper.cpp
    mlx/
      mlx.nim                     # high-level MLX tensor API
      mlx_capi.nim                # mlx-c FFI bindings
    ggml/
      ggml_bindings.nim           # ggml C FFI
      gguf_loader.nim             # GGUF model file loader
    models/
      kokoro.nim                  # Kokoro via GGML
      kokoro_mlx.nim              # Kokoro via MLX (quantization-aware)
    phonem/
      phonemizer.nim              # unified phonemizer (routes to espeak/bopomofo)
      espeak.nim                  # espeak-ng static bindings
      bopomofo.nim                # native Chinese phonemizer
vendor/
  mlx-c-src/                      # mlx-c submodule (Apple Silicon only)
  ggml/                           # ggml submodule (Linux/Intel)
  espeak-ng/                      # espeak-ng submodule
  whisper.cpp/                    # whisper.cpp submodule (GGML backend)
  miniaudio/                      # miniaudio single-header audio library
```

## Models

Platform is auto-detected: Apple Silicon gets MLX models, everything else gets GGML.

**MLX (Apple Silicon)** -- 4-bit quantized safetensors:

| Model | Size | Languages | Voices |
|-------|------|-----------|--------|
| `kokoro-mlx-q4` | 131 MB | en | 28 |
| `kokoro-zh-mlx-q4` | 134 MB | zh, en | 103 |

**GGML (Linux / Intel Mac)** -- Q5 quantized GGUF from [TTS.cpp](https://github.com/mmwillet2/TTS.cpp):

| Model | Size | Languages | Voices |
|-------|------|-----------|--------|
| `kokoro-en-q5.gguf` | 191 MB | en | 28 |
| `kokoro-v1.1-zh-q5.gguf` | 198 MB | zh, en | 103 |

Models are downloaded automatically during `nimble install`. Set `TTS_MODEL_DIR` to override the search path.

## License

MIT
