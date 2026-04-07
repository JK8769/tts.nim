# tts.nim

Native text-to-speech engine for Nim. Uses [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) via [ggml](https://github.com/ggerganov/ggml) for inference. No Python, no ONNX, no runtime dependencies -- just C and Nim.

- 54+ voices across 9 languages (English, Chinese, Spanish, French, Hindi, Italian, Japanese, Portuguese, + more via espeak-ng)
- ~200MB quantized models (Q5)
- Metal acceleration on macOS, CPU on Linux
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
tts_cli models                    # list downloaded models
tts_cli download kokoro-en        # download English model
tts_cli download kokoro-zh        # download Chinese model
```

## Install

### From nimble (builds everything automatically)

```bash
nimble install tts
```

Requires: Nim 2.0+, CMake, a C compiler. On macOS, Metal is enabled automatically.

### From source

```bash
git clone --recurse-submodules https://github.com/JK8769/tts.nim
cd tts.nim
nimble build_deps     # build ggml + espeak-ng from vendor source
nimble download       # download default models (~400MB)
nimble build          # build tts_cli
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

Run `tts_cli voices -m <model>` for the full list from a specific model.

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
  tts_cli.nim                     # CLI binary
  tts/
    common.nim                    # AudioOutput, WAV writer, path utils
    engine.nim                    # TTSEngine API
    tokenizer.nim                 # phoneme string -> token IDs
    ggml/
      ggml_bindings.nim           # ggml C FFI
      gguf_loader.nim             # GGUF model file loader
    models/
      kokoro.nim                  # Kokoro model runner
    phonem/
      phonemizer.nim              # unified phonemizer (routes to espeak/bopomofo)
      espeak.nim                  # espeak-ng static bindings
      bopomofo.nim                # native Chinese phonemizer
vendor/
  ggml/                           # ggml submodule (mmwillet fork, support-for-tts)
  espeak-ng/                      # espeak-ng submodule
```

## Models

Models are GGUF files from [TTS.cpp](https://github.com/mmwillet2/TTS.cpp)'s conversions:

| Model | Size | Languages | Voices |
|-------|------|-----------|--------|
| `kokoro-en-q5.gguf` | 191 MB | en | 40+ |
| `kokoro-v1.1-zh-q5.gguf` | 198 MB | zh, en | 10+ |

Models are downloaded from GitHub releases. Set `TTS_MODEL_DIR` to override the search path.

## License

MIT
