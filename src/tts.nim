## tts — Native TTS engine for Nim
##
## Supports Kokoro (multi-language, 54+ voices) and Orpheus (emotion tags).
## Uses ggml for inference — no Python, no ONNX, just C + Nim.
##
## Quick start:
##   import tts
##   var engine = newTTSEngine()
##   engine.loadModel("kokoro-en-q5.gguf")
##   let audio = engine.synthesize("Hello world!", voice = "af_heart")
##   audio.writeWav("output.wav")
##   engine.close()

import tts/common
import tts/engine

export common, engine
