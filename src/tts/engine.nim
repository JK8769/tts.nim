## Unified TTS engine — loads models and synthesizes speech.
## Supports both GGML (.gguf) and MLX (safetensors) backends.

import std/os
import common

when defined(useMlx):
  import models/kokoro_mlx
  export kokoro_mlx.SynthCallback
else:
  import models/kokoro
  export kokoro.SynthCallback

type
  TTSEngine* = ref object
    when defined(useMlx):
      mlxModel: KokoroModel
    else:
      kokoro: KokoroModel
    loaded: bool

proc newTTSEngine*(): TTSEngine =
  TTSEngine(loaded: false)

proc loadModel*(e: TTSEngine, path: string, voice: string = "af_heart") =
  ## Load a model. For MLX: path is a directory with safetensors.
  ## For GGML: path is a .gguf file.
  when defined(useMlx):
    if e.loaded:
      e.mlxModel.close()
    let resolved = if dirExists(path): path
                   else: findModel(path)
    if resolved.len == 0:
      raise newException(IOError, "Model not found: " & path)
    e.mlxModel = loadKokoroMlx(resolved, voice)
    e.loaded = true
  else:
    let resolved = findModel(path)
    if resolved.len == 0:
      raise newException(IOError,
        "Model not found: " & path & "\n" &
        "  Searched: ./, " & pkgModelDir & "/\n" &
        "  Run: tts_cli download kokoro-en")
    if e.loaded:
      e.kokoro.close()
    e.kokoro = loadKokoro(resolved, voice)
    e.kokoro.postLoadInit()
    e.loaded = true

proc isLoaded*(e: TTSEngine): bool = e.loaded

proc listVoices*(e: TTSEngine): seq[string] =
  if not e.loaded: return @[]
  when defined(useMlx):
    return e.mlxModel.listVoices()
  else:
    return e.kokoro.listVoices()

proc synthesize*(e: TTSEngine, text: string, voice: string = "af_heart",
                 speed: float32 = 1.0,
                 callback: SynthCallback = nil): AudioOutput =
  if not e.loaded:
    raise newException(ValueError, "No model loaded. Call loadModel() first.")
  when defined(useMlx):
    e.mlxModel.synthesize(text, voice, speed, callback)
  else:
    e.kokoro.synthesize(text, voice, speed, callback)

proc mixVoice*(e: TTSEngine, voice1, voice2: string,
               weight: float32 = 0.5, name: string = ""): string =
  ## Blend two voices. Returns the mixed voice name.
  if not e.loaded:
    raise newException(ValueError, "No model loaded. Call loadModel() first.")
  when defined(useMlx):
    e.mlxModel.mixVoice(voice1, voice2, weight, name)
  else:
    e.kokoro.mixVoice(voice1, voice2, weight, name)

proc close*(e: TTSEngine) =
  if e.loaded:
    when defined(useMlx):
      e.mlxModel.close()
    else:
      e.kokoro.close()
    e.loaded = false
