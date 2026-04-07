## Unified TTS engine — loads models and synthesizes speech.
## Currently supports Kokoro. Orpheus coming soon.

import common
import models/kokoro

type
  TTSEngine* = ref object
    kokoro: KokoroModel
    loaded: bool

proc newTTSEngine*(): TTSEngine =
  TTSEngine(loaded: false)

proc loadModel*(e: TTSEngine, path: string, voice: string = "af_heart") =
  ## Load a GGUF model. Path can be absolute, relative, or just a filename
  ## (searched in repo models/ dir, then TTS_MODEL_DIR env).
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
  return e.kokoro.listVoices()

proc synthesize*(e: TTSEngine, text: string, voice: string = "af_heart",
                 speed: float32 = 1.0): AudioOutput =
  if not e.loaded:
    raise newException(ValueError, "No model loaded. Call loadModel() first.")
  e.kokoro.synthesize(text, voice, speed)

proc close*(e: TTSEngine) =
  if e.loaded:
    e.kokoro.close()
    e.loaded = false
