## Minimal tts_nim example — synthesize text to WAV.
## Usage: nim c -r examples/hello.nim <model.gguf> [voice] [text]

import std/os
import tts

when isMainModule:
  if paramCount() < 1:
    echo "Usage: hello <model.gguf> [voice] [text]"
    echo "  voice: af_heart, af_maple, zf_001, etc."
    echo "  text:  any text to synthesize"
    quit(1)

  let modelPath = paramStr(1)
  let voice = if paramCount() >= 2: paramStr(2) else: "af_heart"
  let text = if paramCount() >= 3: paramStr(3) else: "Hello! This is tts nim."

  var engine = newTTSEngine()
  engine.loadModel(modelPath)
  echo "Loaded model: ", modelPath
  echo "Voices: ", engine.listVoices().len

  let audio = engine.synthesize(text, voice)
  audio.writeWav("output.wav")
  echo "Audio: ", audio.samples.len, " samples @ ", audio.sampleRate, " Hz"
  echo "Saved: output.wav"
  engine.close()
