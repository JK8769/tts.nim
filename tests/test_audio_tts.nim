## Test: synthesize speech and play directly via miniaudio (no WAV file).

import tts/engine
import tts/audio/device

proc main() =
  var e = newTTSEngine()
  e.loadModel("kokoro-en-q5.gguf", "af_heart")

  var player = newAudioPlayback(24000)
  player.start()

  echo "Synthesizing and playing..."
  let audio = e.synthesize("Hello! This is direct audio playback through miniaudio. No temp files needed.", "af_heart", 1.0)
  player.writeAll(audio.samples)
  player.waitUntilDone()

  echo "Done."
  player.close()
  e.close()

main()
