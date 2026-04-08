## Quick test: play a 440Hz sine wave for 1 second via miniaudio.

import std/math
import tts/audio/device

proc main() =
  echo "Creating playback device (24kHz)..."
  var player = newAudioPlayback(24000)

  # Generate 1 second of 440Hz sine wave
  const sr = 24000
  const dur = 1.0
  let n = int(sr.float * dur)
  var samples = newSeq[float32](n)
  for i in 0..<n:
    samples[i] = 0.3'f32 * sin(2.0'f32 * PI.float32 * 440.0'f32 * i.float32 / sr.float32)

  echo "Playing 440Hz tone for 1s..."
  player.play(samples)
  player.waitUntilDone()

  echo "Done."
  player.close()

main()
