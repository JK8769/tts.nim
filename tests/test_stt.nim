## Round-trip test: TTS → resample to 16kHz → Whisper STT

import std/[math, strutils]
import tts/engine
import tts/stt/whisper

proc resample(src: seq[float32], srcRate, dstRate: int): seq[float32] =
  ## Simple linear interpolation resampler.
  let ratio = srcRate.float / dstRate.float
  let outLen = int(src.len.float / ratio)
  result = newSeq[float32](outLen)
  for i in 0..<outLen:
    let srcPos = i.float * ratio
    let idx = int(srcPos)
    let frac = srcPos - idx.float
    if idx + 1 < src.len:
      result[i] = src[idx] * (1.0'f32 - frac.float32) + src[idx + 1] * frac.float32
    elif idx < src.len:
      result[i] = src[idx]

proc main() =
  echo "Loading TTS engine..."
  var e = newTTSEngine()
  e.loadModel("kokoro-en-q5.gguf", "af_heart")

  echo "Synthesizing speech..."
  let text = "The quick brown fox jumps over the lazy dog."
  let audio = e.synthesize(text, "af_heart", 1.0)
  echo "  TTS: ", audio.samples.len, " samples at ", audio.sampleRate, "Hz"

  # Resample from 24kHz to 16kHz for whisper
  let samples16k = resample(audio.samples, audio.sampleRate.int, WHISPER_SAMPLE_RATE)
  echo "  Resampled: ", samples16k.len, " samples at 16kHz"

  echo "Loading whisper model..."
  var rec = newSpeechRecognizer("ggml-base.en.bin", "en")

  echo "Transcribing..."
  let result = rec.transcribe(samples16k)
  echo "  Input:  \"", text, "\""
  echo "  Output: \"", result.strip(), "\""

  rec.close()
  e.close()
  echo "Done."

main()
