## Common types for tts.nim

import std/os

const
  # src/tts/common.nim → tts/ → src/ (or installed pkg root)
  pkgRoot* = currentSourcePath().parentDir().parentDir()
  pkgModelDir* = pkgRoot / "res" / "models"
    ## models/ inside the package dir (works for both git clone and nimble install).

proc findModel*(name: string): string =
  ## Find a model file. Searches:
  ##   1. exact/relative path
  ##   2. package models/ dir (nimble pkg or git repo)
  ##   3. TTS_MODEL_DIR env var
  if fileExists(name): return name
  let inPkg = pkgModelDir / name
  if fileExists(inPkg): return inPkg
  let envDir = getEnv("TTS_MODEL_DIR")
  if envDir.len > 0:
    let p = envDir / name
    if fileExists(p): return p
  return ""

type
  AudioOutput* = object
    samples*: seq[float32]
    sampleRate*: int32
    channels*: int32

proc writeLE16(f: File, v: int16) =
  var val = v
  discard f.writeBuffer(addr val, 2)

proc writeLE32(f: File, v: int32) =
  var val = v
  discard f.writeBuffer(addr val, 4)

proc writeWav*(output: AudioOutput, path: string) =
  ## Write AudioOutput to a WAV file (16-bit PCM)
  let numSamples = output.samples.len
  let dataSize = int32(numSamples * 2)
  let fileSize = int32(36 + dataSize)

  var f = open(path, fmWrite)
  defer: f.close()

  # RIFF header
  f.write("RIFF")
  f.writeLE32(fileSize)
  f.write("WAVE")

  # fmt chunk
  f.write("fmt ")
  f.writeLE32(16)
  f.writeLE16(1)                                    # PCM format
  f.writeLE16(int16(output.channels))
  f.writeLE32(output.sampleRate)
  f.writeLE32(output.sampleRate * int32(output.channels) * 2)  # byte rate
  f.writeLE16(int16(output.channels) * 2)           # block align
  f.writeLE16(16)                                   # bits per sample

  # data chunk
  f.write("data")
  f.writeLE32(dataSize)

  var buf = newSeq[int16](numSamples)
  for i, s in output.samples:
    let clamped = max(-1.0'f32, min(1.0'f32, s))
    buf[i] = int16(clamped * 32767.0'f32)
  if numSamples > 0:
    discard f.writeBuffer(addr buf[0], numSamples * 2)
