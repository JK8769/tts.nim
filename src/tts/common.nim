## Common types for tts.nim

import std/os

const
  # src/tts/common.nim → tts/ → src/ (or installed pkg root)
  pkgRoot* = currentSourcePath().parentDir().parentDir()
  pkgModelDir* = pkgRoot / "res" / "models"
    ## models/ inside the package dir (works for both git clone and nimble install).

proc findModel*(name: string): string =
  ## Find a model file or directory. Searches:
  ##   1. exact/relative path (file or directory)
  ##   2. package models/ dir (nimble pkg or git repo)
  ##   3. TTS_MODEL_DIR env var
  if fileExists(name) or dirExists(name): return name
  let inPkg = pkgModelDir / name
  if fileExists(inPkg) or dirExists(inPkg): return inPkg
  let envDir = getEnv("TTS_MODEL_DIR")
  if envDir.len > 0:
    let p = envDir / name
    if fileExists(p) or dirExists(p): return p
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

proc writeWav*(output: AudioOutput, f: File) =
  ## Write AudioOutput as WAV (16-bit PCM) to an open File handle.
  let numSamples = output.samples.len
  let dataSize = int32(numSamples * 2)
  let fileSize = int32(36 + dataSize)

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

proc writeWav*(output: AudioOutput, path: string) =
  ## Write AudioOutput to a WAV file (16-bit PCM).
  var f = open(path, fmWrite)
  defer: f.close()
  output.writeWav(f)

proc readLE16(f: File): int16 =
  discard f.readBuffer(addr result, 2)

proc readLE32(f: File): int32 =
  discard f.readBuffer(addr result, 4)

proc readWav*(path: string): AudioOutput =
  ## Read a WAV file (16-bit PCM, mono or stereo) into float32 samples.
  var f = open(path, fmRead)
  defer: f.close()
  # RIFF header
  var hdr: array[4, char]
  discard f.readBuffer(addr hdr, 4)
  discard f.readLE32()  # file size
  discard f.readBuffer(addr hdr, 4)  # "WAVE"
  # fmt chunk
  discard f.readBuffer(addr hdr, 4)  # "fmt "
  let fmtSize = f.readLE32()
  discard f.readLE16()  # audio format (1=PCM)
  let channels = f.readLE16()
  let sampleRate = f.readLE32()
  discard f.readLE32()  # byte rate
  discard f.readLE16()  # block align
  let bitsPerSample = f.readLE16()
  if fmtSize > 16:
    f.setFilePos(f.getFilePos() + fmtSize - 16)
  # Find data chunk
  while true:
    discard f.readBuffer(addr hdr, 4)
    let chunkSize = f.readLE32()
    if hdr == ['d', 'a', 't', 'a']:
      let numSamples = chunkSize div (bitsPerSample div 8) div channels
      result.sampleRate = sampleRate
      result.channels = channels.int32
      result.samples = newSeq[float32](numSamples)
      for i in 0..<numSamples:
        var sum: float32 = 0
        for c in 0..<channels:
          let s = f.readLE16()
          sum += s.float32 / 32768.0'f32
        result.samples[i] = sum / channels.float32
      return
    else:
      f.setFilePos(f.getFilePos() + chunkSize)
