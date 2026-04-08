## Voice Activity Detection — energy-based with state machine.
## Detects speech onset/offset from a stream of float32 PCM frames.

import std/math

type
  VadState* = enum
    vsSilence   ## No speech detected
    vsSpeech    ## Speech in progress
    vsTrailing  ## Speech ended, waiting holdoff before confirming silence

  VadConfig* = object
    energyThreshold*: float32   ## RMS energy above this = speech (0.01 is a good start)
    speechPadFrames*: int       ## Frames of silence to keep before speech onset
    holdoffFrames*: int         ## Frames of silence before transitioning speech→silence
    frameSize*: int             ## Samples per analysis frame (e.g. 480 = 20ms at 24kHz)

  VadEvent* = enum
    veNone          ## No state change
    veSpeechStart   ## Transitioned to speech
    veSpeechEnd     ## Transitioned to silence (after holdoff)

  Vad* = ref object
    config*: VadConfig
    state*: VadState
    holdoffCounter: int
    ## Ring buffer for speech padding (keeps pre-speech audio)
    padBuf: seq[float32]
    padPos: int
    padFull: bool

proc defaultVadConfig*(): VadConfig =
  VadConfig(
    energyThreshold: 0.012,
    speechPadFrames: 3,       # ~60ms of pre-speech audio at 20ms frames
    holdoffFrames: 25,        # ~500ms of silence before ending speech
    frameSize: 480,           # 20ms at 24kHz
  )

proc newVad*(config: VadConfig = defaultVadConfig()): Vad =
  let padSize = config.speechPadFrames * config.frameSize
  Vad(
    config: config,
    state: vsSilence,
    holdoffCounter: 0,
    padBuf: newSeq[float32](padSize),
    padPos: 0,
    padFull: false,
  )

proc rmsEnergy(samples: openArray[float32]): float32 =
  if samples.len == 0: return 0.0
  var sum: float32 = 0.0
  for s in samples:
    sum += s * s
  sqrt(sum / samples.len.float32)

proc pushPad(v: Vad, frame: openArray[float32]) =
  ## Store frame in the circular pad buffer.
  let fs = min(frame.len, v.padBuf.len)
  for i in 0..<fs:
    v.padBuf[v.padPos] = frame[i]
    v.padPos = (v.padPos + 1) mod v.padBuf.len
  if v.padPos == 0 or fs >= v.padBuf.len:
    v.padFull = true

proc drainPad*(v: Vad): seq[float32] =
  ## Return the pre-speech padding audio and clear the buffer.
  if v.padBuf.len == 0: return @[]
  if not v.padFull:
    result = v.padBuf[0..<v.padPos]
  else:
    result = newSeq[float32](v.padBuf.len)
    let start = v.padPos
    for i in 0..<v.padBuf.len:
      result[i] = v.padBuf[(start + i) mod v.padBuf.len]
  v.padPos = 0
  v.padFull = false

proc processFrame*(v: Vad, frame: openArray[float32]): VadEvent =
  ## Feed one frame of audio. Returns event if state changed.
  let energy = rmsEnergy(frame)
  let isSpeech = energy > v.config.energyThreshold

  case v.state
  of vsSilence:
    if isSpeech:
      v.state = vsSpeech
      v.holdoffCounter = 0
      return veSpeechStart
    else:
      v.pushPad(frame)
      return veNone

  of vsSpeech:
    if not isSpeech:
      v.state = vsTrailing
      v.holdoffCounter = 1
    return veNone

  of vsTrailing:
    if isSpeech:
      # False alarm — back to speech
      v.state = vsSpeech
      v.holdoffCounter = 0
      return veNone
    else:
      inc v.holdoffCounter
      if v.holdoffCounter >= v.config.holdoffFrames:
        v.state = vsSilence
        v.holdoffCounter = 0
        return veSpeechEnd
      return veNone

proc reset*(v: Vad) =
  v.state = vsSilence
  v.holdoffCounter = 0
  v.padPos = 0
  v.padFull = false
