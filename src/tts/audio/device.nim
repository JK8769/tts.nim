## Audio I/O via miniaudio — playback and capture with ring buffers.
## Uses a thin C bridge (ma_bridge.c) to avoid mapping miniaudio's large structs.

import std/os

const bridgeDir = currentSourcePath().parentDir()
const bridgeSrc = bridgeDir / "ma_bridge.c"

{.passC: "-I" & bridgeDir.}
{.compile: bridgeSrc.}
when defined(macosx):
  {.passL: "-framework CoreAudio -framework AudioToolbox -framework CoreFoundation".}
elif defined(linux):
  {.passL: "-lpthread -lm -ldl".}

type
  MaPlaybackObj {.importc: "MaPlayback", header: "ma_bridge.h", incompleteStruct.} = object
  MaPlaybackPtr = ptr MaPlaybackObj
  MaCaptureObj {.importc: "MaCapture", header: "ma_bridge.h", incompleteStruct.} = object
  MaCapturePtr = ptr MaCaptureObj

# ---- Playback FFI ----
proc ma_playback_create(sampleRate: uint32): MaPlaybackPtr {.importc, header: "ma_bridge.h".}
proc ma_playback_write(p: MaPlaybackPtr, samples: ptr float32, frameCount: uint32): uint32 {.importc, header: "ma_bridge.h".}
proc ma_playback_start(p: MaPlaybackPtr): cint {.importc, header: "ma_bridge.h".}
proc ma_playback_stop(p: MaPlaybackPtr): cint {.importc, header: "ma_bridge.h".}
proc ma_playback_is_idle(p: MaPlaybackPtr): cint {.importc, header: "ma_bridge.h".}
proc ma_playback_frames_queued(p: MaPlaybackPtr): uint32 {.importc, header: "ma_bridge.h".}
proc ma_playback_flush(p: MaPlaybackPtr) {.importc, header: "ma_bridge.h".}
proc ma_playback_destroy(p: MaPlaybackPtr) {.importc, header: "ma_bridge.h".}

# ---- Capture FFI ----
proc ma_capture_create(sampleRate: uint32): MaCapturePtr {.importc, header: "ma_bridge.h".}
proc ma_capture_read(c: MaCapturePtr, samples: ptr float32, frameCount: uint32): uint32 {.importc, header: "ma_bridge.h".}
proc ma_capture_start(c: MaCapturePtr): cint {.importc, header: "ma_bridge.h".}
proc ma_capture_stop(c: MaCapturePtr): cint {.importc, header: "ma_bridge.h".}
proc ma_capture_frames_available(c: MaCapturePtr): uint32 {.importc, header: "ma_bridge.h".}
proc ma_capture_destroy(c: MaCapturePtr) {.importc, header: "ma_bridge.h".}

# ---- Nim API ----

type
  AudioPlayback* = ref object
    handle: MaPlaybackPtr
    sampleRate*: uint32

  AudioCapture* = ref object
    handle: MaCapturePtr
    sampleRate*: uint32

proc newAudioPlayback*(sampleRate: uint32 = 24000): AudioPlayback =
  let h = ma_playback_create(sampleRate)
  if h == nil:
    raise newException(IOError, "Failed to create audio playback device")
  result = AudioPlayback(handle: h, sampleRate: sampleRate)

proc write*(p: AudioPlayback, samples: openArray[float32]): int =
  ## Write float32 PCM samples to the playback ring buffer. Returns frames written.
  if samples.len == 0: return 0
  result = ma_playback_write(p.handle, unsafeAddr samples[0], uint32(samples.len)).int

proc writeAll*(p: AudioPlayback, samples: openArray[float32]) =
  ## Write all samples to the ring buffer, blocking if needed.
  var offset = 0
  while offset < samples.len:
    let remaining = samples.len - offset
    let written = ma_playback_write(p.handle, unsafeAddr samples[offset], uint32(remaining)).int
    offset += written
    if written == 0 and offset < samples.len:
      sleep(1)  # ring buffer full, wait for drain

proc start*(p: AudioPlayback) =
  if ma_playback_start(p.handle) != 0:
    raise newException(IOError, "Failed to start audio playback")

proc stop*(p: AudioPlayback) =
  discard ma_playback_stop(p.handle)

proc isIdle*(p: AudioPlayback): bool =
  ma_playback_is_idle(p.handle) != 0

proc framesQueued*(p: AudioPlayback): int =
  ma_playback_frames_queued(p.handle).int

proc flush*(p: AudioPlayback) =
  ## Discard all buffered audio (for interrupt).
  ma_playback_flush(p.handle)

proc close*(p: AudioPlayback) =
  if p.handle != nil:
    ma_playback_destroy(p.handle)
    p.handle = nil

proc play*(p: AudioPlayback, samples: openArray[float32]) =
  ## Convenience: write all samples and start playback.
  p.writeAll(samples)
  p.start()

proc waitUntilDone*(p: AudioPlayback) =
  ## Block until all buffered audio has been played.
  while not p.isIdle:
    sleep(10)

# ---- Capture ----

proc newAudioCapture*(sampleRate: uint32 = 24000): AudioCapture =
  let h = ma_capture_create(sampleRate)
  if h == nil:
    raise newException(IOError, "Failed to create audio capture device")
  result = AudioCapture(handle: h, sampleRate: sampleRate)

proc read*(c: AudioCapture, samples: var seq[float32], maxFrames: int): int =
  ## Read up to maxFrames from capture buffer. Returns frames actually read.
  if samples.len < maxFrames: samples.setLen(maxFrames)
  result = ma_capture_read(c.handle, addr samples[0], uint32(maxFrames)).int

proc start*(c: AudioCapture) =
  if ma_capture_start(c.handle) != 0:
    raise newException(IOError, "Failed to start audio capture")

proc stop*(c: AudioCapture) =
  discard ma_capture_stop(c.handle)

proc framesAvailable*(c: AudioCapture): int =
  ma_capture_frames_available(c.handle).int

proc close*(c: AudioCapture) =
  if c.handle != nil:
    ma_capture_destroy(c.handle)
    c.handle = nil
