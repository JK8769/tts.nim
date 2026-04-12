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
proc ma_playback_set_mute(p: MaPlaybackPtr, mute: cint) {.importc, header: "ma_bridge.h".}
proc ma_playback_destroy(p: MaPlaybackPtr) {.importc, header: "ma_bridge.h".}

# ---- Capture FFI ----
proc ma_capture_create(sampleRate: uint32): MaCapturePtr {.importc, header: "ma_bridge.h".}
proc ma_capture_create_aec(sampleRate: uint32): MaCapturePtr {.importc, header: "ma_bridge.h".}
proc ma_capture_read(c: MaCapturePtr, samples: ptr float32, frameCount: uint32): uint32 {.importc, header: "ma_bridge.h".}
proc ma_capture_start(c: MaCapturePtr): cint {.importc, header: "ma_bridge.h".}
proc ma_capture_stop(c: MaCapturePtr): cint {.importc, header: "ma_bridge.h".}
proc ma_capture_frames_available(c: MaCapturePtr): uint32 {.importc, header: "ma_bridge.h".}
proc ma_capture_flush(c: MaCapturePtr) {.importc, header: "ma_bridge.h".}
proc ma_capture_set_speech_detect(c: MaCapturePtr, threshold: cfloat) {.importc, header: "ma_bridge.h".}
proc ma_capture_set_speech_range(c: MaCapturePtr, low: cfloat, high: cfloat) {.importc, header: "ma_bridge.h".}
proc ma_capture_speech_detected(c: MaCapturePtr): cint {.importc, header: "ma_bridge.h".}
proc ma_capture_speech_frames(c: MaCapturePtr): uint32 {.importc, header: "ma_bridge.h".}
proc ma_capture_speech_reset(c: MaCapturePtr) {.importc, header: "ma_bridge.h".}
proc ma_capture_destroy(c: MaCapturePtr) {.importc, header: "ma_bridge.h".}

# ---- Duplex FFI ----
proc ma_duplex_create_vpio(playbackRate, captureRate: uint32,
    outPlayback: ptr MaPlaybackPtr, outCapture: ptr MaCapturePtr): cint
    {.importc, header: "ma_bridge.h".}

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

proc setMute*(p: AudioPlayback, mute: bool) =
  ## Mux ring buffer output: true=silence (data preserved), false=real audio.
  ma_playback_set_mute(p.handle, if mute: 1 else: 0)

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

proc newAudioCapture*(sampleRate: uint32 = 24000, aec: bool = false): AudioCapture =
  ## Create a capture device. When aec=true on macOS, uses Apple's
  ## VoiceProcessingIO for hardware echo cancellation.
  let h = if aec: ma_capture_create_aec(sampleRate)
          else: ma_capture_create(sampleRate)
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

proc flush*(c: AudioCapture) =
  ## Discard all buffered capture audio.
  ma_capture_flush(c.handle)

proc setSpeechDetect*(c: AudioCapture, threshold: float32 = 0.02) =
  ## Enable energy-based speech detection in the capture thread.
  ## threshold: RMS energy level (0.01-0.05 typical). 0 to disable.
  ma_capture_set_speech_detect(c.handle, threshold)

proc setSpeechRange*(c: AudioCapture, low: float32, high: float32) =
  ## Set speech energy band [low, high]. Only triggers when RMS falls within range.
  ## Rejects quiet noise (below low) and loud transients (above high).
  ma_capture_set_speech_range(c.handle, low, high)

proc speechDetected*(c: AudioCapture): bool =
  ## Returns true if speech was detected since last reset. Lock-free.
  ma_capture_speech_detected(c.handle) != 0

proc speechFrames*(c: AudioCapture): uint32 =
  ## Monotonic count of capture buffers that exceeded the energy threshold.
  ma_capture_speech_frames(c.handle)

proc speechReset*(c: AudioCapture) =
  ## Reset the speech detection flag.
  ma_capture_speech_reset(c.handle)

proc close*(c: AudioCapture) =
  if c.handle != nil:
    ma_capture_destroy(c.handle)
    c.handle = nil

# ---- Duplex ----

proc newAudioDuplex*(playbackRate: uint32 = 24000,
                     captureRate: uint32 = 16000):
    tuple[playback: AudioPlayback, capture: AudioCapture] =
  ## Create paired playback + capture through Apple's VoiceProcessingIO.
  ## Both handles share a single VPIO audio unit for optimal echo cancellation.
  ## On non-macOS, falls back to separate devices with standalone AEC capture.
  var ph: MaPlaybackPtr
  var ch: MaCapturePtr
  if ma_duplex_create_vpio(playbackRate, captureRate, addr ph, addr ch) == 0:
    result.playback = AudioPlayback(handle: ph, sampleRate: playbackRate)
    result.capture = AudioCapture(handle: ch, sampleRate: captureRate)
  else:
    # Fallback: separate devices
    result.playback = newAudioPlayback(playbackRate)
    result.capture = newAudioCapture(captureRate, aec = true)
