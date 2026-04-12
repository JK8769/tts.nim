## Conversation loop — mic capture → VAD → Whisper STT → callback → TTS → speaker.
## Supports barge-in: user speech interrupts agent playback.

import std/[os, strutils, math, terminal, unicode]
import audio/device
import engine
import common

when defined(useMlx):
  import stt/qwen3_asr
  import audio/silero_vad
  import audio/smart_turn
  export silero_vad.VadState, silero_vad.VadEvent
else:
  import audio/vad
  import stt/whisper
  export vad.VadState, vad.VadEvent

const TTS_SAMPLE_RATE = 24000
when defined(useMlx):
  const STT_SAMPLE_RATE = QWEN3_SAMPLE_RATE  # 16000
else:
  const STT_SAMPLE_RATE = WHISPER_SAMPLE_RATE  # 16000

type
  ConverseTurnKind* = enum
    ctUserSpeech    ## User said something (text in `text`)
    ctSilenceTimeout ## Silence timeout (no speech detected for a while)

  ConverseTurn* = object
    kind*: ConverseTurnKind
    text*: string

  ResponseCallback* = proc(turn: ConverseTurn): string
    ## Called when user finishes speaking. Return text for the agent to speak,
    ## or empty string to stay silent.

  ConverseConfig* = object
    voice*: string
    zhVoice*: string          ## Voice for Chinese TTS (e.g. "zf_001")
    speed*: float32
    language*: string         ## STT language ("en", "zh", "auto")
    when not defined(useMlx):
      vadConfig*: VadConfig
    silenceTimeoutMs*: int    ## How long to wait in silence before ending conversation (0 = never)
    greeting*: string         ## Optional greeting spoken at start
    zhModel*: string          ## Path to Chinese TTS model (auto-switch when text has CJK)
    when defined(useMlx):
      vadModelDir*: string    ## Path to Silero VAD model directory
      vadThreshold*: float32  ## Speech probability threshold
      smartTurnModelDir*: string  ## Path to Smart Turn model directory

  ConverseLoop* = ref object
    engine: TTSEngine
    enModel: string           ## Path to English TTS model (for switching back)
    when defined(useMlx):
      recognizer: Qwen3Asr
    else:
      recognizer: SpeechRecognizer
    speaker: AudioPlayback
    mic: AudioCapture
    when defined(useMlx):
      vadInst: SileroVad
      smartTurnInst: SmartTurn
    else:
      vadInst: Vad
    config: ConverseConfig
    running: bool

proc hasChinese(text: string): bool =
  for r in text.runes:
    let c = r.int
    if (c >= 0x4E00 and c <= 0x9FFF) or
       (c >= 0x3400 and c <= 0x4DBF) or
       (c >= 0x20000 and c <= 0x2A6DF):
      return true

proc stripKana(text: string): string =
  ## Remove Japanese hiragana/katakana from STT output.
  for r in text.runes:
    let c = r.int
    if (c >= 0x3040 and c <= 0x309F) or
       (c >= 0x30A0 and c <= 0x30FF) or
       (c >= 0xFF65 and c <= 0xFF9F):
      continue
    result.add r

proc defaultConverseConfig*(): ConverseConfig =
  result = ConverseConfig(
    voice: "af_heart",
    zhVoice: "zf_001",
    speed: 1.0,
    language: "auto",
    silenceTimeoutMs: 0,
    greeting: "",
  )
  when not defined(useMlx):
    result.vadConfig = defaultVadConfig()
  when defined(useMlx):
    result.vadThreshold = 0.5

proc newConverseLoop*(ttsModel, sttModel: string,
                      config: ConverseConfig = defaultConverseConfig()): ConverseLoop =
  var e = newTTSEngine()
  e.loadModel(ttsModel, config.voice)
  when defined(useMlx):
    let rec = loadQwen3Asr(sttModel)
  else:
    let rec = newSpeechRecognizer(sttModel, config.language)
  let (speaker, mic) = newAudioDuplex(TTS_SAMPLE_RATE.uint32, STT_SAMPLE_RATE.uint32)
  when defined(useMlx):
    let v = loadSileroVad(config.vadModelDir, config.vadThreshold,
                          holdoffChunks = 25, padChunks = 8)
    let st = if config.smartTurnModelDir.len > 0:
               loadSmartTurn(config.smartTurnModelDir)
             else: nil
  else:
    let v = newVad(config.vadConfig)
  let enPath = e.currentModelPath
  when defined(useMlx):
    ConverseLoop(
      engine: e, enModel: enPath, recognizer: rec,
      speaker: speaker, mic: mic,
      vadInst: v, smartTurnInst: st,
      config: config, running: false,
    )
  else:
    ConverseLoop(
      engine: e, enModel: enPath, recognizer: rec,
      speaker: speaker, mic: mic,
      vadInst: v, config: config, running: false,
    )

proc speak(cl: ConverseLoop, text: string) =
  ## Streaming synthesis: play each sentence chunk as it's generated.
  ## Auto-switches between EN/ZH models based on text content.
  if text.len == 0: return
  let voice = if cl.config.zhModel.len > 0 and hasChinese(text):
                cl.engine.loadModel(cl.config.zhModel, cl.config.zhVoice)
                cl.config.zhVoice
              else:
                cl.engine.loadModel(cl.enModel, cl.config.voice)
                cl.config.voice
  let speaker = cl.speaker
  let cb = proc(chunk: AudioOutput, index, total: int) {.closure.} =
    speaker.writeAll(chunk.samples)
  discard cl.engine.synthesize(text, voice, cl.config.speed, cb)

proc bargeIn(cl: ConverseLoop) =
  ## Interrupt agent speech when user starts talking.
  if not cl.speaker.isIdle:
    cl.speaker.flush()

proc rms(samples: openArray[float32]): float32 =
  ## Root mean square of audio samples — used as a simple volume level.
  if samples.len == 0: return 0
  var sum = 0.0
  for s in samples: sum += s * s
  sqrt(sum / samples.len.float).float32

proc levelBar(level: float32, width: int = 20): string =
  ## Render a volume level bar: ▓▓▓▓░░░░░░
  # RMS of speech is typically 0.01-0.1, scale accordingly
  let clamped = min(1.0, level * 5.0)
  let filled = int(clamped * width.float)
  result = "▓".repeat(filled) & "░".repeat(width - filled)

var lastStatusKey = ""

proc showStatus(status: string, level: float32 = 0, duration: float32 = 0) =
  ## Show status updates. Uses a key to avoid spamming duplicate states.
  let bar = if level > 0: " " & levelBar(level) else: ""
  let dur = if duration > 0: " " & duration.formatFloat(ffDecimal, 1) & "s" else: ""
  # Only deduplicate on the status text + rounded duration to avoid spam
  let key = status & (if duration > 0: " " & $(int(duration * 2)) else: "")
  if key == lastStatusKey: return
  lastStatusKey = key
  stderr.writeLine "  " & status & bar & dur

proc clearStatus() =
  lastStatusKey = ""

proc run*(cl: ConverseLoop, onTurn: ResponseCallback) =
  ## Main conversation loop. Blocks until stopped.
  ## onTurn is called each time the user finishes a sentence.
  cl.running = true
  cl.speaker.start()
  cl.mic.start()

  # Optional greeting
  if cl.config.greeting.len > 0:
    cl.speak(cl.config.greeting)

  when defined(useMlx):
    const frameSize = SILERO_CHUNK_SIZE  # 512 samples = 32ms at 16kHz
  else:
    let frameSize = cl.config.vadConfig.frameSize

  var speechBuf: seq[float32] = @[]
  var frame = newSeq[float32](frameSize)
  var accumBuf: seq[float32] = @[]
  var silenceFrames = 0
  var pauseFrames = 0  # frames of silence since Smart Turn paused
  const pauseTimeoutMs = 5000  # force transcription after 5s of silence with pending speech

  stderr.writeLine ""
  showStatus("🎧 Listening...")

  while cl.running:
    # Accumulate mic samples until we have a full VAD frame
    var tmp = newSeq[float32](frameSize)
    let got = cl.mic.read(tmp, frameSize)
    if got > 0:
      accumBuf.add tmp[0..<got]
    if accumBuf.len < frameSize:
      sleep(2)
      continue
    # Extract one frame and keep the remainder
    for j in 0..<frameSize:
      frame[j] = accumBuf[j]
    accumBuf = accumBuf[frameSize..^1]

    let level = rms(frame)

    # AEC handles echo cancellation on macOS (VoiceProcessingIO).
    # No hard mic-mute — barge-in is possible during agent speech.
    let event = cl.vadInst.processFrame(frame)

    case event
    of veSpeechStart:
      # Include the pre-speech padding
      let pad = cl.vadInst.drainPad()
      if speechBuf.len > 0:
        # Resuming after Smart Turn pause — append to existing buffer
        speechBuf.add pad
        speechBuf.add frame[0..<got]
      else:
        speechBuf = pad & frame[0..<got]
      pauseFrames = 0
      showStatus("🎙 Speech", level)

    of veSpeechEnd:
      # User stopped talking — check Smart Turn and min duration before transcribing
      if speechBuf.len > 0:
        let dur = speechBuf.len.float / STT_SAMPLE_RATE.float
        # Minimum 0.8s speech for reliable transcription
        if dur < 0.8:
          showStatus("⏸ Too short, waiting...", 0, dur)
          continue
        when defined(useMlx):
          if cl.smartTurnInst != nil:
            let (turnComplete, prob) = cl.smartTurnInst.predictEndpoint(speechBuf)
            if not turnComplete:
              showStatus("⏸ Paused...", 0, dur)
              continue
        showStatus("⏳ Transcribing...", 0, dur)
        let text = cl.recognizer.transcribe(speechBuf).stripKana().strip()
        speechBuf = @[]
        if text.len > 0:
          clearStatus()
          stderr.writeLine "  You: " & text
          silenceFrames = 0
          let turn = ConverseTurn(kind: ctUserSpeech, text: text)
          let response = onTurn(turn)
          if response.len > 0:
            stderr.writeLine "  Agent: " & response
            cl.speak(response)
          showStatus("🎧 Listening...")
        else:
          showStatus("🎧 Listening...")

    of veNone:
      # Accumulate speech if currently recording
      if cl.vadInst.state in {vsSpeech, vsTrailing}:
        speechBuf.add frame[0..<got]
        pauseFrames = 0
        let dur = speechBuf.len.float / STT_SAMPLE_RATE.float
        showStatus("🎙 Speech", level, dur)
      else:
        # In silence — check if Smart Turn has pending speech to force-transcribe
        if speechBuf.len > 0:
          inc pauseFrames
          let pauseMs = pauseFrames * frameSize * 1000 div STT_SAMPLE_RATE
          if pauseMs >= pauseTimeoutMs:
            let dur = speechBuf.len.float / STT_SAMPLE_RATE.float
            showStatus("⏳ Transcribing...", 0, dur)
            let text = cl.recognizer.transcribe(speechBuf).stripKana().strip()
            speechBuf = @[]
            pauseFrames = 0
            if text.len > 0:
              clearStatus()
              stderr.writeLine "  You: " & text
              silenceFrames = 0
              let turn = ConverseTurn(kind: ctUserSpeech, text: text)
              let response = onTurn(turn)
              if response.len > 0:
                stderr.writeLine "  Agent: " & response
                cl.speak(response)
            showStatus("🎧 Listening...")
        else:
          # Show mic level even in silence
          showStatus("🎧 Listening...", level)
        # In silence — check timeout
        inc silenceFrames
        if cl.config.silenceTimeoutMs > 0:
          let silenceMs = silenceFrames * frameSize * 1000 div STT_SAMPLE_RATE
          if silenceMs >= cl.config.silenceTimeoutMs:
            clearStatus()
            discard onTurn(ConverseTurn(kind: ctSilenceTimeout, text: ""))
            break

  clearStatus()
  cl.mic.stop()
  cl.speaker.waitUntilDone()
  cl.speaker.stop()

proc stop*(cl: ConverseLoop) =
  cl.running = false

proc close*(cl: ConverseLoop) =
  cl.stop()
  cl.mic.close()
  cl.speaker.close()
  cl.recognizer.close()
  cl.engine.close()
