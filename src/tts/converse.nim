## Conversation loop — mic capture → VAD → Whisper STT → callback → TTS → speaker.
## Supports barge-in: user speech interrupts agent playback.

import std/[os, strutils]
import audio/device
import engine
import common

when defined(useMlx):
  import stt/whisper_mlx
  import audio/silero_vad
  export silero_vad.VadState, silero_vad.VadEvent
else:
  import audio/vad
  import stt/whisper
  export vad.VadState, vad.VadEvent

const
  TTS_SAMPLE_RATE = 24000
  STT_SAMPLE_RATE = WHISPER_SAMPLE_RATE  # 16000

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
    speed*: float32
    language*: string         ## Whisper language ("en", "zh", "auto")
    when not defined(useMlx):
      vadConfig*: VadConfig
    silenceTimeoutMs*: int    ## How long to wait in silence before ending conversation (0 = never)
    greeting*: string         ## Optional greeting spoken at start
    when defined(useMlx):
      vadModelDir*: string    ## Path to Silero VAD model directory
      vadThreshold*: float32  ## Speech probability threshold

  ConverseLoop* = ref object
    engine: TTSEngine
    recognizer: SpeechRecognizer
    speaker: AudioPlayback
    mic: AudioCapture
    when defined(useMlx):
      vadInst: SileroVad
    else:
      vadInst: Vad
    config: ConverseConfig
    running: bool

proc defaultConverseConfig*(): ConverseConfig =
  result = ConverseConfig(
    voice: "af_heart",
    speed: 1.0,
    language: "en",
    silenceTimeoutMs: 0,
    greeting: "",
  )
  when not defined(useMlx):
    result.vadConfig = defaultVadConfig()
  when defined(useMlx):
    result.vadThreshold = 0.5

proc newConverseLoop*(ttsModel, whisperModel: string,
                      config: ConverseConfig = defaultConverseConfig()): ConverseLoop =
  var e = newTTSEngine()
  e.loadModel(ttsModel, config.voice)
  let rec = newSpeechRecognizer(whisperModel, config.language)
  let speaker = newAudioPlayback(TTS_SAMPLE_RATE.uint32)
  let mic = newAudioCapture(STT_SAMPLE_RATE.uint32)
  when defined(useMlx):
    let v = loadSileroVad(config.vadModelDir, config.vadThreshold)
  else:
    let v = newVad(config.vadConfig)
  ConverseLoop(
    engine: e, recognizer: rec,
    speaker: speaker, mic: mic,
    vadInst: v, config: config, running: false,
  )

proc speak(cl: ConverseLoop, text: string) =
  ## Streaming synthesis: play each sentence chunk as it's generated.
  ## First chunk starts playing while the rest are still being synthesized.
  if text.len == 0: return
  let speaker = cl.speaker
  let cb = proc(chunk: AudioOutput, index, total: int) {.closure.} =
    speaker.writeAll(chunk.samples)
  discard cl.engine.synthesize(text, cl.config.voice, cl.config.speed, cb)

proc bargeIn(cl: ConverseLoop) =
  ## Interrupt agent speech when user starts talking.
  if not cl.speaker.isIdle:
    cl.speaker.flush()

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
  var silenceFrames = 0

  while cl.running:
    # Read a frame from the mic
    let got = cl.mic.read(frame, frameSize)
    if got < frameSize:
      sleep(5)
      continue

    # Mic-mute gate: discard mic input while agent is speaking to prevent echo
    if not cl.speaker.isIdle:
      cl.vadInst.reset()
      speechBuf = @[]
      continue

    let event = cl.vadInst.processFrame(frame[0..<got])

    case event
    of veSpeechStart:
      # Include the pre-speech padding
      let pad = cl.vadInst.drainPad()
      speechBuf = pad & frame[0..<got]

    of veSpeechEnd:
      # User stopped talking — transcribe what they said
      if speechBuf.len > 0:
        let text = cl.recognizer.transcribe(speechBuf).strip()
        speechBuf = @[]
        if text.len > 0:
          silenceFrames = 0
          let turn = ConverseTurn(kind: ctUserSpeech, text: text)
          let response = onTurn(turn)
          if response.len > 0:
            cl.speak(response)

    of veNone:
      # Accumulate speech if currently recording
      if cl.vadInst.state in {vsSpeech, vsTrailing}:
        speechBuf.add frame[0..<got]
      else:
        # In silence — check timeout
        inc silenceFrames
        if cl.config.silenceTimeoutMs > 0:
          let silenceMs = silenceFrames * frameSize * 1000 div STT_SAMPLE_RATE
          if silenceMs >= cl.config.silenceTimeoutMs:
            discard onTurn(ConverseTurn(kind: ctSilenceTimeout, text: ""))
            break

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
