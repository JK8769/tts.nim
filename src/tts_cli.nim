## tts_cli — Command-line TTS synthesis and MCP server.
## Install: nimble install tts
## Usage:
##   tts_cli synth "Hello world" -v af_heart -m kokoro-en -o output.wav
##   tts_cli voices -m kokoro-en
##   tts_cli download kokoro-en
##   tts_cli serve       # MCP stdio server for AI agents
##   tts_cli models

import std/[os, strutils, strformat, sequtils, json, algorithm, terminal, times, posix, math, osproc]
import tts/common
import tts/engine
import tts/audio/device
when defined(useMlx):
  import tts/stt/qwen3_asr
  import tts/audio/silero_vad
else:
  import tts/audio/vad
  import tts/stt/whisper
import tts/converse
import docopt

const Doc = """
tts_cli — Native TTS engine for Nim.

Usage:
  tts_cli synth <text>... [-m <model>] [-v <voice>] [-s <speed>] [-o <output>] [--json] [--stream]
  tts_cli batch <input> [-m <model>] [-v <voice>] [-s <speed>] [-d <dir>] [--json]
  tts_cli voices [-m <model>] [--male] [--female] [--en] [--zh] [--json]
  tts_cli download [<name>]
  tts_cli models [--json]
  tts_cli converse [-m <model>] [-v <voice>] [-s <speed>] [--whisper <wmodel>] [--lang <lang>] [--greeting <text>]
  tts_cli serve
  tts_cli mcp [--print]
  tts_cli schema [--per-command]
  tts_cli (-h | --help)

Options:
  -m, --model <model>    Model file or shorthand (kokoro-en) [default: kokoro-en].
  -v, --voice <voice>    Voice name, or mix: voice1+voice2:weight [default: af_heart].
  -s, --speed <speed>    Speed multiplier [default: 1.0].
  -o, --output <output>  Output WAV file, or - for stdout/raw PCM [default: output.wav].
  --stream               Stream raw PCM (s16le, 24kHz, mono) to stdout per sentence.
  -d, --dir <dir>        Output directory for batch mode [default: .].
  --male                 Show only male voices.
  --female               Show only female voices.
  --en                   Show only English voices.
  --zh                   Show only Chinese voices.
  --json                 Output as JSON (for agent/programmatic use).
  --per-command           Output per-command schema.
  --print                 Print MCP config JSON to stdout instead of writing .mcp.json.
  --whisper <wmodel>     Whisper model for STT [default: auto].
  --lang <lang>          Language for speech recognition [default: auto].
  --greeting <text>      Greeting spoken at conversation start.
  -h, --help             Show this help.
"""

when defined(useMlx):
  const Models = {
    "kokoro-en": "kokoro-mlx-q4",
    "kokoro-zh": "kokoro-zh-mlx-q4",
  }
  const DefaultQwen3Asr = "qwen3-asr-0.6b-4bit"
  const STT_RATE = QWEN3_SAMPLE_RATE
else:
  const Models = {
    "kokoro-en": "kokoro-en-q5.gguf",
    "kokoro-zh": "kokoro-v1.1-zh-q5.gguf",
  }
  const DefaultWhisper = "ggml-base.en.bin"
  const STT_RATE = WHISPER_SAMPLE_RATE
const Repo = "JK8769/tts.nim"

proc listModels(asJson: bool) =
  if dirExists(pkgModelDir):
    var jsonArr = newJArray()
    var count = 0
    for f in walkDir(pkgModelDir):
      when defined(useMlx):
        # MLX models are directories with safetensors
        if f.kind == pcDir:
          let name = extractFilename(f.path)
          var voiceCount = 0
          try:
            var e = newTTSEngine()
            e.loadModel(f.path, "af_heart")
            voiceCount = e.listVoices().len
            e.close()
          except: discard
          if voiceCount > 0:
            inc count
            if asJson:
              jsonArr.add %*{"model": name, "path": f.path, "voices": voiceCount}
            else:
              echo "  ", name, "  (", voiceCount, " voices)"
      else:
        if f.path.endsWith(".gguf"):
          inc count
          let name = extractFilename(f.path)
          let bytes = getFileSize(f.path)
          let mb = bytes div (1024 * 1024)
          var voiceCount = 0
          try:
            var e = newTTSEngine()
            e.loadModel(f.path, "af_heart")
            voiceCount = e.listVoices().len
            e.close()
          except: discard
          if asJson:
            jsonArr.add %*{"model": name, "path": f.path,
                           "size_bytes": bytes, "voices": voiceCount}
          else:
            echo "  ", name, "  (", mb, " MB, ", voiceCount, " voices)"
    if asJson:
      echo jsonArr.pretty
    elif count == 0:
      echo "  (empty — run: tts_cli download kokoro-en)"
  else:
    if asJson:
      echo newJArray().pretty
    else:
      echo "Models dir: ", pkgModelDir
      echo "  (empty — run: tts_cli download kokoro-en)"

proc downloadModel(name: string) =
  createDir(pkgModelDir)
  if name.len == 0:
    echo "Available: kokoro-en, kokoro-zh"
    return
  var file = ""
  for (k, v) in Models:
    if k == name: file = v
  if file.len == 0:
    echo "Unknown model: ", name, " (available: kokoro-en, kokoro-zh)"
    quit(1)
  let dest = pkgModelDir / file
  if fileExists(dest):
    echo name, " ✓ ", dest
    return
  let url = "https://github.com/" & Repo & "/releases/latest/download/" & file
  echo "Downloading ", name, " → ", dest
  let code = execShellCmd("curl -L --progress-bar --fail -o " &
                          quoteShell(dest) & " " & quoteShell(url))
  if code != 0:
    removeFile(dest)
    echo "Failed. Download manually from: https://github.com/", Repo, "/releases"
    quit(1)

proc resolveModel(name: string): string =
  ## Resolve shorthand like "kokoro-zh" to full filename, or pass through as-is.
  for (k, v) in Models:
    if k == name: return v
  return name

proc printColumns(items: seq[string], indent = 2) =
  ## Print items in aligned columns that fit the terminal width.
  if items.len == 0: return
  let width = try: terminalWidth() except: 80
  let maxItem = items.mapIt(it.len).max
  let colW = maxItem + 2
  let cols = max(1, (width - indent) div colW)
  let pad = " ".repeat(indent)
  for i, item in items:
    if i mod cols == 0 and i > 0: echo ""
    if i mod cols == 0: stdout.write pad
    stdout.write item.alignLeft(colW)
  echo ""

proc voiceLang(v: string): string =
  if v.len >= 1:
    case v[0]
    of 'a', 'b': return "en"
    of 'z': return "zh"
    else: discard
  return "unknown"

proc voiceGender(v: string): string =
  if v.len >= 2:
    case v[1]
    of 'f': return "female"
    of 'm': return "male"
    else: discard
  return "unknown"

proc parseVoice(spec: string): tuple[voice1, voice2: string, weight: float32, isMix: bool] =
  ## Parse voice spec: "af_heart" or "af_heart+am_adam:0.3"
  let plusIdx = spec.find('+')
  if plusIdx < 0:
    return (spec, "", 0.0'f32, false)
  let v1 = spec[0..<plusIdx]
  let rest = spec[plusIdx + 1 .. ^1]
  let colonIdx = rest.find(':')
  if colonIdx < 0:
    return (v1, rest, 0.5'f32, true)
  let v2 = rest[0..<colonIdx]
  let w = parseFloat(rest[colonIdx + 1 .. ^1]).float32
  return (v1, v2, w, true)

proc die(msg: string, asJson: bool, code = 1) =
  if asJson:
    stderr.writeLine((%*{"error": msg}).pretty)
  else:
    stderr.writeLine("Error: ", msg)
  quit(code)

proc matchVoice(v: string, wantMale, wantFemale, wantEn, wantZh: bool): bool =
  let hasFilter = wantMale or wantFemale or wantEn or wantZh
  if v.len < 3 or v[1] notin {'f', 'm'}: return not hasFilter
  if wantMale and v[1] != 'm': return false
  if wantFemale and v[1] != 'f': return false
  if wantEn and v[0] notin {'a', 'b'}: return false
  if wantZh and v[0] != 'z': return false
  return true

when isMainModule:
  let args = docopt(Doc)

  let jsonOut = args["--json"]

  if args.isCommand("synth"):
    let rawText = @(args["<text>"]).join(" ")
    let text = if rawText == "-": stdin.readAll().strip() else: rawText
    if text.len == 0: die("no text provided", jsonOut)
    let model = resolveModel($args["--model"])
    let voiceSpec = $args["--voice"]
    let speed = parseFloat($args["--speed"]).float32
    let output = $args["--output"]
    let stream = args["--stream"]
    var e = newTTSEngine()
    try:
      e.loadModel(model, voiceSpec)
    except CatchableError as ex:
      die("failed to load model '" & model & "': " & ex.msg, jsonOut)
    # Parse voice mix syntax: "af_heart+am_adam:0.3"
    let vp = parseVoice(voiceSpec)
    var voice = vp.voice1
    if vp.isMix:
      let allVoices = e.listVoices()
      if vp.voice1 notin allVoices:
        die("voice '" & vp.voice1 & "' not found", jsonOut)
      if vp.voice2 notin allVoices:
        die("voice '" & vp.voice2 & "' not found", jsonOut)
      voice = e.mixVoice(vp.voice1, vp.voice2, vp.weight)
    else:
      let allVoices = e.listVoices()
      if voice notin allVoices:
        die("voice '" & voice & "' not found in model '" & model & "'", jsonOut)
    # Streaming: write raw PCM (s16le) to stdout per sentence
    if stream:
      var totalSamples = 0
      let cb = proc(chunk: AudioOutput, index, total: int) {.gcsafe.} =
        var buf = newSeq[int16](chunk.samples.len)
        for i, s in chunk.samples:
          let clamped = max(-1.0'f32, min(1.0'f32, s))
          buf[i] = int16(clamped * 32767.0'f32)
        if buf.len > 0:
          discard stdout.writeBuffer(addr buf[0], buf.len * 2)
          stdout.flushFile()
        totalSamples += chunk.samples.len
      discard e.synthesize(text, voice, speed, cb)
      if jsonOut:
        stderr.writeLine((%*{
          "sample_rate": 24000, "channels": 1, "format": "s16le",
          "samples": totalSamples,
          "duration": (totalSamples.float / 24000.0).formatFloat(ffDecimal, 2).parseFloat,
        }).pretty)
    else:
      let audio = e.synthesize(text, voice, speed)
      if output == "-":
        audio.writeWav(stdout)
      else:
        audio.writeWav(output)
      let dur = audio.samples.len.float / audio.sampleRate.float
      if output != "-":
        if jsonOut:
          echo (%*{
            "output": output,
            "duration": dur.formatFloat(ffDecimal, 2).parseFloat,
            "sample_rate": audio.sampleRate,
            "samples": audio.samples.len,
            "voice": voiceSpec,
            "model": model,
            "speed": speed,
            "size_bytes": getFileSize(output),
          }).pretty
        else:
          echo "Audio: ", dur.formatFloat(ffDecimal, 1), "s → ", output
    e.close()

  elif args.isCommand("voices"):
    let wantMale = args["--male"]
    let wantFemale = args["--female"]
    let wantEn = args["--en"]
    let wantZh = args["--zh"]
    let rawModel = $args["--model"]
    let model = resolveModel(rawModel)
    let filterModel = rawModel != "kokoro-en"
    if dirExists(pkgModelDir):
      var jsonModels = newJArray()
      for f in walkDir(pkgModelDir):
        when defined(useMlx):
          if f.kind != pcDir: continue
        else:
          if not f.path.endsWith(".gguf"): continue
        let name = extractFilename(f.path)
        if filterModel and name != model: continue
        var e = newTTSEngine()
        e.loadModel(f.path, "af_heart")
        let allVoices = e.listVoices()
        var filtered: seq[string]
        for v in allVoices:
          if matchVoice(v, wantMale, wantFemale, wantEn, wantZh):
            filtered.add v
        filtered.sort()
        if filtered.len > 0:
          if jsonOut:
            var jVoices = newJArray()
            for v in filtered:
              jVoices.add %*{"name": v, "language": voiceLang(v), "gender": voiceGender(v)}
            jsonModels.add %*{"model": name, "voices": jVoices}
          else:
            echo name, " (", filtered.len, " voices):"
            printColumns(filtered)
        e.close()
      if jsonOut:
        echo jsonModels.pretty
    else:
      echo "No models found. Run: tts_cli download kokoro-en"

  elif args.isCommand("batch"):
    let inputFile = $args["<input>"]
    let model = resolveModel($args["--model"])
    let voiceSpec = $args["--voice"]
    let speed = parseFloat($args["--speed"]).float32
    let outDir = $args["--dir"]
    if not fileExists(inputFile): die("file not found: " & inputFile, jsonOut)
    createDir(outDir)
    var e = newTTSEngine()
    try:
      e.loadModel(model, voiceSpec)
    except CatchableError as ex:
      die("failed to load model '" & model & "': " & ex.msg, jsonOut)
    let vp = parseVoice(voiceSpec)
    var voice = vp.voice1
    if vp.isMix:
      voice = e.mixVoice(vp.voice1, vp.voice2, vp.weight)
    let lines = readFile(inputFile).strip().splitLines()
    var results = newJArray()
    var count = 0
    for i, line in lines:
      let text = line.strip()
      if text.len == 0: continue
      inc count
      let outFile = outDir / &"{i+1:04d}.wav"
      let audio = e.synthesize(text, voice, speed)
      audio.writeWav(outFile)
      let dur = audio.samples.len.float / audio.sampleRate.float
      if jsonOut:
        results.add %*{"index": i+1, "output": outFile,
                        "duration": dur.formatFloat(ffDecimal, 2).parseFloat,
                        "text": text}
      else:
        echo &"[{i+1}/{lines.len}] {dur.formatFloat(ffDecimal, 1)}s → {outFile}"
    if jsonOut:
      echo results.pretty
    else:
      echo "Done: ", count, " files in ", outDir
    e.close()

  elif args.isCommand("download"):
    downloadModel(if args["<name>"]: $args["<name>"] else: "")

  elif args.isCommand("models"):
    if not jsonOut: echo "Models dir: ", pkgModelDir
    listModels(jsonOut)

  elif args.isCommand("converse"):
    let model = resolveModel($args["--model"])
    let voiceSpec = $args["--voice"]
    let speed = parseFloat($args["--speed"]).float32
    when defined(useMlx):
      let sttRaw = if args["--whisper"]: $args["--whisper"] else: DefaultQwen3Asr
      let sttModel = if sttRaw == "auto": findModel(DefaultQwen3Asr)
                     else: findModel(sttRaw)
    else:
      let sttRaw = $args["--whisper"]
      let sttModel = if sttRaw == "auto": findModel(DefaultWhisper)
                     else: findModel(sttRaw)
    if sttModel.len == 0:
      die("STT model not found: " & sttRaw & " (run: nimble download)", false)
    let lang = $args["--lang"]
    let greeting = if args["--greeting"]: $args["--greeting"] else: ""
    var config = defaultConverseConfig()
    config.voice = voiceSpec
    config.speed = speed
    config.language = lang
    config.greeting = greeting
    when defined(useMlx):
      # Auto-detect Chinese model for bilingual TTS
      let zhModelPath = findModel(resolveModel("kokoro-zh"))
      if zhModelPath.len > 0:
        config.zhModel = zhModelPath
    echo "Starting conversation (speak to begin, Ctrl+C to exit)..."
    echo "  TTS model: ", model, " | Voice: ", voiceSpec
    echo "  STT model: ", sttModel, " | Language: ", lang
    var cl = newConverseLoop(model, sttModel, config)
    cl.run(proc(turn: ConverseTurn): string =
      if turn.kind == ctUserSpeech:
        # Echo mode — repeat back what was said (replace with LLM later)
        return "You said: " & turn.text
      return ""
    )
    cl.close()

  elif args.isCommand("serve"):
    # MCP stdio server — JSON-RPC 2.0 over stdin/stdout (newline-delimited)

    # Kill any stale tts_cli processes (old servers, lingering converse sessions)
    let myPid = getCurrentProcessId()
    try:
      let pgrepOut = execProcess("pgrep -f \"tts_cli (serve|converse)\"").strip()
      for line in pgrepOut.splitLines():
        let pid = line.strip()
        if pid.len > 0 and pid != $myPid:
          discard execProcess("kill " & pid)
          stderr.writeLine "Killed stale tts_cli process: ", pid
    except: discard

    var e = newTTSEngine()
    # Model loaded lazily on first synth/speak call for fast MCP startup

    # Audio constants shared across listen/converse tools
    const CHUNK = 512                  # mic read chunk size (samples)
    const SILENCE_HOLDOFF_MS = 800     # ms of silence to trigger speech-end
    const PRE_ROLL_MAX = 3200          # 200ms at 16kHz — captures word onsets
    const CONTINUATION_WINDOW = 1.5    # seconds to wait for follow-up speech
    const MIN_SPEECH_SAMPLES = 1600    # 100ms at 16kHz — ignore shorter buffers as noise
    const VERIFY_HIT_RATIO = 0.25     # barge-in: fraction of reads above threshold to confirm
    const VAD_SPEECH_RATIO = 0.3      # barge-in: fraction of VAD chunks that must be speech
    const VERIFY_DURATION_1 = 2.0     # first barge-in verify window (seconds)
    const VERIFY_DURATION_2 = 4.0     # second (escalated) verify window
    const BARGE_IN_GRACE = 2.0        # AEC stabilization / post-resume cooldown (seconds)
    const MAX_SPEECH_BUF = 16000 * 30 # 30s at 16kHz — cap cvSpeechBuf

    # Shared VPIO duplex pair (speaker + mic through one audio unit for AEC)
    var speaker: AudioPlayback = nil
    var mic: AudioCapture = nil
    var duplexReady = false

    proc ensureDuplex() =
      if not duplexReady:
        let (s, m) = newAudioDuplex(24000, STT_RATE.uint32)
        speaker = s
        mic = m
        speaker.start()
        duplexReady = true

    # Cached STT for listen tool — avoid reloading on every call
    when defined(useMlx):
      var cachedQwen3: Qwen3Asr
      var cachedVad: SileroVad
    else:
      var cachedWhisper: SpeechRecognizer
    var cachedRecLoaded = false
    var cachedRecLang = ""

    # Full-duplex converse session state
    var cvActive = false
    var cvVoice = "af_heart"
    var cvSpeed: float32 = 1.0
    var cvSpeechBuf: seq[float32] = @[]

    # Last send state for recv
    var cvLastVoice = ""
    var cvLastSamples = 0
    var cvBargedIn = false

    const SttArtifacts = ["[BLANK_AUDIO]", "[blank_audio]", "*sigh*",
      "(silence)", "[silence]", "[music]", "[applause]", "[laughter]"]

    proc extractArtifacts(text: string): tuple[clean: string, artifacts: seq[string]] =
      ## Strip STT artifacts (hallucinated tags) from transcribed text.
      result.clean = text
      for artifact in SttArtifacts:
        if artifact in result.clean:
          result.artifacts.add artifact
          result.clean = result.clean.replace(artifact, "")
      result.clean = result.clean.strip()

    proc sttTranscribe(samples: openArray[float32]): string =
      when defined(useMlx):
        return cachedQwen3.transcribe(samples)
      else:
        return cachedWhisper.transcribe(samples)

    proc ensureStt(language: string = "auto") =
      when defined(useMlx):
        if not cachedRecLoaded or cachedRecLang != language:
          if cachedRecLoaded and cachedQwen3 != nil:
            cachedQwen3.close()
          let asrModel = findModel(DefaultQwen3Asr)
          cachedQwen3 = loadQwen3Asr(asrModel)
          cachedRecLoaded = true
          cachedRecLang = language
      else:
        if not cachedRecLoaded or cachedRecLang != language:
          if cachedRecLoaded:
            cachedWhisper.close()
          let whisperModel = findModel(DefaultWhisper)
          cachedWhisper = newSpeechRecognizer(whisperModel, language)
          cachedRecLoaded = true
          cachedRecLang = language

    var cvEnergyThreshold: float32 = 0.02

    const calibrationPath = pkgRoot / "res" / "calibration.json"

    proc saveCalibration() =
      ## Persist calibration threshold to disk.
      let j = %*{"energy_threshold": cvEnergyThreshold}
      writeFile(calibrationPath, $j)
      stderr.writeLine "🎚 calibration saved to " & calibrationPath

    proc loadCalibration(): bool =
      ## Load saved calibration. Returns true if loaded.
      if not fileExists(calibrationPath): return false
      try:
        let j = parseJson(readFile(calibrationPath))
        cvEnergyThreshold = j["energy_threshold"].getFloat().float32
        stderr.writeLine "🎚 loaded calibration: threshold=" &
          cvEnergyThreshold.formatFloat(ffDecimal, 4)
        return true
      except:
        return false

    proc applySpeechDetect() =
      ## Apply energy threshold to mic. Just a low gate — VAD does the real work.
      mic.setSpeechDetect(cvEnergyThreshold)

    when defined(useMlx):
      proc ensureVad() =
        if cachedVad == nil:
          let vadDir = findModel("silero-vad")
          cachedVad = loadSileroVad(vadDir)

    # ---- MCP I/O ----

    proc mcpSend(j: JsonNode) =
      let s = $j
      try:
        stdout.write(s & "\n")
        stdout.flushFile()
      except IOError:
        quit(0)

    proc mcpRecv(): JsonNode =
      var line: string
      while true:
        try:
          line = stdin.readLine().strip()
        except EOFError:
          return nil
        except IOError:
          return nil
        if line.len > 0:
          return parseJson(line)

    proc stdinReady(): bool =
      var fds: array[1, TPollfd]
      fds[0].fd = cint(0)
      fds[0].events = POLLIN
      return poll(addr fds[0], 1, 0) > 0

    proc mcpResult(id: JsonNode, res: JsonNode): JsonNode =
      %*{"jsonrpc": "2.0", "id": id, "result": res}

    proc mcpError(id: JsonNode, code: int, msg: string): JsonNode =
      %*{"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": msg}}

    var logFile: File
    discard logFile.open("/tmp/tts_mcp.log", fmAppend)

    proc mcpLog(level: string, msg: string, event: string = "", data: JsonNode = nil) =
      let ts = now().format("HH:mm:ss'.'fff")
      stderr.writeLine "[" & ts & "] " & msg
      if logFile != nil:
        var j = %*{"ts": ts, "level": level, "msg": msg}
        if event.len > 0: j["event"] = %event
        if data != nil: j["data"] = data
        logFile.writeLine $j
        logFile.flushFile()
      mcpSend(%*{"jsonrpc": "2.0", "method": "notifications/message",
                 "params": {"level": level, "logger": "tts",
                            "data": {"message": msg}}})

    # ---- Helpers ----

    proc stopSpeaking(): bool =
      if speaker != nil and not speaker.isIdle:
        speaker.flush()
        return true
      return false

    proc drainMic(buf: var seq[float32]) =
      ## Read all available mic data into buf, capped at MAX_SPEECH_BUF.
      var tmp = newSeq[float32](CHUNK)
      var got = mic.read(tmp, CHUNK)
      while got > 0 and buf.len < MAX_SPEECH_BUF:
        buf.add tmp[0..<got]
        got = mic.read(tmp, CHUNK)

    type VerifyResult = object
      confirmed: bool
      speechBuf: seq[float32]
      hits, reads: int

    proc vadCheckSpeech(samples: openArray[float32]): tuple[speechChunks, totalChunks: int] =
      ## Run Silero VAD on accumulated audio. Returns speech/total chunk counts.
      when defined(useMlx):
        if cachedVad == nil: return (0, 0)
        cachedVad.reset()
        var i = 0
        while i + SILERO_CHUNK_SIZE <= samples.len:
          let prob = cachedVad.processChunk(samples[i ..< i + SILERO_CHUNK_SIZE])
          result.totalChunks += 1
          if prob > cachedVad.threshold:
            result.speechChunks += 1
          i += SILERO_CHUNK_SIZE

    const VAD_QUICK_CHECK_MS = 500  # quick mute duration for VAD gate (ms)

    proc vadQuickCheck(): bool =
      ## Brief mute + VAD + energy to distinguish speech from noise.
      ## Mutes for 500ms, requires both VAD majority AND sustained energy.
      speaker.setMute(true)
      mic.flush()  # discard AEC-contaminated audio
      let startFrames = mic.speechFrames()
      var buf: seq[float32] = @[]
      var tmp = newSeq[float32](CHUNK)
      let deadline = epochTime() + VAD_QUICK_CHECK_MS.float / 1000.0
      while epochTime() < deadline:
        let got = mic.read(tmp, CHUNK)
        if got > 0:
          buf.add tmp[0..<got]
        else:
          sleep(5)
      let energyHits = int(mic.speechFrames() - startFrames)
      if buf.len < SILERO_CHUNK_SIZE or energyHits < 3:
        speaker.setMute(false)
        mcpLog("info", "quick-vad: energy=" & $energyHits & " — skipped",
          "quick_vad", %*{"energy": energyHits, "result": "skipped"})
        return false
      let (speechChunks, totalChunks) = vadCheckSpeech(buf)
      let minChunks = max(2, int(totalChunks.float * VAD_SPEECH_RATIO))
      result = speechChunks >= minChunks
      if not result:
        speaker.setMute(false)  # not speech — unmute immediately
      let verdict = if result: "speech" else: "noise"
      mcpLog("info", "quick-vad: vad=" & $speechChunks & "/" & $totalChunks &
        " energy=" & $energyHits & " → " & verdict,
        "quick_vad", %*{"vad": speechChunks, "vad_total": totalChunks,
          "energy": energyHits, "samples": buf.len, "result": verdict})

    proc verifyBargeIn(duration: float): VerifyResult =
      ## Mute speaker and listen for sustained speech over `duration` seconds.
      ## Gate 1: C-level energy counter. Gate 2: Silero neural VAD.
      speaker.setMute(true)
      var tmp = newSeq[float32](CHUNK)
      let startFrames = mic.speechFrames()
      let confirmDeadline = epochTime() + duration
      while epochTime() < confirmDeadline:
        let got = mic.read(tmp, CHUNK)
        if got > 0:
          result.reads += 1
          result.speechBuf.add tmp[0..<got]
        else:
          sleep(10)
      result.hits = int(mic.speechFrames() - startFrames)
      let minHits = max(3, int(result.reads.float * VERIFY_HIT_RATIO))
      if result.hits < minHits:
        mcpLog("info", "verify: energy=" & $result.hits & "/" & $minHits & " — rejected",
          "verify_bargein", %*{"energy": result.hits, "energy_min": minHits, "result": "rejected"})
        return
      # Gate 2: neural VAD on accumulated audio
      let (speechChunks, totalChunks) = vadCheckSpeech(result.speechBuf)
      let minVad = max(2, int(totalChunks.float * VAD_SPEECH_RATIO))
      result.confirmed = speechChunks >= minVad
      let vResult = if result.confirmed: "confirmed" else: "rejected"
      mcpLog("info", "verify: energy=" & $result.hits & "/" & $minHits &
        " vad=" & $speechChunks & "/" & $minVad & " → " & vResult,
        "verify_bargein", %*{"energy": result.hits, "energy_min": minHits,
          "vad": speechChunks, "vad_min": minVad, "vad_total": totalChunks,
          "result": vResult})

    type ListenResult = object
      parts: seq[string]
      bargedIn: bool
      speechSamples: int
      timedOut: bool

    proc listenMic(silenceMs: int = SILENCE_HOLDOFF_MS,
                   timeoutMs: int = 30000,
                   initialTimeoutMs: int = 0,
                   detectBargeIn: bool = false,
                   carryBuf: var seq[float32]): ListenResult =
      ## Core mic listening loop: pre-roll, energy detection, silence holdoff, transcription.
      ## If detectBargeIn, flushes speaker on speech start during playback.
      ## carryBuf: speech already captured (e.g. from barge-in during synthesis).
      ## initialTimeoutMs: shorter deadline for first speech onset (0 = use timeoutMs).
      var readBuf = newSeq[float32](CHUNK)
      var preRoll: seq[float32] = @[]
      var silenceStart = 0.0
      var continuationDeadline = 0.0
      let initMs = if initialTimeoutMs > 0: initialTimeoutMs else: timeoutMs
      var deadline = epochTime() + initMs.float / 1000.0
      var inSpeech = carryBuf.len > 0
      var playbackDone = not detectBargeIn or speaker == nil or speaker.isIdle

      # Don't reset speech state if we already have carry buffer (barge-in audio)
      if carryBuf.len == 0:
        mic.speechReset()
      applySpeechDetect()

      if detectBargeIn and not playbackDone:
        mcpLog("info", "playing...", "playback_start")

      block loop:
        while epochTime() < deadline:
          if continuationDeadline > 0 and epochTime() > continuationDeadline:
            break loop
          if detectBargeIn and stdinReady():
            break loop

          # Track playback drain
          if not playbackDone and speaker != nil and speaker.isIdle:
            playbackDone = true
            if detectBargeIn:
              mcpLog("info", "playback finished", "playback_done")

          let got = mic.read(readBuf, CHUNK)
          if got > 0:
            if inSpeech:
              carryBuf.add readBuf[0..<got]
            else:
              preRoll.add readBuf[0..<got]
              if preRoll.len > PRE_ROLL_MAX:
                preRoll = preRoll[preRoll.len - PRE_ROLL_MAX .. ^1]

            if mic.speechDetected():
              mic.speechReset()
              if not inSpeech:
                if not playbackDone and speaker != nil and not speaker.isIdle:
                  if vadQuickCheck():
                    mcpLog("info", "speech? verifying...", "bargein_check")
                    let vr = verifyBargeIn(VERIFY_DURATION_1)
                    if vr.confirmed:
                      inSpeech = true
                      carryBuf = preRoll & vr.speechBuf
                      preRoll = @[]
                      result.bargedIn = true
                      speaker.flush()
                      playbackDone = true
                      # Extend to full timeout from speech start
                      deadline = epochTime() + timeoutMs.float / 1000.0
                      mcpLog("info", "barge-in confirmed — flushing", "bargein_confirmed")
                    else:
                      speaker.setMute(false)
                      mcpLog("info", "false alarm — unmuted", "bargein_rejected")
                else:
                  inSpeech = true
                  carryBuf = preRoll & readBuf[0..<got]
                  preRoll = @[]
                  # Extend to full timeout from speech start
                  deadline = epochTime() + timeoutMs.float / 1000.0
                  if detectBargeIn:
                    mcpLog("info", "speech detected", "speech_start")
              else:
                # During active speech: sliding 10s window to avoid cutting off long utterances
                let minDeadline = epochTime() + 10.0
                if minDeadline > deadline:
                  deadline = minDeadline
              silenceStart = 0
              continuationDeadline = 0
            elif inSpeech:
              if silenceStart == 0:
                silenceStart = epochTime()
              elif (epochTime() - silenceStart) * 1000 > silenceMs.float:
                if carryBuf.len >= MIN_SPEECH_SAMPLES:
                  result.speechSamples += carryBuf.len
                  let durSec = carryBuf.len.float / STT_RATE.float
                  if detectBargeIn:
                    mcpLog("info", "speech ended (" &
                      durSec.formatFloat(ffDecimal, 1) & "s), transcribing...",
                      "speech_end", %*{"duration_s": durSec, "samples": carryBuf.len})
                  let partial = sttTranscribe(carryBuf).strip()
                  if partial.len > 0:
                    result.parts.add(partial)
                    if detectBargeIn:
                      mcpLog("info", "transcript: \"" & partial & "\"",
                        "transcript", %*{"text": partial})
                carryBuf = @[]
                inSpeech = false
                silenceStart = 0
                if result.parts.len > 0:
                  continuationDeadline = epochTime() + CONTINUATION_WINDOW
          else:
            if not detectBargeIn and not inSpeech and result.parts.len == 0:
              # listen tool: timeout on no data
              result.timedOut = true
            sleep(2)

      mic.setSpeechDetect(0)

      # Flush remaining speech
      if carryBuf.len >= MIN_SPEECH_SAMPLES:
        result.speechSamples += carryBuf.len
        if detectBargeIn:
          mcpLog("info", "speech ended (timeout), transcribing...", "speech_end",
            %*{"reason": "timeout", "samples": carryBuf.len})
        let trailing = sttTranscribe(carryBuf).strip()
        if trailing.len > 0:
          result.parts.add(trailing)
      carryBuf = @[]

    # ---- Tool definitions ----

    let tools = %*[
      {"name": "synth", "description": "Synthesize speech from text and return a WAV file. Supports voice mixing (voice1+voice2:weight) and returns per-sentence timing chunks.",
       "inputSchema": {"type": "object",
         "properties": {
           "text": {"type": "string", "description": "Text to synthesize"},
           "voice": {"type": "string", "description": "Voice name or mix (e.g. af_heart+am_adam:0.3)", "default": "af_heart"},
           "model": {"type": "string", "description": "Model name or shorthand", "default": "kokoro-en"},
           "speed": {"type": "number", "description": "Speed multiplier", "default": 1.0},
           "output": {"type": "string", "description": "Output WAV path", "default": "output.wav"}},
         "required": ["text"]}},
      {"name": "voices", "description": "List available voices with language and gender metadata",
       "inputSchema": {"type": "object",
         "properties": {
           "model": {"type": "string", "description": "Model name or shorthand"},
           "language": {"type": "string", "enum": ["en", "zh"], "description": "Filter by language"},
           "gender": {"type": "string", "enum": ["male", "female"], "description": "Filter by gender"}}}},
      {"name": "speak", "description": "Speak text aloud through the system speakers. Interrupts any previous speech automatically. Returns after playback finishes.",
       "inputSchema": {"type": "object",
         "properties": {
           "text": {"type": "string", "description": "Text to speak"},
           "voice": {"type": "string", "description": "Voice name or mix", "default": "af_heart"},
           "model": {"type": "string", "description": "Model name or shorthand", "default": "kokoro-en"},
           "speed": {"type": "number", "description": "Speed multiplier", "default": 1.0}},
         "required": ["text"]}},
      {"name": "stop", "description": "Stop any speech currently playing. No-op if nothing is playing.",
       "inputSchema": {"type": "object", "properties": {}}},
      {"name": "listen", "description": "Listen on the microphone until the user stops speaking. Returns transcribed text. Uses hardware energy detection with 800ms silence holdoff. Pass test_audio with a WAV file path to feed audio from file instead of mic (for testing).",
       "inputSchema": {"type": "object",
         "properties": {
           "language": {"type": "string", "description": "Language code (en, zh, auto)", "default": "auto"},
           "timeout_ms": {"type": "integer", "description": "Max silence before giving up (ms)", "default": 10000},
           "test_audio": {"type": "string", "description": "WAV file path for testing (bypasses mic)"}}}},
      {"name": "models", "description": "List downloaded TTS models",
       "inputSchema": {"type": "object", "properties": {}}},
      {"name": "converse_start", "description": "Start a full-duplex voice conversation. Opens mic with hardware echo cancellation (AEC). Use converse_send to speak and converse_recv to listen. The mic stays hot between calls.",
       "inputSchema": {"type": "object",
         "properties": {
           "voice": {"type": "string", "description": "Voice name or mix", "default": "af_heart"},
           "model": {"type": "string", "description": "TTS model", "default": "kokoro-en"},
           "speed": {"type": "number", "description": "Speed multiplier", "default": 1.0},
           "language": {"type": "string", "description": "STT language (en, zh, auto)", "default": "auto"},
           "energy_threshold": {"type": "number", "description": "Mic energy threshold for speech detection (0.01-0.05)", "default": 0.02}}}},
      {"name": "converse_send", "description": "Speak one utterance in the active conversation. Blocks until playback finishes or user barge-in is detected (pause-verify). Returns barged_in=true if user interrupted. Call converse_recv next to listen for user speech.",
       "inputSchema": {"type": "object",
         "properties": {
           "text": {"type": "string", "description": "Text to speak"},
           "model": {"type": "string", "description": "TTS model override (e.g. kokoro-zh for Chinese)"},
           "voice": {"type": "string", "description": "Voice override (e.g. zf_001 for Chinese)"}},
         "required": ["text"]}},
      {"name": "converse_recv", "description": "Wait for user speech and return transcript. IMPORTANT: Always show the full transcribed text to the user immediately after each call — never skip or abbreviate. Monitors playback drain and barge-in. For multi-agent: call send→recv per agent sequentially.",
       "inputSchema": {"type": "object",
         "properties": {
           "timeout_ms": {"type": "integer", "description": "Max total wait time (ms)", "default": 30000},
           "initial_timeout_ms": {"type": "integer", "description": "Max wait for speech onset before returning empty (ms). Extends to timeout_ms once speech starts.", "default": 8000},
           "silence_ms": {"type": "integer", "description": "Silence duration to end speech (ms)", "default": 800}}}},
      {"name": "converse_stop", "description": "End the voice conversation session. Stops playback and mic capture.",
       "inputSchema": {"type": "object", "properties": {}}},
      {"name": "calibrate", "description": "Measure mic audio levels. Phase 1: ambient noise (silence). Phase 2: user makes the requested sound. Sets energy threshold at dB midpoint. Use 'request' to tell the user what sound to make (speech, cough, typing, etc).",
       "inputSchema": {"type": "object",
         "properties": {
           "request": {"type": "string", "description": "TTS prompt telling the user what sound to make for phase 2", "default": "Now, please say a few words so I can measure your voice level."},
           "duration_ms": {"type": "integer", "description": "Measurement duration per phase (ms)", "default": 2000},
           "margin_db": {"type": "number", "description": "Fallback: dB above ambient if no sound detected in phase 2", "default": 15}}}}
      ,{"name": "log", "description": "Read the JSONL debug log. Each entry has: ts, level, msg, event (optional), data (optional). Filter by event type or grep msg. Events: quick_vad, verify_bargein, bargein_check, bargein_confirmed, bargein_rejected, playback_start, playback_done, speech_start, speech_end, transcript, synth_start, recv_start, cal_*.",
       "inputSchema": {"type": "object",
         "properties": {
           "tail": {"type": "integer", "description": "Number of entries from end to return (default 50)", "default": 50},
           "event": {"type": "string", "description": "Filter by event type (exact match, e.g. 'quick_vad', 'transcript')"},
           "grep": {"type": "string", "description": "Filter entries whose msg contains this substring (case-insensitive)"},
           "clear": {"type": "boolean", "description": "Clear the log after reading (default false)", "default": false}}}}
      ,{"name": "script", "description": "Build and render JSONL screenplays. Actions: new (create script), append (add lines), read (view lines), edit (update/delete/insert/cast/header), render (synthesize to WAV/MP3), video (render to MP4 via Remotion — requires audio from render). Line types: line (dialogue), narration, scene (setting/action — silent by default, narrate:true to voice it; extra fields: setting, characters, mood), chapter (section), pause (silence). Cast maps names→voices.",
       "inputSchema": {"type": "object", "required": ["file", "action"],
         "properties": {
           "file": {"type": "string", "description": "Path to the .jsonl script file"},
           "action": {"type": "string", "description": "new, append, read, edit, render, video"},
           "title": {"type": "string", "description": "[new] Title of the script"},
           "format": {"type": "string", "description": "[new] show, podcast, storybook, drama, audiobook", "default": "show"},
           "cast": {"type": "object", "description": "[new/edit:cast] Character name → voice ID map"},
           "defaults": {"type": "object", "description": "[new] Default model, speed, pause, narrator_voice"},
           "lines": {"type": "array", "description": "[append/render] Array of line objects",
             "items": {"type": "object",
               "properties": {
                 "type": {"type": "string", "description": "line, narration, scene, chapter, pause", "default": "line"},
                 "narrate": {"type": "boolean", "description": "[scene] Narrate the scene text (default false)"},
                 "setting": {"type": "string", "description": "[scene] Location/environment"},
                 "characters": {"type": "array", "description": "[scene] Characters present", "items": {"type": "string"}},
                 "mood": {"type": "string", "description": "[scene] Mood/atmosphere"},
                 "name": {"type": "string", "description": "Character name (resolved from cast)"},
                 "voice": {"type": "string", "description": "Voice ID override"},
                 "text": {"type": "string", "description": "Text to speak / direction / chapter title"},
                 "model": {"type": "string", "description": "Model override"},
                 "speed": {"type": "number", "description": "Speed override"},
                 "pause": {"type": "number", "description": "Silence after (seconds)"},
                 "emotion": {"type": "string", "description": "Emotion hint"},
                 "duration": {"type": "number", "description": "[pause type] Silence duration (seconds)"}}}},
           "from": {"type": "integer", "description": "[read/render] Start line index (0-based)"},
           "count": {"type": "integer", "description": "[read/render] Number of lines"},
           "name": {"type": "string", "description": "[read] Filter by character name"},
           "line_type": {"type": "string", "description": "[read] Filter by line type"},
           "edit_action": {"type": "string", "description": "[edit] update, delete, insert, cast, header"},
           "index": {"type": "integer", "description": "[edit] Line index for update/delete/insert"},
           "line": {"type": "object", "description": "[edit] Line data for update/insert"},
           "fields": {"type": "object", "description": "[edit:header] Header fields to update"},
           "output": {"type": "string", "description": "[render/video] Output path (WAV/MP3/MP4)", "default": "render.wav"},
           "audio": {"type": "string", "description": "[video] Path to rendered audio (MP3/WAV)"},
           "theme": {"type": "string", "description": "[video] Visual theme: radio, storybook, podcast", "default": "radio"},
           "concurrency": {"type": "integer", "description": "[video] Render concurrency (default 8)", "default": 8},
           "frames": {"type": "string", "description": "[video] Frame range e.g. '0-900' for partial render"}}}}
    ]

    while true:
      let req = mcpRecv()
      if req == nil: break
      let id = req.getOrDefault("id")
      let meth = req.getOrDefault("method")
      if meth == nil: continue
      let methStr = meth.getStr()

      if methStr == "initialize":
        mcpSend(mcpResult(id, %*{
          "protocolVersion": "2024-11-05",
          "capabilities": {"tools": {}},
          "serverInfo": {"name": "tts", "version": "0.1.0"}
        }))
      elif methStr == "notifications/initialized":
        discard
      elif methStr == "tools/list":
        mcpSend(mcpResult(id, %*{"tools": tools}))
      elif methStr == "tools/call":
        let params = req["params"]
        let toolName = params["name"].getStr()
        let toolArgs = params.getOrDefault("arguments")
        try:
          if toolName == "synth":
            let text = toolArgs["text"].getStr()
            let voiceSpec = toolArgs.getOrDefault("voice").getStr("af_heart")
            let model = resolveModel(toolArgs.getOrDefault("model").getStr("kokoro-en"))
            let speed = toolArgs.getOrDefault("speed").getFloat(1.0).float32
            let output = toolArgs.getOrDefault("output").getStr("output.wav")
            e.loadModel(model, voiceSpec)
            # Handle voice mixing
            let vp = parseVoice(voiceSpec)
            var voice = vp.voice1
            if vp.isMix:
              voice = e.mixVoice(vp.voice1, vp.voice2, vp.weight)
            # Collect per-sentence chunk timing
            var chunks = newJArray()
            var sampleOffset = 0
            let sr = 24000.0
            let cb = proc(chunk: AudioOutput, index, total: int) {.closure.} =
              let chunkDur = chunk.samples.len.float / sr
              chunks.add %*{
                "index": index,
                "start": (sampleOffset.float / sr).formatFloat(ffDecimal, 3).parseFloat,
                "duration": chunkDur.formatFloat(ffDecimal, 3).parseFloat,
                "samples": chunk.samples.len,
              }
              sampleOffset += chunk.samples.len
            let audio = e.synthesize(text, voice, speed, cb)
            audio.writeWav(output)
            let dur = audio.samples.len.float / audio.sampleRate.float
            mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $(%*{
              "output": output, "duration": dur.formatFloat(ffDecimal, 2).parseFloat,
              "sample_rate": audio.sampleRate, "voice": voiceSpec, "model": model,
              "size_bytes": getFileSize(output),
              "chunks": chunks})}]}))
          elif toolName == "voices":
            let model = resolveModel(toolArgs.getOrDefault("model").getStr(""))
            let wantEn = toolArgs.getOrDefault("language").getStr("") == "en"
            let wantZh = toolArgs.getOrDefault("language").getStr("") == "zh"
            let wantMale = toolArgs.getOrDefault("gender").getStr("") == "male"
            let wantFemale = toolArgs.getOrDefault("gender").getStr("") == "female"
            var jsonModels = newJArray()
            if dirExists(pkgModelDir):
              for f in walkDir(pkgModelDir):
                when defined(useMlx):
                  if f.kind != pcDir: continue
                  # Only TTS model dirs (kokoro-*), skip STT/VAD/other models
                  if not extractFilename(f.path).startsWith("kokoro"): continue
                else:
                  if not f.path.endsWith(".gguf"): continue
                let name = extractFilename(f.path)
                if model.len > 0 and name != model: continue
                # Scan voice files directly — no need to load the whole model
                let voicesDir = f.path / "voices"
                var filtered: seq[string]
                if dirExists(voicesDir):
                  for vf in walkDir(voicesDir):
                    if vf.path.endsWith(".safetensors"):
                      let vName = vf.path.splitFile().name
                      if matchVoice(vName, wantMale, wantFemale, wantEn, wantZh):
                        filtered.add vName
                filtered.sort()
                if filtered.len > 0:
                  var jv = newJArray()
                  for v in filtered:
                    jv.add %*{"name": v, "language": voiceLang(v), "gender": voiceGender(v)}
                  jsonModels.add %*{"model": name, "voices": jv}
            mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $jsonModels}]}))
          elif toolName == "speak":
            # Interrupt any previous playback
            let interrupted = stopSpeaking()
            let text = toolArgs["text"].getStr()
            let voiceSpec = toolArgs.getOrDefault("voice").getStr("af_heart")
            let model = resolveModel(toolArgs.getOrDefault("model").getStr("kokoro-en"))
            let speed = toolArgs.getOrDefault("speed").getFloat(1.0).float32
            e.loadModel(model, voiceSpec)
            let vp = parseVoice(voiceSpec)
            var voice = vp.voice1
            if vp.isMix:
              voice = e.mixVoice(vp.voice1, vp.voice2, vp.weight)
            # Streaming synthesis: play each sentence chunk as it's generated
            ensureDuplex()
            var totalSamples = 0
            var firstChunkPlaying = false
            let cb = proc(chunk: AudioOutput, index, total: int) {.closure.} =
              speaker.writeAll(chunk.samples)
              totalSamples += chunk.samples.len
              if not firstChunkPlaying:
                firstChunkPlaying = true
            discard e.synthesize(text, voice, speed, cb)
            let dur = totalSamples.float / 24000.0
            mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $(%*{
              "playing": true, "duration": dur.formatFloat(ffDecimal, 2).parseFloat,
              "voice": voiceSpec, "samples": totalSamples,
              "interrupted_previous": interrupted})}]}))
          elif toolName == "stop":
            let stopped = stopSpeaking()
            mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $(%*{
              "stopped": stopped})}]}))
          elif toolName == "listen":
            let language = toolArgs.getOrDefault("language").getStr("auto")
            let timeoutMs = toolArgs.getOrDefault("timeout_ms").getInt(10000)
            let testAudio = toolArgs.getOrDefault("test_audio").getStr("")
            ensureStt(language)

            var parts: seq[string] = @[]

            if testAudio.len > 0:
              # File mode: feed audio directly to STT
              stderr.writeLine "[listen] test_audio: " & testAudio
              let wav = readWav(testAudio)
              var samples16k: seq[float32]
              if wav.sampleRate == STT_RATE.int32:
                samples16k = wav.samples
              else:
                let ratio = wav.sampleRate.float64 / STT_RATE.float64
                let outLen = int(wav.samples.len.float64 / ratio)
                samples16k = newSeq[float32](outLen)
                for i in 0..<outLen:
                  let srcPos = i.float64 * ratio
                  let idx = int(srcPos)
                  let frac = float32(srcPos - idx.float64)
                  if idx + 1 < wav.samples.len:
                    samples16k[i] = wav.samples[idx] * (1.0'f32 - frac) + wav.samples[idx + 1] * frac
                  elif idx < wav.samples.len:
                    samples16k[i] = wav.samples[idx]
              stderr.writeLine "[listen] " & $samples16k.len & " samples at 16kHz → STT"
              let text = sttTranscribe(samples16k).strip()
              if text.len > 0:
                parts.add(text)
            else:
              # Mic mode: hardware energy detection + ASR
              ensureDuplex()
              mic.start()
              var buf: seq[float32] = @[]
              let lr = listenMic(timeoutMs = timeoutMs, carryBuf = buf)
              parts = lr.parts
              mic.stop()
            let text = parts.join(" ")
            let (cleanText, artifacts) = extractArtifacts(text)
            var resp = %*{"text": cleanText, "heard": cleanText.len > 0}
            if artifacts.len > 0:
              resp["artifacts"] = %artifacts
            mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $resp}]}))
          elif toolName == "converse_start":
            ensureDuplex()
            when defined(useMlx): ensureVad()
            let voiceSpec = toolArgs.getOrDefault("voice").getStr("af_heart")
            let model = resolveModel(toolArgs.getOrDefault("model").getStr("kokoro-en"))
            let language = toolArgs.getOrDefault("language").getStr("auto")
            if toolArgs.hasKey("energy_threshold"):
              cvEnergyThreshold = toolArgs["energy_threshold"].getFloat().float32
            else:
              discard loadCalibration()
            e.loadModel(model, voiceSpec)
            cvVoice = voiceSpec
            cvSpeed = toolArgs.getOrDefault("speed").getFloat(1.0).float32
            ensureStt(language)
            mic.start()
            cvActive = true
            cvSpeechBuf = @[]
            mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $(%*{
              "started": true, "voice": voiceSpec, "model": model,
              "language": language, "energy_threshold": cvEnergyThreshold})}]}))

          elif toolName == "converse_send":
            if not cvActive:
              mcpSend(mcpError(id, -32000, "No active converse session — call converse_start first"))
            else:
              let text = toolArgs["text"].getStr()
              let voiceOv = toolArgs.getOrDefault("voice").getStr("")
              let modelOv = toolArgs.getOrDefault("model").getStr("")

              # Flush stale mic audio before speaking
              mic.flush()
              cvSpeechBuf = @[]

              # Apply voice/model overrides
              if modelOv.len > 0 or voiceOv.len > 0:
                let m = if modelOv.len > 0: resolveModel(modelOv) else: ""
                let v = if voiceOv.len > 0: voiceOv else: cvVoice
                if m.len > 0: e.loadModel(m, v)
                if voiceOv.len > 0: cvVoice = v
              let vp = parseVoice(cvVoice)
              var voice = vp.voice1
              if vp.isMix:
                voice = e.mixVoice(vp.voice1, vp.voice2, vp.weight)

              # Enable speech detection during synthesis for early barge-in
              speaker.setMute(false)  # ensure unmuted from any prior barge-in
              mic.speechReset()
              applySpeechDetect()
              cvBargedIn = false

              # Synthesize and queue audio to ring buffer with barge-in monitoring
              cvLastSamples = 0
              var bargeInGrace = epochTime() + BARGE_IN_GRACE
              var bargeInStrikes = 0
              mcpLog("info", "synthesizing with barge-in", "synth_start",
                %*{"voice": cvVoice, "text_len": text.len})

              proc checkBargeIn() =
                ## VAD pre-check → mute-verify barge-in with escalation.
                if not mic.speechDetected(): return
                mic.speechReset()
                if epochTime() < bargeInGrace: return

                # Quick mute+VAD — 300ms to distinguish speech from noise
                if not vadQuickCheck():
                  mic.speechReset()
                  bargeInGrace = epochTime() + 0.5  # brief cooldown
                  return

                # VAD confirmed speech on clean audio — barge-in
                cvBargedIn = true
                speaker.setMute(false)
                speaker.flush()
                drainMic(cvSpeechBuf)
                mcpLog("info", "barge-in confirmed by VAD", "bargein_confirmed")

              let cb = proc(chunk: AudioOutput, index, total: int) {.closure.} =
                if cvBargedIn: return
                # Write samples with barge-in check on every ring buffer wait
                var offset = 0
                while offset < chunk.samples.len and not cvBargedIn:
                  let n = speaker.write(chunk.samples.toOpenArray(offset, chunk.samples.len - 1))
                  offset += n
                  cvLastSamples += n
                  if n == 0:
                    sleep(1)
                  checkBargeIn()
              discard e.synthesize(text, voice, cvSpeed, cb)
              cvLastVoice = if voiceOv.len > 0: voiceOv else: cvVoice

              # Wait for remaining ring buffer to drain (up to ~10.9s tail)
              while not speaker.isIdle and not cvBargedIn:
                sleep(50)
                checkBargeIn()

              let dur = cvLastSamples.float / 24000.0
              var resp = %*{
                "spoken": true,
                "duration": dur.formatFloat(ffDecimal, 2).parseFloat,
                "samples": cvLastSamples,
                "voice": cvLastVoice}
              if cvBargedIn:
                resp["barged_in"] = %true
              mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $resp}]}))

              # Bridge the gap: drain mic into cvSpeechBuf while waiting for
              # the next MCP call. The LLM takes 1-5s to process and call
              # converse_recv — without this, speech during the gap is lost.
              # Keeps a pre-roll buffer so speech onset before detection is preserved.
              var bridgeBuf = newSeq[float32](CHUNK)
              var bridgePreRoll: seq[float32] = @[]
              mic.speechReset()
              applySpeechDetect()
              var bridgeSpeech = cvBargedIn  # already speaking if barged in
              while not stdinReady():
                let got = mic.read(bridgeBuf, CHUNK)
                if got > 0:
                  if bridgeSpeech:
                    cvSpeechBuf.add bridgeBuf[0..<got]
                  else:
                    # Keep rolling pre-roll so speech onset is captured
                    bridgePreRoll.add bridgeBuf[0..<got]
                    if bridgePreRoll.len > PRE_ROLL_MAX:
                      bridgePreRoll = bridgePreRoll[bridgePreRoll.len - PRE_ROLL_MAX .. ^1]
                    if mic.speechDetected():
                      mic.speechReset()
                      bridgeSpeech = true
                      # Prepend pre-roll to capture word onset
                      cvSpeechBuf.add bridgePreRoll
                      bridgePreRoll = @[]
                else:
                  sleep(2)

          elif toolName == "converse_recv":
            if not cvActive:
              mcpSend(mcpError(id, -32000, "No active converse session — call converse_start first"))
            else:
              let timeoutMs = toolArgs.getOrDefault("timeout_ms").getInt(30000)
              let silenceMs = toolArgs.getOrDefault("silence_ms").getInt(SILENCE_HOLDOFF_MS)
              # Shorter initial wait for speech onset; extends to timeoutMs once speech starts
              let initialMs = toolArgs.getOrDefault("initial_timeout_ms").getInt(
                if cvBargedIn: timeoutMs  # already speaking — use full timeout
                else: 8000)              # default 8s wait for speech onset

              if cvBargedIn:
                mcpLog("info", "barge-in carry — listening for speech", "recv_start",
                  %*{"barged_in": true, "carry_samples": cvSpeechBuf.len})

              let lr = listenMic(
                silenceMs = silenceMs,
                timeoutMs = timeoutMs,
                initialTimeoutMs = initialMs,
                detectBargeIn = true,
                carryBuf = cvSpeechBuf)

              let bargedIn = cvBargedIn or lr.bargedIn
              let rawText = lr.parts.join(" ")
              let (cleanText, artifacts) = extractArtifacts(rawText)
              let speechDur = if lr.speechSamples > 0:
                (lr.speechSamples.float / STT_RATE.float).formatFloat(ffDecimal, 2).parseFloat
              else: 0.0
              var resp = %*{
                "text": cleanText,
                "heard": cleanText.len > 0,
                "barged_in": bargedIn}
              if speechDur > 0:
                resp["speech_duration"] = %speechDur
              if artifacts.len > 0:
                resp["artifacts"] = %artifacts
              mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $resp}]}))

          elif toolName == "converse_stop":
            if cvActive:
              mic.stop()
              if speaker != nil and not speaker.isIdle:
                speaker.flush()
              cvActive = false
              cvSpeechBuf = @[]
              mic.setSpeechDetect(0)
            mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $(%*{
              "stopped": true})}]}))

          elif toolName == "calibrate":
            ensureDuplex()
            let durationMs = toolArgs.getOrDefault("duration_ms").getInt(2000)
            let marginDb = toolArgs.getOrDefault("margin_db").getFloat(15.0)
            var tmp = newSeq[float32](CHUNK)

            proc toDb(rms: float64): float64 =
              if rms > 0: 20.0 * log10(rms) else: -120.0

            # Helper: compute RMS of a chunk
            proc chunkRms(buf: openArray[float32], count: int): float32 =
              var sumSq = 0.0'f32
              for i in 0..<count:
                sumSq += buf[i] * buf[i]
              sqrt(sumSq / count.float32)

            # Measure RMS over a fixed duration (for ambient — no trigger needed)
            proc measureRms(durationMs: int): tuple[peak, avg: float32, chunks: int] =
              var peakRms = 0.0'f32
              var sumRms = 0.0'f64
              var nChunks = 0
              mic.flush()
              let deadline = epochTime() + durationMs.float / 1000.0
              while epochTime() < deadline:
                let got = mic.read(tmp, CHUNK)
                if got > 0:
                  let rms = chunkRms(tmp, got)
                  if rms > peakRms: peakRms = rms
                  sumRms += rms.float64
                  nChunks += 1
                else:
                  sleep(5)
              let avgRms = if nChunks > 0: (sumRms / nChunks.float64).float32 else: 0.0'f32
              (peakRms, avgRms, nChunks)

            # Triggered measure: wait for sound onset (above trigger level),
            # then measure for durationMs. Times out after timeoutMs total.
            proc measureTriggered(triggerRms: float32, durationMs: int,
                timeoutMs: int = 8000): tuple[peak, avg: float32, chunks: int] =
              var peakRms = 0.0'f32
              var sumRms = 0.0'f64
              var nChunks = 0
              var triggered = false
              var measureDeadline = 0.0
              mic.flush()
              let absDeadline = epochTime() + timeoutMs.float / 1000.0
              while epochTime() < absDeadline:
                if triggered and epochTime() > measureDeadline:
                  break
                let got = mic.read(tmp, CHUNK)
                if got > 0:
                  let rms = chunkRms(tmp, got)
                  if not triggered:
                    if rms > triggerRms * 3.0:  # 3x ambient = sound onset
                      triggered = true
                      measureDeadline = epochTime() + durationMs.float / 1000.0
                      mcpLog("info", "triggered at " &
                        toDb(rms.float64).formatFloat(ffDecimal, 1) & " dB", "cal_trigger")
                    else:
                      continue  # still waiting for onset
                  if rms > peakRms: peakRms = rms
                  sumRms += rms.float64
                  nChunks += 1
                else:
                  sleep(5)
              if not triggered:
                mcpLog("info", "no sound detected (timeout)", "cal_timeout")
              let avgRms = if nChunks > 0: (sumRms / nChunks.float64).float32 else: 0.0'f32
              (peakRms, avgRms, nChunks)

            # TTS helper
            let voiceSpec = "af_heart"
            let calModel = resolveModel("kokoro-en")
            e.loadModel(calModel, voiceSpec)
            let vp = parseVoice(voiceSpec)
            proc sayAndWait(text: string) =
              discard e.synthesize(text, vp.voice1, 1.0, proc(chunk: AudioOutput, index, total: int) =
                speaker.writeAll(chunk.samples))
              speaker.start()
              speaker.waitUntilDone()
              sleep(200)

            proc muteAndMeasure(label: string, durationMs: int,
                triggerRms: float32 = 0): tuple[peak, avg: float32, chunks: int] =
              if speaker != nil: speaker.setMute(true)
              sleep(50)
              mcpLog("info", "measuring: " & label &
                (if triggerRms > 0: " (waiting for sound...)" else: "") & "...", "cal_measure",
                %*{"label": label})
              if triggerRms > 0:
                result = measureTriggered(triggerRms, durationMs)
              else:
                result = measureRms(durationMs)
              if speaker != nil: speaker.setMute(false)
              let avgDb = toDb(result.avg.float64)
              let peakDb = toDb(result.peak.float64)
              mcpLog("info", label & ": avg=" & avgDb.formatFloat(ffDecimal, 1) &
                " dB, peak=" & peakDb.formatFloat(ffDecimal, 1) & " dB", "cal_result",
                %*{"label": label, "avg_db": avgDb, "peak_db": peakDb, "chunks": result.chunks})

            mic.start()

            # Phase 1: Ambient
            sayAndWait("Calibrating. Please stay quiet while I measure background noise.")
            let ambient = muteAndMeasure("ambient", durationMs)
            let ambientDb = toDb(ambient.avg.float64)

            # Phase 2: Speech (triggered — waits for user to start speaking)
            sayAndWait("Now, please say a few words so I can measure your voice level.")
            let speech = muteAndMeasure("speech", durationMs, ambient.avg)
            let speechDb = toDb(speech.avg.float64)

            # Phase 3: Coughing (triggered)
            sayAndWait("Now, please cough a few times.")
            let cough = muteAndMeasure("cough", durationMs, ambient.avg)
            let coughDb = toDb(cough.avg.float64)

            # Phase 4: Typing (triggered)
            sayAndWait("Now, please type on your keyboard.")
            let typing = muteAndMeasure("typing", durationMs, ambient.avg)
            let typingDb = toDb(typing.avg.float64)

            # Phase 5: AEC residual — measure mic while speaker plays, user stays quiet
            # This is the noise floor the energy gate sees during active playback
            sayAndWait("Now please stay quiet while I measure the echo cancellation level.")
            discard e.synthesize(
              "This is a measurement of the echo cancellation residual energy. Please stay quiet and do not speak during this measurement. I need to know what the microphone picks up when only the speaker is active.",
              vp.voice1, 1.0, proc(chunk: AudioOutput, index, total: int) =
                speaker.writeAll(chunk.samples))
            speaker.start()
            sleep(500)  # let AEC settle
            mcpLog("info", "measuring: AEC residual (speaker active, user quiet)...", "cal_measure",
              %*{"label": "aec_residual"})
            let aecResidual = measureRms(durationMs)
            speaker.flush()
            let aecDb = toDb(aecResidual.avg.float64)
            let aecPeakDb = toDb(aecResidual.peak.float64)
            mcpLog("info", "AEC residual: avg=" & aecDb.formatFloat(ffDecimal, 1) &
              " dB, peak=" & aecPeakDb.formatFloat(ffDecimal, 1) & " dB", "cal_result",
              %*{"label": "aec_residual", "avg_db": aecDb, "peak_db": aecPeakDb})

            # Phase 6: Barge-in test — measure speech during active playback
            sayAndWait("Now I will keep talking. Please interrupt me by saying something.")
            discard e.synthesize(
              "This is a test of the barge-in detection system. I am going to keep talking for a while so that you can try interrupting me. Please speak now to test the barge-in calibration. Say anything you like, as long as you speak over me.",
              vp.voice1, 1.0, proc(chunk: AudioOutput, index, total: int) =
                speaker.writeAll(chunk.samples))
            speaker.start()
            mcpLog("info", "measuring: barge-in speech (waiting for sound, speaker active)...", "cal_measure",
              %*{"label": "bargein_speech"})
            let bargeIn = measureTriggered(aecResidual.avg, durationMs)
            speaker.flush()
            let bargeInDb = toDb(bargeIn.avg.float64)
            let bargeInPeakDb = toDb(bargeIn.peak.float64)
            mcpLog("info", "barge-in speech: avg=" & bargeInDb.formatFloat(ffDecimal, 1) &
              " dB, peak=" & bargeInPeakDb.formatFloat(ffDecimal, 1) & " dB", "cal_result",
              %*{"label": "bargein_speech", "avg_db": bargeInDb, "peak_db": bargeInPeakDb})

            sayAndWait("Calibration complete.")

            # Threshold: max(ambient, aec_residual) + margin_db
            # Use the higher noise floor so it works both during playback and silence
            let noiseFloorDb = max(ambientDb, aecDb)
            let threshDb = noiseFloorDb + marginDb
            cvEnergyThreshold = max(0.001, pow(10.0, threshDb / 20.0).float32)

            mcpLog("info", "calibrated: threshold=" & threshDb.formatFloat(ffDecimal, 1) &
              " dB, headroom=" & (bargeInDb - threshDb).formatFloat(ffDecimal, 1) & " dB",
              "cal_done", %*{"ambient_db": ambientDb, "aec_db": aecDb,
                "floor_db": noiseFloorDb, "threshold_db": threshDb,
                "margin_db": marginDb, "headroom_db": bargeInDb - threshDb})

            if cvActive:
              applySpeechDetect()
            saveCalibration()
            mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $(%*{
              "calibrated": true,
              "levels": {
                "ambient_db": ambientDb.formatFloat(ffDecimal, 1),
                "aec_residual_db": aecDb.formatFloat(ffDecimal, 1),
                "speech_db": speechDb.formatFloat(ffDecimal, 1),
                "barge_in_db": bargeInDb.formatFloat(ffDecimal, 1),
                "cough_db": coughDb.formatFloat(ffDecimal, 1),
                "typing_db": typingDb.formatFloat(ffDecimal, 1)},
              "threshold_db": threshDb.formatFloat(ffDecimal, 1),
              "headroom_db": (bargeInDb - threshDb).formatFloat(ffDecimal, 1),
              "margin_db": marginDb})}]}))

          elif toolName == "models":
            var arr = newJArray()
            if dirExists(pkgModelDir):
              for f in walkDir(pkgModelDir):
                when defined(useMlx):
                  if f.kind != pcDir: continue
                else:
                  if not f.path.endsWith(".gguf"): continue
                let name = extractFilename(f.path)
                var vc = 0
                try:
                  var eng = newTTSEngine()
                  eng.loadModel(f.path, "af_heart")
                  vc = eng.listVoices().len
                  eng.close()
                except: discard
                if vc > 0:
                  when defined(useMlx):
                    arr.add %*{"model": name, "voices": vc}
                  else:
                    arr.add %*{"model": name, "size_bytes": getFileSize(f.path), "voices": vc}
            mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $arr}]}))
          elif toolName == "script":
            let file = toolArgs["file"].getStr()
            let action = toolArgs["action"].getStr()

            if action == "new":
              let title = toolArgs["title"].getStr()
              let fmt = toolArgs.getOrDefault("format").getStr("show")
              let castJ = toolArgs.getOrDefault("cast")
              let defaults = toolArgs.getOrDefault("defaults")
              var header = %*{"type": "header", "title": title, "format": fmt,
                "created": now().format("yyyy-MM-dd'T'HH:mm:ss")}
              if castJ != nil: header["cast"] = castJ
              if defaults != nil: header["defaults"] = defaults
              writeFile(file, $header & "\n")
              mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $(%*{
                "created": file, "title": title, "format": fmt})}]}))

            elif action == "append":
              let lines = toolArgs["lines"]
              var lineCount = 0
              if fileExists(file):
                for l in file.lines:
                  lineCount += 1
                lineCount = max(0, lineCount - 1)
              var f = open(file, fmAppend)
              defer: f.close()
              var added = 0
              for line in lines:
                var entry = line.copy()
                if not entry.hasKey("type"):
                  entry["type"] = %"line"
                f.writeLine($entry)
                added += 1
              mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $(%*{
                "appended": added, "total_lines": lineCount + added})}]}))

            elif action == "read":
              let fromIdx = toolArgs.getOrDefault("from").getInt(0)
              let countMax = toolArgs.getOrDefault("count").getInt(int.high)
              let nameFilter = toolArgs.getOrDefault("name").getStr("")
              let typeFilter = toolArgs.getOrDefault("line_type").getStr("")
              var allLines: seq[string]
              if fileExists(file):
                for l in file.lines:
                  allLines.add l
              var header: JsonNode = nil
              if allLines.len > 0:
                try: header = parseJson(allLines[0])
                except: discard
              var readResult = newJArray()
              var idx = 0
              var returned = 0
              for i in 1 ..< allLines.len:
                if allLines[i].len == 0: continue
                let j = try: parseJson(allLines[i]) except: continue
                if nameFilter.len > 0 and j.getOrDefault("name").getStr("") != nameFilter:
                  idx += 1
                  continue
                if typeFilter.len > 0 and j.getOrDefault("type").getStr("line") != typeFilter:
                  idx += 1
                  continue
                if idx >= fromIdx and returned < countMax:
                  var entry = j.copy()
                  entry["_index"] = %idx
                  readResult.add entry
                  returned += 1
                idx += 1
              mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $(%*{
                "header": header, "lines": readResult, "total": idx})}]}))

            elif action == "edit":
              let editAct = toolArgs["edit_action"].getStr()
              var allLines: seq[string]
              if fileExists(file):
                for l in file.lines:
                  allLines.add l
              if editAct == "update":
                let idx = toolArgs["index"].getInt() + 1
                let line = toolArgs["line"]
                if idx < 1 or idx >= allLines.len:
                  mcpSend(mcpError(id, -32000, "index out of range: " & $(idx - 1)))
                else:
                  allLines[idx] = $line
                  writeFile(file, allLines.join("\n") & "\n")
                  mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $(%*{
                    "updated": idx - 1})}]}))
              elif editAct == "delete":
                let idx = toolArgs["index"].getInt() + 1
                if idx < 1 or idx >= allLines.len:
                  mcpSend(mcpError(id, -32000, "index out of range: " & $(idx - 1)))
                else:
                  allLines.delete(idx)
                  writeFile(file, allLines.join("\n") & "\n")
                  mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $(%*{
                    "deleted": idx - 1, "total_lines": allLines.len - 1})}]}))
              elif editAct == "insert":
                let idx = toolArgs["index"].getInt() + 1
                let line = toolArgs["line"]
                if idx < 1 or idx > allLines.len:
                  mcpSend(mcpError(id, -32000, "index out of range: " & $(idx - 1)))
                else:
                  allLines.insert($line, idx)
                  writeFile(file, allLines.join("\n") & "\n")
                  mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $(%*{
                    "inserted": idx - 1, "total_lines": allLines.len - 1})}]}))
              elif editAct == "cast":
                let castUpd = toolArgs["cast"]
                if allLines.len > 0:
                  var header = try: parseJson(allLines[0]) except: %*{"type": "header"}
                  if not header.hasKey("cast"):
                    header["cast"] = newJObject()
                  for k, v in castUpd:
                    header["cast"][k] = v
                  allLines[0] = $header
                  writeFile(file, allLines.join("\n") & "\n")
                  mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $(%*{
                    "cast": header["cast"]})}]}))
                else:
                  mcpSend(mcpError(id, -32000, "empty script file"))
              elif editAct == "header":
                let fields = toolArgs["fields"]
                if allLines.len > 0:
                  var header = try: parseJson(allLines[0]) except: %*{"type": "header"}
                  for k, v in fields:
                    header[k] = v
                  allLines[0] = $header
                  writeFile(file, allLines.join("\n") & "\n")
                  mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $header}]}))
                else:
                  mcpSend(mcpError(id, -32000, "empty script file"))
              else:
                mcpSend(mcpError(id, -32000, "unknown edit_action: " & editAct))

            elif action == "render":
              let output = toolArgs.getOrDefault("output").getStr("render.wav")
              let renderFrom = toolArgs.getOrDefault("from").getInt(0)
              let renderCount = toolArgs.getOrDefault("count").getInt(int.high)
              let sr = 24000'i32

              # Load lines from script file
              var scriptLines: seq[JsonNode]
              var castMap = newJObject()
              var defModel = ""
              var defSpeed = 1.0
              var defPause = 0.3
              var defNarrator = ""

              if fileExists(file):
                var first = true
                for l in file.lines:
                  if l.len == 0: continue
                  let j = try: parseJson(l) except: continue
                  if first:
                    first = false
                    if j.getOrDefault("type").getStr("") == "header":
                      let c = j.getOrDefault("cast")
                      if c != nil: castMap = c
                      let d = j.getOrDefault("defaults")
                      if d != nil:
                        defModel = d.getOrDefault("model").getStr("")
                        defSpeed = d.getOrDefault("speed").getFloat(1.0)
                        defPause = d.getOrDefault("pause").getFloat(0.3)
                        defNarrator = d.getOrDefault("narrator_voice").getStr("")
                      continue
                  scriptLines.add j
              else:
                # Also support inline lines for quick renders
                let inlineLines = toolArgs.getOrDefault("lines")
                if inlineLines != nil:
                  for line in inlineLines:
                    scriptLines.add line

              var allSamples: seq[float32]
              var lineResults = newJArray()
              var rendered = 0
              for i in 0 ..< scriptLines.len:
                if i < renderFrom: continue
                if rendered >= renderCount: break
                let line = scriptLines[i]
                let lineType = line.getOrDefault("type").getStr("line")

                if lineType == "scene":
                  let narrate = line.getOrDefault("narrate").getBool(false)
                  if narrate:
                    let text = line.getOrDefault("text").getStr("")
                    if text.len > 0:
                      var nVoice = line.getOrDefault("voice").getStr(defNarrator)
                      if nVoice.len == 0: nVoice = "af_sky"  # fallback narrator
                      let nModel = resolveModel(line.getOrDefault("model").getStr(
                        if defModel.len > 0: defModel
                        elif nVoice.startsWith("z"): "kokoro-zh"
                        else: "kokoro-en"))
                      let nSpeed = line.getOrDefault("speed").getFloat(defSpeed).float32
                      let nPause = line.getOrDefault("pause").getFloat(0.8)
                      e.loadModel(nModel, nVoice)
                      let vp = parseVoice(nVoice)
                      let startSample = allSamples.len
                      let audio = e.synthesize(text, vp.voice1, nSpeed)
                      allSamples.add audio.samples
                      let silSamples = int(nPause * sr.float)
                      for j in 0 ..< silSamples:
                        allSamples.add 0.0'f32
                      let dur = audio.samples.len.float / sr.float
                      lineResults.add %*{
                        "index": i, "name": "narrator", "voice": nVoice, "type": "scene",
                        "start": (startSample.float / sr.float).formatFloat(ffDecimal, 2).parseFloat,
                        "duration": dur.formatFloat(ffDecimal, 2).parseFloat,
                        "text": text[0 ..< min(text.len, 60)]}
                      mcpLog("info", "render: " & $rendered & " scene dur=" &
                        dur.formatFloat(ffDecimal, 1) & "s",
                        "render_line", %*{"index": i, "name": "narrator", "duration": dur})
                  else:
                    # Non-narrated scene: 1s silence
                    let silSamples = int(1.0 * sr.float)
                    for j in 0 ..< silSamples:
                      allSamples.add 0.0'f32
                  rendered += 1
                  continue
                if lineType == "direction":
                  # Legacy compat — treat as non-narrated scene
                  let silSamples = int(0.5 * sr.float)
                  for j in 0 ..< silSamples:
                    allSamples.add 0.0'f32
                  rendered += 1
                  continue
                if lineType == "chapter":
                  let silSamples = int(1.5 * sr.float)
                  for j in 0 ..< silSamples:
                    allSamples.add 0.0'f32
                  rendered += 1
                  continue
                if lineType == "pause":
                  let dur = line.getOrDefault("duration").getFloat(1.0)
                  let silSamples = int(dur * sr.float)
                  for j in 0 ..< silSamples:
                    allSamples.add 0.0'f32
                  rendered += 1
                  continue

                let text = line.getOrDefault("text").getStr("")
                if text.len == 0:
                  rendered += 1
                  continue

                var voiceSpec = line.getOrDefault("voice").getStr("")
                if voiceSpec.len == 0:
                  let name = line.getOrDefault("name").getStr("")
                  if name.len > 0 and castMap.hasKey(name):
                    voiceSpec = castMap[name].getStr()
                  elif lineType in ["narration", "scene"] and defNarrator.len > 0:
                    voiceSpec = defNarrator
                  else:
                    voiceSpec = "af_heart"

                let modelName = line.getOrDefault("model").getStr(
                  if defModel.len > 0: defModel
                  elif voiceSpec.startsWith("z"): "kokoro-zh"
                  else: "kokoro-en")
                let speed = line.getOrDefault("speed").getFloat(defSpeed).float32
                let pauseSec = line.getOrDefault("pause").getFloat(defPause)

                let model = resolveModel(modelName)
                e.loadModel(model, voiceSpec)
                let vp = parseVoice(voiceSpec)
                var voice = vp.voice1
                if vp.isMix:
                  voice = e.mixVoice(vp.voice1, vp.voice2, vp.weight)
                let startSample = allSamples.len
                let audio = e.synthesize(text, voice, speed)
                allSamples.add audio.samples
                let silenceSamples = int(pauseSec * sr.float)
                for j in 0 ..< silenceSamples:
                  allSamples.add 0.0'f32
                let dur = audio.samples.len.float / sr.float
                let displayName = line.getOrDefault("name").getStr(voiceSpec)
                lineResults.add %*{
                  "index": i, "name": displayName, "voice": voiceSpec,
                  "start": (startSample.float / sr.float).formatFloat(ffDecimal, 2).parseFloat,
                  "duration": dur.formatFloat(ffDecimal, 2).parseFloat,
                  "text": text[0 ..< min(text.len, 60)]}
                mcpLog("info", "render: " & $rendered & " " & displayName &
                  " dur=" & dur.formatFloat(ffDecimal, 1) & "s",
                  "render_line", %*{"index": i, "name": displayName, "duration": dur})
                rendered += 1
              let combined = AudioOutput(samples: allSamples, sampleRate: sr, channels: 1)
              let totalDur = allSamples.len.float / sr.float
              var finalOutput = output
              if output.endsWith(".mp3"):
                # Render WAV to temp, convert to MP3 via ffmpeg
                let tmpWav = output & ".tmp.wav"
                combined.writeWav(tmpWav)
                let ffCmd = "ffmpeg -y -i " & tmpWav & " -codec:a libmp3lame -q:a 2 " & output
                let ffResult = execProcess(ffCmd)
                if not fileExists(output):
                  mcpSend(mcpError(id, -32000, "ffmpeg failed: " & ffResult))
                  removeFile(tmpWav)
                else:
                  removeFile(tmpWav)
              else:
                combined.writeWav(output)
              # Save timeline for video action
              let renderResult = %*{
                "output": output,
                "duration": totalDur.formatFloat(ffDecimal, 2).parseFloat,
                "lines_rendered": rendered,
                "size_bytes": getFileSize(output),
                "timeline": lineResults}
              let timelinePath = output & ".timeline.json"
              writeFile(timelinePath, $renderResult)
              mcpLog("info", "saved timeline to " & timelinePath, "timeline_saved")
              mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $renderResult}]}))

            elif action == "video":
              # Render script to MP4 video via Remotion
              let output = toolArgs.getOrDefault("output").getStr("render.mp4")
              let audioInput = toolArgs.getOrDefault("audio").getStr("")
              let theme = toolArgs.getOrDefault("theme").getStr("radio")
              let concurrency = toolArgs.getOrDefault("concurrency").getInt(8)
              let framesArg = toolArgs.getOrDefault("frames").getStr("") # e.g. "0-900"

              # We need the audio rendered first — check for it
              if audioInput.len == 0 or not fileExists(audioInput):
                mcpSend(mcpError(id, -32000,
                  "audio file required — render the script to MP3 first, then pass audio=\"path.mp3\""))
              else:
                # Find video template dir
                let templateDir = pkgRoot / "res" / "video-template"
                if not dirExists(templateDir):
                  mcpSend(mcpError(id, -32000, "video template not found at " & templateDir))
                else:
                  # Set up working dir next to the output
                  let workDir = output.parentDir() / ".remotion-work"
                  if not dirExists(workDir):
                    createDir(workDir)
                    # Copy template files
                    for f in @["package.json", "tsconfig.json", "remotion.config.ts", "prep.ts"]:
                      copyFile(templateDir / f, workDir / f)
                    createDir(workDir / "src")
                    for f in walkDir(templateDir / "src"):
                      if f.kind == pcFile:
                        copyFile(f.path, workDir / "src" / f.path.extractFilename)
                  # Install node deps if needed
                  if not dirExists(workDir / "node_modules"):
                    mcpLog("info", "installing video deps...", "video_install")
                    let installExe = if findExe("bun").len > 0: findExe("bun") else: findExe("npm")
                    let installResult = execProcess(installExe,
                      args = @["install"],
                      workingDir = workDir,
                      options = {poUsePath, poStdErrToStdOut})
                    mcpLog("info", "deps installed: " & installResult[0..min(200, installResult.len-1)], "video_install_done")

                  # Run prep.ts to generate data.json + copy audio
                  # First, we need the render timeline — re-render audio to get it,
                  # or read from an existing render result
                  let timelineFile = audioInput & ".timeline.json"

                  # If no timeline exists, do a quick render to generate it
                  if not fileExists(timelineFile):
                    # Build timeline from script + audio duration
                    # For now, require the user to save the render result
                    mcpSend(mcpError(id, -32000,
                      "timeline file not found at " & timelineFile &
                      ". After render, save the result JSON to this path."))
                  else:
                    # Run prep
                    let prepExe = if findExe("bun").len > 0: findExe("bun") else: findExe("npx")
                    var prepArgs: seq[string]
                    if findExe("bun").len > 0:
                      prepArgs = @["prep.ts", file.absolutePath, audioInput.absolutePath, timelineFile.absolutePath]
                    else:
                      prepArgs = @["tsx", "prep.ts", file.absolutePath, audioInput.absolutePath, timelineFile.absolutePath]
                    let prepResult = execProcess(prepExe,
                      args = prepArgs,
                      workingDir = workDir,
                      options = {poUsePath, poStdErrToStdOut})
                    mcpLog("info", "prep: " & prepResult, "video_prep")

                    # Render video
                    createDir(output.parentDir())
                    let npxExe = findExe("npx")
                    var renderArgs = @["remotion", "render", "Storybook", output.absolutePath,
                      "--concurrency=" & $concurrency]
                    if framesArg.len > 0:
                      renderArgs.add "--frames=" & framesArg
                    mcpLog("info", "rendering video: npx " & renderArgs.join(" "), "video_render")
                    let renderResult = execProcess(npxExe,
                      args = renderArgs,
                      workingDir = workDir,
                      options = {poUsePath, poStdErrToStdOut})
                    mcpLog("info", "render done: " & renderResult[0..min(500, renderResult.len-1)], "video_render_done")

                    if fileExists(output):
                      mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $(%*{
                        "output": output,
                        "size_bytes": getFileSize(output),
                        "theme": theme})}]}))
                    else:
                      mcpSend(mcpError(id, -32000, "video render failed: " & renderResult))

            else:
              mcpSend(mcpError(id, -32000, "unknown script action: " & action))
          elif toolName == "log":
            let logPath = "/tmp/tts_mcp.log"
            let tailN = toolArgs.getOrDefault("tail").getInt(50)
            let grepPat = toolArgs.getOrDefault("grep").getStr("")
            let eventFilter = toolArgs.getOrDefault("event").getStr("")
            let doClear = toolArgs.getOrDefault("clear").getBool(false)
            var entries: seq[JsonNode]
            if fileExists(logPath):
              for line in logPath.lines:
                if line.len == 0: continue
                try:
                  let j = parseJson(line)
                  if eventFilter.len > 0:
                    if j.getOrDefault("event").getStr("") != eventFilter: continue
                  if grepPat.len > 0:
                    if not j.getOrDefault("msg").getStr("").toLowerAscii.contains(grepPat.toLowerAscii):
                      continue
                  entries.add j
                except:
                  # Legacy plain-text line — wrap it
                  if grepPat.len > 0 and not line.toLowerAscii.contains(grepPat.toLowerAscii):
                    continue
                  if eventFilter.len > 0: continue  # can't match event on plain text
                  entries.add %*{"msg": line}
            # Take last N entries
            let start = max(0, entries.len - tailN)
            var arr = newJArray()
            for i in start ..< entries.len:
              arr.add entries[i]
            if doClear and fileExists(logPath):
              writeFile(logPath, "")
              mcpLog("info", "log cleared", "log_clear")
            mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $arr}]}))
          else:
            mcpSend(mcpError(id, -32601, "unknown tool: " & toolName))
        except CatchableError as ex:
          mcpSend(mcpError(id, -32000, ex.msg))
      elif methStr == "ping":
        mcpSend(mcpResult(id, %*{}))
      else:
        mcpSend(mcpError(id, -32601, "method not found: " & methStr))

  elif args.isCommand("mcp"):
    let binaryPath = getAppFilename()
    let config = %*{
      "mcpServers": {
        "tts": {
          "command": binaryPath,
          "args": ["serve"]
        }
      }
    }
    if args["--print"]:
      echo config.pretty
    else:
      let dest = getCurrentDir() / ".mcp.json"
      writeFile(dest, config.pretty & "\n")
      echo "Wrote ", dest
      echo "  command: ", binaryPath, " serve"

  elif args.isCommand("schema"):
    if args["--per-command"]:
      echo perCommandSchema(Doc).pretty
    else:
      echo schema(Doc).pretty
