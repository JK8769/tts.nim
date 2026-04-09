## tts_cli — Command-line TTS synthesis tool.
## Install: nimble install tts
## Usage:
##   tts_cli synth "Hello world" -v af_heart -m kokoro-en-q5.gguf -o output.wav
##   tts_cli voices -m kokoro-en-q5.gguf
##   tts_cli download kokoro-en
##   tts_cli models

import std/[os, strutils, strformat, sequtils, json, algorithm, terminal]
import tts/common
import tts/engine
import tts/audio/device
when defined(useMlx):
  import tts/audio/silero_vad
  import tts/stt/whisper_mlx
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
  tts_cli converse [-m <model>] [-v <voice>] [-s <speed>] [--whisper <wmodel>] [--vad-model <vmodel>] [--lang <lang>] [--greeting <text>]
  tts_cli serve
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
  --whisper <wmodel>     Whisper model for STT [default: auto].
  --vad-model <vmodel>   Silero VAD model directory (MLX backend).
  --lang <lang>          Language for speech recognition [default: auto].
  --greeting <text>      Greeting spoken at conversation start.
  -h, --help             Show this help.
"""

when defined(useMlx):
  const Models = {
    "kokoro-en": "kokoro-mlx-q4",
    "kokoro-zh": "kokoro-zh-mlx-q4",
  }
  const DefaultWhisper = "whisper-base.en-mlx"
  const DefaultVadModel = "silero-vad"
else:
  const Models = {
    "kokoro-en": "kokoro-en-q5.gguf",
    "kokoro-zh": "kokoro-v1.1-zh-q5.gguf",
  }
  const DefaultWhisper = "ggml-base.en.bin"
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
    let whisperRaw = $args["--whisper"]
    let whisperModel = if whisperRaw == "auto": findModel(DefaultWhisper)
                       else: findModel(whisperRaw)
    if whisperModel.len == 0:
      die("Whisper model not found: " & whisperRaw & " (run: nimble download)", false)
    let lang = $args["--lang"]
    let greeting = if args["--greeting"]: $args["--greeting"] else: ""
    var config = defaultConverseConfig()
    config.voice = voiceSpec
    config.speed = speed
    config.language = lang
    config.greeting = greeting
    when defined(useMlx):
      let vadRaw = if args["--vad-model"]: $args["--vad-model"] else: DefaultVadModel
      config.vadModelDir = findModel(vadRaw)
      if config.vadModelDir.len == 0:
        die("VAD model not found: " & vadRaw & " (run: nimble download)", false)
    echo "Starting conversation (speak to begin, Ctrl+C to exit)..."
    echo "  TTS model: ", model, " | Voice: ", voiceSpec
    echo "  Whisper model: ", whisperModel, " | Language: ", lang
    var cl = newConverseLoop(model, whisperModel, config)
    cl.run(proc(turn: ConverseTurn): string =
      if turn.kind == ctUserSpeech:
        echo "You: ", turn.text
        # Echo mode — repeat back what was said (replace with LLM later)
        let response = "You said: " & turn.text
        echo "Agent: ", response
        return response
      return ""
    )
    cl.close()

  elif args.isCommand("serve"):
    # MCP stdio server — JSON-RPC over stdin/stdout with Content-Length framing
    var e = newTTSEngine()
    # Model loaded lazily on first synth/speak call for fast MCP startup

    var speaker: AudioPlayback = nil

    proc stopSpeaking(): bool =
      ## Flush and stop any active playback. Returns true if something was playing.
      if speaker != nil and not speaker.isIdle:
        speaker.flush()
        return true
      return false

    proc mcpSend(j: JsonNode) =
      let s = $j
      stdout.write(s & "\n")
      stdout.flushFile()

    proc mcpRecv(): JsonNode =
      var line: string
      while true:
        try:
          line = stdin.readLine().strip()
        except EOFError:
          return nil
        if line.len > 0:
          return parseJson(line)

    proc mcpResult(id: JsonNode, res: JsonNode): JsonNode =
      %*{"jsonrpc": "2.0", "id": id, "result": res}

    proc mcpError(id: JsonNode, code: int, msg: string): JsonNode =
      %*{"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": msg}}

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
      {"name": "listen", "description": "Listen on the microphone until the user stops speaking. Returns transcribed text. Automatically interrupts any playing speech. Uses VAD for speech detection and Whisper for transcription.",
       "inputSchema": {"type": "object",
         "properties": {
           "whisper_model": {"type": "string", "description": "Whisper model name"},
           "language": {"type": "string", "description": "Language code (en, zh, auto). Default: auto (mixed-language)", "default": "auto"},
           "timeout_ms": {"type": "integer", "description": "Max silence before giving up (ms)", "default": 10000}}}},
      {"name": "models", "description": "List downloaded TTS models",
       "inputSchema": {"type": "object", "properties": {}}}
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
                if not f.path.endsWith(".gguf"): continue
                let name = extractFilename(f.path)
                if model.len > 0 and name != model: continue
                var eng = newTTSEngine()
                eng.loadModel(f.path, "af_heart")
                var filtered: seq[string]
                for v in eng.listVoices():
                  if matchVoice(v, wantMale, wantFemale, wantEn, wantZh):
                    filtered.add v
                filtered.sort()
                if filtered.len > 0:
                  var jv = newJArray()
                  for v in filtered:
                    jv.add %*{"name": v, "language": voiceLang(v), "gender": voiceGender(v)}
                  jsonModels.add %*{"model": name, "voices": jv}
                eng.close()
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
            if speaker == nil:
              speaker = newAudioPlayback(24000)
              speaker.start()
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
            # Mic-mute gate: wait for playback to finish before opening mic
            # This prevents echo (mic hearing the speakers)
            if speaker != nil and not speaker.isIdle:
              speaker.waitUntilDone()
            let whisperRaw = toolArgs.getOrDefault("whisper_model").getStr(DefaultWhisper)
            let whisperModelName = findModel(whisperRaw)
            let language = toolArgs.getOrDefault("language").getStr("auto")
            let timeoutMs = toolArgs.getOrDefault("timeout_ms").getInt(10000)
            var rec = newSpeechRecognizer(whisperModelName, language)
            var mic = newAudioCapture(WHISPER_SAMPLE_RATE.uint32)
            when defined(useMlx):
              let vadModelDir = findModel(toolArgs.getOrDefault("vad_model").getStr(DefaultVadModel))
              var v = loadSileroVad(vadModelDir)
              const frameSize = SILERO_CHUNK_SIZE
            else:
              var v = newVad()
              let frameSize = v.config.frameSize
            mic.start()
            var speechBuf: seq[float32] = @[]
            var frame = newSeq[float32](frameSize)
            var accumBuf: seq[float32] = @[]
            var silenceFrames = 0
            var heard = false
            var parts: seq[string] = @[]
            var noDataMs = 0
            block listenLoop:
              while true:
                var tmp = newSeq[float32](frameSize)
                let got = mic.read(tmp, frameSize)
                if got > 0:
                  accumBuf.add tmp[0..<got]
                  noDataMs = 0
                else:
                  noDataMs += 2
                  if noDataMs >= timeoutMs:
                    break listenLoop
                if accumBuf.len < frameSize:
                  sleep(2)
                  continue
                for j in 0..<frameSize:
                  frame[j] = accumBuf[j]
                accumBuf = accumBuf[frameSize..^1]
                let event = v.processFrame(frame)
                case event
                of veSpeechStart:
                  heard = true
                  silenceFrames = 0
                  let pad = v.drainPad()
                  speechBuf = pad & frame[0..<got]
                of veSpeechEnd:
                  # Transcribe this segment. Non-verbal sounds (cough, etc.)
                  # produce empty text and are discarded. Real speech is
                  # accumulated — user can pause, make a sound to stay alive,
                  # and continue talking.
                  if speechBuf.len > 0:
                    let partial = rec.transcribe(speechBuf).strip()
                    if partial.len > 0:
                      parts.add(partial)
                  speechBuf = @[]
                  silenceFrames = 0  # reset timeout after any sound
                of veNone:
                  if v.state in {vsSpeech, vsTrailing}:
                    speechBuf.add frame[0..<got]
                  else:
                    inc silenceFrames
                    let silMs = silenceFrames * frameSize * 1000 div WHISPER_SAMPLE_RATE
                    if silMs >= timeoutMs:
                      break listenLoop
            mic.stop()
            mic.close()
            # Transcribe any trailing speech not yet ended by VAD
            if speechBuf.len > 0:
              let trailing = rec.transcribe(speechBuf).strip()
              if trailing.len > 0:
                parts.add(trailing)
            let text = parts.join(" ")
            rec.close()
            mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $(%*{
              "text": text,
              "heard": heard,
              "samples": speechBuf.len,
              "duration": (speechBuf.len.float / WHISPER_SAMPLE_RATE.float).formatFloat(ffDecimal, 2).parseFloat})}]}))
          elif toolName == "models":
            var arr = newJArray()
            if dirExists(pkgModelDir):
              for f in walkDir(pkgModelDir):
                if f.path.endsWith(".gguf"):
                  let name = extractFilename(f.path)
                  var vc = 0
                  try:
                    var eng = newTTSEngine()
                    eng.loadModel(f.path, "af_heart")
                    vc = eng.listVoices().len
                    eng.close()
                  except: discard
                  arr.add %*{"model": name, "size_bytes": getFileSize(f.path), "voices": vc}
            mcpSend(mcpResult(id, %*{"content": [{"type": "text", "text": $arr}]}))
          else:
            mcpSend(mcpError(id, -32601, "unknown tool: " & toolName))
        except CatchableError as ex:
          mcpSend(mcpError(id, -32000, ex.msg))
      elif methStr == "ping":
        mcpSend(mcpResult(id, %*{}))
      else:
        mcpSend(mcpError(id, -32601, "method not found: " & methStr))

  elif args.isCommand("schema"):
    if args["--per-command"]:
      echo perCommandSchema(Doc).pretty
    else:
      echo schema(Doc).pretty
