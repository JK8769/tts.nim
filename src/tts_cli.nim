## tts_cli — Command-line TTS synthesis tool.
## Install: nimble install tts
## Usage:
##   tts_cli synth "Hello world" -v af_heart -m kokoro-en-q5.gguf -o output.wav
##   tts_cli voices -m kokoro-en-q5.gguf
##   tts_cli download kokoro-en
##   tts_cli models

import std/[os, strutils]
import tts/common
import tts/engine

const Usage = """
tts_cli — Native TTS engine for Nim

Usage:
  tts_cli synth <text> [options]    Synthesize speech to WAV
  tts_cli voices [options]          List available voices
  tts_cli download [name]           Download a model (kokoro-en, kokoro-zh)
  tts_cli models                    List downloaded models
Options:
  -m, --model <file>    Model file (default: kokoro-en-q5.gguf)
  -v, --voice <name>    Voice name (default: af_heart)
  -s, --speed <float>   Speed multiplier (default: 1.0)
  -o, --output <file>   Output WAV file (default: output.wav)
  -h, --help            Show this help
"""

proc listModels() =
  echo "Models dir: ", pkgModelDir
  if dirExists(pkgModelDir):
    for f in walkDir(pkgModelDir):
      if f.path.endsWith(".gguf"):
        let mb = getFileSize(f.path) div (1024 * 1024)
        echo "  ", extractFilename(f.path), "  (", mb, " MB)"
  else:
    echo "  (empty — run: tts_cli download kokoro-en)"

const Models = {
  "kokoro-en": "kokoro-en-q5.gguf",
  "kokoro-zh": "kokoro-v1.1-zh-q5.gguf",
}
const Repo = "JK8769/tts.nim"

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

proc main() =
  var model = "kokoro-en-q5.gguf"
  var voice = "af_heart"
  var speed = 1.0'f32
  var output = "output.wav"
  var args = commandLineParams()

  if args.len == 0 or args[0] in ["-h", "--help", "help"]:
    echo Usage; quit(0)

  let cmd = args[0]
  args.delete(0)

  # Simple arg parser: consume -flag value pairs, collect positionals
  var positionals: seq[string]
  var i = 0
  while i < args.len:
    let a = args[i]
    if a in ["-m", "--model"] and i + 1 < args.len:
      model = args[i + 1]; i += 2
    elif a in ["-v", "--voice"] and i + 1 < args.len:
      voice = args[i + 1]; i += 2
    elif a in ["-s", "--speed"] and i + 1 < args.len:
      speed = parseFloat(args[i + 1]).float32; i += 2
    elif a in ["-o", "--output"] and i + 1 < args.len:
      output = args[i + 1]; i += 2
    elif a in ["-h", "--help"]:
      echo Usage; quit(0)
    elif a.startsWith("-"):
      echo "Unknown option: ", a; quit(1)
    else:
      positionals.add a; inc i

  case cmd
  of "synth", "say":
    if positionals.len == 0:
      echo "Usage: tts_cli synth \"Hello world\" [-m model.gguf] [-v voice]"
      quit(1)
    let text = positionals.join(" ")
    var e = newTTSEngine()
    e.loadModel(model, voice)
    let audio = e.synthesize(text, voice, speed)
    audio.writeWav(output)
    let dur = formatFloat(audio.samples.len.float / audio.sampleRate.float, ffDecimal, 1)
    echo "Audio: ", dur, "s → ", output
    e.close()

  of "voices":
    var e = newTTSEngine()
    e.loadModel(model, voice)
    let voices = e.listVoices()
    echo voices.len, " voices:"
    for v in voices: echo "  ", v
    e.close()

  of "download":
    downloadModel(if positionals.len > 0: positionals[0] else: "")

  of "models", "list":
    listModels()

  else:
    echo "Unknown command: ", cmd
    echo Usage; quit(1)

when isMainModule:
  main()
