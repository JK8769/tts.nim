# Package
version       = "0.1.0"
author        = "owaf"
description   = "Native TTS engine for Nim — Kokoro via ggml. No Python, no ONNX."
license       = "MIT"
srcDir        = "src"
binDir        = "bin"
bin           = @["tts_cli"]
installDirs   = @["tts", "lib", "res", "include"]
installFiles  = @["tts.nim"]

# Dependencies
requires "nim >= 2.0.0"
requires "zippy >= 0.10.0"
requires "https://github.com/JK8769/docopt.nim >= 0.8.0"

proc buildGgml() =
  let root = thisDir()
  let ggmlSrc = root & "/vendor/ggml"
  let buildDir = ggmlSrc & "/build"
  let libDir = root & "/src/lib"
  mkDir buildDir
  mkDir libDir

  var args = "-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF " &
             "-DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF"
  if hostOS == "macosx":
    args &= " -DGGML_METAL=ON -DGGML_ACCELERATE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple"
  else:
    args &= " -DGGML_METAL=OFF"

  exec "cmake -S " & ggmlSrc & " -B " & buildDir & " " & args
  exec "cmake --build " & buildDir & " --config Release -j4"
  exec "find " & buildDir & " -name 'libggml*.a' -exec cp {} " & libDir & "/ \\;"

proc buildEspeakNg() =
  let root = thisDir()
  let espeakSrc = root & "/vendor/espeak-ng"
  let buildDir = espeakSrc & "/build"
  let libDir = root & "/src/lib"
  let dataDir = root & "/src/res/data/espeak"
  mkDir buildDir
  mkDir libDir
  mkDir dataDir

  let args = "-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF " &
             "-DENABLE_TESTS=OFF -DCOMPILE_INTONATIONS=ON " &
             "-DUSE_MBROLA=OFF -DUSE_LIBPCAUDIO=OFF -DUSE_LIBSONIC=OFF " &
             "-DUSE_ASYNC=OFF -DUSE_KLATT=OFF -DUSE_SPEECHPLAYER=OFF"

  exec "cmake -S " & espeakSrc & " -B " & buildDir & " " & args
  exec "cmake --build " & buildDir & " --config Release -j4"

  # Copy static libs
  exec "cp " & buildDir & "/src/libespeak-ng/libespeak-ng.a " & libDir & "/"
  exec "cp " & buildDir & "/src/ucd-tools/libucd.a " & libDir & "/"

  # Copy compiled data (core files + needed language dicts only)
  let builtData = buildDir & "/espeak-ng-data"
  for f in @["phondata", "phonindex", "phontab", "intonations", "phondata-manifest"]:
    exec "cp " & builtData & "/" & f & " " & dataDir & "/"
  # Ship all lang families (584K) so espeak can init any voice.
  # Default dict: en only. Chinese uses native Bopomofo (no espeak dict needed).
  # Add more with: nimble lang add es fr ja
  exec "cp -r " & builtData & "/lang " & dataDir & "/"
  for lang in @["en"]:
    exec "cp " & builtData & "/" & lang & "_dict " & dataDir & "/ 2>/dev/null || true"

proc downloadModel(name, file: string) =
  let dest = thisDir() & "/src/res/models/" & file
  if fileExists(dest):
    echo name & " ✓"
    return
  mkDir thisDir() & "/src/res/models"
  let repo = "JK8769/tts.nim"
  echo "Downloading " & name & "..."
  exec "curl -L --progress-bar --fail -o " & dest &
       " https://github.com/" & repo & "/releases/latest/download/" & file

proc stageHeaders() =
  ## Copy vendor headers into src/include/ so nimble installDirs picks them up.
  let root = thisDir()
  let incDir = root & "/src/include"
  mkDir incDir
  exec "cp -r " & root & "/vendor/ggml/include/. " & incDir & "/"
  exec "cp -r " & root & "/vendor/espeak-ng/src/include/. " & incDir & "/"

before install:
  buildGgml()
  buildEspeakNg()
  downloadModel("kokoro-en", "kokoro-en-q5.gguf")
  downloadModel("kokoro-zh", "kokoro-v1.1-zh-q5.gguf")
  stageHeaders()

task build_deps, "Build ggml and espeak-ng from vendor source":
  buildGgml()
  buildEspeakNg()

task download, "Download default TTS models":
  downloadModel("kokoro-en", "kokoro-en-q5.gguf")
  downloadModel("kokoro-zh", "kokoro-v1.1-zh-q5.gguf")

proc basename(path: string): string =
  let i = path.rfind('/')
  if i >= 0: path[i+1..^1] else: path

task lang, "Manage espeak languages: nimble lang [list|add|remove] [lang...]":
  let root = thisDir()
  let dataDir = root & "/src/res/data/espeak"
  let buildData = root & "/vendor/espeak-ng/build/espeak-ng-data"
  # commandLineParams is provided by nimble — contains args after task name
  var positional: seq[string]
  for p in commandLineParams:
    if not p.startsWith("-"): positional.add p
  let subcmd = if positional.len > 0: positional[0] else: "list"

  if subcmd == "list":
    echo "Phonemizers:"
    echo "  Bopomofo (native):    zh (Kokoro z* voices)"
    var installed: seq[string]
    if dirExists(dataDir):
      for f in listFiles(dataDir):
        let name = basename(f)
        if name.endsWith("_dict"):
          installed.add name[0 ..< name.len - 5]
    echo "  espeak-ng (installed): ", if installed.len > 0: installed.join(", ") else: "(none)"
    if dirExists(buildData):
      var available: seq[string]
      for f in listFiles(buildData):
        let name = basename(f)
        if name.endsWith("_dict") and name[0 ..< name.len - 5] notin installed:
          available.add name[0 ..< name.len - 5]
      if available.len > 0:
        echo "  espeak-ng (available): ", available.join(", ")
    else:
      echo "Run 'nimble build_deps' first to see available languages."

  elif subcmd == "add":
    if not dirExists(buildData):
      echo "Run 'nimble build_deps' first."
      quit(1)
    mkDir dataDir
    for lang in positional[1..^1]:
      let src = buildData & "/" & lang & "_dict"
      if not fileExists(src):
        echo "Not found: ", lang
      else:
        exec "cp " & src & " " & dataDir & "/"
        echo "Added: ", lang

  elif subcmd == "remove":
    for lang in positional[1..^1]:
      let path = dataDir & "/" & lang & "_dict"
      if fileExists(path):
        rmFile path
        echo "Removed: ", lang
      else:
        echo "Not installed: ", lang

  else:
    echo "Usage: nimble lang [list|add|remove] [lang...]"
    echo "  nimble lang              List installed & available languages"
    echo "  nimble lang add es fr    Add Spanish and French dicts"
    echo "  nimble lang remove es    Remove Spanish dict"
