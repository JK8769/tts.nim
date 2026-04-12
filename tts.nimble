# Package
version       = "0.2.0"
author        = "owaf"
description   = "Native TTS engine for Nim — Kokoro via MLX/ggml. No Python, no ONNX."
license       = "MIT"
srcDir        = "src"
binDir        = "bin"
bin           = @["tts_cli"]
installDirs   = @["tts", "lib", "res", "include"]
installFiles  = @["tts.nim", "config.nims"]

# Dependencies
requires "nim >= 2.0.0"
requires "zippy >= 0.10.0"
requires "https://github.com/JK8769/docopt.nim >= 0.8.0"

# ── Platform detection ────────────────────────────────────────────

proc isAppleSilicon(): bool =
  hostOS == "macosx" and hostCPU == "arm64"

proc ensureSubmodules() =
  ## nimble doesn't clone submodules — init them if missing.
  let root = thisDir()
  if not fileExists(root & "/vendor/espeak-ng/CMakeLists.txt"):
    echo "Initializing git submodules..."
    exec "git -C " & root & " submodule update --init --recursive"

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

proc downloadFile(name, url, dest: string, required = true) =
  if fileExists(dest):
    echo name & " ✓"
    return
  echo "Downloading " & name & "..."
  if required:
    exec "curl -L --progress-bar --fail -o " & dest & " " & url
  else:
    # Optional download — warn but don't fail on 404
    let code = gorgeEx("curl -L --progress-bar --fail -o " & dest & " " & url)
    if code.exitCode != 0:
      echo "  ⚠ Download failed for " & name & " (place manually in " & dest & ")"
      if fileExists(dest): rmFile dest  # remove partial download

proc downloadModel(name, file: string) =
  let dest = thisDir() & "/src/res/models/" & file
  mkDir thisDir() & "/src/res/models"
  let repo = "JK8769/tts.nim"
  downloadFile(name, "https://github.com/" & repo &
    "/releases/latest/download/" & file, dest)

proc downloadMlxModel(name, file: string) =
  ## Download and extract a tar.gz model archive for MLX backend.
  let modelsDir = thisDir() & "/src/res/models"
  let dirName = file.replace(".tar.gz", "")
  let dest = modelsDir & "/" & dirName
  if dirExists(dest):
    echo name & " ✓"
    return
  mkDir modelsDir
  let repo = "JK8769/tts.nim"
  let tarball = modelsDir & "/" & file
  downloadFile(name, "https://github.com/" & repo &
    "/releases/latest/download/" & file, tarball, required = false)
  if fileExists(tarball):
    exec "tar -xzf " & tarball & " -C " & modelsDir
    rmFile tarball

proc downloadWhisperMlx() =
  ## Download Whisper base.en model (safetensors) from HuggingFace.
  ## Uses English-only model — multilingual requires mel spectrogram alignment work.
  let dest = thisDir() & "/src/res/models/whisper-base.en-mlx"
  if dirExists(dest) and fileExists(dest & "/model.safetensors"):
    echo "whisper-base.en-mlx ✓"
    return
  mkDir dest
  let base = "https://huggingface.co/openai/whisper-base.en/resolve/main"
  for f in @["config.json", "model.safetensors", "vocab.json"]:
    downloadFile("whisper " & f, base & "/" & f, dest & "/" & f)

proc buildWhisper() =
  let root = thisDir()
  let whisperSrc = root & "/vendor/whisper.cpp"
  let buildDir = whisperSrc & "/build"
  let libDir = root & "/src/lib/whisper"
  mkDir buildDir
  mkDir libDir

  var args = "-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON " &
             "-DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=OFF"
  if hostOS == "macosx":
    args &= " -DGGML_METAL=ON -DGGML_ACCELERATE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple"
  else:
    args &= " -DGGML_METAL=OFF"

  exec "cmake -S " & whisperSrc & " -B " & buildDir & " " & args
  exec "cmake --build " & buildDir & " --config Release -j4"
  # Copy shared libs
  exec "find " & buildDir & " -name 'libwhisper*.dylib' -o -name 'libwhisper*.so*' | " &
       "xargs -I{} cp -P {} " & libDir & "/"
  exec "find " & buildDir & " -name 'libggml*.dylib' -o -name 'libggml*.so*' | " &
       "xargs -I{} cp -P {} " & libDir & "/"

proc stageHeaders() =
  ## Copy vendor headers into src/include/ so nimble installDirs picks them up.
  let root = thisDir()
  let incDir = root & "/src/include"
  mkDir incDir
  # ggml headers — only needed for GGML backend
  if not isAppleSilicon():
    if fileExists(root & "/vendor/ggml/include/ggml.h"):
      exec "cp -r " & root & "/vendor/ggml/include/. " & incDir & "/"
    elif fileExists(root & "/vendor/whisper.cpp/ggml/include/ggml.h"):
      exec "cp -r " & root & "/vendor/whisper.cpp/ggml/include/. " & incDir & "/"
  exec "cp -r " & root & "/vendor/espeak-ng/src/include/. " & incDir & "/"

proc downloadWhisperModel(name, file: string) =
  let dest = thisDir() & "/src/res/models/" & file
  if fileExists(dest):
    echo name & " ✓"
    return
  mkDir thisDir() & "/src/res/models"
  echo "Downloading " & name & "..."
  exec "curl -L --progress-bar --fail -o " & dest &
       " https://huggingface.co/ggerganov/whisper.cpp/resolve/main/" & file

proc buildMlx() =
  let root = thisDir()
  let mlxSrc = root & "/vendor/mlx-c-src"
  let buildDir = mlxSrc & "/build"
  let installDir = root & "/vendor/mlx"
  if fileExists(installDir & "/lib/libmlxc.a") and fileExists(installDir & "/lib/libmlx.a"):
    echo "mlx-c ✓ (already built)"
    return
  mkDir buildDir
  exec "cmake -S " & mlxSrc & " -B " & buildDir &
       " -DCMAKE_BUILD_TYPE=Release -DMLX_C_BUILD_EXAMPLES=OFF"
  exec "cmake --build " & buildDir & " -j"
  exec "cmake --install " & buildDir & " --prefix " & installDir
  # Copy MLX core libs that mlx-c depends on
  exec "cp " & buildDir & "/_deps/mlx-build/libmlx.a " & installDir & "/lib/"
  exec "cp " & buildDir & "/_deps/mlx-build/mlx/io/libgguflib.a " & installDir & "/lib/"
  exec "cp " & buildDir & "/_deps/fmt-build/libfmt.a " & installDir & "/lib/"
  exec "cp " & buildDir & "/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib " &
       installDir & "/lib/"

before install:
  ensureSubmodules()
  buildEspeakNg()
  if isAppleSilicon():
    echo "=== Apple Silicon detected — building MLX backend ==="
    buildMlx()
    downloadMlxModel("kokoro-mlx-q4", "kokoro-mlx-q4.tar.gz")
    downloadMlxModel("kokoro-zh-mlx-q4", "kokoro-zh-mlx-q4.tar.gz")
    downloadWhisperMlx()
    downloadMlxModel("silero-vad", "silero-vad.tar.gz")
  else:
    echo "=== Building GGML backend ==="
    buildGgml()
    buildWhisper()
    downloadModel("kokoro-en", "kokoro-en-q5.gguf")
    downloadModel("kokoro-zh", "kokoro-v1.1-zh-q5.gguf")
    downloadWhisperModel("whisper-base.en", "ggml-base.en.bin")
  stageHeaders()

task build_deps, "Build native deps (auto-detects platform: MLX on Apple Silicon, GGML elsewhere)":
  ensureSubmodules()
  buildEspeakNg()
  if isAppleSilicon():
    buildMlx()
  else:
    buildGgml()
    buildWhisper()

task build_mlx, "Build only mlx-c from vendor source":
  buildMlx()

task build_ggml, "Build only ggml from vendor source":
  buildGgml()
  buildWhisper()

task download, "Download models (auto-detects platform)":
  if isAppleSilicon():
    downloadMlxModel("kokoro-mlx-q4", "kokoro-mlx-q4.tar.gz")
    downloadMlxModel("kokoro-zh-mlx-q4", "kokoro-zh-mlx-q4.tar.gz")
    downloadWhisperMlx()
    downloadMlxModel("silero-vad", "silero-vad.tar.gz")
  else:
    downloadModel("kokoro-en", "kokoro-en-q5.gguf")
    downloadModel("kokoro-zh", "kokoro-v1.1-zh-q5.gguf")
    downloadWhisperModel("whisper-base.en", "ggml-base.en.bin")

task voices, "Download all English voices from HuggingFace (requires torch, safetensors)":
  let modelsDir = thisDir() & "/src/res/models"
  for model in ["kokoro-mlx", "kokoro-mlx-q4"]:
    let dir = modelsDir & "/" & model
    if dirExists(dir):
      exec "python3 " & thisDir() & "/scripts/download_voices.py " & dir & " --english-only"

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
