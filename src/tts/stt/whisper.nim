## Speech-to-text via whisper.cpp (shared library).
## Whisper uses its own ggml (in libwhisper.dylib) to avoid symbol
## conflicts with the TTS ggml (linked statically).

import std/os
import ../common

const
  bridgeDir = currentSourcePath().parentDir()
  bridgeSrc = bridgeDir / "whisper_bridge.c"
  ttsSrcRoot = bridgeDir.parentDir().parentDir()
  whisperInclude = ttsSrcRoot.parentDir() / "vendor" / "whisper.cpp" / "include"
  whisperLib = ttsSrcRoot / "lib" / "whisper"

{.passC: "-I" & bridgeDir & " -I" & whisperInclude.}
{.compile: bridgeSrc.}
{.passL: "-L" & whisperLib & " -lwhisper -Wl,-rpath," & whisperLib.}

const WHISPER_SAMPLE_RATE* = 16000

type
  WhisperSTTObj {.importc: "WhisperSTT", header: "whisper_bridge.h", incompleteStruct.} = object
  WhisperSTTPtr = ptr WhisperSTTObj

proc whisper_stt_create(modelPath: cstring): WhisperSTTPtr {.importc, header: "whisper_bridge.h".}
proc whisper_stt_transcribe(w: WhisperSTTPtr, samples: ptr cfloat, nSamples: cint,
                            language: cstring): cstring {.importc, header: "whisper_bridge.h".}
proc whisper_stt_free_text(text: cstring) {.importc, header: "whisper_bridge.h".}
proc whisper_stt_destroy(w: WhisperSTTPtr) {.importc, header: "whisper_bridge.h".}

# ---- Nim API ----

type
  SpeechRecognizer* = ref object
    handle: WhisperSTTPtr
    language*: string

proc newSpeechRecognizer*(modelPath: string, language: string = "en"): SpeechRecognizer =
  ## Load a whisper model. Uses same search paths as TTS models (findModel).
  let resolved = findModel(modelPath)
  if resolved.len == 0:
    raise newException(IOError, "Whisper model not found: " & modelPath &
      "\n  Searched: ./, " & pkgModelDir & "/")
  let h = whisper_stt_create(resolved.cstring)
  if h == nil:
    raise newException(IOError, "Failed to load whisper model: " & resolved)
  SpeechRecognizer(handle: h, language: language)

proc transcribe*(r: SpeechRecognizer, samples: openArray[float32]): string =
  ## Transcribe float32 PCM audio (must be 16kHz mono).
  ## Returns the recognized text, or empty string on failure.
  if samples.len == 0: return ""
  let text = whisper_stt_transcribe(r.handle, unsafeAddr samples[0],
                                    cint(samples.len), r.language.cstring)
  if text == nil: return ""
  result = $text
  whisper_stt_free_text(text)

proc close*(r: SpeechRecognizer) =
  if r.handle != nil:
    whisper_stt_destroy(r.handle)
    r.handle = nil
