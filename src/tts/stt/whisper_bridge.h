/* whisper_bridge.h — thin C bridge for whisper.cpp.
 * Hides whisper's large param structs behind a simple transcribe API. */

#ifndef WHISPER_BRIDGE_H
#define WHISPER_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WhisperSTT WhisperSTT;

/* Create a whisper context from a GGUF model file.
 * Returns NULL on failure. */
WhisperSTT* whisper_stt_create(const char* modelPath);

/* Transcribe float32 PCM audio (16kHz mono).
 * Returns a malloc'd UTF-8 string (caller must free with whisper_stt_free_text).
 * language: ISO code ("en", "zh", "auto") or NULL for auto-detect.
 * Returns NULL on failure. */
char* whisper_stt_transcribe(WhisperSTT* w, const float* samples, int nSamples,
                             const char* language);

/* Free a string returned by whisper_stt_transcribe. */
void whisper_stt_free_text(char* text);

/* Destroy the whisper context. */
void whisper_stt_destroy(WhisperSTT* w);

#ifdef __cplusplus
}
#endif

#endif /* WHISPER_BRIDGE_H */
