/* whisper_bridge.c — whisper.cpp bridge implementation.
 * Links against libwhisper.dylib (shared) to avoid ggml symbol conflicts
 * with the TTS ggml (linked statically). */

#include "whisper.h"
#include "whisper_bridge.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct WhisperSTT {
    struct whisper_context* ctx;
};

WhisperSTT* whisper_stt_create(const char* modelPath) {
    WhisperSTT* w = (WhisperSTT*)calloc(1, sizeof(WhisperSTT));
    if (!w) return NULL;

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;

    w->ctx = whisper_init_from_file_with_params(modelPath, cparams);
    if (!w->ctx) {
        free(w);
        return NULL;
    }
    return w;
}

char* whisper_stt_transcribe(WhisperSTT* w, const float* samples, int nSamples,
                             const char* language) {
    if (!w || !w->ctx || !samples || nSamples <= 0) return NULL;

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_realtime   = false;
    params.print_progress   = false;
    params.print_special    = false;
    params.print_timestamps = false;
    params.single_segment   = false;
    params.no_timestamps    = true;
    params.n_threads        = 4;

    if (language && strcmp(language, "auto") != 0) {
        params.language = language;
    } else {
        params.language = "en";
    }

    if (whisper_full(w->ctx, params, samples, nSamples) != 0) {
        return NULL;
    }

    /* Concatenate all segments */
    int n_segments = whisper_full_n_segments(w->ctx);
    if (n_segments <= 0) return NULL;

    /* First pass: compute total length */
    size_t total_len = 0;
    for (int i = 0; i < n_segments; i++) {
        const char* text = whisper_full_get_segment_text(w->ctx, i);
        if (text) total_len += strlen(text);
    }
    if (total_len == 0) return NULL;

    char* result = (char*)malloc(total_len + 1);
    if (!result) return NULL;

    /* Second pass: copy */
    size_t pos = 0;
    for (int i = 0; i < n_segments; i++) {
        const char* text = whisper_full_get_segment_text(w->ctx, i);
        if (text) {
            size_t len = strlen(text);
            memcpy(result + pos, text, len);
            pos += len;
        }
    }
    result[pos] = '\0';
    return result;
}

void whisper_stt_free_text(char* text) {
    free(text);
}

void whisper_stt_destroy(WhisperSTT* w) {
    if (!w) return;
    if (w->ctx) whisper_free(w->ctx);
    free(w);
}
