/* ma_bridge.h — thin C bridge between miniaudio and Nim.
 * Hides miniaudio's large structs behind opaque pointers.
 * Compile ma_bridge.c alongside your Nim project. */

#ifndef MA_BRIDGE_H
#define MA_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles */
typedef struct MaPlayback MaPlayback;
typedef struct MaCapture  MaCapture;

/* ---- Playback ---- */

/* Create a playback device. Mono float32 at the given sample rate.
 * Returns NULL on failure. */
MaPlayback* ma_playback_create(uint32_t sampleRate);

/* Feed PCM samples (float32, mono). Copies into internal ring buffer.
 * Returns number of frames actually written. */
uint32_t ma_playback_write(MaPlayback* p, const float* samples, uint32_t frameCount);

/* Start / stop the device (audio flows from ring buffer to speakers). */
int ma_playback_start(MaPlayback* p);
int ma_playback_stop(MaPlayback* p);

/* Returns 1 if the device has finished playing all buffered data. */
int ma_playback_is_idle(MaPlayback* p);

/* Number of frames still queued for playback. */
uint32_t ma_playback_frames_queued(MaPlayback* p);

/* Discard all buffered audio (for interrupt). */
void ma_playback_flush(MaPlayback* p);

/* Destroy the device and free all memory. */
void ma_playback_destroy(MaPlayback* p);

/* ---- Capture ---- */

/* Create a capture device. Mono float32 at the given sample rate.
 * Returns NULL on failure. */
MaCapture* ma_capture_create(uint32_t sampleRate);

/* Read captured PCM samples (float32, mono). Returns frames actually read. */
uint32_t ma_capture_read(MaCapture* c, float* samples, uint32_t frameCount);

/* Start / stop the capture device. */
int ma_capture_start(MaCapture* c);
int ma_capture_stop(MaCapture* c);

/* Number of frames available to read. */
uint32_t ma_capture_frames_available(MaCapture* c);

/* Destroy the capture device and free all memory. */
void ma_capture_destroy(MaCapture* c);

#ifdef __cplusplus
}
#endif

#endif /* MA_BRIDGE_H */
