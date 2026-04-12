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

/* Mute/unmute: mux ring buffer output to silence (1) or real data (0).
 * Ring buffer data is preserved while muted — unmute resumes playback. */
void ma_playback_set_mute(MaPlayback* p, int mute);

/* Destroy the device and free all memory. */
void ma_playback_destroy(MaPlayback* p);

/* ---- Capture ---- */

/* Create a capture device. Mono float32 at the given sample rate.
 * Returns NULL on failure. */
MaCapture* ma_capture_create(uint32_t sampleRate);

/* Create a capture device with echo cancellation.
 * On macOS, uses Apple's VoiceProcessingIO audio unit for hardware AEC.
 * On other platforms, falls back to ma_capture_create(). */
MaCapture* ma_capture_create_aec(uint32_t sampleRate);

/* Read captured PCM samples (float32, mono). Returns frames actually read. */
uint32_t ma_capture_read(MaCapture* c, float* samples, uint32_t frameCount);

/* Start / stop the capture device. */
int ma_capture_start(MaCapture* c);
int ma_capture_stop(MaCapture* c);

/* Number of frames available to read. */
uint32_t ma_capture_frames_available(MaCapture* c);

/* Discard all buffered capture audio. */
void ma_capture_flush(MaCapture* c);

/* Enable energy-based speech detection in the audio capture thread.
 * threshold: RMS energy level (0.01-0.05 typical). Set 0 to disable. */
void ma_capture_set_speech_detect(MaCapture* c, float threshold);

/* Set a speech energy band: only trigger when RMS is in [low, high].
 * Rejects both quiet noise (below low) and loud transients (above high). */
void ma_capture_set_speech_range(MaCapture* c, float low, float high);

/* Returns 1 if speech was detected since last reset. Lock-free. */
int ma_capture_speech_detected(MaCapture* c);

/* Monotonic count of audio buffers that exceeded the energy threshold.
 * Read at start and end of a window to count speech frames without resetting. */
uint32_t ma_capture_speech_frames(MaCapture* c);

/* Reset the speech detection flag (not the frame counter). */
void ma_capture_speech_reset(MaCapture* c);

/* Destroy the capture device and free all memory. */
void ma_capture_destroy(MaCapture* c);

/* ---- Duplex (VPIO) ---- */

/* Create paired playback + capture through a single VoiceProcessingIO audio unit.
 * On macOS: both handles share one VPIO for optimal echo cancellation.
 * The VPIO feeds the exact playback signal as AEC reference — better than
 * standalone AEC which relies on system output tap.
 * On other platforms: returns -1 (use separate devices instead).
 * Returns 0 on success. */
int ma_duplex_create_vpio(uint32_t playbackRate, uint32_t captureRate,
                          MaPlayback **outPlayback, MaCapture **outCapture);

#ifdef __cplusplus
}
#endif

#endif /* MA_BRIDGE_H */
