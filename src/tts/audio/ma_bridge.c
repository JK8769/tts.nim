/* ma_bridge.c — miniaudio bridge implementation.
 * Ring-buffer based playback and capture for Nim. */

#define MINIAUDIO_IMPLEMENTATION
#include "../../../vendor/miniaudio/miniaudio.h"
#include "ma_bridge.h"

#include <stdlib.h>
#include <string.h>

/* ---- Ring buffer (lock-free, single producer / single consumer) ---- */

#define RING_FRAMES (1 << 18)  /* 256K frames ~= 10.9s at 24kHz */
#define RING_MASK   (RING_FRAMES - 1)

typedef struct {
    float    buf[RING_FRAMES];
    uint32_t head;  /* written by producer, read by consumer */
    uint32_t tail;  /* written by consumer, read by producer */
} RingBuf;

static uint32_t ring_avail(const RingBuf* r) {
    return r->head - r->tail;  /* works with unsigned wrap */
}

static uint32_t ring_free(const RingBuf* r) {
    return RING_FRAMES - ring_avail(r);
}

static uint32_t ring_write(RingBuf* r, const float* src, uint32_t n) {
    uint32_t avail = ring_free(r);
    if (n > avail) n = avail;
    for (uint32_t i = 0; i < n; i++) {
        r->buf[(r->head + i) & RING_MASK] = src[i];
    }
    __atomic_store_n(&r->head, r->head + n, __ATOMIC_RELEASE);
    return n;
}

static uint32_t ring_read(RingBuf* r, float* dst, uint32_t n) {
    uint32_t avail = ring_avail(r);
    if (n > avail) n = avail;
    for (uint32_t i = 0; i < n; i++) {
        dst[i] = r->buf[(r->tail + i) & RING_MASK];
    }
    __atomic_store_n(&r->tail, r->tail + n, __ATOMIC_RELEASE);
    return n;
}

static void ring_flush(RingBuf* r) {
    __atomic_store_n(&r->tail, r->head, __ATOMIC_RELEASE);
}

/* ---- Playback ---- */

struct MaPlayback {
    ma_device device;
    RingBuf   ring;
    int       idle;  /* 1 when ring is empty and device has drained */
};

static void playback_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    MaPlayback* p = (MaPlayback*)pDevice->pUserData;
    float* out = (float*)pOutput;
    uint32_t read = ring_read(&p->ring, out, frameCount);
    /* Zero-fill remainder */
    if (read < frameCount) {
        memset(out + read, 0, (frameCount - read) * sizeof(float));
        if (ring_avail(&p->ring) == 0) {
            p->idle = 1;
        }
    }
    (void)pInput;
}

MaPlayback* ma_playback_create(uint32_t sampleRate) {
    MaPlayback* p = (MaPlayback*)calloc(1, sizeof(MaPlayback));
    if (!p) return NULL;

    ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format   = ma_format_f32;
    cfg.playback.channels = 1;
    cfg.sampleRate        = sampleRate;
    cfg.dataCallback      = playback_callback;
    cfg.pUserData         = p;

    if (ma_device_init(NULL, &cfg, &p->device) != MA_SUCCESS) {
        free(p);
        return NULL;
    }
    p->idle = 1;
    return p;
}

uint32_t ma_playback_write(MaPlayback* p, const float* samples, uint32_t frameCount) {
    p->idle = 0;
    return ring_write(&p->ring, samples, frameCount);
}

int ma_playback_start(MaPlayback* p) {
    p->idle = 0;
    return ma_device_start(&p->device) == MA_SUCCESS ? 0 : -1;
}

int ma_playback_stop(MaPlayback* p) {
    return ma_device_stop(&p->device) == MA_SUCCESS ? 0 : -1;
}

int ma_playback_is_idle(MaPlayback* p) {
    return p->idle;
}

uint32_t ma_playback_frames_queued(MaPlayback* p) {
    return ring_avail(&p->ring);
}

void ma_playback_flush(MaPlayback* p) {
    ring_flush(&p->ring);
    p->idle = 1;
}

void ma_playback_destroy(MaPlayback* p) {
    if (!p) return;
    ma_device_uninit(&p->device);
    free(p);
}

/* ---- Capture ---- */

struct MaCapture {
    ma_device device;
    RingBuf   ring;
};

static void capture_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    MaCapture* c = (MaCapture*)pDevice->pUserData;
    ring_write(&c->ring, (const float*)pInput, frameCount);
    (void)pOutput;
}

MaCapture* ma_capture_create(uint32_t sampleRate) {
    MaCapture* c = (MaCapture*)calloc(1, sizeof(MaCapture));
    if (!c) return NULL;

    ma_device_config cfg = ma_device_config_init(ma_device_type_capture);
    cfg.capture.format   = ma_format_f32;
    cfg.capture.channels = 1;
    cfg.sampleRate       = sampleRate;
    cfg.dataCallback     = capture_callback;
    cfg.pUserData        = c;

    if (ma_device_init(NULL, &cfg, &c->device) != MA_SUCCESS) {
        free(c);
        return NULL;
    }
    return c;
}

uint32_t ma_capture_read(MaCapture* c, float* samples, uint32_t frameCount) {
    return ring_read(&c->ring, samples, frameCount);
}

int ma_capture_start(MaCapture* c) {
    return ma_device_start(&c->device) == MA_SUCCESS ? 0 : -1;
}

int ma_capture_stop(MaCapture* c) {
    return ma_device_stop(&c->device) == MA_SUCCESS ? 0 : -1;
}

uint32_t ma_capture_frames_available(MaCapture* c) {
    return ring_avail(&c->ring);
}

void ma_capture_destroy(MaCapture* c) {
    if (!c) return;
    ma_device_uninit(&c->device);
    free(c);
}
