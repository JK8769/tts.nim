/* ma_bridge.c — miniaudio bridge implementation.
 * Ring-buffer based playback and capture for Nim. */

#define MINIAUDIO_IMPLEMENTATION
#include "../../../vendor/miniaudio/miniaudio.h"
#include "ma_bridge.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef __APPLE__
#include <AudioToolbox/AudioToolbox.h>
#endif

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

/* VPIO duplex state (macOS) — shared between playback and capture handles.
 * Uses pointers to MaPlayback/MaCapture (opaque, declared in header). */
#ifdef __APPLE__
typedef struct VPIODuplex {
    AudioComponentInstance auVPIO;
    MaPlayback *playback;   /* points to the playback handle (ring + idle) */
    MaCapture  *capture;    /* points to the capture handle (ring) */
    /* Sample rate mapping (VPIO runs at hardware rate, app uses different rates) */
    uint32_t playAppRate;   /* app-facing playback rate (e.g. 24000) */
    uint32_t playHwRate;    /* VPIO playback rate (e.g. 48000) */
    uint32_t capAppRate;    /* app-facing capture rate (e.g. 16000) */
    uint32_t capHwRate;     /* VPIO capture rate (e.g. 48000) */
} VPIODuplex;
#endif

/* ---- Playback ---- */

#define FADE_MS 200  /* fade-in duration after unmute (milliseconds) */

struct MaPlayback {
    ma_device device;
    RingBuf   ring;
    int       idle;  /* 1 when ring is empty and device has drained */
    int       muted; /* 1 = output silence, ring data preserved (mux) */
    volatile uint32_t fadeRemaining;  /* frames left in fade-in ramp (0 = done) */
    uint32_t  fadeTotal;              /* total fade frames (set from sample rate) */
#ifdef __APPLE__
    int useVPIO;            /* 0=miniaudio, 2=duplex VPIO */
    VPIODuplex *duplex;
#endif
};

/* Apply linear fade-in ramp to output buffer. Called from audio callbacks. */
static void apply_fade_in(MaPlayback* p, float* out, uint32_t frameCount) {
    if (p->fadeRemaining == 0) return;
    uint32_t total = p->fadeTotal;
    uint32_t rem = p->fadeRemaining;
    uint32_t n = rem < frameCount ? rem : frameCount;
    for (uint32_t i = 0; i < n; i++) {
        float gain = (float)(total - rem + i) / (float)total;
        out[i] *= gain;
    }
    p->fadeRemaining = rem - n;
}

static void playback_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    MaPlayback* p = (MaPlayback*)pDevice->pUserData;
    float* out = (float*)pOutput;
    if (p->muted) {
        memset(out, 0, frameCount * sizeof(float));
    } else {
        uint32_t read = ring_read(&p->ring, out, frameCount);
        if (read < frameCount) {
            memset(out + read, 0, (frameCount - read) * sizeof(float));
            if (ring_avail(&p->ring) == 0)
                p->idle = 1;
        }
        apply_fade_in(p, out, frameCount);
    }
    (void)pInput;
}

MaPlayback* ma_playback_create(uint32_t sampleRate) {
    MaPlayback* p = (MaPlayback*)calloc(1, sizeof(MaPlayback));
    if (!p) return NULL;
#ifdef __APPLE__
    p->useVPIO = 0;
#endif

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
    p->fadeTotal = sampleRate * FADE_MS / 1000;
    return p;
}

uint32_t ma_playback_write(MaPlayback* p, const float* samples, uint32_t frameCount) {
    p->idle = 0;
#ifdef __APPLE__
    if (p->useVPIO == 2 && p->duplex->playHwRate != p->duplex->playAppRate) {
        /* Upsample from app rate to VPIO hardware rate before writing to ring.
         * The ring stores hw-rate data so the VPIO callback reads directly. */
        double ratio = (double)p->duplex->playHwRate / (double)p->duplex->playAppRate;
        uint32_t outFrames = (uint32_t)(frameCount * ratio) + 1;
        float *buf = (float*)malloc(outFrames * sizeof(float));
        if (!buf) return 0;
        for (uint32_t i = 0; i < outFrames; i++) {
            double srcPos = (double)i / ratio;
            uint32_t idx = (uint32_t)srcPos;
            float frac = (float)(srcPos - idx);
            if (idx + 1 < frameCount)
                buf[i] = samples[idx] * (1.0f - frac) + samples[idx + 1] * frac;
            else if (idx < frameCount)
                buf[i] = samples[idx];
            else
                buf[i] = 0.0f;
        }
        uint32_t written = ring_write(&p->ring, buf, outFrames);
        free(buf);
        /* Return app-rate frame count consumed */
        return (uint32_t)(written / ratio);
    }
#endif
    return ring_write(&p->ring, samples, frameCount);
}

int ma_playback_start(MaPlayback* p) {
    p->idle = 0;
    p->muted = 0;
    p->fadeRemaining = 0;  /* clean start, no fade */
#ifdef __APPLE__
    if (p->useVPIO) return 0;  /* VPIO already running */
#endif
    return ma_device_start(&p->device) == MA_SUCCESS ? 0 : -1;
}

int ma_playback_stop(MaPlayback* p) {
    p->muted = 1;
#ifdef __APPLE__
    if (p->useVPIO) return 0;  /* VPIO stays running for capture */
#endif
    return ma_device_stop(&p->device) == MA_SUCCESS ? 0 : -1;
}

void ma_playback_set_mute(MaPlayback* p, int mute) {
    if (!mute && p->muted) {
        /* Unmuting: set up fade-in before clearing muted flag */
        p->fadeRemaining = p->fadeTotal;
    }
    p->muted = mute;
}

int ma_playback_is_idle(MaPlayback* p) {
    return p->idle;
}

uint32_t ma_playback_frames_queued(MaPlayback* p) {
    uint32_t hw = ring_avail(&p->ring);
#ifdef __APPLE__
    if (p->useVPIO == 2 && p->duplex->playHwRate != p->duplex->playAppRate) {
        return (uint32_t)(hw * (double)p->duplex->playAppRate / (double)p->duplex->playHwRate);
    }
#endif
    return hw;
}

void ma_playback_flush(MaPlayback* p) {
    ring_flush(&p->ring);
    p->idle = 1;
}

void ma_playback_destroy(MaPlayback* p) {
    if (!p) return;
#ifdef __APPLE__
    if (p->useVPIO) {
        VPIODuplex *d = p->duplex;
        d->playback = NULL;
        if (d->capture == NULL) {
            /* Last reference — tear down VPIO */
            AudioOutputUnitStop(d->auVPIO);
            AudioUnitUninitialize(d->auVPIO);
            AudioComponentInstanceDispose(d->auVPIO);
            free(d);
        }
        free(p);
        return;
    }
#endif
    ma_device_uninit(&p->device);
    free(p);
}

/* ---- Capture ---- */

struct MaCapture {
    RingBuf   ring;
    ma_device device;      /* used when useVPIO == 0 */
    volatile int speechDetected;  /* 1 if consecutive energy exceeds threshold since last reset */
    volatile uint32_t speechFrames; /* count of buffers exceeding threshold (never resets to 0) */
    uint32_t speechConsecutive;   /* consecutive above-threshold buffers (resets on silence) */
    float energyThreshold;        /* RMS low threshold for speech detection */
    float energyThresholdHigh;    /* RMS high threshold (0 = no upper bound) */
#ifdef __APPLE__
    int useVPIO;            /* 0=miniaudio, 1=standalone VPIO, 2=duplex VPIO */
    AudioComponentInstance auVPIO;  /* standalone AEC */
    VPIODuplex *duplex;             /* duplex AEC */
#endif
};

/* Minimum consecutive above-threshold buffers to trigger speechDetected.
 * At ~5ms/callback (VPIO 48kHz, 256 frames), 10 buffers ≈ 50ms.
 * Filters transient noise (key clicks) while catching speech onset quickly. */
#define SPEECH_MIN_CONSECUTIVE 10

static void capture_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    MaCapture* c = (MaCapture*)pDevice->pUserData;
    ring_write(&c->ring, (const float*)pInput, frameCount);
    /* Lightweight energy-based speech detection in audio thread */
    if (c->energyThreshold > 0) {
        const float *samples = (const float*)pInput;
        float sumSq = 0;
        for (uint32_t i = 0; i < frameCount; i++)
            sumSq += samples[i] * samples[i];
        float rms = sqrtf(sumSq / (float)frameCount);
        if (rms > c->energyThreshold && (c->energyThresholdHigh == 0 || rms < c->energyThresholdHigh)) {
            c->speechFrames++;
            if (++c->speechConsecutive >= SPEECH_MIN_CONSECUTIVE)
                c->speechDetected = 1;
        } else {
            c->speechConsecutive = 0;
        }
    }
    (void)pOutput;
}

MaCapture* ma_capture_create(uint32_t sampleRate) {
    MaCapture* c = (MaCapture*)calloc(1, sizeof(MaCapture));
    if (!c) return NULL;
#ifdef __APPLE__
    c->useVPIO = 0;
#endif

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

/* ---- VPIO echo-cancelling capture (macOS) ---- */

#ifdef __APPLE__

static OSStatus vpio_input_callback(
    void *inRefCon,
    AudioUnitRenderActionFlags *ioActionFlags,
    const AudioTimeStamp *inTimeStamp,
    UInt32 inBusNumber,
    UInt32 inNumberFrames,
    AudioBufferList *ioData)
{
    MaCapture *c = (MaCapture *)inRefCon;
    float buf[4096];
    uint32_t n = inNumberFrames > 4096 ? 4096 : (uint32_t)inNumberFrames;

    AudioBufferList abl;
    abl.mNumberBuffers = 1;
    abl.mBuffers[0].mNumberChannels = 1;
    abl.mBuffers[0].mDataByteSize = n * sizeof(float);
    abl.mBuffers[0].mData = buf;

    AudioUnitRenderActionFlags flags = 0;
    if (AudioUnitRender(c->auVPIO, &flags, inTimeStamp, 1, n, &abl) == noErr) {
        ring_write(&c->ring, buf, n);
        /* Lightweight energy-based speech detection in audio thread */
        if (c->energyThreshold > 0) {
            float sumSq = 0;
            for (uint32_t i = 0; i < n; i++)
                sumSq += buf[i] * buf[i];
            float rms = sqrtf(sumSq / (float)n);
            if (rms > c->energyThreshold && (c->energyThresholdHigh == 0 || rms < c->energyThresholdHigh)) {
                c->speechFrames++;
                if (++c->speechConsecutive >= SPEECH_MIN_CONSECUTIVE)
                    c->speechDetected = 1;
            } else {
                c->speechConsecutive = 0;
            }
        }
    }
    return noErr;
}

MaCapture* ma_capture_create_aec(uint32_t sampleRate) {
    MaCapture* c = (MaCapture*)calloc(1, sizeof(MaCapture));
    if (!c) return NULL;
    c->useVPIO = 1;

    /* Find VoiceProcessingIO audio unit */
    AudioComponentDescription desc = {
        .componentType         = kAudioUnitType_Output,
        .componentSubType      = kAudioUnitSubType_VoiceProcessingIO,
        .componentManufacturer = kAudioUnitManufacturer_Apple,
    };
    AudioComponent comp = AudioComponentFindNext(NULL, &desc);
    if (!comp) goto fallback;

    if (AudioComponentInstanceNew(comp, &c->auVPIO) != noErr) goto fallback;

    /* Enable input on bus 1 (microphone) */
    UInt32 one = 1;
    if (AudioUnitSetProperty(c->auVPIO, kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Input, 1, &one, sizeof(one)) != noErr)
        goto fail;

    /* Disable output on bus 0 — we only need capture.
     * On macOS, VPIO still performs AEC by tapping the system audio output
     * through CoreAudio's HAL, even with its own output bus disabled. */
    UInt32 zero = 0;
    AudioUnitSetProperty(c->auVPIO, kAudioOutputUnitProperty_EnableIO,
                         kAudioUnitScope_Output, 0, &zero, sizeof(zero));

    /* Set stream format: float32, mono, requested sample rate */
    AudioStreamBasicDescription fmt = {0};
    fmt.mSampleRate       = (Float64)sampleRate;
    fmt.mFormatID         = kAudioFormatLinearPCM;
    fmt.mFormatFlags      = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    fmt.mFramesPerPacket  = 1;
    fmt.mChannelsPerFrame = 1;
    fmt.mBitsPerChannel   = 32;
    fmt.mBytesPerPacket   = 4;
    fmt.mBytesPerFrame    = 4;
    if (AudioUnitSetProperty(c->auVPIO, kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Output, 1, &fmt, sizeof(fmt)) != noErr)
        goto fail;

    /* Set input callback — called when mic audio is available */
    AURenderCallbackStruct cb = {
        .inputProc       = vpio_input_callback,
        .inputProcRefCon = c,
    };
    if (AudioUnitSetProperty(c->auVPIO, kAudioOutputUnitProperty_SetInputCallback,
            kAudioUnitScope_Global, 0, &cb, sizeof(cb)) != noErr)
        goto fail;

    if (AudioUnitInitialize(c->auVPIO) != noErr) goto fail;

    fprintf(stderr, "[AEC] VoiceProcessingIO capture initialized (%u Hz)\n", sampleRate);
    return c;

fail:
    AudioComponentInstanceDispose(c->auVPIO);
fallback:
    free(c);
    fprintf(stderr, "[AEC] VoiceProcessingIO unavailable, falling back to standard capture\n");
    return ma_capture_create(sampleRate);
}

#else

MaCapture* ma_capture_create_aec(uint32_t sampleRate) {
    /* Non-macOS: no VPIO, use standard capture */
    return ma_capture_create(sampleRate);
}

#endif

uint32_t ma_capture_read(MaCapture* c, float* samples, uint32_t frameCount) {
#ifdef __APPLE__
    if (c->useVPIO == 2 && c->duplex->capHwRate != c->duplex->capAppRate) {
        /* Ring stores hw-rate data. Downsample to app rate on read. */
        double ratio = (double)c->duplex->capHwRate / (double)c->duplex->capAppRate;
        uint32_t hwNeed = (uint32_t)(frameCount * ratio) + 1;
        float *hwBuf = (float*)malloc(hwNeed * sizeof(float));
        if (!hwBuf) return 0;
        uint32_t hwGot = ring_read(&c->ring, hwBuf, hwNeed);
        uint32_t outFrames = (uint32_t)(hwGot / ratio);
        if (outFrames > frameCount) outFrames = frameCount;
        for (uint32_t i = 0; i < outFrames; i++) {
            double srcPos = (double)i * ratio;
            uint32_t idx = (uint32_t)srcPos;
            float frac = (float)(srcPos - idx);
            if (idx + 1 < hwGot)
                samples[i] = hwBuf[idx] * (1.0f - frac) + hwBuf[idx + 1] * frac;
            else if (idx < hwGot)
                samples[i] = hwBuf[idx];
            else
                samples[i] = 0.0f;
        }
        free(hwBuf);
        return outFrames;
    }
#endif
    return ring_read(&c->ring, samples, frameCount);
}

int ma_capture_start(MaCapture* c) {
#ifdef __APPLE__
    if (c->useVPIO == 2) {
        ring_flush(&c->ring);  /* discard stale audio from before start */
        return 0;              /* VPIO already running */
    }
    if (c->useVPIO == 1)
        return AudioOutputUnitStart(c->auVPIO) == noErr ? 0 : -1;
#endif
    return ma_device_start(&c->device) == MA_SUCCESS ? 0 : -1;
}

int ma_capture_stop(MaCapture* c) {
#ifdef __APPLE__
    if (c->useVPIO == 2) return 0;  /* VPIO stays running for playback */
    if (c->useVPIO == 1)
        return AudioOutputUnitStop(c->auVPIO) == noErr ? 0 : -1;
#endif
    return ma_device_stop(&c->device) == MA_SUCCESS ? 0 : -1;
}

uint32_t ma_capture_frames_available(MaCapture* c) {
    uint32_t hw = ring_avail(&c->ring);
#ifdef __APPLE__
    if (c->useVPIO == 2 && c->duplex->capHwRate != c->duplex->capAppRate) {
        return (uint32_t)(hw / ((double)c->duplex->capHwRate / (double)c->duplex->capAppRate));
    }
#endif
    return hw;
}

void ma_capture_flush(MaCapture* c) {
    if (!c) return;
    ring_flush(&c->ring);
}

void ma_capture_set_speech_detect(MaCapture* c, float threshold) {
    if (!c) return;
    c->energyThreshold = threshold;
    c->energyThresholdHigh = 0;  /* no upper bound */
    c->speechDetected = 0;
}

void ma_capture_set_speech_range(MaCapture* c, float low, float high) {
    if (!c) return;
    c->energyThreshold = low;
    c->energyThresholdHigh = high;
    c->speechDetected = 0;
}

int ma_capture_speech_detected(MaCapture* c) {
    if (!c) return 0;
    return c->speechDetected;
}

uint32_t ma_capture_speech_frames(MaCapture* c) {
    if (!c) return 0;
    return c->speechFrames;
}

void ma_capture_speech_reset(MaCapture* c) {
    if (!c) return;
    c->speechDetected = 0;
    c->speechConsecutive = 0;
}

void ma_capture_destroy(MaCapture* c) {
    if (!c) return;
#ifdef __APPLE__
    if (c->useVPIO == 2) {
        VPIODuplex *d = c->duplex;
        d->capture = NULL;
        if (d->playback == NULL) {
            /* Last reference — tear down VPIO */
            AudioOutputUnitStop(d->auVPIO);
            AudioUnitUninitialize(d->auVPIO);
            AudioComponentInstanceDispose(d->auVPIO);
            free(d);
        }
        free(c);
        return;
    }
    if (c->useVPIO == 1) {
        AudioOutputUnitStop(c->auVPIO);
        AudioComponentInstanceDispose(c->auVPIO);
        free(c);
        return;
    }
#endif
    ma_device_uninit(&c->device);
    free(c);
}

/* ---- VPIO Duplex (macOS) ---- */

#ifdef __APPLE__

/* Output callback: feeds playback ring buffer to VPIO output (speakers).
 * Ring buffer stores data at the VPIO hardware rate (pre-resampled by
 * ma_playback_write), so the callback is a simple ring read. */
static OSStatus vpio_duplex_render(
    void *inRefCon,
    AudioUnitRenderActionFlags *ioActionFlags,
    const AudioTimeStamp *inTimeStamp,
    UInt32 inBusNumber,
    UInt32 inNumberFrames,
    AudioBufferList *ioData)
{
    VPIODuplex *d = (VPIODuplex *)inRefCon;
    float *out = (float *)ioData->mBuffers[0].mData;
    if (d->playback->muted) {
        memset(out, 0, inNumberFrames * sizeof(float));
    } else {
        uint32_t read = ring_read(&d->playback->ring, out, (uint32_t)inNumberFrames);
        if (read < inNumberFrames) {
            memset(out + read, 0, (inNumberFrames - read) * sizeof(float));
            if (ring_avail(&d->playback->ring) == 0)
                d->playback->idle = 1;
        }
        apply_fade_in(d->playback, out, (uint32_t)inNumberFrames);
    }
    return noErr;
}

/* Input callback: pulls AEC-processed mic audio into capture ring buffer.
 * Stores at hardware rate; ma_capture_read downsamples to app rate. */
static OSStatus vpio_duplex_input(
    void *inRefCon,
    AudioUnitRenderActionFlags *ioActionFlags,
    const AudioTimeStamp *inTimeStamp,
    UInt32 inBusNumber,
    UInt32 inNumberFrames,
    AudioBufferList *ioData)
{
    VPIODuplex *d = (VPIODuplex *)inRefCon;
    float buf[8192];
    uint32_t n = inNumberFrames > 8192 ? 8192 : (uint32_t)inNumberFrames;

    AudioBufferList abl;
    abl.mNumberBuffers = 1;
    abl.mBuffers[0].mNumberChannels = 1;
    abl.mBuffers[0].mDataByteSize = n * sizeof(float);
    abl.mBuffers[0].mData = buf;

    AudioUnitRenderActionFlags flags = 0;
    if (AudioUnitRender(d->capture->auVPIO, &flags, inTimeStamp, 1, n, &abl) == noErr) {
        ring_write(&d->capture->ring, buf, n);
        /* Lightweight energy-based speech detection in audio thread */
        MaCapture *c = d->capture;
        if (c->energyThreshold > 0) {
            float sumSq = 0;
            for (uint32_t i = 0; i < n; i++)
                sumSq += buf[i] * buf[i];
            float rms = sqrtf(sumSq / (float)n);
            if (rms > c->energyThreshold && (c->energyThresholdHigh == 0 || rms < c->energyThresholdHigh)) {
                c->speechFrames++;
                if (++c->speechConsecutive >= SPEECH_MIN_CONSECUTIVE)
                    c->speechDetected = 1;
            } else {
                c->speechConsecutive = 0;
            }
        }
    }
    return noErr;
}

static AudioStreamBasicDescription make_f32_mono(uint32_t sampleRate) {
    AudioStreamBasicDescription fmt = {0};
    fmt.mSampleRate       = (Float64)sampleRate;
    fmt.mFormatID         = kAudioFormatLinearPCM;
    fmt.mFormatFlags      = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    fmt.mFramesPerPacket  = 1;
    fmt.mChannelsPerFrame = 1;
    fmt.mBitsPerChannel   = 32;
    fmt.mBytesPerPacket   = 4;
    fmt.mBytesPerFrame    = 4;
    return fmt;
}

int ma_duplex_create_vpio(uint32_t playbackRate, uint32_t captureRate,
                          MaPlayback **outPlayback, MaCapture **outCapture)
{
    VPIODuplex *d = (VPIODuplex *)calloc(1, sizeof(VPIODuplex));
    MaPlayback *p = (MaPlayback *)calloc(1, sizeof(MaPlayback));
    MaCapture  *c = (MaCapture *)calloc(1, sizeof(MaCapture));
    if (!d || !p || !c) goto oom;

    /* Find VoiceProcessingIO component */
    AudioComponentDescription desc = {
        .componentType         = kAudioUnitType_Output,
        .componentSubType      = kAudioUnitSubType_VoiceProcessingIO,
        .componentManufacturer = kAudioUnitManufacturer_Apple,
    };
    AudioComponent comp = AudioComponentFindNext(NULL, &desc);
    if (!comp) {
        fprintf(stderr, "[AEC] duplex: VPIO component not found\n");
        goto oom;
    }

    OSStatus err;
    err = AudioComponentInstanceNew(comp, &d->auVPIO);
    if (err != noErr) {
        fprintf(stderr, "[AEC] duplex: AudioComponentInstanceNew failed: %d\n", (int)err);
        goto oom;
    }

    /* Enable input (bus 1) and output (bus 0) */
    UInt32 one = 1;
    err = AudioUnitSetProperty(d->auVPIO, kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Input, 1, &one, sizeof(one));
    if (err != noErr) {
        fprintf(stderr, "[AEC] duplex: enable input failed: %d\n", (int)err);
        goto fail;
    }
    /* Output is enabled by default, but be explicit */
    err = AudioUnitSetProperty(d->auVPIO, kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Output, 0, &one, sizeof(one));
    if (err != noErr) {
        fprintf(stderr, "[AEC] duplex: enable output failed: %d\n", (int)err);
        goto fail;
    }

    /* Query the VPIO's hardware sample rate. VPIO requires both buses at the
     * same rate — use the hardware rate and resample in our callbacks. */
    AudioStreamBasicDescription hwFmt = {0};
    UInt32 fmtSize = sizeof(hwFmt);
    AudioUnitGetProperty(d->auVPIO, kAudioUnitProperty_StreamFormat,
                         kAudioUnitScope_Output, 0, &hwFmt, &fmtSize);
    uint32_t hwRate = (uint32_t)hwFmt.mSampleRate;
    if (hwRate == 0) hwRate = 48000;

    AudioStreamBasicDescription vpioFmt = make_f32_mono(hwRate);

    /* Bus 0 input scope = playback feed (at hardware rate) */
    err = AudioUnitSetProperty(d->auVPIO, kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Input, 0, &vpioFmt, sizeof(vpioFmt));
    if (err != noErr) {
        fprintf(stderr, "[AEC] duplex: set playback format (%u Hz) failed: %d\n",
                hwRate, (int)err);
        goto fail;
    }
    /* Bus 1 output scope = capture output (at hardware rate) */
    err = AudioUnitSetProperty(d->auVPIO, kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Output, 1, &vpioFmt, sizeof(vpioFmt));
    if (err != noErr) {
        fprintf(stderr, "[AEC] duplex: set capture format (%u Hz) failed: %d\n",
                hwRate, (int)err);
        goto fail;
    }
    uint32_t actualPlayRate = hwRate;
    uint32_t actualCapRate  = hwRate;

    /* Wire up handles */
    d->playback    = p;
    d->capture     = c;
    d->playAppRate = playbackRate;
    d->playHwRate  = actualPlayRate;
    d->capAppRate  = captureRate;
    d->capHwRate   = actualCapRate;
    p->useVPIO     = 2;
    p->duplex      = d;
    p->idle        = 1;
    p->fadeTotal    = actualPlayRate * FADE_MS / 1000;
    c->useVPIO     = 2;
    c->duplex      = d;
    /* Store VPIO ref in capture for AudioUnitRender in input callback */
    c->auVPIO      = d->auVPIO;

    /* Output render callback (bus 0): provides playback audio to VPIO */
    AURenderCallbackStruct renderCb = {
        .inputProc       = vpio_duplex_render,
        .inputProcRefCon = d,
    };
    err = AudioUnitSetProperty(d->auVPIO, kAudioUnitProperty_SetRenderCallback,
            kAudioUnitScope_Input, 0, &renderCb, sizeof(renderCb));
    if (err != noErr) {
        fprintf(stderr, "[AEC] duplex: set render callback failed: %d\n", (int)err);
        goto fail;
    }

    /* Input callback: receives AEC-processed mic audio */
    AURenderCallbackStruct inputCb = {
        .inputProc       = vpio_duplex_input,
        .inputProcRefCon = d,
    };
    err = AudioUnitSetProperty(d->auVPIO, kAudioOutputUnitProperty_SetInputCallback,
            kAudioUnitScope_Global, 0, &inputCb, sizeof(inputCb));
    if (err != noErr) {
        fprintf(stderr, "[AEC] duplex: set input callback failed: %d\n", (int)err);
        goto fail;
    }

    err = AudioUnitInitialize(d->auVPIO);
    if (err != noErr) {
        fprintf(stderr, "[AEC] duplex: AudioUnitInitialize failed: %d\n", (int)err);
        goto fail;
    }
    err = AudioOutputUnitStart(d->auVPIO);
    if (err != noErr) {
        fprintf(stderr, "[AEC] duplex: AudioOutputUnitStart failed: %d\n", (int)err);
        goto fail;
    }

    *outPlayback = p;
    *outCapture  = c;
    fprintf(stderr, "[AEC] VPIO duplex: play %u→%u Hz, cap %u→%u Hz\n",
            playbackRate, actualPlayRate, captureRate, actualCapRate);
    return 0;

fail:
    AudioComponentInstanceDispose(d->auVPIO);
oom:
    free(d); free(p); free(c);
    return -1;
}

#else

int ma_duplex_create_vpio(uint32_t playbackRate, uint32_t captureRate,
                          MaPlayback **outPlayback, MaCapture **outCapture)
{
    (void)playbackRate; (void)captureRate;
    (void)outPlayback; (void)outCapture;
    return -1;  /* not available on this platform */
}

#endif
