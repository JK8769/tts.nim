#!/usr/bin/env python3
"""Generate WAV files from both implementations for quality comparison."""

import sys, os, json, glob, struct, types
os.environ["TOKENIZERS_PARALLELISM"] = "false"

VOICE_NAME = "af_heart"
PHONEMES = "ðə kwˈɪk bɹˈaʊn fˈɑːks dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ."


def write_wav(path, samples, sr=24000):
    """Write float32 samples to 16-bit WAV."""
    n = len(samples)
    data = b""
    for s in samples:
        v = max(-1.0, min(1.0, s))
        data += struct.pack("<h", int(v * 32767))
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)
    print(f"  Wrote {path} ({n} samples, {n/sr:.2f}s)")


def get_model_path():
    from huggingface_hub import snapshot_download
    return snapshot_download("mlx-community/Kokoro-82M-bf16",
                             allow_patterns=["*.json", "*.safetensors"])


def gen_kokoro_mlx(model_path):
    sys.path.insert(0, "/Users/owaf/Work/Agents/kokoro-mlx/src")
    import mlx.core as mx
    from kokoro_mlx.model import KokoroModel

    model = KokoroModel.from_pretrained(model_path)
    voice = mx.load(os.path.join(model_path, "voices", f"{VOICE_NAME}.safetensors"))
    voice = voice.get("voice", voice.get(list(voice.keys())[0]))

    n_tokens = len([c for c in PHONEMES if c in model.vocab]) + 2
    idx = min(n_tokens - 1, voice.shape[0] - 1)
    ref_s = voice[idx:idx+1]
    if ref_s.ndim == 3:
        ref_s = ref_s.squeeze(1)

    audio = model.forward(PHONEMES, ref_s, speed=1.0)
    mx.eval(audio)
    samples = audio.tolist()
    if isinstance(samples[0], list):
        samples = samples[0]
    return samples


def gen_mlx_audio(model_path):
    sys.path.insert(0, "/Users/owaf/Work/Agents/mlx-audio")
    import mlx.core as mx
    if "mlx_audio.tts.models.kokoro.pipeline" not in sys.modules:
        fake_pipeline = types.ModuleType("mlx_audio.tts.models.kokoro.pipeline")
        fake_pipeline.KokoroPipeline = type("KokoroPipeline", (), {})
        sys.modules["mlx_audio.tts.models.kokoro.pipeline"] = fake_pipeline
    from mlx_audio.tts.models.kokoro.kokoro import Model, ModelConfig

    with open(os.path.join(model_path, "config.json")) as f:
        config_data = json.load(f)
    config = ModelConfig(**config_data)
    model = Model(config)
    st_path = glob.glob(os.path.join(model_path, "*.safetensors"))[0]
    weights = mx.load(st_path)
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    voice = mx.load(os.path.join(model_path, "voices", f"{VOICE_NAME}.safetensors"))
    voice = voice.get("voice", voice.get(list(voice.keys())[0]))

    n_tokens = len([c for c in PHONEMES if c in config.vocab]) + 2
    idx = min(n_tokens - 1, voice.shape[0] - 1)
    ref_s = voice[idx:idx+1]
    if ref_s.ndim == 3:
        ref_s = ref_s.squeeze(1)

    audio = model(PHONEMES, ref_s, speed=1.0)
    if hasattr(audio, 'audio'):
        audio = audio.audio
    mx.eval(audio)
    samples = audio.flatten().tolist()
    return samples


if __name__ == "__main__":
    model_path = get_model_path()

    print("kokoro-mlx:")
    s1 = gen_kokoro_mlx(model_path)
    write_wav("/tmp/kokoro_mlx.wav", s1)

    print("mlx-audio:")
    s2 = gen_mlx_audio(model_path)
    write_wav("/tmp/mlx_audio.wav", s2)

    # Stats
    import statistics
    for name, s in [("kokoro-mlx", s1), ("mlx-audio", s2)]:
        mn, mx_v = min(s), max(s)
        rms = (sum(x*x for x in s) / len(s)) ** 0.5
        print(f"  {name}: len={len(s)}  min={mn:.4f}  max={mx_v:.4f}  rms={rms:.4f}")

    # Cross-correlation to see if they're producing the same signal
    minlen = min(len(s1), len(s2))
    dot = sum(a*b for a,b in zip(s1[:minlen], s2[:minlen]))
    norm1 = sum(a*a for a in s1[:minlen]) ** 0.5
    norm2 = sum(b*b for b in s2[:minlen]) ** 0.5
    corr = dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
    print(f"\n  Correlation: {corr:.4f}  (1.0 = identical signal)")
    print(f"\n  Play: afplay /tmp/kokoro_mlx.wav && afplay /tmp/mlx_audio.wav")
