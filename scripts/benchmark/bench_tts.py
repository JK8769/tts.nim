#!/usr/bin/env python3
"""Benchmark kokoro-mlx vs mlx-audio — forward pass only (no phonemization)."""

import sys, time, os, json, glob
os.environ["TOKENIZERS_PARALLELISM"] = "false"

WARMUP = 1
RUNS = 5
VOICE_NAME = "af_heart"

TEXTS = {
    "short": "ðə kwˈɪk bɹˈaʊn fˈɑːks dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ.",
    "medium": "ðə kwˈɪk bɹˈaʊn fˈɑːks dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ. ɪt wʌz ðə bˈɛst ʌv tˈaɪmz, ɪt wʌz ðə wˈɜːst ʌv tˈaɪmz. ðə sˈʌn ʃˈoʊn bɹˈaɪtli ˌoʊvɚ ðə kwˈaɪət vˈɪlɪdʒ.",
    "long": "ðə kwˈɪk bɹˈaʊn fˈɑːks dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ. ɪt wʌz ðə bˈɛst ʌv tˈaɪmz, ɪt wʌz ðə wˈɜːst ʌv tˈaɪmz. ðə sˈʌn ʃˈoʊn bɹˈaɪtli ˌoʊvɚ ðə kwˈaɪət vˈɪlɪdʒ. ʃiː wˈɔːkt slˈoʊli θɹuː ðə ɡˈɑːɹdən, ɛndʒˈɔɪɪŋ ðə fɹˈeɪɡɹəns ʌv fɹˈɛʃli kˈʌt flˈaʊɚz. ðə tʃˈɪldɹən plˈeɪd hˈæpɪli ɪn ðə pˈɑːɹk wˈaɪl ðɛɹ pˈɛɹənts wˈɑːtʃt fɹʌm ə bˈɛntʃ nˈɪɹbaɪ.",
}


def get_model_path():
    from huggingface_hub import snapshot_download
    return snapshot_download("mlx-community/Kokoro-82M-bf16",
                             allow_patterns=["*.json", "*.safetensors"])


# Cache loaded models across runs
_kokoro_model = None
_kokoro_voice = None
_mlx_audio_model = None
_mlx_audio_config = None
_mlx_audio_voice = None


def bench_kokoro_mlx(model_path, phonemes):
    global _kokoro_model, _kokoro_voice
    sys.path.insert(0, "/Users/owaf/Work/Agents/kokoro-mlx/src")
    import mlx.core as mx
    from kokoro_mlx.model import KokoroModel

    if _kokoro_model is None:
        _kokoro_model = KokoroModel.from_pretrained(model_path)
        voice = mx.load(os.path.join(model_path, "voices", f"{VOICE_NAME}.safetensors"))
        _kokoro_voice = voice.get("voice", voice.get(list(voice.keys())[0]))

    model = _kokoro_model
    voice = _kokoro_voice

    n_tokens = len([c for c in phonemes if c in model.vocab]) + 2
    idx = min(n_tokens - 1, voice.shape[0] - 1)
    ref_s = voice[idx:idx+1]
    if ref_s.ndim == 3:
        ref_s = ref_s.squeeze(1)

    for _ in range(WARMUP):
        audio = model.forward(phonemes, ref_s, speed=1.0)
        mx.eval(audio)

    times = []
    for i in range(RUNS):
        t0 = time.perf_counter()
        audio = model.forward(phonemes, ref_s, speed=1.0)
        mx.eval(audio)
        times.append(time.perf_counter() - t0)

    samples = audio.size
    duration = samples / 24000
    avg = sum(times) / len(times)
    print(f"  kokoro-mlx:  {avg*1000:6.1f}ms avg  {min(times)*1000:6.1f}ms min  {samples:6d} samples  {duration:.2f}s audio  {duration/avg:.1f}x RT")
    return avg, duration


def bench_mlx_audio(model_path, phonemes):
    global _mlx_audio_model, _mlx_audio_config, _mlx_audio_voice
    sys.path.insert(0, "/Users/owaf/Work/Agents/mlx-audio")
    import mlx.core as mx
    import types
    if "mlx_audio.tts.models.kokoro.pipeline" not in sys.modules:
        fake_pipeline = types.ModuleType("mlx_audio.tts.models.kokoro.pipeline")
        fake_pipeline.KokoroPipeline = type("KokoroPipeline", (), {})
        sys.modules["mlx_audio.tts.models.kokoro.pipeline"] = fake_pipeline
    from mlx_audio.tts.models.kokoro.kokoro import Model, ModelConfig

    if _mlx_audio_model is None:
        with open(os.path.join(model_path, "config.json")) as f:
            config_data = json.load(f)
        _mlx_audio_config = ModelConfig(**config_data)
        _mlx_audio_model = Model(_mlx_audio_config)
        st_path = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(st_path):
            st_path = glob.glob(os.path.join(model_path, "*.safetensors"))[0]
        weights = mx.load(st_path)
        weights = _mlx_audio_model.sanitize(weights)
        _mlx_audio_model.load_weights(list(weights.items()))
        mx.eval(_mlx_audio_model.parameters())
        voice = mx.load(os.path.join(model_path, "voices", f"{VOICE_NAME}.safetensors"))
        _mlx_audio_voice = voice.get("voice", voice.get(list(voice.keys())[0]))

    model = _mlx_audio_model
    config = _mlx_audio_config
    voice = _mlx_audio_voice

    n_tokens = len([c for c in phonemes if c in config.vocab]) + 2
    idx = min(n_tokens - 1, voice.shape[0] - 1)
    ref_s = voice[idx:idx+1]
    if ref_s.ndim == 3:
        ref_s = ref_s.squeeze(1)

    for _ in range(WARMUP):
        audio = model(phonemes, ref_s, speed=1.0)
        if hasattr(audio, 'audio'):
            audio = audio.audio
        mx.eval(audio)

    times = []
    for i in range(RUNS):
        t0 = time.perf_counter()
        audio = model(phonemes, ref_s, speed=1.0)
        if hasattr(audio, 'audio'):
            audio = audio.audio
        mx.eval(audio)
        times.append(time.perf_counter() - t0)

    samples = audio.size
    duration = samples / 24000
    avg = sum(times) / len(times)
    print(f"  mlx-audio:   {avg*1000:6.1f}ms avg  {min(times)*1000:6.1f}ms min  {samples:6d} samples  {duration:.2f}s audio  {duration/avg:.1f}x RT")
    return avg, duration


if __name__ == "__main__":
    model_path = get_model_path()
    print(f"Model: {model_path}\n")

    for label, text in TEXTS.items():
        print(f"--- {label} ({len(text)} phoneme chars) ---")
        r1_avg, r1_dur = bench_kokoro_mlx(model_path, text)
        r2_avg, r2_dur = bench_mlx_audio(model_path, text)
        ratio = r1_avg / r2_avg
        winner = f"kokoro-mlx {1/ratio:.2f}x faster" if ratio < 1 else f"mlx-audio {ratio:.2f}x faster"
        print(f"  → {winner}\n")
