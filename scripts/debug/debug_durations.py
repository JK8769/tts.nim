#!/usr/bin/env python3
"""Compare durations from Python reference vs Nim."""
import sys, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "/Users/owaf/Work/Agents/kokoro-mlx/src")

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from kokoro_mlx.model import KokoroModel

model_path = snapshot_download("mlx-community/Kokoro-82M-bf16",
                               allow_patterns=["*.json", "*.safetensors"])
model = KokoroModel.from_pretrained(model_path)

# Use pre-phonemized IPA for "Hello world"
ph = "həlˈoʊ wˈɜːld"
print(f"Phonemes: '{ph}'")
voice_name = "af_heart"
voice = mx.load(os.path.join(model_path, "voices", f"{voice_name}.safetensors"))
voice = voice.get("voice", voice.get(list(voice.keys())[0]))

# Tokenize exactly like model.forward does
input_ids_list = [model.vocab[c] for c in ph if c in model.vocab]
input_ids = mx.array([[0, *input_ids_list, 0]])
T = input_ids.shape[1]
print(f"Tokens: {T} (matched {len(input_ids_list)} phoneme chars)")
print(f"Token IDs: {input_ids.tolist()[0]}")

n_tokens = len([c for c in ph if c in model.vocab]) + 2
idx = min(n_tokens - 1, voice.shape[0] - 1)
ref_s = voice[idx:idx+1]
if ref_s.ndim == 3:
    ref_s = ref_s.squeeze(1)

s_prosody = ref_s[:, 128:]
s_decoder = ref_s[:, :128]
text_mask = mx.zeros((1, T), dtype=mx.bool_)
input_lengths = mx.array([T])

bert_dur = model.bert(input_ids)
d_en = model.bert_encoder(bert_dur).transpose(0, 2, 1)

d = model.predictor.text_encoder(d_en, s_prosody, input_lengths, text_mask)
x = model.predictor.lstm(d)
duration = model.predictor.duration_proj(x)
duration = mx.sigmoid(duration).sum(axis=-1) / 1.0
pred_dur = mx.clip(mx.round(duration), 0, None).astype(mx.int32).squeeze(0)
pred_dur = mx.maximum(pred_dur, mx.ones_like(pred_dur))

mx.eval(pred_dur)
print(f"Durations: {pred_dur.tolist()}")
print(f"Total frames: {sum(pred_dur.tolist())}")

# Also generate full audio
audio = model.forward(ph, ref_s, speed=1.0)
mx.eval(audio)
print(f"Audio samples: {audio.size}, duration: {audio.size/24000:.2f}s")
