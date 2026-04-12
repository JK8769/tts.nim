## Smart Turn v3 — endpoint detection for conversational audio turns.
## Predicts whether the user's turn is complete (done talking) or incomplete (mid-pause).
## Architecture: Whisper encoder (4 layers, 384d) + attention pooling + MLP classifier.
## Input: up to 8s of 16kHz mono audio → output: probability [0.0=incomplete, 1.0=complete].

import std/[tables, strutils, math, os, json]
import ../mlx/mlx
import ../stt/whisper_mlx  # reuse mel spectrogram

const
  ST_SAMPLE_RATE = 16000
  ST_MAX_SECONDS = 8
  ST_MAX_SAMPLES = ST_MAX_SECONDS * ST_SAMPLE_RATE  # 128000
  ST_D_MODEL = 384
  ST_N_HEADS = 6
  ST_N_LAYERS = 4
  ST_FFN_DIM = 1536
  ST_N_MELS = 80
  ST_MAX_POS = 400  # max source positions after stride-2 conv

type
  SmartTurn* = ref object
    weights: Table[string, Tensor]
    threshold*: float32
    posEmb: Tensor         # (MAX_POS, D_MODEL) learned positional embeddings
    melFilters: Tensor     # reuse Whisper mel filterbank

proc w(m: SmartTurn, key: string): Tensor {.inline.} =
  if key notin m.weights:
    raise newException(KeyError, "SmartTurn missing weight: " & key)
  m.weights[key]

# ── LayerNorm, GELU ─────────────────────────────────────────────

proc layerNorm(x, weight, bias: Tensor, eps: float32 = 1e-5): Tensor =
  let m = mean(x, axis = -1, keepdims = true)
  let v = variance(x, axis = -1, keepdims = true)
  (x - m) / sqrt(v + scalar(eps)) * weight + bias

proc gelu(x: Tensor): Tensor =
  x * scalar(0.5'f32) * (scalar(1.0'f32) + erf(x * scalar(0.7071067811865476'f32)))

# ── Multi-head self-attention (no k_proj bias) ──────────────────

proc selfAttn(m: SmartTurn, x: Tensor, prefix: string): Tensor =
  let B = x.dim(0)
  let T = x.dim(1)
  let headDim = ST_D_MODEL div ST_N_HEADS

  let qW = m.w(prefix & "q_proj.weight")
  let qB = m.w(prefix & "q_proj.bias")
  let q = (x @ transpose(qW)) + qB

  let kW = m.w(prefix & "k_proj.weight")
  let k = x @ transpose(kW)  # no bias for k_proj

  let vW = m.w(prefix & "v_proj.weight")
  let vB = m.w(prefix & "v_proj.bias")
  let v = (x @ transpose(vW)) + vB

  let scale = 1.0'f32 / sqrt(float32(headDim))
  let qr = q.reshape([B, T, ST_N_HEADS, headDim]).transpose([0, 2, 1, 3])
  let kr = k.reshape([B, T, ST_N_HEADS, headDim]).transpose([0, 2, 3, 1])
  let vr = v.reshape([B, T, ST_N_HEADS, headDim]).transpose([0, 2, 1, 3])

  let attn = softmax(qr @ kr * scalar(scale), axis = -1, precise = true)
  let attnOut = (attn @ vr).transpose([0, 2, 1, 3]).reshape([B, T, ST_D_MODEL])

  let oW = m.w(prefix & "out_proj.weight")
  let oB = m.w(prefix & "out_proj.bias")
  (attnOut @ transpose(oW)) + oB

# ── Encoder layer ────────────────────────────────────────────────

proc encoderLayer(m: SmartTurn, x: Tensor, prefix: string): Tensor =
  # Pre-norm self-attention
  let ln1W = m.w(prefix & "self_attn_layer_norm.weight")
  let ln1B = m.w(prefix & "self_attn_layer_norm.bias")
  var cur = x + selfAttn(m, layerNorm(x, ln1W, ln1B), prefix & "self_attn.")

  # Pre-norm FFN
  let ln2W = m.w(prefix & "final_layer_norm.weight")
  let ln2B = m.w(prefix & "final_layer_norm.bias")
  let normed = layerNorm(cur, ln2W, ln2B)
  let fc1W = m.w(prefix & "fc1.weight")
  let fc1B = m.w(prefix & "fc1.bias")
  let fc2W = m.w(prefix & "fc2.weight")
  let fc2B = m.w(prefix & "fc2.bias")
  cur + (gelu((normed @ transpose(fc1W)) + fc1B) @ transpose(fc2W)) + fc2B

# ── Encoder ──────────────────────────────────────────────────────

proc encode(m: SmartTurn, mel: Tensor): Tensor =
  ## mel: (1, nMels, nFrames) → (1, T, D_MODEL)
  # Transpose to (1, nFrames, nMels) for Conv1d
  var x = mel.transpose([0, 2, 1])

  let c1W = m.w("encoder.conv1.weight")
  let c1B = m.w("encoder.conv1.bias")
  x = gelu(conv1d(x, c1W, padding = 1) + c1B)

  let c2W = m.w("encoder.conv2.weight")
  let c2B = m.w("encoder.conv2.bias")
  x = gelu(conv1d(x, c2W, stride = 2, padding = 1) + c2B)

  # Learned positional embeddings
  let T = x.dim(1)
  let posSlice = m.posEmb.slice([0, 0], [T, ST_D_MODEL]).expandDims(0)
  x = x + posSlice

  # Encoder layers
  for i in 0..<ST_N_LAYERS:
    x = encoderLayer(m, x, "encoder.layers." & $i & ".")

  # Final layer norm
  let lnW = m.w("encoder.layer_norm.weight")
  let lnB = m.w("encoder.layer_norm.bias")
  layerNorm(x, lnW, lnB)

# ── Attention pooling + classifier ───────────────────────────────

proc classify(m: SmartTurn, hidden: Tensor): float32 =
  ## hidden: (1, T, D_MODEL) → probability [0.0, 1.0]
  # Attention pooling
  let pa0W = m.w("pool_attention_0.weight")
  let pa0B = m.w("pool_attention_0.bias")
  let pa2W = m.w("pool_attention_2.weight")
  let pa2B = m.w("pool_attention_2.bias")

  # tanh → linear → softmax → weighted sum
  let attnScores = ((hidden @ transpose(pa0W)) + pa0B)
  # tanh
  let tanhOut = (exp(attnScores * scalar(2.0'f32)) - scalar(1.0'f32)) /
                (exp(attnScores * scalar(2.0'f32)) + scalar(1.0'f32))
  let attnWeights = softmax((tanhOut @ transpose(pa2W)) + pa2B, axis = 1, precise = true)
  let pooled = sum(hidden * attnWeights, axis = 1, keepdims = false)  # (1, D_MODEL)

  # Classifier MLP
  let c0W = m.w("classifier_0.weight")
  let c0B = m.w("classifier_0.bias")
  let c1W = m.w("classifier_1.weight")
  let c1B = m.w("classifier_1.bias")
  let c4W = m.w("classifier_4.weight")
  let c4B = m.w("classifier_4.bias")
  let c6W = m.w("classifier_6.weight")
  let c6B = m.w("classifier_6.bias")

  var x = (pooled @ transpose(c0W)) + c0B
  # LayerNorm (classifier_1)
  x = layerNorm(x, c1W, c1B)
  x = gelu(x)
  x = gelu((x @ transpose(c4W)) + c4B)
  let logits = (x @ transpose(c6W)) + c6B

  # Sigmoid
  let sig = scalar(1.0'f32) / (scalar(1.0'f32) + exp(scalar(0.0'f32) - logits))
  eval(sig)
  sig.itemFloat32()

# ── Public API ───────────────────────────────────────────────────

proc loadSmartTurn*(modelDir: string, threshold: float32 = 0.5): SmartTurn =
  ## Load Smart Turn model from directory with config.json and model.safetensors.
  initMlx()
  initDefaultStream()

  result = SmartTurn(threshold: threshold)

  # Load config (just for threshold override)
  let configPath = modelDir / "config.json"
  if fileExists(configPath):
    let cfg = parseJson(readFile(configPath))
    if cfg.hasKey("threshold"):
      result.threshold = cfg["threshold"].getFloat().float32

  # Load weights
  result.weights = initTable[string, Tensor]()
  for f in walkDir(modelDir):
    if f.path.endsWith(".safetensors"):
      let tensors = loadSafetensors(f.path)
      for k, v in tensors:
        result.weights[k] = v

  # Sanitize keys: strip "inner." prefix, remap pool/classifier names
  var sanitized = initTable[string, Tensor]()
  for k, v in result.weights:
    var key = k
    var val = v
    if key.startsWith("inner."):
      key = key[6..^1]

    key = key.replace("pool_attention.0.", "pool_attention_0.")
    key = key.replace("pool_attention.2.", "pool_attention_2.")
    key = key.replace("classifier.0.", "classifier_0.")
    key = key.replace("classifier.1.", "classifier_1.")
    key = key.replace("classifier.4.", "classifier_4.")
    key = key.replace("classifier.6.", "classifier_6.")

    # Transpose conv weights: (out, in, kernel) → (out, kernel, in) for MLX
    if ("conv1.weight" in key or "conv2.weight" in key) and val.ndim == 3:
      val = val.transpose([0, 2, 1])

    sanitized[key] = val
  result.weights = sanitized

  # Extract positional embeddings
  result.posEmb = result.w("encoder.embed_positions.weight")

  # Build mel filterbank (same as Whisper)
  result.melFilters = buildMelFilters(ST_SAMPLE_RATE, N_FFT, ST_N_MELS)

  stderr.writeLine "Smart Turn loaded (threshold=", result.threshold, ")"

proc predictEndpoint*(m: SmartTurn, samples: openArray[float32]): tuple[complete: bool, probability: float32] =
  ## Predict whether the user's turn is complete.
  ## Input: 16kHz mono float32 PCM (up to 8 seconds — longer is truncated, shorter is left-padded).
  if samples.len == 0:
    return (true, 1.0)  # empty audio = complete

  # Take last 8 seconds, left-pad if shorter
  var audio: seq[float32]
  if samples.len > ST_MAX_SAMPLES:
    audio = @(samples[samples.len - ST_MAX_SAMPLES ..< samples.len])
  elif samples.len < ST_MAX_SAMPLES:
    audio = newSeq[float32](ST_MAX_SAMPLES)
    let offset = ST_MAX_SAMPLES - samples.len
    for i in 0..<samples.len:
      audio[offset + i] = samples[i]
  else:
    audio = @samples

  # Normalize
  var mean = 0.0'f64
  for s in audio: mean += float64(s)
  mean /= float64(audio.len)
  var std = 0.0'f64
  for s in audio: std += (float64(s) - mean) * (float64(s) - mean)
  std = sqrt(std / float64(audio.len))
  if std < 1e-7: std = 1e-7
  for i in 0..<audio.len:
    audio[i] = float32((float64(audio[i]) - mean) / std)

  # Compute mel spectrogram using Whisper's mel computation
  # Need to create a temporary WhisperModel-like object for melFilters
  var audioT = fromSeq(audio, [audio.len])
  let mel = logMelSpectrogramWith(audioT, m.melFilters)

  # Target frames for 8 seconds: 128000 / 160 = 800 frames
  let targetFrames = ST_MAX_SAMPLES div HOP_LENGTH  # 800
  let frames = mel.dim(0)
  var melPadded: Tensor
  if frames > targetFrames:
    melPadded = mel.slice([frames - targetFrames, 0], [frames, ST_N_MELS])
  elif frames < targetFrames:
    let padFrames = zeros([targetFrames - frames, ST_N_MELS])
    melPadded = concatenate([padFrames, mel], axis = 0)
  else:
    melPadded = mel

  # Shape: (1, nMels, nFrames) — HF convention
  let melInput = melPadded.transpose([1, 0]).expandDims(0)

  # Forward pass
  let hidden = m.encode(melInput)
  let prob = m.classify(hidden)

  (prob > m.threshold, prob)
