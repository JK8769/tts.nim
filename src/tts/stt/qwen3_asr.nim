## Qwen3 ASR speech-to-text using Apple MLX backend.
## Encoder-decoder model: audio → 128-bin mel → Conv2d frontend → transformer encoder
## → Qwen3 LLM decoder (autoregressive).
## Port from mlx-audio's Qwen3 ASR implementation to Nim, using mlx-c bindings.
## 0.6B model (8-bit quantized decoder), supports 30 languages.

import std/[tables, strutils, math, os, json]
import ../mlx/mlx
import ./whisper_mlx  # reuse mel spectrogram + buildMelFilters

const
  QWEN3_SAMPLE_RATE* = 16000

# ── Configuration (loaded from config.json) ─────────────────────

type
  AudioEncoderCfg = object
    numMelBins: int         # 128
    encoderLayers: int      # 18
    encoderHeads: int       # 14
    encoderFfnDim: int      # 3584
    dModel: int             # 896
    maxSourcePositions: int # 1500
    nWindow: int            # 50
    nWindowInfer: int       # 800
    convChunksize: int      # 500
    downsampleHiddenSize: int  # 480
    outputDim: int          # 1024

  TextDecoderCfg = object
    vocabSize: int          # 151936
    hiddenSize: int         # 1024
    intermediateSize: int   # 3072
    numLayers: int          # 28
    numHeads: int           # 16
    numKvHeads: int         # 8
    headDim: int            # 128
    rmsNormEps: float32     # 1e-6
    ropeTheta: float64      # 1000000.0
    maxPositionEmbeddings: int  # 65536

  Qwen3AsrConfig = object
    audio: AudioEncoderCfg
    text: TextDecoderCfg
    audioTokenId: int       # 151676
    audioStartTokenId: int  # 151669
    audioEndTokenId: int    # 151670
    quantGroupSize: int     # 64
    quantBits: int          # 8

proc parseConfig(configPath: string): Qwen3AsrConfig =
  let j = parseJson(readFile(configPath))
  let thinker = if j.hasKey("thinker_config"): j["thinker_config"] else: j
  let ac = thinker["audio_config"]
  let tc = thinker["text_config"]
  let quant = if j.hasKey("quantization"): j["quantization"] else: newJObject()

  result.audio = AudioEncoderCfg(
    numMelBins: ac.getOrDefault("num_mel_bins").getInt(128),
    encoderLayers: ac.getOrDefault("encoder_layers").getInt(18),
    encoderHeads: ac.getOrDefault("encoder_attention_heads").getInt(14),
    encoderFfnDim: ac.getOrDefault("encoder_ffn_dim").getInt(3584),
    dModel: ac.getOrDefault("d_model").getInt(896),
    maxSourcePositions: ac.getOrDefault("max_source_positions").getInt(1500),
    nWindow: ac.getOrDefault("n_window").getInt(50),
    nWindowInfer: ac.getOrDefault("n_window_infer").getInt(800),
    convChunksize: ac.getOrDefault("conv_chunksize").getInt(500),
    downsampleHiddenSize: ac.getOrDefault("downsample_hidden_size").getInt(480),
    outputDim: ac.getOrDefault("output_dim").getInt(1024),
  )
  result.text = TextDecoderCfg(
    vocabSize: tc.getOrDefault("vocab_size").getInt(151936),
    hiddenSize: tc.getOrDefault("hidden_size").getInt(1024),
    intermediateSize: tc.getOrDefault("intermediate_size").getInt(3072),
    numLayers: tc.getOrDefault("num_hidden_layers").getInt(28),
    numHeads: tc.getOrDefault("num_attention_heads").getInt(16),
    numKvHeads: tc.getOrDefault("num_key_value_heads").getInt(8),
    headDim: tc.getOrDefault("head_dim").getInt(128),
    rmsNormEps: tc.getOrDefault("rms_norm_eps").getFloat(1e-6).float32,
    ropeTheta: tc.getOrDefault("rope_theta").getFloat(1000000.0),
    maxPositionEmbeddings: tc.getOrDefault("max_position_embeddings").getInt(65536),
  )
  result.audioTokenId = thinker.getOrDefault("audio_token_id").getInt(151676)
  result.audioStartTokenId = thinker.getOrDefault("audio_start_token_id").getInt(151669)
  result.audioEndTokenId = thinker.getOrDefault("audio_end_token_id").getInt(151670)
  result.quantGroupSize = quant.getOrDefault("group_size").getInt(64)
  result.quantBits = quant.getOrDefault("bits").getInt(8)

# ── Main type ────────────────────────────────────────────────────

type
  KvCache* = object
    keys: Tensor      # accumulated keys: (1, numKvHeads, seqLen, headDim)
    values: Tensor     # accumulated values: same shape
    offset*: int       # current sequence position

  Qwen3Asr* = ref object
    cfg: Qwen3AsrConfig
    weights: Table[string, Tensor]
    melFilters: Tensor          # (n_freqs, 128) mel filterbank
    sinPosEmb: Tensor           # (maxSourcePositions, dModel) precomputed
    ropeFreqs: Tensor           # precomputed RoPE frequencies
    cachedEmbedTable: Tensor    # lazily dequantized embedding table
    embedTableReady: bool
    tokens: Table[int, string]  # token id → string for decoding
    # Prompt token IDs (precomputed)
    promptPrefix: seq[int32]    # <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
    promptSuffix: seq[int32]    # <|audio_end|><|im_end|>\n<|im_start|>assistant\n
    eosTokenIds*: seq[int32]

proc w(m: Qwen3Asr, key: string): Tensor {.inline.} =
  if key notin m.weights:
    raise newException(KeyError, "Qwen3ASR missing weight: " & key)
  m.weights[key]

proc hasWeight(m: Qwen3Asr, key: string): bool {.inline.} =
  key in m.weights

# ── Building blocks ──────────────────────────────────────────────

proc rmsNorm(x, weight: Tensor, eps: float32): Tensor =
  ## RMS normalization (no mean subtraction, no bias).
  let v = mean(x * x, axis = -1, keepdims = true)
  x * rsqrt(v + scalar(eps)) * weight

proc gelu(x: Tensor): Tensor =
  x * scalar(0.5'f32) * (scalar(1.0'f32) + erf(x * scalar(0.7071067811865476'f32)))

proc silu(x: Tensor): Tensor =
  ## SiLU/Swish: x * sigmoid(x)
  x * (scalar(1.0'f32) / (scalar(1.0'f32) + exp(scalar(0.0'f32) - x)))

# ── Quantized linear helper ─────────────────────────────────────

proc qlinear(m: Qwen3Asr, x: Tensor, prefix: string): Tensor =
  ## Quantized linear layer: x @ W^T using quantized matmul.
  let w = m.w(prefix & "weight")
  let s = m.w(prefix & "scales")
  let b = m.w(prefix & "biases")
  quantizedMatmul(x, w, s, b, transpose = true,
                  groupSize = m.cfg.quantGroupSize, bits = m.cfg.quantBits)

proc ensureEmbedTable(m: Qwen3Asr) =
  ## Lazily dequantize and cache the embedding table on first use.
  if not m.embedTableReady:
    let eW = m.w("model.embed_tokens.weight")
    let eS = m.w("model.embed_tokens.scales")
    let eB = m.w("model.embed_tokens.biases")
    m.cachedEmbedTable = dequantize(eW, eS, eB,
                                     groupSize = m.cfg.quantGroupSize, bits = m.cfg.quantBits)
    eval(m.cachedEmbedTable)
    m.embedTableReady = true

proc embedTokens*(m: Qwen3Asr, ids: Tensor): Tensor =
  ## Look up token embeddings using cached dequantized table.
  m.ensureEmbedTable()
  m.cachedEmbedTable.take(ids, axis = 0)

proc lmHead(m: Qwen3Asr, hidden: Tensor): Tensor =
  ## Compute logits: hidden @ embed_tokens^T (tied weights).
  m.ensureEmbedTable()
  hidden @ transpose(m.cachedEmbedTable)

# ── Feature extraction (128-bin mel spectrogram) ─────────────────

const WHISPER_CHUNK_SAMPLES = 480000  ## 30 seconds at 16kHz

proc computeFeatures(m: Qwen3Asr, audio: openArray[float32]): tuple[features: Tensor, featureLen: int] =
  ## Compute 128-bin log-mel spectrogram features.
  ## Pads raw audio to 30 seconds before mel computation (WhisperFeatureExtractor behavior).
  ## Returns (1, numMelBins, 3000) features and featureLen=3000.
  # Pad or truncate raw audio to exactly 30 seconds
  var audioBuf: seq[float32]
  if audio.len < WHISPER_CHUNK_SAMPLES:
    audioBuf = @audio
    audioBuf.setLen(WHISPER_CHUNK_SAMPLES)  # zero-pads
  elif audio.len > WHISPER_CHUNK_SAMPLES:
    audioBuf = audio[0..<WHISPER_CHUNK_SAMPLES]
  else:
    audioBuf = @audio
  var audioT = fromSeq(audioBuf, [WHISPER_CHUNK_SAMPLES])
  let mel = logMelSpectrogramWith(audioT, m.melFilters)
  # mel shape: (numFrames, numMelBins) — should be 3000 frames
  let numFrames = mel.dim(0)
  let features = mel.transpose([1, 0]).expandDims(0)  # (1, numMelBins, numFrames)
  (features, numFrames)

# ── Output length after Conv2d frontend ──────────────────────────

proc fdiv(a, b: int): int {.inline.} =
  ## Floor division matching Python semantics (round toward negative infinity).
  floorDiv(a, b)

proc getOutputLength(inputLen: int): int =
  ## Compute output length of Conv2d layers for a given input length.
  let leave = inputLen mod 100
  let featLen = fdiv(leave - 1, 2) + 1
  let partial = fdiv(fdiv(featLen - 1, 2) + 1 - 1, 2) + 1
  partial + (inputLen div 100) * 13

# ── Sinusoidal position embeddings ───────────────────────────────

proc buildSinPosEmb(length, channels: int, maxTimescale: float64 = 10000.0): Tensor =
  ## Build sinusoidal position embeddings (length, channels).
  assert channels mod 2 == 0
  let halfC = channels div 2
  let logInc = ln(maxTimescale) / float64(halfC - 1)

  var data = newSeq[float32](length * channels)
  for pos in 0..<length:
    for i in 0..<halfC:
      let angle = float64(pos) * exp(-logInc * float64(i))
      data[pos * channels + i] = sin(angle).float32
      data[pos * channels + halfC + i] = cos(angle).float32

  fromSeq(data, [length, channels])

# ── Audio encoder ────────────────────────────────────────────────

proc layerNorm(x, weight, bias: Tensor, eps: float32 = 1e-5): Tensor =
  let m = mean(x, axis = -1, keepdims = true)
  let v = variance(x, axis = -1, keepdims = true)
  (x - m) / sqrt(v + scalar(eps)) * weight + bias

proc audioAttention(m: Qwen3Asr, x: Tensor, mask: Tensor, prefix: string): Tensor =
  ## Multi-head attention for audio encoder.
  let B = x.dim(0)
  let T = x.dim(1)
  let embedDim = m.cfg.audio.dModel
  let numHeads = m.cfg.audio.encoderHeads
  let headDim = embedDim div numHeads
  let scale = 1.0'f32 / sqrt(float32(headDim))

  let q = (x @ transpose(m.w(prefix & "q_proj.weight"))) + m.w(prefix & "q_proj.bias")
  let k = (x @ transpose(m.w(prefix & "k_proj.weight"))) + m.w(prefix & "k_proj.bias")
  let v = (x @ transpose(m.w(prefix & "v_proj.weight"))) + m.w(prefix & "v_proj.bias")

  # Reshape to (B, numHeads, T, headDim)
  let qr = q.reshape([B, T, numHeads, headDim]).transpose([0, 2, 1, 3]) * scalar(scale)
  let kr = k.reshape([B, T, numHeads, headDim]).transpose([0, 2, 1, 3])
  let vr = v.reshape([B, T, numHeads, headDim]).transpose([0, 2, 1, 3])

  # Attention with mask
  let scores = (qr @ kr.transpose([0, 1, 3, 2])) + mask
  let attn = softmax(scores, axis = -1, precise = true)
  let attnOut = (attn @ vr).transpose([0, 2, 1, 3]).reshape([B, T, embedDim])

  (attnOut @ transpose(m.w(prefix & "out_proj.weight"))) + m.w(prefix & "out_proj.bias")

proc audioEncoderLayer(m: Qwen3Asr, x: Tensor, mask: Tensor, prefix: string): Tensor =
  ## Single audio encoder transformer layer (pre-norm).
  let ln1W = m.w(prefix & "self_attn_layer_norm.weight")
  let ln1B = m.w(prefix & "self_attn_layer_norm.bias")
  let normed1 = layerNorm(x, ln1W, ln1B)
  var cur = x + audioAttention(m, normed1, mask, prefix & "self_attn.")

  let ln2W = m.w(prefix & "final_layer_norm.weight")
  let ln2B = m.w(prefix & "final_layer_norm.bias")
  let normed2 = layerNorm(cur, ln2W, ln2B)

  let fc1W = m.w(prefix & "fc1.weight")
  let fc1B = m.w(prefix & "fc1.bias")
  let fc2W = m.w(prefix & "fc2.weight")
  let fc2B = m.w(prefix & "fc2.bias")
  cur + (gelu((normed2 @ transpose(fc1W)) + fc1B) @ transpose(fc2W)) + fc2B

proc encodeAudio*(m: Qwen3Asr, features: Tensor, featureLen: int): Tensor =
  ## Full audio encoder: features (1, nMels, nFrames) → audio hidden (numTokens, outputDim).
  let chunkSize = m.cfg.audio.nWindow * 2  # 100
  let numChunks = (featureLen + chunkSize - 1) div chunkSize

  # Split into chunks and process through Conv2d frontend
  var chunks: seq[Tensor]
  var chunkLens: seq[int]
  for j in 0..<numChunks:
    let pos = j * chunkSize
    let clen = if j == numChunks - 1:
      let remainder = featureLen mod chunkSize
      if remainder == 0: chunkSize else: remainder
    else:
      chunkSize

    # Extract chunk: (1, nMels, clen)
    let chunk = features.slice([0, 0, pos], [1, m.cfg.audio.numMelBins, pos + clen])
    chunks.add(chunk)
    chunkLens.add(clen)

  # Pad chunks to max length and stack
  let maxChunkLen = max(chunkLens)
  var paddedChunks: seq[Tensor]
  for i, chunk in chunks:
    if chunkLens[i] < maxChunkLen:
      let padWidth = maxChunkLen - chunkLens[i]
      let padded = pad(chunk, axes = [2], lowPad = [0], highPad = [padWidth])
      paddedChunks.add(padded)
    else:
      paddedChunks.add(chunk)

  # Stack: (numChunks, nMels, maxChunkLen) → Conv2d input (N, H, W, C)
  let stacked = concatenate(paddedChunks, axis = 0)  # (numChunks, nMels, maxChunkLen)
  # Python: padded_feature[:, :, :, None] → (N, nMels, maxChunkLen, 1) i.e. H=nMels, W=time, C=1
  var x = stacked.expandDims(3)  # (numChunks, nMels, maxChunkLen, 1)

  # 3 Conv2d layers with stride 2, padding 1
  let c1W = m.w("audio_tower.conv2d1.weight")
  let c1B = m.w("audio_tower.conv2d1.bias")
  x = gelu(conv2d(x, c1W, stride = [2, 2], padding = [1, 1]) + c1B)

  let c2W = m.w("audio_tower.conv2d2.weight")
  let c2B = m.w("audio_tower.conv2d2.bias")
  x = gelu(conv2d(x, c2W, stride = [2, 2], padding = [1, 1]) + c2B)

  let c3W = m.w("audio_tower.conv2d3.weight")
  let c3B = m.w("audio_tower.conv2d3.bias")
  x = gelu(conv2d(x, c3W, stride = [2, 2], padding = [1, 1]) + c3B)

  # x shape: (N, F', T', C) where F'=nMels/8, T'=time/8, C=480
  # Python: b,f,t,c = x.shape; x = x.transpose(0,2,3,1).reshape(b, t, c*f)
  let ncB = x.dim(0)
  let fDim = x.dim(1)  # frequency (height) after conv
  let tDim = x.dim(2)  # time (width) after conv
  let cDim = x.dim(3)  # channels
  x = x.transpose([0, 2, 3, 1]).reshape([ncB, tDim, cDim * fDim])

  # Linear projection to dModel
  let convOutW = m.w("audio_tower.conv_out.weight")
  x = x @ transpose(convOutW)  # (numChunks, T', dModel)

  # Add sinusoidal position embeddings
  let posEmb = m.sinPosEmb.slice([0, 0], [tDim, m.cfg.audio.dModel]).expandDims(0)
  x = x + posEmb

  # Compute output lengths per chunk for valid frame extraction
  var cnnLens: seq[int]
  for cl in chunkLens:
    cnnLens.add(getOutputLength(cl))

  # Extract valid frames from each chunk and concatenate
  var hiddenList: seq[Tensor]
  for i in 0..<numChunks:
    let validLen = cnnLens[i]
    hiddenList.add(x.slice([i, 0, 0], [i + 1, validLen, m.cfg.audio.dModel]).squeeze(0))

  var hidden = concatenate(hiddenList, axis = 0)  # (totalTokens, dModel)

  # Build block attention mask for windowed processing
  let totalCnnLen = getOutputLength(featureLen)
  let maxLenAfterCnn = max(cnnLens)
  let windowAfterCnn = maxLenAfterCnn * (m.cfg.audio.nWindowInfer div (m.cfg.audio.nWindow * 2))

  var cuChunkLens: seq[int] = @[0]
  let numFullWindows = totalCnnLen div windowAfterCnn
  for _ in 0..<numFullWindows:
    cuChunkLens.add(windowAfterCnn)
  let remainder = totalCnnLen mod windowAfterCnn
  if remainder != 0:
    cuChunkLens.add(remainder)

  # Cumulative sum for cu_seqlens (cuChunkLens starts with 0 to produce leading 0)
  var cuSeqlens: seq[int]
  var cumSum = 0
  for cl in cuChunkLens:
    cumSum += cl
    cuSeqlens.add(cumSum)

  # Build block attention mask (seqLen, seqLen)
  let seqLen = hidden.dim(0)
  var maskData = newSeq[float32](seqLen * seqLen)
  for i in 0..<seqLen * seqLen:
    maskData[i] = -1e9'f32
  # Allow attention within each block
  for blockIdx in 0..<cuSeqlens.len - 1:
    let blockStart = cuSeqlens[blockIdx]
    let blockEnd = cuSeqlens[blockIdx + 1]
    for r in blockStart..<blockEnd:
      for c in blockStart..<blockEnd:
        maskData[r * seqLen + c] = 0.0'f32

  let attMask = fromSeq(maskData, [seqLen, seqLen])
  let attMask4d = attMask.expandDims(0).expandDims(0)  # (1, 1, seqLen, seqLen)

  # Add batch dim for transformer layers
  hidden = hidden.expandDims(0)  # (1, seqLen, dModel)

  # Run through encoder layers
  for i in 0..<m.cfg.audio.encoderLayers:
    hidden = audioEncoderLayer(m, hidden, attMask4d, "audio_tower.layers." & $i & ".")

  hidden = hidden.squeeze(0)  # (seqLen, dModel)

  # Post-encoder: LayerNorm → GELU(proj1) → proj2
  let lnW = m.w("audio_tower.ln_post.weight")
  let lnB = m.w("audio_tower.ln_post.bias")
  hidden = layerNorm(hidden, lnW, lnB)

  let p1W = m.w("audio_tower.proj1.weight")
  let p1B = m.w("audio_tower.proj1.bias")
  hidden = gelu((hidden @ transpose(p1W)) + p1B)

  let p2W = m.w("audio_tower.proj2.weight")
  let p2B = m.w("audio_tower.proj2.bias")
  hidden = (hidden @ transpose(p2W)) + p2B  # (totalTokens, outputDim)

  hidden

# ── RoPE (Rotary Position Embeddings) ───────────────────────────

proc applyRope(x: Tensor, freqs: Tensor, offset: int): Tensor =
  ## Apply rotary position embeddings (non-traditional, split at midpoint).
  ## x: (B, numHeads, seqLen, headDim)
  ## freqs: precomputed (maxPos, headDim/2) angle table
  let seqLen = x.dim(2)
  let headDim = x.dim(3)
  let halfDim = headDim div 2

  # Get angles for this sequence range: (seqLen, halfDim) → (1, 1, seqLen, halfDim)
  let angles = freqs.slice([offset, 0], [offset + seqLen, halfDim]).expandDims(0).expandDims(0)
  let cosF = cos(angles)
  let sinF = sin(angles)

  # Split at midpoint: x[..., :halfDim] and x[..., halfDim:]
  let x1 = x.slice([0, 0, 0, 0], [x.dim(0), x.dim(1), seqLen, halfDim])
  let x2 = x.slice([0, 0, 0, halfDim], [x.dim(0), x.dim(1), seqLen, headDim])

  # Rotate
  let o1 = x1 * cosF - x2 * sinF
  let o2 = x1 * sinF + x2 * cosF

  # Concatenate halves back
  concatenate([o1, o2], axis = -1)

proc buildRopeFreqs(headDim: int, maxPos: int, theta: float64): Tensor =
  ## Build RoPE frequency table (maxPos, headDim/2).
  let halfDim = headDim div 2
  var data = newSeq[float32](maxPos * halfDim)
  for pos in 0..<maxPos:
    for i in 0..<halfDim:
      let freq = float64(pos) / pow(theta, float64(2 * i) / float64(headDim))
      data[pos * halfDim + i] = freq.float32
  fromSeq(data, [maxPos, halfDim])

# ── Text decoder ─────────────────────────────────────────────────

proc textAttention(m: Qwen3Asr, x: Tensor, cache: var KvCache, layerIdx: int): Tensor =
  ## GQA attention with RoPE and Q/K norms for text decoder.
  let B = x.dim(0)
  let L = x.dim(1)
  let prefix = "model.layers." & $layerIdx & ".self_attn."
  let numHeads = m.cfg.text.numHeads
  let numKvHeads = m.cfg.text.numKvHeads
  let headDim = m.cfg.text.headDim
  let scale = 1.0'f32 / sqrt(float32(headDim))

  # Project Q, K, V (quantized)
  var q = m.qlinear(x, prefix & "q_proj.")
  var k = m.qlinear(x, prefix & "k_proj.")
  var v = m.qlinear(x, prefix & "v_proj.")

  # Reshape
  q = q.reshape([B, L, numHeads, headDim])
  k = k.reshape([B, L, numKvHeads, headDim])
  v = v.reshape([B, L, numKvHeads, headDim])

  # Q/K RMSNorm (per-head)
  let qNormW = m.w(prefix & "q_norm.weight")
  let kNormW = m.w(prefix & "k_norm.weight")
  q = rmsNorm(q, qNormW, m.cfg.text.rmsNormEps)
  k = rmsNorm(k, kNormW, m.cfg.text.rmsNormEps)

  # Transpose to (B, numHeads, L, headDim)
  q = q.transpose([0, 2, 1, 3])
  k = k.transpose([0, 2, 1, 3])
  v = v.transpose([0, 2, 1, 3])

  # Apply RoPE
  q = applyRope(q, m.ropeFreqs, cache.offset)
  k = applyRope(k, m.ropeFreqs, cache.offset)

  # Update KV cache
  if cache.offset == 0:
    cache.keys = k
    cache.values = v
  else:
    cache.keys = concatenate([cache.keys, k], axis = 2)
    cache.values = concatenate([cache.values, v], axis = 2)
  cache.offset += L

  let allK = cache.keys    # (B, numKvHeads, totalLen, headDim)
  let allV = cache.values

  # GQA: repeat KV heads to match Q heads
  let kvRepeat = numHeads div numKvHeads
  var expandedK, expandedV: Tensor
  if kvRepeat > 1:
    # (B, numKvHeads, totalLen, headDim) → (B, numHeads, totalLen, headDim)
    expandedK = allK.repeat(kvRepeat, axis = 1)
    expandedV = allV.repeat(kvRepeat, axis = 1)
  else:
    expandedK = allK
    expandedV = allV

  # Scaled dot-product attention with causal mask
  let totalLen = expandedK.dim(2)
  let scores = (q * scalar(scale)) @ expandedK.transpose([0, 1, 3, 2])
  # Causal mask: query at position i can attend to positions 0..i
  # For KV cache, query positions are [offset-L .. offset-1], key positions are [0 .. totalLen-1]
  if L > 1:
    # Prefill: need causal mask
    var maskData = newSeq[float32](L * totalLen)
    for qi in 0..<L:
      let qPos = cache.offset - L + qi  # actual position of this query
      for ki in 0..<totalLen:
        if ki > qPos:
          maskData[qi * totalLen + ki] = -1e9'f32
    let mask = fromSeq(maskData, [1, 1, L, totalLen])
    let attn = softmax(scores + mask, axis = -1, precise = true)
    let attnOut = (attn @ expandedV).transpose([0, 2, 1, 3]).reshape([B, L, numHeads * headDim])
    m.qlinear(attnOut, prefix & "o_proj.")
  else:
    # Decode step: single query, no mask needed (can attend to everything)
    let attn = softmax(scores, axis = -1, precise = true)
    let attnOut = (attn @ expandedV).transpose([0, 2, 1, 3]).reshape([B, L, numHeads * headDim])
    m.qlinear(attnOut, prefix & "o_proj.")

proc textMLP(m: Qwen3Asr, x: Tensor, layerIdx: int): Tensor =
  ## SwiGLU MLP for text decoder (all quantized).
  let prefix = "model.layers." & $layerIdx & ".mlp."
  let gate = m.qlinear(x, prefix & "gate_proj.")
  let up = m.qlinear(x, prefix & "up_proj.")
  m.qlinear(silu(gate) * up, prefix & "down_proj.")

proc textDecoderLayer(m: Qwen3Asr, x: Tensor, cache: var KvCache, layerIdx: int): Tensor =
  ## Single text decoder transformer layer.
  let prefix = "model.layers." & $layerIdx & "."

  # Pre-norm self-attention
  let lnW = m.w(prefix & "input_layernorm.weight")
  var cur = x + textAttention(m, rmsNorm(x, lnW, m.cfg.text.rmsNormEps), cache, layerIdx)

  # Pre-norm MLP
  let ln2W = m.w(prefix & "post_attention_layernorm.weight")
  cur + textMLP(m, rmsNorm(cur, ln2W, m.cfg.text.rmsNormEps), layerIdx)

proc textForward*(m: Qwen3Asr, inputEmbeds: Tensor, cache: var seq[KvCache]): Tensor =
  ## Run text decoder. Returns logits.
  var hidden = inputEmbeds
  for i in 0..<m.cfg.text.numLayers:
    hidden = textDecoderLayer(m, hidden, cache[i], i)

  # Final norm
  let normW = m.w("model.norm.weight")
  hidden = rmsNorm(hidden, normW, m.cfg.text.rmsNormEps)

  # LM head (tied with embedding weights — quantized matmul)
  m.lmHead(hidden)

# ── Prompt building ──────────────────────────────────────────────

proc buildPrompt*(m: Qwen3Asr, numAudioTokens: int): seq[int32] =
  ## Build input_ids: prefix + audio_pad tokens + suffix.
  result = m.promptPrefix
  for _ in 0..<numAudioTokens:
    result.add(m.cfg.audioTokenId.int32)
  result.add(m.promptSuffix)

# ── Embedding with audio feature merging ─────────────────────────

proc buildInputEmbeds*(m: Qwen3Asr, inputIds: seq[int32], audioFeatures: Tensor): Tensor =
  ## Build input embeddings, replacing audio_pad tokens with audio features.
  var idsBuf = inputIds
  let idsT = fromSeq(idsBuf, [inputIds.len]).astype(MLX_INT32)

  # Embed all tokens
  let embeds = m.embedTokens(idsT).expandDims(0)  # (1, seqLen, hiddenSize)
  # embeds: (1, seqLen, hiddenSize)

  # Find audio token positions and replace with audio features
  let audioFeat = audioFeatures.astype(embeds.dtype)
  var resultEmbeds = embeds

  # Build replacement: for each position, use audio feature if it's an audio_pad token
  var audioIdx = 0
  let seqLen = inputIds.len
  let hiddenSize = m.cfg.text.hiddenSize

  # Collect slices (each is 1 × hiddenSize)
  var slices: seq[Tensor]
  for i in 0..<seqLen:
    if inputIds[i] == m.cfg.audioTokenId.int32 and audioIdx < audioFeatures.dim(0):
      slices.add(audioFeat.slice([audioIdx, 0], [audioIdx + 1, hiddenSize]))
      audioIdx += 1
    else:
      slices.add(resultEmbeds.slice([0, i, 0], [1, i + 1, hiddenSize]).squeeze(0))

  concatenate(slices, axis = 0).expandDims(0)  # (1, seqLen, hiddenSize)

# ── Token decoding ───────────────────────────────────────────────

proc decodeToken(m: Qwen3Asr, tokenId: int): string =
  m.tokens.getOrDefault(tokenId, "")

proc decodeTokens*(m: Qwen3Asr, tokenIds: seq[int32]): string =
  for tid in tokenIds:
    result.add(m.decodeToken(tid.int))

# ── Public API ───────────────────────────────────────────────────

proc loadQwen3Asr*(modelDir: string): Qwen3Asr =
  ## Load Qwen3 ASR model from directory.
  initMlx()
  initDefaultStream()

  result = Qwen3Asr()
  result.cfg = parseConfig(modelDir / "config.json")

  # Load weights
  result.weights = initTable[string, Tensor]()
  for f in walkDir(modelDir):
    if f.path.endsWith(".safetensors"):
      let tensors = loadSafetensors(f.path)
      for k, v in tensors:
        var key = k
        # Strip "thinker." prefix if present
        if key.startsWith("thinker."):
          key = key[8..^1]
        # Skip lm_head (tied with embeddings)
        if key == "lm_head.weight":
          continue
        result.weights[key] = v

  # Load mel filterbank (128 bins) — must match WhisperFeatureExtractor exactly
  let melPath = modelDir / "mel_filters_128.bin"
  if fileExists(melPath):
    var melData = readFile(melPath)
    let nFreqs = N_FFT div 2 + 1  # 201
    let nMels = result.cfg.audio.numMelBins  # 128
    assert melData.len == nFreqs * nMels * sizeof(float32)
    var melBuf = newSeq[float32](nFreqs * nMels)
    copyMem(addr melBuf[0], addr melData[0], melData.len)
    result.melFilters = fromSeq(melBuf, [nFreqs, nMels])
  else:
    result.melFilters = buildMelFilters(QWEN3_SAMPLE_RATE, N_FFT, result.cfg.audio.numMelBins)

  # Build sinusoidal position embeddings for encoder
  result.sinPosEmb = buildSinPosEmb(result.cfg.audio.maxSourcePositions, result.cfg.audio.dModel)

  # Build RoPE frequency table for decoder (8192 positions is plenty for STT)
  result.ropeFreqs = buildRopeFreqs(result.cfg.text.headDim, 8192,
                                     result.cfg.text.ropeTheta)

  # Load tokenizer — tokens.json (id→token) or vocab.json (token→id)
  result.tokens = initTable[int, string]()
  let tokensPath = modelDir / "tokens.json"
  let vocabPath = modelDir / "vocab.json"
  if fileExists(tokensPath):
    let tokJ = parseJson(readFile(tokensPath))
    for k, v in tokJ:
      result.tokens[parseInt(k)] = v.getStr()
  elif fileExists(vocabPath):
    let vocJ = parseJson(readFile(vocabPath))
    for tok, idNode in vocJ:
      result.tokens[idNode.getInt()] = tok
  else:
    stderr.writeLine "WARNING: tokens.json/vocab.json not found, decoding will fail"

  # Precomputed prompt token IDs
  result.promptPrefix = @[151644'i32, 8948, 198, 151645, 198, 151644, 872, 198, 151669]
  result.promptSuffix = @[151670'i32, 151645, 198, 151644, 77091, 198]
  result.eosTokenIds = @[151643'i32, 151645]

  stderr.writeLine "Qwen3 ASR loaded (encoder: " & $result.cfg.audio.encoderLayers &
    " layers, decoder: " & $result.cfg.text.numLayers & " layers)"

proc makeCache*(m: Qwen3Asr): seq[KvCache] =
  result = newSeq[KvCache](m.cfg.text.numLayers)
  for i in 0..<m.cfg.text.numLayers:
    result[i] = KvCache(offset: 0)

proc transcribe*(m: Qwen3Asr, samples: openArray[float32],
                 maxTokens: int = 4096): string =
  ## Transcribe audio to text.
  ## Input: 16kHz mono float32 PCM.
  if samples.len == 0:
    return ""

  # Step 1: Compute mel features
  let (features, featureLen) = m.computeFeatures(samples)

  # Step 2: Encode audio
  let audioFeatures = m.encodeAudio(features, featureLen)
  eval(audioFeatures)
  let numAudioTokens = audioFeatures.dim(0)

  # Step 3: Build prompt
  let promptIds = m.buildPrompt(numAudioTokens)

  # Step 4: Build input embeddings with audio merged
  let inputEmbeds = m.buildInputEmbeds(promptIds, audioFeatures)
  eval(inputEmbeds)

  # Step 5: Prefill — run all prompt tokens through decoder
  var cache = m.makeCache()
  let prefillLogits = m.textForward(inputEmbeds, cache)
  eval(prefillLogits)

  # Get first generated token (greedy: argmax of last position)
  let lastLogits = prefillLogits.slice([0, prefillLogits.dim(1) - 1, 0],
                                        [1, prefillLogits.dim(1), m.cfg.text.vocabSize])
  var nextToken = argmax(lastLogits.squeeze(0), axis = -1)
  eval(nextToken)
  var tokenId = nextToken.itemInt32()

  var generatedTokens: seq[int32]

  # Step 6: Autoregressive decode
  for step in 0..<maxTokens:
    if tokenId in m.eosTokenIds:
      break
    generatedTokens.add(tokenId)

    # Embed next token
    var tokBuf = @[tokenId]
    let tokenIds = fromSeq(tokBuf, [1]).astype(MLX_INT32)
    let tokenEmbed = m.embedTokens(tokenIds).expandDims(0)  # (1, 1, hidden)

    # Forward through decoder
    let logits = m.textForward(tokenEmbed, cache)
    eval(logits)

    # Greedy decode
    let tok = argmax(logits.slice([0, 0, 0], [1, 1, m.cfg.text.vocabSize]).squeeze(0), axis = -1)
    eval(tok)
    tokenId = tok.itemInt32()

  # Step 7: Decode tokens to text
  var text = m.decodeTokens(generatedTokens)

  # Extract text after <asr_text> tag if present (language auto-detect mode)
  let asrTag = "<asr_text>"
  let tagIdx = text.find(asrTag)
  if tagIdx >= 0:
    text = text[tagIdx + asrTag.len..^1]

  text.strip()

proc close*(m: Qwen3Asr) =
  ## Release model resources.
  m.weights.clear()
