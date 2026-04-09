## Whisper speech-to-text using Apple MLX backend.
## Encoder-decoder transformer: audio → mel spectrogram → encoder → decoder → tokens.
## Port from mlx-audio's Whisper implementation to Nim, using mlx-c bindings.

import std/[tables, strutils, math, os, json, sequtils]
import ../mlx/mlx

const WHISPER_SAMPLE_RATE* = 16000
const N_FFT = 400
const HOP_LENGTH = 160
const CHUNK_LENGTH = 30
const N_SAMPLES = CHUNK_LENGTH * WHISPER_SAMPLE_RATE  # 480000
const N_FRAMES* = N_SAMPLES div HOP_LENGTH             # 3000

# ── Configuration ────────────────────────────────────────────────

type
  WhisperConfig* = object
    nMels*: int           # 80
    nAudioCtx*: int       # 1500 (30s @ 10fps after stride-2 conv)
    nAudioState*: int     # 512 (base), 384 (tiny), 768 (small), 1024 (medium), 1280 (large)
    nAudioHead*: int      # 8 (base)
    nAudioLayer*: int     # 6 (base)
    nVocab*: int          # 51864 (.en) or 51865 (multilingual)
    nTextCtx*: int        # 448
    nTextState*: int      # 512 (base)
    nTextHead*: int       # 8 (base)
    nTextLayer*: int      # 6 (base)

proc configFromJson*(data: JsonNode): WhisperConfig =
  ## Parse config.json — handles both MLX and HuggingFace formats.
  if data.hasKey("d_model") or data.hasKey("encoder_layers"):
    # HuggingFace format
    WhisperConfig(
      nMels: data.getOrDefault("num_mel_bins").getInt(80),
      nAudioCtx: data.getOrDefault("max_source_positions").getInt(1500),
      nAudioState: data.getOrDefault("d_model").getInt(512),
      nAudioHead: data.getOrDefault("encoder_attention_heads").getInt(8),
      nAudioLayer: data.getOrDefault("encoder_layers").getInt(6),
      nVocab: data.getOrDefault("vocab_size").getInt(51865),
      nTextCtx: data.getOrDefault("max_target_positions").getInt(448),
      nTextState: data.getOrDefault("d_model").getInt(512),
      nTextHead: data.getOrDefault("decoder_attention_heads").getInt(8),
      nTextLayer: data.getOrDefault("decoder_layers").getInt(6),
    )
  else:
    # MLX native format
    WhisperConfig(
      nMels: data.getOrDefault("n_mels").getInt(80),
      nAudioCtx: data.getOrDefault("n_audio_ctx").getInt(1500),
      nAudioState: data.getOrDefault("n_audio_state").getInt(512),
      nAudioHead: data.getOrDefault("n_audio_head").getInt(8),
      nAudioLayer: data.getOrDefault("n_audio_layer").getInt(6),
      nVocab: data.getOrDefault("n_vocab").getInt(51865),
      nTextCtx: data.getOrDefault("n_text_ctx").getInt(448),
      nTextState: data.getOrDefault("n_text_state").getInt(512),
      nTextHead: data.getOrDefault("n_text_head").getInt(8),
      nTextLayer: data.getOrDefault("n_text_layer").getInt(6),
    )

# ── Whisper model store ──────────────────────────────────────────

type
  WhisperModel* = object
    config*: WhisperConfig
    weights*: Table[string, Tensor]
    melFilters*: Tensor        # (n_mels, n_fft/2+1) precomputed
    posEmbEncoder*: Tensor     # sinusoidal, (n_audio_ctx, n_audio_state)
    causalMask*: Tensor        # (n_text_ctx, n_text_ctx) additive mask

proc w*(m: WhisperModel, key: string): Tensor {.inline.} =
  if key notin m.weights:
    raise newException(KeyError, "Missing weight: " & key)
  m.weights[key]

proc hasW*(m: WhisperModel, key: string): bool {.inline.} =
  key in m.weights

# ── Utility: LayerNorm, GELU ─────────────────────────────────────

proc layerNorm(x, weight, bias: Tensor, eps: float32 = 1e-5): Tensor =
  let m = mean(x, axis = -1, keepdims = true)
  let v = variance(x, axis = -1, keepdims = true)
  let norm = (x - m) / sqrt(v + scalar(eps))
  norm * weight + bias

proc gelu(x: Tensor): Tensor =
  ## GELU activation: x * 0.5 * (1 + erf(x / sqrt(2)))
  x * scalar(0.5'f32) * (scalar(1.0'f32) + erf(x * scalar(0.7071067811865476'f32)))

# ── Mel spectrogram ──────────────────────────────────────────────

proc hzToMel(freq: float64): float64 =
  ## HTK mel scale
  2595.0 * log10(1.0 + freq / 700.0)

proc melToHz(mel: float64): float64 =
  700.0 * (pow(10.0, mel / 2595.0) - 1.0)

proc buildMelFilters*(sampleRate: int, nFft: int, nMels: int): Tensor =
  ## Build mel filterbank matrix (n_freqs, n_mels) with slaney normalization.
  let nFreqs = nFft div 2 + 1
  let fMax = float64(sampleRate) / 2.0

  # Mel points
  let mMin = hzToMel(0.0)
  let mMax = hzToMel(fMax)

  # Frequency bin centers
  var allFreqs = newSeq[float32](nFreqs)
  for i in 0..<nFreqs:
    allFreqs[i] = float32(float64(i) * fMax / float64(nFreqs - 1))

  # Mel breakpoints
  var mPts = newSeq[float64](nMels + 2)
  for i in 0..nMels+1:
    mPts[i] = mMin + float64(i) * (mMax - mMin) / float64(nMels + 1)

  var fPts = newSeq[float64](nMels + 2)
  for i in 0..nMels+1:
    fPts[i] = melToHz(mPts[i])

  # Build triangular filterbank (n_freqs × n_mels)
  var fb = newSeq[float32](nFreqs * nMels)
  for m in 0..<nMels:
    let fDiffLow = fPts[m+1] - fPts[m]
    let fDiffHigh = fPts[m+2] - fPts[m+1]
    for f in 0..<nFreqs:
      let freq = float64(allFreqs[f])
      let downSlope = (freq - fPts[m]) / fDiffLow
      let upSlope = (fPts[m+2] - freq) / fDiffHigh
      let val = max(0.0, min(downSlope, upSlope))
      fb[f * nMels + m] = float32(val)

  # Slaney normalization: 2 / (f[m+2] - f[m])
  for m in 0..<nMels:
    let enorm = 2.0 / (fPts[m+2] - fPts[m])
    for f in 0..<nFreqs:
      fb[f * nMels + m] *= float32(enorm)

  fromSeq(fb, [nFreqs, nMels])

proc logMelSpectrogram*(audio: Tensor, model: WhisperModel): Tensor =
  ## Compute log-mel spectrogram from 16kHz PCM audio.
  ## Input: (N_SAMPLES,) float32. Output: (n_frames, n_mels).
  let window = hanning(N_FFT)

  # Zero-pad for centered STFT (good enough for Whisper)
  let padLen = N_FFT div 2
  let padded = concatenate([zeros([padLen]), audio, zeros([padLen])], axis = 0)

  # Frame the audio: shape (num_frames, N_FFT)
  let totalLen = padded.dim(0)
  let numFrames = 1 + (totalLen - N_FFT) div HOP_LENGTH

  # Build frame indices: each frame starts at i*HOP_LENGTH
  var frameData = newSeq[int32](numFrames)
  for i in 0..<numFrames:
    frameData[i] = int32(i * HOP_LENGTH)
  var frameOffsets = fromSeq(frameData, [numFrames])

  # Use take to gather frames: expand to (numFrames, N_FFT)
  let sampleIdx = arange(N_FFT, MLX_INT32)  # (N_FFT,)
  let indices = frameOffsets.expandDims(1) + sampleIdx.expandDims(0)  # (numFrames, N_FFT)
  let frames = padded.take(indices.flatten(), axis = 0).reshape([numFrames, N_FFT])

  # Apply window and FFT
  let windowed = frames * window
  let freqs = rfft(windowed, N_FFT)  # (numFrames, N_FFT/2+1) complex

  # Magnitude squared — abs of complex, then square
  let magnitudes = square(absVal(freqs))  # (numFrames, N_FFT/2+1)
  # Drop last frame to match reference (freqs[:-1, :])
  let mags = magnitudes.slice([0, 0], [numFrames - 1, magnitudes.dim(1)])

  # Apply mel filterbank: (numFrames-1, N_FFT/2+1) @ (N_FFT/2+1, n_mels) → (numFrames-1, n_mels)
  let melSpec = mags @ model.melFilters

  # Log scale with dynamic range compression
  let logSpec = log10(maximum(melSpec, scalar(1e-10'f32)))
  let logMax = maxVal(logSpec, axis = -1, keepdims = true)
  # Broadcast max across all positions
  let globalMax = maxVal(logMax, axis = 0, keepdims = true)
  let clamped = maximum(logSpec, globalMax - scalar(8.0'f32))
  (clamped + scalar(4.0'f32)) / scalar(4.0'f32)

# ── Sinusoidal positional embeddings ─────────────────────────────

proc sinusoids*(length, channels: int): Tensor =
  ## Sinusoidal positional embeddings for the audio encoder.
  ## Returns (length, channels) float32.
  let halfC = channels div 2
  let logIncr = ln(10000.0) / float64(halfC - 1)
  var data = newSeq[float32](length * channels)
  for t in 0..<length:
    for c in 0..<halfC:
      let angle = float64(t) * exp(-logIncr * float64(c))
      data[t * channels + c] = float32(sin(angle))
      data[t * channels + halfC + c] = float32(cos(angle))
  fromSeq(data, [length, channels])

# ── Additive causal mask ─────────────────────────────────────────

proc buildCausalMask*(size: int): Tensor =
  ## Create additive causal mask: 0 for allowed, -inf for masked.
  let onesM = ones([size, size])
  let upper = triu(onesM, 1)  # strictly upper triangular = 1
  upper * scalar(-1e9'f32)    # large negative for masked positions

# ── Multi-head attention ─────────────────────────────────────────

proc multiHeadAttention(model: WhisperModel, x: Tensor,
                        prefix: string, nHead: int,
                        xa: Tensor = Tensor(), # cross-attention input
                        mask: Tensor = Tensor(),
                        kvCache: tuple[k, v: Tensor] = (Tensor(), Tensor())
                        ): tuple[output: Tensor, kv: tuple[k, v: Tensor]] =
  ## Multi-head attention with optional cross-attention and KV cache.
  let qW = model.w(prefix & "query.weight")
  let qB = model.w(prefix & "query.bias")
  let q = (x @ transpose(qW)) + qB

  var k, v: Tensor
  let isCross = not xa.isNil and xa.ndim > 0

  if not isCross:
    # Self-attention
    let kW = model.w(prefix & "key.weight")
    k = x @ transpose(kW)  # key has no bias
    let vW = model.w(prefix & "value.weight")
    let vB = model.w(prefix & "value.bias")
    v = (x @ transpose(vW)) + vB
    # Append to KV cache
    if not kvCache.k.isNil and kvCache.k.ndim > 0:
      k = concatenate([kvCache.k, k], axis = 1)
      v = concatenate([kvCache.v, v], axis = 1)
  elif kvCache.k.isNil or kvCache.k.ndim == 0:
    # First cross-attention: compute K,V from xa
    let kW = model.w(prefix & "key.weight")
    k = xa @ transpose(kW)
    let vW = model.w(prefix & "value.weight")
    let vB = model.w(prefix & "value.bias")
    v = (xa @ transpose(vW)) + vB
  else:
    # Cached cross-attention
    k = kvCache.k
    v = kvCache.v

  let nBatch = q.dim(0)
  let nCtxQ = q.dim(1)
  let nState = q.dim(2)
  let headDim = nState div nHead
  let scale = pow(float32(headDim), -0.25'f32)

  # Reshape and transpose for multi-head: (B, T, nHead, headDim) → (B, nHead, T, headDim)
  let qr = q.reshape([nBatch, nCtxQ, nHead, headDim]).transpose([0, 2, 1, 3]) * scalar(scale)
  let nCtxK = k.dim(1)
  let kr = k.reshape([nBatch, nCtxK, nHead, headDim]).transpose([0, 2, 3, 1]) * scalar(scale)
  let vr = v.reshape([nBatch, nCtxK, nHead, headDim]).transpose([0, 2, 1, 3])

  # Attention scores
  var qk = qr @ kr  # (B, nHead, nCtxQ, nCtxK)
  if not mask.isNil and mask.ndim > 0:
    # mask is (nTextCtx, nTextCtx), slice to query context size
    # During autoregressive decode (nCtxQ=1), this gives [[0]] which has no effect
    let m = mask.slice([0, 0], [nCtxQ, nCtxQ])
    qk = qk + m

  let w = softmax(qk, axis = -1, precise = true)
  var attnOut = (w @ vr).transpose([0, 2, 1, 3])  # (B, nCtxQ, nHead, headDim)
  attnOut = attnOut.reshape([nBatch, nCtxQ, nState])

  # Output projection
  let oW = model.w(prefix & "out.weight")
  let oB = model.w(prefix & "out.bias")
  let output = (attnOut @ transpose(oW)) + oB

  (output, (k, v))

# ── Residual attention block ─────────────────────────────────────

type BlockCache = object
  selfKV*: tuple[k, v: Tensor]
  crossKV*: tuple[k, v: Tensor]

proc emptyBlockCache(): BlockCache =
  BlockCache(selfKV: (Tensor(), Tensor()), crossKV: (Tensor(), Tensor()))

proc residualAttentionBlock(model: WhisperModel, x: Tensor,
                            prefix: string, nHead: int,
                            xa: Tensor = Tensor(),
                            mask: Tensor = Tensor(),
                            cache: BlockCache = emptyBlockCache(),
                            hasCrossAttn: bool = false
                            ): tuple[output: Tensor, cache: BlockCache] =
  var cur = x
  var newCache = cache

  # Self-attention
  let attnLnW = model.w(prefix & "attn_ln.weight")
  let attnLnB = model.w(prefix & "attn_ln.bias")
  let normed = layerNorm(cur, attnLnW, attnLnB)
  let (attnOut, selfKV) = multiHeadAttention(model, normed,
    prefix & "attn.", nHead, mask = mask, kvCache = cache.selfKV)
  cur = cur + attnOut
  newCache.selfKV = selfKV

  # Cross-attention (decoder only)
  if hasCrossAttn:
    let crossLnW = model.w(prefix & "cross_attn_ln.weight")
    let crossLnB = model.w(prefix & "cross_attn_ln.bias")
    let crossNormed = layerNorm(cur, crossLnW, crossLnB)
    let (crossOut, crossKV) = multiHeadAttention(model, crossNormed,
      prefix & "cross_attn.", nHead, xa = xa, kvCache = cache.crossKV)
    cur = cur + crossOut
    newCache.crossKV = crossKV

  # MLP: LayerNorm → Linear → GELU → Linear
  let mlpLnW = model.w(prefix & "mlp_ln.weight")
  let mlpLnB = model.w(prefix & "mlp_ln.bias")
  let mlpNormed = layerNorm(cur, mlpLnW, mlpLnB)
  let mlp1W = model.w(prefix & "mlp1.weight")
  let mlp1B = model.w(prefix & "mlp1.bias")
  let mlp2W = model.w(prefix & "mlp2.weight")
  let mlp2B = model.w(prefix & "mlp2.bias")
  let h = gelu(((mlpNormed @ transpose(mlp1W)) + mlp1B))
  cur = cur + ((h @ transpose(mlp2W)) + mlp2B)

  (cur, newCache)

# ── Audio encoder ────────────────────────────────────────────────

proc encodeAudio*(model: WhisperModel, mel: Tensor): Tensor =
  ## Encode mel spectrogram → audio features.
  ## mel: (1, n_frames, n_mels) → output: (1, n_audio_ctx, n_audio_state)
  let cfg = model.config

  # Conv1: (1, n_frames, n_mels) → (1, n_frames, n_audio_state) with GELU
  let c1W = model.w("encoder.conv1.weight")
  let c1B = model.w("encoder.conv1.bias")
  let conv1Out = conv1d(mel, c1W, padding = 1)
  var x = gelu(conv1Out + c1B)

  # Conv2: stride=2, (1, n_frames, n_audio_state) → (1, n_frames/2, n_audio_state)
  let c2W = model.w("encoder.conv2.weight")
  let c2B = model.w("encoder.conv2.bias")
  x = gelu(conv1d(x, c2W, stride = 2, padding = 1) + c2B)

  # Add sinusoidal positional embeddings
  x = x + model.posEmbEncoder

  # Transformer blocks
  for i in 0..<cfg.nAudioLayer:
    let prefix = "encoder.blocks." & $i & "."
    let (blkOut, _) = residualAttentionBlock(model, x, prefix, cfg.nAudioHead)
    x = blkOut

  # Final layer norm
  let lnW = model.w("encoder.ln_post.weight")
  let lnB = model.w("encoder.ln_post.bias")
  layerNorm(x, lnW, lnB)

# ── Text decoder ─────────────────────────────────────────────────

proc decodeTokens*(model: WhisperModel, tokens: Tensor, audioFeatures: Tensor,
                    caches: var seq[BlockCache]): Tensor =
  ## Decode token IDs given audio features.
  ## tokens: (1, T) int32, audioFeatures: (1, n_audio_ctx, n_audio_state)
  ## Returns logits: (1, T, n_vocab)
  let cfg = model.config

  # Token embedding + positional embedding
  let tokEmb = model.w("decoder.token_embedding.weight")
  let posEmb = model.w("decoder.positional_embedding")

  # Determine offset from KV cache
  let offset = if caches.len > 0 and not caches[0].selfKV.k.isNil and caches[0].selfKV.k.ndim > 0:
    caches[0].selfKV.k.dim(1)
  else: 0

  var x = take(tokEmb, tokens, axis = 0)  # (1, T, n_text_state)
  let tLen = tokens.dim(1)
  let posSlice = posEmb.slice([offset, 0], [offset + tLen, cfg.nTextState])
  x = x + posSlice

  # Initialize caches if needed
  if caches.len == 0:
    for i in 0..<cfg.nTextLayer:
      caches.add(emptyBlockCache())

  # Transformer blocks with cross-attention
  for i in 0..<cfg.nTextLayer:
    let prefix = "decoder.blocks." & $i & "."
    let (blkOut, newC) = residualAttentionBlock(model, x, prefix, cfg.nTextHead,
      xa = audioFeatures, mask = model.causalMask, cache = caches[i], hasCrossAttn = true)
    x = blkOut
    caches[i] = newC

  # Final layer norm
  let lnW = model.w("decoder.ln.weight")
  let lnB = model.w("decoder.ln.bias")
  x = layerNorm(x, lnW, lnB)

  # Output logits via weight-tied embedding: x @ tokEmb.T
  x @ transpose(tokEmb)

# ── Greedy decoding ──────────────────────────────────────────────

# Token IDs for Whisper models (51864 = English-only, 51865 = multilingual)
const
  SOT_TOKEN = 50257          # <|startoftranscript|>
  EOT_TOKEN = 50256          # <|endoftext|>
  TRANSCRIBE_TOKEN = 50358   # <|transcribe|>
  NO_TIMESTAMPS_TOKEN = 50362 # <|notimestamps|>
  EN_TOKEN = 50259           # <|en|>
  # For multilingual models, language tokens start at 50259

proc getInitialTokens(cfg: WhisperConfig, language: string): seq[int32] =
  ## Build the initial prompt tokens: SOT [+ language + task] + notimestamps
  ## language="auto" omits the language token, letting the model freely mix languages.
  result = @[int32(SOT_TOKEN)]
  if cfg.nVocab >= 51865 and language != "auto":
    # Multilingual model — add language and task tokens
    # Language token = SOT + 1 + language_index
    let langToken = case language
      of "en": EN_TOKEN
      of "zh": 50260
      of "de": 50261
      of "es": 50262
      of "ru": 50263
      of "ko": 50264
      of "fr": 50265
      of "ja": 50266
      of "pt": 50267
      of "tr": 50268
      of "pl": 50269
      of "it": 50274
      else: EN_TOKEN  # default to English
    result.add(int32(langToken))
    result.add(int32(TRANSCRIBE_TOKEN))
  result.add(int32(NO_TIMESTAMPS_TOKEN))

proc greedyDecode*(model: WhisperModel, audioFeatures: Tensor,
                   language: string = "en", maxTokens: int = 224): seq[int32] =
  ## Greedy decode: pick argmax at each step until EOT or maxTokens.
  let cfg = model.config
  let initTokens = getInitialTokens(cfg, language)

  var tokensList = initTokens
  var caches: seq[BlockCache] = @[]

  # First pass: encode all initial tokens at once
  var inputIds = newSeq[int32](tokensList.len)
  for i, t in tokensList: inputIds[i] = t
  var inputTensor = fromSeq(inputIds, [1, tokensList.len])

  let logits = model.decodeTokens(inputTensor, audioFeatures, caches)
  eval(logits)

  # Get last token prediction
  let lastLogits = logits.slice([0, tokensList.len - 1, 0],
                                 [1, tokensList.len, cfg.nVocab]).squeeze(0).squeeze(0)
  let nextToken = argmax(lastLogits, axis = -1).itemInt32
  tokensList.add(nextToken)

  if nextToken == int32(EOT_TOKEN):
    return tokensList

  # Auto-regressive loop
  for step in 0..<maxTokens:
    var singleToken = @[tokensList[^1]]
    var singleInput = fromSeq(singleToken, [1, 1])

    let stepLogits = model.decodeTokens(singleInput, audioFeatures, caches)
    eval(stepLogits)

    let stepLast = stepLogits.slice([0, 0, 0], [1, 1, cfg.nVocab]).squeeze(0).squeeze(0)
    let nextTok = argmax(stepLast, axis = -1).itemInt32

    tokensList.add(nextTok)
    if nextTok == int32(EOT_TOKEN):
      break

  tokensList

# ── Tokenizer (vocabulary decode) ────────────────────────────────

type WhisperTokenizer* = object
  vocab*: seq[string]          # token_id → string
  merges*: seq[tuple[a, b: string]]  # BPE merges (not needed for decode)

proc loadTokenizer*(modelDir: string): WhisperTokenizer =
  ## Load tokenizer from vocab.json in the model directory.
  let vocabPath = modelDir / "vocab.json"
  if fileExists(vocabPath):
    let data = parseJson(readFile(vocabPath))
    # vocab.json maps string → id; invert it
    var maxId = 0
    for key, val in data:
      maxId = max(maxId, val.getInt)
    result.vocab = newSeq[string](maxId + 1)
    for key, val in data:
      result.vocab[val.getInt] = key

proc decodeTokens*(tok: WhisperTokenizer, tokens: seq[int32]): string =
  ## Decode token IDs to text. Filters out special tokens (id >= 50257).
  var pieces: seq[string]
  for t in tokens:
    if t >= 50256: continue  # skip special tokens (EOT=50256, SOT=50257, etc)
    if t >= 0 and t < int32(tok.vocab.len):
      pieces.add(tok.vocab[t])
  result = pieces.join("")
  # BPE uses "Ġ" (U+0120) for space — convert
  result = result.replace("Ġ", " ")
  result = result.strip()

# ── Weight sanitization ──────────────────────────────────────────

proc sanitizeWeights*(weights: Table[string, Tensor]): Table[string, Tensor] =
  ## Remap HuggingFace weight keys to MLX naming convention.
  let keyMap = [
    ("encoder.layer_norm.", "encoder.ln_post."),
    ("decoder.layer_norm.", "decoder.ln."),
    ("encoder.layers.", "encoder.blocks."),
    ("decoder.layers.", "decoder.blocks."),
    (".self_attn_layer_norm.", ".attn_ln."),
    (".final_layer_norm.", ".mlp_ln."),
    (".encoder_attn_layer_norm.", ".cross_attn_ln."),
    (".fc1.", ".mlp1."),
    (".fc2.", ".mlp2."),
    (".self_attn.q_proj.", ".attn.query."),
    (".self_attn.k_proj.", ".attn.key."),
    (".self_attn.v_proj.", ".attn.value."),
    (".self_attn.out_proj.", ".attn.out."),
    (".encoder_attn.q_proj.", ".cross_attn.query."),
    (".encoder_attn.k_proj.", ".cross_attn.key."),
    (".encoder_attn.v_proj.", ".cross_attn.value."),
    (".encoder_attn.out_proj.", ".cross_attn.out."),
    ("decoder.embed_tokens.", "decoder.token_embedding."),
  ]

  let isHf = weights.keys.toSeq.anyIt(it.startsWith("model."))
  result = initTable[string, Tensor]()

  for key, val in weights:
    var k = key
    var v = val

    if isHf:
      if k.startsWith("model."):
        k = k[6..^1]
      # Skip encoder positional embeddings (computed from sinusoids)
      if "encoder.embed_positions.weight" in k:
        continue
      # Remap decoder positional embeddings
      if "decoder.embed_positions.weight" in k:
        k = "decoder.positional_embedding"

      for (old, rep) in keyMap:
        if old in k:
          k = k.replace(old, rep)

      # Transpose conv1d weights: HF (out, in, kernel) → MLX (out, kernel, in)
      if ("conv1.weight" in k or "conv2.weight" in k) and v.ndim == 3:
        v = v.transpose([0, 2, 1])

    result[k] = v

# ── Model loading ────────────────────────────────────────────────

proc loadWhisperMlx*(modelDir: string): WhisperModel =
  ## Load a Whisper model from a directory containing config.json and *.safetensors.
  initMlx()
  initDefaultStream()

  # Load config
  let configPath = modelDir / "config.json"
  if not fileExists(configPath):
    raise newException(IOError, "config.json not found in: " & modelDir)
  let configData = parseJson(readFile(configPath))
  result.config = configFromJson(configData)
  let cfg = result.config

  echo "Whisper config: ", cfg.nAudioState, "d, ", cfg.nAudioLayer, " enc layers, ",
       cfg.nTextLayer, " dec layers, vocab=", cfg.nVocab

  # Load weights
  result.weights = initTable[string, Tensor]()
  for f in walkDir(modelDir):
    if f.path.endsWith(".safetensors"):
      let tensors = loadSafetensors(f.path)
      for k, v in tensors:
        result.weights[k] = v
      echo "Loaded ", tensors.len, " tensors from ", extractFilename(f.path)

  # Sanitize keys (HuggingFace → MLX naming)
  result.weights = sanitizeWeights(result.weights)

  # Build mel filterbank
  result.melFilters = buildMelFilters(WHISPER_SAMPLE_RATE, N_FFT, cfg.nMels)

  # Build sinusoidal positional embeddings for encoder
  result.posEmbEncoder = sinusoids(cfg.nAudioCtx, cfg.nAudioState)

  # Build causal mask for decoder
  result.causalMask = buildCausalMask(cfg.nTextCtx)

  echo "Whisper MLX model loaded: ", cfg.nAudioLayer, " encoder + ",
       cfg.nTextLayer, " decoder layers"

# ── Public API (matches whisper.nim interface) ────────────────────

type
  SpeechRecognizer* = ref object
    model*: WhisperModel
    tokenizer*: WhisperTokenizer
    language*: string

proc newSpeechRecognizer*(modelPath: string, language: string = "en"): SpeechRecognizer =
  ## Load a Whisper MLX model from a directory.
  if not dirExists(modelPath):
    raise newException(IOError, "Whisper model directory not found: " & modelPath)
  var rec = SpeechRecognizer(language: language)
  rec.model = loadWhisperMlx(modelPath)
  rec.tokenizer = loadTokenizer(modelPath)
  rec

proc transcribe*(r: SpeechRecognizer, samples: openArray[float32]): string =
  ## Transcribe float32 PCM audio (must be 16kHz mono).
  if samples.len == 0: return ""

  # Pad or trim to N_SAMPLES (30 seconds)
  var audio: seq[float32]
  if samples.len >= N_SAMPLES:
    audio = @(samples[0..<N_SAMPLES])
  else:
    audio = @samples
    audio.setLen(N_SAMPLES)

  var audioTensor = fromSeq(audio, [N_SAMPLES])

  # Compute mel spectrogram: (n_frames, n_mels)
  let mel = logMelSpectrogram(audioTensor, r.model)
  let melBatched = mel.expandDims(0)  # (1, n_frames, n_mels)

  # Encode audio
  let audioFeatures = encodeAudio(r.model, melBatched)
  eval(audioFeatures)

  # Greedy decode
  let tokens = greedyDecode(r.model, audioFeatures, r.language)

  # Decode tokens to text
  if r.tokenizer.vocab.len > 0:
    r.tokenizer.decodeTokens(tokens)
  else:
    # Fallback: return token IDs as string
    var parts: seq[string]
    for t in tokens:
      if t < 50257:
        parts.add($t)
    parts.join(" ")

proc close*(r: SpeechRecognizer) =
  r.model.weights.clear()

# ── Smoke test ───────────────────────────────────────────────────

when isMainModule:
  import std/times

  if paramCount() < 1:
    echo "Usage: whisper_mlx <model_dir> [audio.wav]"
    quit(1)

  let modelDir = paramStr(1)
  var rec = newSpeechRecognizer(modelDir)

  if paramCount() >= 2:
    # Load WAV file: find "data" chunk, read 16-bit PCM samples
    let wavPath = paramStr(2)
    let data = readFile(wavPath)
    var samples: seq[float32]
    # Find "data" chunk by searching for the marker
    var dataOffset = -1
    for i in 0..<data.len - 8:
      if data[i] == 'd' and data[i+1] == 'a' and data[i+2] == 't' and data[i+3] == 'a':
        dataOffset = i + 8  # skip "data" + 4-byte size
        break
    if dataOffset < 0:
      echo "Invalid WAV: no data chunk found"
      quit(1)
    let numSamples = (data.len - dataOffset) div 2
    samples = newSeq[float32](numSamples)
    for i in 0..<numSamples:
      let lo = uint8(data[dataOffset + i * 2])
      let hi = uint8(data[dataOffset + i * 2 + 1])
      let sample = cast[int16](lo.uint16 or (hi.uint16 shl 8))
      samples[i] = float32(sample) / 32768.0
    echo "Audio: ", numSamples, " samples (", numSamples.float / 16000.0, "s)"

    let t0 = cpuTime()
    let text = rec.transcribe(samples)
    let elapsed = cpuTime() - t0
    echo "Transcription (", (elapsed * 1000).int, "ms): ", text
  else:
    echo "Model loaded. Pass an audio file to transcribe."

  rec.close()
