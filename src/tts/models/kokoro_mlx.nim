## Kokoro TTS model runner using Apple MLX backend.
## Non-autoregressive: text → ALBERT → duration → F0/N → decoder → iSTFT → audio.
## Port from kokoro-mlx (Python) to Nim, using mlx-c bindings.

import std/[tables, strutils, math, os, json, unicode, times]
import ../mlx/mlx
import ../common
import ../phonem/phonemizer

type SynthCallback* = proc(chunk: AudioOutput, index, total: int)

# ── Configuration ────────────────────────────────────────────────

type
  KokoroConfig* = object
    vocabSize*: int
    hiddenSize*: int          # ALBERT hidden = 768
    embeddingSize*: int       # ALBERT embedding = 128
    nAttnHeads*: int          # 12
    intermediateSize*: int    # 2048
    nRecurrence*: int         # 12 (shared layer iterations)
    maxPositionEmbeddings*: int  # 512
    styleDim*: int            # 256 (128 prosody + 128 decoder)
    dHidden*: int             # 512
    nDurationLayers*: int     # 3
    maxDur*: int              # 50
    textEncoderKernelSize*: int  # 5
    nConvLayers*: int         # 3
    f0NBlocks*: int           # 3
    nDecoderBlocks*: int      # 4
    nKernels*: int            # 3
    nUpsamples*: int          # 2
    upsampleRates*: seq[int]  # [10, 6]
    upsampleKernelSizes*: seq[int]  # [20, 12]
    resblockKernelSizes*: seq[int]  # [3, 7, 11]
    resblockDilationSizes*: seq[seq[int]]  # [[1,3,5],[1,3,5],[1,3,5]]
    genIstftNFft*: int        # 20
    genIstftHop*: int         # 5
    upsampleInitialChannel*: int  # 512
    harmonicNum*: int         # 8
    samplingRate*: float32    # 24000

proc defaultConfig*(): KokoroConfig =
  KokoroConfig(
    vocabSize: 178,
    hiddenSize: 768,
    embeddingSize: 128,
    nAttnHeads: 12,
    intermediateSize: 2048,
    nRecurrence: 12,
    maxPositionEmbeddings: 512,
    styleDim: 256,
    dHidden: 512,
    nDurationLayers: 3,
    maxDur: 50,
    textEncoderKernelSize: 5,
    nConvLayers: 3,
    f0NBlocks: 3,
    nDecoderBlocks: 4,
    nKernels: 3,
    nUpsamples: 2,
    upsampleRates: @[10, 6],
    upsampleKernelSizes: @[20, 12],
    resblockKernelSizes: @[3, 7, 11],
    resblockDilationSizes: @[@[1, 3, 5], @[1, 3, 5], @[1, 3, 5]],
    genIstftNFft: 20,
    genIstftHop: 5,
    upsampleInitialChannel: 512,
    harmonicNum: 8,
    samplingRate: 24000.0,
  )

# ── Weight store ─────────────────────────────────────────────────
# Instead of building a class hierarchy, we store all weights in a flat
# Table[string, Tensor] keyed by safetensors name. Forward functions
# look up weights by name prefix.

type
  StftBasis* = object
    forwardBasis*, cosMat*, sinMat*, window*: Tensor
    valid*: bool

  KokoroModel* = object
    config*: KokoroConfig
    weights*: Table[string, Tensor]
    voices*: Table[string, Tensor]  # voice name → (510, 1, 256) or (256,) per token
    vocab*: Table[string, int]
    phmzr*: Phonemizer
    stftBasis*: StftBasis  # cached STFT basis matrices
    quantCfg*: QuantConfig

  QuantConfig* = object
    bits*: int
    groupSize*: int
    quantized*: bool  ## true if model uses quantized weights

proc w*(model: KokoroModel, key: string): Tensor {.inline.} =
  ## Look up a weight tensor by safetensors key.
  if key notin model.weights:
    raise newException(KeyError, "weight not found: " & key)
  model.weights[key]

proc hasW*(model: KokoroModel, key: string): bool {.inline.} =
  key in model.weights

proc isQuantized*(model: KokoroModel, key: string): bool {.inline.} =
  ## Check if a weight has quantized scales companion.
  (key & ".scales") in model.weights

proc linear*(model: KokoroModel, x: Tensor, weightKey: string): Tensor =
  ## Quantization-aware linear: x @ transpose(weight).
  ## Uses quantizedMatmul when weight is quantized, regular matmul otherwise.
  if model.isQuantized(weightKey):
    quantizedMatmul(x, model.w(weightKey),
                    model.w(weightKey & ".scales"),
                    model.w(weightKey & ".biases"),
                    transpose = true,
                    groupSize = model.quantCfg.groupSize,
                    bits = model.quantCfg.bits)
  else:
    x @ transpose(model.w(weightKey))

proc deqW*(model: KokoroModel, key: string): Tensor =
  ## Get weight, dequantizing if needed (for conv1d etc that need full tensors).
  if model.isQuantized(key):
    dequantize(model.w(key), model.w(key & ".scales"), model.w(key & ".biases"),
               groupSize = model.quantCfg.groupSize, bits = model.quantCfg.bits)
  else:
    model.w(key)

# ── Utility functions ────────────────────────────────────────────

proc getPadding(kernelSize: int, dilation: int = 1): int =
  (kernelSize * dilation - dilation) div 2

proc leakyRelu(x: Tensor, negSlope: float32 = 0.2): Tensor =
  ## LeakyReLU: max(0,x) + negSlope * min(0,x)
  let zero = scalar(0.0'f32)
  let alpha = scalar(negSlope)
  maximum(x, zero) + alpha * minimum(x, zero)

proc instanceNorm(x: Tensor, eps: float32 = 1e-5): Tensor =
  ## Instance norm over last dim (channel dim in NCL after transpose).
  ## x shape: (B, C, L) — normalize over L.
  let m = mean(x, axis = -1, keepdims = true)
  let v = variance(x, axis = -1, keepdims = true)
  (x - m) / sqrt(v + scalar(eps))

proc layerNorm(x: Tensor, weight, bias: Tensor, eps: float32 = 1e-12): Tensor =
  ## Layer normalization over last dim.
  let m = mean(x, axis = -1, keepdims = true)
  let v = variance(x, axis = -1, keepdims = true)
  let normalized = (x - m) / sqrt(v + scalar(eps))
  normalized * weight + bias

# ── Weight-normalized Conv1d ─────────────────────────────────────
# At load time, we fuse weight_g and weight_v into a single effective weight
# in MLX layout, stored as "prefix.weight_fused". This eliminates the norm
# computation on every forward call.

proc fuseWnConv(weightG, weightV: Tensor): Tensor =
  ## Precompute fused weight: g * v / ||v|| in PyTorch layout (out, in, kernel).
  let outCh = weightV.dim(0)
  let vFlat = weightV.reshape([outCh, -1])
  let norms = norm(vFlat, ord = 2.0, axis = 1, keepdims = true).expandDims(2)
  weightG * weightV / norms

proc precomputeWeights*(model: var KokoroModel) =
  ## Fuse all weight_g/weight_v pairs into effective weights and
  ## transpose to MLX layout. Call once after loading.
  var fused: seq[tuple[key: string, val: Tensor]]
  for key, val in model.weights:
    if key.endsWith(".weight_v"):
      let prefix = key[0..^10]  # strip ".weight_v"
      let gKey = prefix & ".weight_g"
      if gKey in model.weights:
        let w = fuseWnConv(model.weights[gKey], val)
        fused.add((prefix & ".weight_fused", w))
  for (k, v) in fused:
    model.weights[k] = v

proc wnConv1d(x: Tensor, model: KokoroModel, prefix: string,
              stride: int = 1, padding: int = 0, dilation: int = 1,
              groups: int = 1): Tensor =
  ## Weight-normalized 1D convolution using precomputed fused weight. NCL in/out.
  let fusedKey = prefix & "weight_fused"
  let wNormed = if model.hasW(fusedKey): model.w(fusedKey)
                else: fuseWnConv(model.w(prefix & "weight_g"), model.w(prefix & "weight_v"))
  let wMlx = wNormed.transpose([0, 2, 1])  # PyTorch → MLX layout
  var res = conv1d(x.transpose([0, 2, 1]), wMlx, stride, padding, dilation, groups)
  let biasKey = prefix & "bias"
  if model.hasW(biasKey):
    res = res + model.w(biasKey)
  res.transpose([0, 2, 1])

proc wnConvTranspose1d(x: Tensor, model: KokoroModel, prefix: string,
                        stride: int = 1, padding: int = 0,
                        outputPadding: int = 0, groups: int = 1): Tensor =
  ## Weight-normalized transposed 1D convolution. NCL in/out.
  let fusedKey = prefix & "weight_fused"
  let wNormed = if model.hasW(fusedKey): model.w(fusedKey)
                else: fuseWnConv(model.w(prefix & "weight_g"), model.w(prefix & "weight_v"))
  var wMlx: Tensor
  if groups > 1:
    wMlx = wNormed.transpose([0, 2, 1])
  else:
    wMlx = wNormed.transpose([1, 2, 0])
  var res = convTranspose1d(x.transpose([0, 2, 1]), wMlx, stride, padding,
                             outputPadding = outputPadding, groups = groups)
  let biasKey = prefix & "bias"
  if model.hasW(biasKey):
    res = res + model.w(biasKey)
  res.transpose([0, 2, 1])

# ── LSTM ─────────────────────────────────────────────────────────

proc lstmCell(x: Tensor, hPrev, cPrev: Tensor,
              wx, wh, bias: Tensor): tuple[h, c: Tensor] =
  ## Single LSTM step.
  ## wx: (4H, input_size), wh: (4H, hidden_size), bias: (4H,)
  let gates = (x @ transpose(wx)) + (hPrev @ transpose(wh)) + bias
  let parts = split(gates, 4, axis = -1)
  let i = sigmoid(parts[0])
  let f = sigmoid(parts[1])
  let g = tanh(parts[2])
  let o = sigmoid(parts[3])
  let newC = f * cPrev + i * g
  let newH = o * tanh(newC)
  (newH, newC)

proc lstmForward(x: Tensor, wx, wh, bias: Tensor): Tensor =
  ## Forward LSTM over sequence. x: (B, T, input_size). Returns (B, T, hidden_size).
  let batchSize = x.dim(0)
  let seqLen = x.dim(1)
  let hiddenSize = wh.dim(0) div 4

  var h = zeros([batchSize, hiddenSize])
  var c = zeros([batchSize, hiddenSize])
  var outputs = newSeq[Tensor](seqLen)

  # Pre-compute input projection
  let xProj = (x @ transpose(wx)) + bias  # (B, T, 4H)

  for t in 0..<seqLen:
    let xt = xProj.slice([0, t, 0], [batchSize, t + 1, xProj.dim(2)]).squeeze(1)
    let gates = xt + (h @ transpose(wh))
    let parts = split(gates, 4, axis = -1)
    let ig = sigmoid(parts[0])
    let fg = sigmoid(parts[1])
    let gg = tanh(parts[2])
    let og = sigmoid(parts[3])
    c = fg * c + ig * gg
    h = og * tanh(c)
    outputs[t] = h.expandDims(1)

  concatenate(outputs, axis = 1)

proc bilstmForward(x: Tensor, fwdWx, fwdWh, fwdBias: Tensor,
                    bwdWx, bwdWh, bwdBias: Tensor): Tensor =
  ## Bidirectional LSTM. x: (B, T, input). Returns (B, T, 2*hidden).
  let fwdOut = lstmForward(x, fwdWx, fwdWh, fwdBias)

  # Backward: reverse sequence, run forward LSTM, reverse output
  let batchSize = x.dim(0)
  let seqLen = x.dim(1)
  let hiddenSize = bwdWh.dim(0) div 4

  var h = zeros([batchSize, hiddenSize])
  var c = zeros([batchSize, hiddenSize])
  var bwdOutputs = newSeq[Tensor](seqLen)

  let xProj = (x @ transpose(bwdWx)) + bwdBias

  for t in countdown(seqLen - 1, 0):
    let xt = xProj.slice([0, t, 0], [batchSize, t + 1, xProj.dim(2)]).squeeze(1)
    let gates = xt + (h @ transpose(bwdWh))
    let parts = split(gates, 4, axis = -1)
    let ig = sigmoid(parts[0])
    let fg = sigmoid(parts[1])
    let gg = tanh(parts[2])
    let og = sigmoid(parts[3])
    c = fg * c + ig * gg
    h = og * tanh(c)
    bwdOutputs[t] = h.expandDims(1)

  let bwdOut = concatenate(bwdOutputs, axis = 1)
  concatenate([fwdOut, bwdOut], axis = -1)

# ── ALBERT Forward ───────────────────────────────────────────────

proc albertForward(model: KokoroModel, inputIds: Tensor, textMask: Tensor): Tensor =
  ## ALBERT text encoder.
  ## inputIds: (B, T) int32, textMask: (B, T) float32
  ## Returns: (B, T, hidden_size)
  let cfg = model.config

  # Embeddings
  let wordEmb = take(model.deqW("bert.embeddings.word_embeddings.weight"), inputIds, axis = 0)
  let posIds = arange(inputIds.dim(1), MLX_INT32).expandDims(0)
  let posEmb = take(model.deqW("bert.embeddings.position_embeddings.weight"), posIds, axis = 0)
  let tokTypeEmb = take(model.deqW("bert.embeddings.token_type_embeddings.weight"),
                         zeros([1, inputIds.dim(1)], MLX_INT32), axis = 0)

  var x = wordEmb + posEmb + tokTypeEmb
  x = layerNorm(x,
    model.w("bert.embeddings.LayerNorm.weight"),
    model.w("bert.embeddings.LayerNorm.bias"))

  # Embedding projection: 128 → 768
  x = model.linear(x, "bert.encoder.embedding_hidden_mapping_in.weight") +
      model.w("bert.encoder.embedding_hidden_mapping_in.bias")

  # Shared ALBERT layer repeated nRecurrence times
  let headSize = cfg.hiddenSize div cfg.nAttnHeads
  let scale = 1.0 / sqrt(float32(headSize))

  for r in 0..<cfg.nRecurrence:
    let residual = x

    # Self-attention
    let prefix = "bert.encoder.albert_layer_groups.0.albert_layers.0."
    var q = model.linear(x, prefix & "attention.query.weight") +
            model.w(prefix & "attention.query.bias")
    var k = model.linear(x, prefix & "attention.key.weight") +
            model.w(prefix & "attention.key.bias")
    var v = model.linear(x, prefix & "attention.value.weight") +
            model.w(prefix & "attention.value.bias")

    let B = x.dim(0)
    let T = x.dim(1)
    q = q.reshape([B, T, cfg.nAttnHeads, headSize]).transpose([0, 2, 1, 3])
    k = k.reshape([B, T, cfg.nAttnHeads, headSize]).transpose([0, 2, 1, 3])
    v = v.reshape([B, T, cfg.nAttnHeads, headSize]).transpose([0, 2, 1, 3])

    var attnWeights = (q @ k.transpose([0, 1, 3, 2])) * scalar(scale)
    # Apply attention mask (0 = attend, -inf = block)
    # textMask is already 0/1 or 0/-inf
    attnWeights = softmax(attnWeights, axis = -1)
    var attnOut = attnWeights @ v
    attnOut = attnOut.transpose([0, 2, 1, 3]).reshape([B, T, cfg.hiddenSize])

    attnOut = model.linear(attnOut, prefix & "attention.dense.weight") +
              model.w(prefix & "attention.dense.bias")

    x = layerNorm(attnOut + residual,
      model.w(prefix & "attention.LayerNorm.weight"),
      model.w(prefix & "attention.LayerNorm.bias"))

    # FFN
    let residualFfn = x
    var ffnOut = model.linear(x, prefix & "ffn.weight") + model.w(prefix & "ffn.bias")
    # GELU activation
    ffnOut = ffnOut * scalar(0.5'f32) * (scalar(1.0'f32) + erf(ffnOut * scalar(0.7071067811865476'f32)))
    ffnOut = model.linear(ffnOut, prefix & "ffn_output.weight") + model.w(prefix & "ffn_output.bias")

    x = layerNorm(ffnOut + residualFfn,
      model.w(prefix & "full_layer_layer_norm.weight"),
      model.w(prefix & "full_layer_layer_norm.bias"))

  return x

# ── Text Encoder ─────────────────────────────────────────────────

proc textEncoderForward(model: KokoroModel, inputIds: Tensor,
                         inputLengths: Tensor, textMask: Tensor): Tensor =
  ## CNN + BiLSTM text encoder. Returns (B, d_hidden, T) in NCL format.
  let prefix = "text_encoder."

  # Embedding lookup
  var x = take(model.deqW(prefix & "embedding.weight"), inputIds, axis = 0)
  # x: (B, T, d_hidden) — transpose to NCL for conv
  x = x.transpose([0, 2, 1])  # (B, d_hidden, T)

  # Conv layers with weight normalization + LayerNorm + LeakyReLU
  for i in 0..<model.config.nConvLayers:
    let cPrefix = prefix & "cnn." & $i & "."
    x = wnConv1d(x, model, cPrefix & "0.", padding = 2)  # kernel=5, padding=2

    # LayerNorm over channel dim
    var xT = x.transpose([0, 2, 1])  # (B, T, C)
    xT = layerNorm(xT,
      model.w(cPrefix & "1.gamma"),
      model.w(cPrefix & "1.beta"), eps = 1e-5)
    x = leakyRelu(xT.transpose([0, 2, 1]), 0.2)

  # BiLSTM
  x = x.transpose([0, 2, 1])  # NCL → NLC (B, T, C)
  x = bilstmForward(x,
    model.deqW(prefix & "lstm.weight_ih_l0"),
    model.deqW(prefix & "lstm.weight_hh_l0"),
    model.w(prefix & "lstm.bias_ih_l0") + model.w(prefix & "lstm.bias_hh_l0"),
    model.deqW(prefix & "lstm.weight_ih_l0_reverse"),
    model.deqW(prefix & "lstm.weight_hh_l0_reverse"),
    model.w(prefix & "lstm.bias_ih_l0_reverse") + model.w(prefix & "lstm.bias_hh_l0_reverse"))

  x.transpose([0, 2, 1])  # NLC → NCL (B, d_hidden, T)

# ── Duration / Prosody Predictor ─────────────────────────────────

proc adaLayerNorm(x, style: Tensor, gammaW, gammaBias, betaW, betaBias: Tensor): Tensor =
  ## Adaptive LayerNorm: (1 + gamma) * norm(x) + beta
  ## x: (B, T, C), style: (B, style_dim)
  let gamma = (style @ transpose(gammaW)) + gammaBias  # (B, C)
  let beta = (style @ transpose(betaW)) + betaBias
  let m = mean(x, axis = -1, keepdims = true)
  let v = variance(x, axis = -1, keepdims = true)
  let xNorm = (x - m) / sqrt(v + scalar(1e-5'f32))
  (scalar(1.0'f32) + gamma.expandDims(1)) * xNorm + beta.expandDims(1)

proc adainResBlk1d(x, style: Tensor, model: KokoroModel, prefix: string,
                    upsample: bool = false): Tensor =
  ## AdaIN residual block for prosody/decoder. NCL format.
  let sqrtTwo = scalar(sqrt(2.0'f32))
  var cur = x

  # AdaIN norm 1: InstanceNorm over time (axis=-1 of NCL)
  let gamma1W = model.deqW(prefix & "norm1.fc.weight")
  let gamma1B = model.w(prefix & "norm1.fc.bias")
  let m1 = mean(cur, axis = -1, keepdims = true)   # mean over time L
  let v1 = variance(cur, axis = -1, keepdims = true)
  var normed = (cur - m1) / sqrt(v1 + scalar(1e-5'f32))  # (B, C, L)
  # FC produces 2*channels: split into gamma and beta
  let h1 = (style @ transpose(gamma1W)) + gamma1B  # (B, 2*C)
  let halfC = h1.dim(-1) div 2
  let gamma1 = h1.slice([0, 0], [h1.dim(0), halfC]).expandDims(-1)  # (B, C, 1)
  let beta1 = h1.slice([0, halfC], [h1.dim(0), h1.dim(-1)]).expandDims(-1)
  cur = (scalar(1.0'f32) + gamma1) * normed + beta1  # NCL

  cur = leakyRelu(cur, 0.2)

  if upsample and model.hasW(prefix & "pool.weight_v"):
    let poolGroups = cur.dim(1)  # depthwise: groups = channels
    cur = wnConvTranspose1d(cur, model, prefix & "pool.",
      stride = 2, padding = 1, outputPadding = 1, groups = poolGroups)

  cur = wnConv1d(cur, model, prefix & "conv1.", padding = 1)

  # AdaIN norm 2: InstanceNorm over time (axis=-1 of NCL)
  let gamma2W = model.deqW(prefix & "norm2.fc.weight")
  let gamma2B = model.w(prefix & "norm2.fc.bias")
  let m2 = mean(cur, axis = -1, keepdims = true)
  let v2 = variance(cur, axis = -1, keepdims = true)
  normed = (cur - m2) / sqrt(v2 + scalar(1e-5'f32))
  let h2 = (style @ transpose(gamma2W)) + gamma2B
  let halfC2 = h2.dim(-1) div 2
  let gamma2 = h2.slice([0, 0], [h2.dim(0), halfC2]).expandDims(-1)
  let beta2 = h2.slice([0, halfC2], [h2.dim(0), h2.dim(-1)]).expandDims(-1)
  cur = (scalar(1.0'f32) + gamma2) * normed + beta2

  cur = leakyRelu(cur, 0.2)
  cur = wnConv1d(cur, model, prefix & "conv2.", padding = 1)

  # Shortcut: upsample FIRST, then conv1x1 (matches Python _shortcut order)
  var res = x
  if upsample:
    res = res.repeat(2, axis = 2)  # nearest-neighbor 2x upsample on time axis

  if model.hasW(prefix & "conv1x1.weight_v"):
    res = wnConv1d(res, model, prefix & "conv1x1.")

  # Align time dims if needed
  let curL = cur.dim(2)
  let resL = res.dim(2)
  if resL > curL:
    res = res.slice([0, 0, 0], [res.dim(0), res.dim(1), curL])
  elif curL > resL:
    cur = cur.slice([0, 0, 0], [cur.dim(0), cur.dim(1), resL])

  (cur + res) / sqrtTwo

proc durationEncoderForward(model: KokoroModel, dEn, style: Tensor,
                             inputLengths: Tensor, textMask: Tensor): Tensor =
  ## Duration encoder: BiLSTM + AdaLayerNorm layers.
  ## dEn: (B, d_hidden, T) NCL, style: (B, style_dim/2)
  ## Returns: (B, T, d_hidden+style_dim/2) NLC
  let prefix = "predictor.text_encoder."
  let T = dEn.dim(2)

  # Concat style to d_en
  let styleExpanded = style.expandDims(1).repeat(T, axis = 1)  # (B, T, style_dim/2)
  var x = concatenate([dEn.transpose([0, 2, 1]), styleExpanded], axis = -1)  # (B, T, d_hidden+style)

  for i in 0..<model.config.nDurationLayers:
    # Even indices in lstms = BiLSTM, odd indices = AdaLayerNorm
    let lstmIdx = i * 2
    let normIdx = i * 2 + 1
    let lPrefix = prefix & "lstms." & $lstmIdx & "."
    x = bilstmForward(x,
      model.deqW(lPrefix & "weight_ih_l0"),
      model.deqW(lPrefix & "weight_hh_l0"),
      model.w(lPrefix & "bias_ih_l0") + model.w(lPrefix & "bias_hh_l0"),
      model.deqW(lPrefix & "weight_ih_l0_reverse"),
      model.deqW(lPrefix & "weight_hh_l0_reverse"),
      model.w(lPrefix & "bias_ih_l0_reverse") + model.w(lPrefix & "bias_hh_l0_reverse"))

    let nPrefix = prefix & "lstms." & $normIdx & "."
    if model.hasW(nPrefix & "fc.weight"):
      let fcW = model.deqW(nPrefix & "fc.weight")
      let fcB = model.w(nPrefix & "fc.bias")
      let h = (style @ transpose(fcW)) + fcB  # parens needed: @ has low precedence
      let halfC = h.dim(-1) div 2
      let gamma = h.slice([0, 0], [h.dim(0), halfC]).expandDims(1)
      let beta = h.slice([0, halfC], [h.dim(0), h.dim(-1)]).expandDims(1)
      let m = mean(x, axis = -1, keepdims = true)
      let v = variance(x, axis = -1, keepdims = true)
      x = (scalar(1.0'f32) + gamma) * ((x - m) / sqrt(v + scalar(1e-5'f32))) + beta

    # Re-concat style
    x = concatenate([x, styleExpanded], axis = -1)

  return x

proc prosodyForward(model: KokoroModel, dEn, style: Tensor,
                     inputLengths, textMask: Tensor,
                     speed: float32): tuple[predDur: Tensor, hiddenStates: Tensor] =
  ## Duration and hidden state prediction.
  ## Returns predicted durations (B, T) and hidden states for generation.
  let prefix = "predictor."

  let d = durationEncoderForward(model, dEn, style, inputLengths, textMask)

  # Duration projection: BiLSTM → linear → sigmoid → sum → round
  var x = bilstmForward(d,
    model.deqW(prefix & "lstm.weight_ih_l0"),
    model.deqW(prefix & "lstm.weight_hh_l0"),
    model.w(prefix & "lstm.bias_ih_l0") + model.w(prefix & "lstm.bias_hh_l0"),
    model.deqW(prefix & "lstm.weight_ih_l0_reverse"),
    model.deqW(prefix & "lstm.weight_hh_l0_reverse"),
    model.w(prefix & "lstm.bias_ih_l0_reverse") + model.w(prefix & "lstm.bias_hh_l0_reverse"))

  # duration_proj: Linear(d_hidden → max_dur)
  var dur = model.linear(x, prefix & "duration_proj.linear_layer.weight") +
            model.w(prefix & "duration_proj.linear_layer.bias")
  dur = sigmoid(dur)
  dur = sum(dur, axis = -1) / scalar(speed)  # (B, T)
  var predDur = clip(round(dur), scalar(0.0'f32), scalar(50.0'f32)).astype(MLX_INT32)
  predDur = maximum(predDur, scalar(1'i32))

  (predDur, d)

proc f0NForward(model: KokoroModel, en, style: Tensor): tuple[f0, n: Tensor] =
  ## F0 and energy prediction from expanded features.
  ## en: (B, d_hidden+style, T') NCL, style: (B, style_dim/2)
  let prefix = "predictor."

  # Shared BiLSTM
  var x = bilstmForward(en.transpose([0, 2, 1]),
    model.deqW(prefix & "shared.weight_ih_l0"),
    model.deqW(prefix & "shared.weight_hh_l0"),
    model.w(prefix & "shared.bias_ih_l0") + model.w(prefix & "shared.bias_hh_l0"),
    model.deqW(prefix & "shared.weight_ih_l0_reverse"),
    model.deqW(prefix & "shared.weight_hh_l0_reverse"),
    model.w(prefix & "shared.bias_ih_l0_reverse") + model.w(prefix & "shared.bias_hh_l0_reverse"))
  x = x.transpose([0, 2, 1])  # → NCL

  # F0 path: 3 AdainResBlk1d blocks (middle one has upsample)
  var f0 = x
  for i in 0..<model.config.f0NBlocks:
    let blkPrefix = prefix & "F0." & $i & "."
    f0 = adainResBlk1d(f0, style, model, blkPrefix, upsample = (i == 1))

  # F0 projection: Conv1d → (B, 1, T')
  let f0W = model.deqW(prefix & "F0_proj.weight")
  let f0B = model.w(prefix & "F0_proj.bias")
  # Plain conv1d (not weight-normed): weight is (out, in, kernel) in PyTorch
  let f0WMlx = f0W.transpose([0, 2, 1])  # → (out, kernel, in)
  f0 = conv1d(f0.transpose([0, 2, 1]), f0WMlx).transpose([0, 2, 1]) + f0B.expandDims(0).expandDims(-1)

  # N path: same structure
  var n = x
  for i in 0..<model.config.f0NBlocks:
    let blkPrefix = prefix & "N." & $i & "."
    n = adainResBlk1d(n, style, model, blkPrefix, upsample = (i == 1))

  let nW = model.deqW(prefix & "N_proj.weight")
  let nB = model.w(prefix & "N_proj.bias")
  let nWMlx = nW.transpose([0, 2, 1])
  n = conv1d(n.transpose([0, 2, 1]), nWMlx).transpose([0, 2, 1]) + nB.expandDims(0).expandDims(-1)

  (f0.squeeze(1), n.squeeze(1))

# ── Generator (iSTFT Vocoder) ────────────────────────────────────

proc interpolate1d(x: Tensor, outLen: int): Tensor =
  ## 1-D linear interpolation along axis=1 (align_corners=False).
  ## x: (B, L_in, D) → (B, outLen, D)
  let lIn = x.dim(1)
  if lIn == outLen: return x
  let idx = (arange(float64(outLen), MLX_FLOAT32) + scalar(0.5'f32)) *
            scalar(float32(lIn) / float32(outLen)) - scalar(0.5'f32)
  let idxClamped = clip(idx, scalar(0.0'f32), scalar(float32(lIn - 1)))
  let lo = floor(idxClamped).astype(MLX_INT32)
  let hi = minimum(lo + scalar(1'i32), scalar(int32(lIn - 1)))
  let frac = (idxClamped - floor(idxClamped)).expandDims(-1)  # (outLen, 1)
  # Gather: x[:, lo, :] and x[:, hi, :]
  # Use take along axis=1
  let xLo = x.take(lo, axis = 1)
  let xHi = x.take(hi, axis = 1)
  xLo * (scalar(1.0'f32) - frac) + xHi * frac

proc sineGen(f0: Tensor, harmonicNum: int, samplingRate: int,
             upsampleScale: int, sineAmp: float32 = 0.1,
             noiseStd: float32 = 0.003, voicedThreshold: float32 = 0.0): tuple[sineWaves, uv, noise: Tensor] =
  ## SineGen: f0 (B, L, 1) → sine_waves (B, L, dim), uv (B, L, 1), noise (B, L, dim)
  let dim = harmonicNum + 1
  let B = f0.dim(0)
  let L = f0.dim(1)

  # Expand to harmonics: f0 * [1, 2, ..., dim]
  let harmonics = arange(1.0, float64(dim + 1), 1.0, MLX_FLOAT32).reshape([1, 1, dim])
  let fn = f0 * harmonics  # (B, L, dim)

  # Fractional cycles per sample
  let radValues = remainder(fn / scalar(float32(samplingRate)), scalar(1.0'f32))

  # Random initial phase for harmonics (fundamental always 0)
  let randIni = randomUniform(scalar(0.0'f32), scalar(1.0'f32), [B, dim])
  # Zero out the first harmonic's random phase
  let zeroCol = zeros([B, 1])
  let randRest = randIni.slice([0, 1], [B, dim])
  let randIniFixed = concatenate([zeroCol, randRest], axis = 1)  # (B, dim)

  # Add initial phase to first sample
  # radValues: (B, L, dim), randIniFixed: (B, dim) → (B, 1, dim)
  var radVal = radValues
  let firstSample = radVal.slice([0, 0, 0], [B, 1, dim]) + randIniFixed.expandDims(1)
  # Replace first sample — build by concatenating
  let restSamples = radVal.slice([0, 1, 0], [B, L, dim])
  radVal = concatenate([firstSample, restSamples], axis = 1)

  # Downsample → cumsum → scale → upsample (matches Python)
  let scale = upsampleScale
  let lDown = L div scale
  let radDown = interpolate1d(radVal, lDown)  # (B, lDown, dim)
  var phase = cumsum(radDown, axis = 1) * scalar(2.0'f32 * PI)
  phase = phase * scalar(float32(scale))
  phase = interpolate1d(phase, L)  # (B, L, dim)

  let sinWaves = sin(phase) * scalar(sineAmp)

  # Voiced/unvoiced
  let uv = (f0 > scalar(voicedThreshold)).astype(MLX_FLOAT32)  # (B, L, 1)
  let noiseAmp = uv * scalar(noiseStd) + (scalar(1.0'f32) - uv) * scalar(sineAmp / 3.0'f32)
  let noise = noiseAmp * randomNormal([B, L, dim])
  let result = sinWaves * uv + noise
  (result, uv, noise)

proc sourceModuleForward(model: KokoroModel, f0Up: Tensor, prefix: string,
                          harmonicNum: int, upsampleScale: int): tuple[sineMerge, noise, uv: Tensor] =
  ## SourceModuleHnNSF: f0Up (B, L, 1) → sineMerge (B, L, 1), noise (B, L, 1), uv (B, L, 1)
  let cfg = model.config
  let (sineWavs, uv, _) = sineGen(f0Up, harmonicNum, int(cfg.samplingRate),
                                    upsampleScale, sineAmp = 0.1,
                                    noiseStd = 0.003, voicedThreshold = 10.0)
  # l_linear: plain Linear (not weight-normed), maps (B, L, dim) → (B, L, 1)
  let linW = model.deqW(prefix & "l_linear.weight")  # (1, dim) for Linear
  let linB = model.w(prefix & "l_linear.bias")     # (1,)
  let sineMerge = tanh((sineWavs @ transpose(linW)) + linB)  # (B, L, 1)
  let noise = randomNormal([uv.dim(0), uv.dim(1), uv.dim(2)]) * scalar(0.1'f32 / 3.0'f32)
  (sineMerge, noise, uv)

proc adainResBlock1(x, style: Tensor, model: KokoroModel, prefix: string,
                     dilations: seq[int], kernelSize: int): Tensor =
  ## Generator AdaIN residual block with Snake activation. NCL format.
  var inpl = x
  let nLayers = dilations.len

  for i in 0..<nLayers:
    let pad = getPadding(kernelSize, dilations[i])
    let iStr = $i

    # AdaIN 1
    let adain1W = model.deqW(prefix & "adain1." & iStr & ".fc.weight")
    let adain1B = model.w(prefix & "adain1." & iStr & ".fc.bias")
    var cur = instanceNorm(inpl)
    let h1 = (style @ transpose(adain1W)) + adain1B
    let halfC = h1.dim(-1) div 2
    let g1 = h1.slice([0, 0], [h1.dim(0), halfC]).expandDims(-1)
    let b1 = h1.slice([0, halfC], [h1.dim(0), h1.dim(-1)]).expandDims(-1)
    cur = (scalar(1.0'f32) + g1) * cur + b1

    # Snake activation
    let alpha1 = model.w(prefix & "alpha1." & iStr)
    cur = snake(cur, alpha1)  # alpha is (1, C, 1) from safetensors

    # Conv1
    cur = wnConv1d(cur, model, prefix & "convs1." & iStr & ".",
      padding = pad, dilation = dilations[i])

    # AdaIN 2
    let adain2W = model.deqW(prefix & "adain2." & iStr & ".fc.weight")
    let adain2B = model.w(prefix & "adain2." & iStr & ".fc.bias")
    cur = instanceNorm(cur)
    let h2 = (style @ transpose(adain2W)) + adain2B
    let halfC2 = h2.dim(-1) div 2
    let g2 = h2.slice([0, 0], [h2.dim(0), halfC2]).expandDims(-1)
    let b2 = h2.slice([0, halfC2], [h2.dim(0), h2.dim(-1)]).expandDims(-1)
    cur = (scalar(1.0'f32) + g2) * cur + b2

    # Snake activation
    let alpha2 = model.w(prefix & "alpha2." & iStr)
    cur = snake(cur, alpha2)  # alpha is (1, C, 1) from safetensors

    # Conv2
    cur = wnConv1d(cur, model, prefix & "convs2." & iStr & ".",
      padding = getPadding(kernelSize, 1))

    inpl = inpl + cur
  inpl

proc buildStftBasis(nFft, winLength: int): tuple[forwardBasis, cosMat, sinMat, window: Tensor] =
  ## Precompute DFT basis matrices for conv-based STFT/iSTFT.
  ## Returns forward_basis (2*nBins, nFft, 1), cos_mat_T (1, nFft, nBins),
  ## sin_mat_T (1, nFft, nBins), window (nFft,).
  let nBins = nFft div 2 + 1

  # Build Hann window (periodic: hanning(N+1)[:-1])
  var winData = newSeq[float32](winLength)
  for i in 0..<winLength:
    winData[i] = 0.5'f32 * (1.0'f32 - cos(2.0'f32 * PI * float32(i) / float32(winLength)))
  let win = fromSeq(winData, [winLength])

  # Build DFT matrix
  var fourierData = newSeq[float32](2 * nBins * nFft)
  for k in 0..<nBins:
    for n in 0..<nFft:
      let angle = -2.0'f32 * PI * float32(k) * float32(n) / float32(nFft)
      fourierData[k * nFft + n] = cos(angle) * winData[n]              # real part
      fourierData[(nBins + k) * nFft + n] = sin(angle) * winData[n]    # imag part
  var fwdBasis = fromSeq(fourierData, [2 * nBins, nFft]).expandDims(-1)  # (2*nBins, nFft, 1)

  # iDFT basis with conjugate scaling
  var cosData = newSeq[float32](nFft * nBins)
  var sinData = newSeq[float32](nFft * nBins)
  for n in 0..<nFft:
    for k in 0..<nBins:
      let angle = 2.0'f32 * PI * float32(k) * float32(n) / float32(nFft)
      var conjScale = 1.0'f32
      if k > 0 and k < nBins - 1: conjScale = 2.0'f32
      cosData[n * nBins + k] = cos(angle) * conjScale / float32(nFft)
      sinData[n * nBins + k] = sin(angle) * conjScale / float32(nFft)
  let cosMat = fromSeq(cosData, [1, nFft, nBins])
  let sinMat = fromSeq(sinData, [1, nFft, nBins])

  (fwdBasis, cosMat, sinMat, win)

proc stftTransform(x: Tensor, forwardBasis: Tensor, nFft, hop: int): tuple[mag, phase: Tensor] =
  ## Conv-based STFT. x: (B, T_audio) → magnitude (B, nBins, frames), phase (B, nBins, frames)
  let padAmt = nFft div 2
  let xPadded = pad(x, axes = [1], lowPad = [padAmt], highPad = [padAmt])
  let xNlc = xPadded.expandDims(-1)  # (B, T_padded, 1)
  # Conv1d: NLC input, (out, kernel, in) weight
  var output = conv1d(xNlc, forwardBasis, stride = hop)
  output = output.transpose([0, 2, 1])  # (B, 2*nBins, frames)

  let nBins = nFft div 2 + 1
  let realPart = output.slice([0, 0, 0], [output.dim(0), nBins, output.dim(2)])
  let imagPart = output.slice([0, nBins, 0], [output.dim(0), 2 * nBins, output.dim(2)])
  let mag = sqrt(realPart * realPart + imagPart * imagPart)
  let phase = arctan2(imagPart, realPart)
  (mag, phase)

proc istftInverse(mag, phase: Tensor, cosMat, sinMat, window: Tensor,
                   nFft, hop: int): Tensor =
  ## Reconstruct waveform from magnitude and phase via precomputed iDFT basis.
  ## mag, phase: (B, nBins, frames). Returns (B, 1, samples).
  let realPart = mag * cos(phase)  # (B, nBins, frames)
  let imagPart = mag * sin(phase)

  let B = realPart.dim(0)
  let numFrames = realPart.dim(2)

  # iDFT: (1, nFft, nBins) @ (B, nBins, frames) = (B, nFft, frames)
  let frames = (cosMat @ realPart) - (sinMat @ imagPart)

  # Apply synthesis window: (B, nFft, frames) * window (nFft,) → broadcast via (nFft, 1)
  let windowedFrames = frames * window.expandDims(-1)

  # Vectorized overlap-add via scatter-add
  let totalLen = nFft + (numFrames - 1) * hop

  # Index map: frame f, sample k → output position f*hop + k
  let frameOffsets = arange(numFrames) * scalar(hop)    # (numFrames,) int32
  let sampleOffsets = arange(nFft)                       # (nFft,) int32
  # (numFrames, 1) + (1, nFft) → (numFrames, nFft) → flatten
  let idx = (frameOffsets.expandDims(-1) + sampleOffsets.expandDims(0)).reshape([-1])

  # Flatten frames: (B, nFft, numFrames) → transpose → (B, numFrames, nFft) → (B, numFrames*nFft)
  let framesFlat = windowedFrames.transpose([0, 2, 1]).reshape([B, -1])

  var output = zeros([B, totalLen])
  output = scatterAdd(output, idx.expandDims(0).broadcastTo([B, idx.dim(0)]), framesFlat, axis = 1)

  # Window-squared normalization
  let winSq = window * window  # (nFft,)
  let winSqBroad = winSq.expandDims(0).broadcastTo([numFrames, nFft]).reshape([-1])
  var winSqSum = zeros([totalLen])
  winSqSum = scatterAdd(winSqSum.expandDims(0), idx.expandDims(0), winSqBroad.expandDims(0), axis = 1).squeeze(0)
  output = output / maximum(winSqSum.expandDims(0), scalar(1e-8'f32))

  # Remove padding
  let padAmt = nFft div 2
  if padAmt > 0:
    result = output.slice([0, padAmt], [B, totalLen - padAmt]).expandDims(1)
  else:
    result = output.expandDims(1)

proc generatorForward(model: KokoroModel, x, style, f0Pred: Tensor): Tensor =
  ## iSTFTNet generator: source module → upsample loop → conv_post → iSTFT.
  ## x: (B, channels, T'), style: (B, style_dim/2), f0Pred: (B, T')
  let cfg = model.config
  let prefix = "decoder.generator."

  # Compute upsample_scale = prod(upsample_rates) * gen_istft_hop
  var upsampleScale = cfg.genIstftHop
  for r in cfg.upsampleRates: upsampleScale *= r

  # ── Source module ──
  # Upsample F0: (B, T') → repeat → (B, T'*scale, 1)
  let f0Up = f0Pred.expandDims(1).repeat(upsampleScale, axis = -1)  # (B, 1, T'*scale)
  let f0UpBLT = f0Up.transpose([0, 2, 1])  # (B, T'*scale, 1)

  let (harSource, _, _) = sourceModuleForward(model, f0UpBLT, prefix & "m_source.",
                                               cfg.harmonicNum, upsampleScale)
  # harSource: (B, T'*scale, 1) → squeeze → (B, T'*scale)
  let harFlat = harSource.squeeze(-1)

  # STFT of harmonic source → har (B, n_fft+2, frames)
  # Use cached basis or build on first call
  let (fwdBasis, cosMat, sinMat, window) =
    if model.stftBasis.valid:
      (model.stftBasis.forwardBasis, model.stftBasis.cosMat,
       model.stftBasis.sinMat, model.stftBasis.window)
    else:
      buildStftBasis(cfg.genIstftNFft, cfg.genIstftNFft)
  let (harSpec, harPhase) = stftTransform(harFlat, fwdBasis, cfg.genIstftNFft, cfg.genIstftHop)
  let har = concatenate([harSpec, harPhase], axis = 1)  # (B, n_fft+2, frames)

  # ── Upsample loop ──
  var cur = x
  for i in 0..<cfg.nUpsamples:
    cur = leakyRelu(cur, 0.1)

    # noise_conv: plain Conv1d on har (NCL → NLC for MLX conv1d)
    let ncPrefix = prefix & "noise_convs." & $i & "."
    let ncWeight = model.w(ncPrefix & "weight")  # (out, in_per_group, kernel) PyTorch layout
    # Conv weight: PyTorch (out, in, k) → MLX (out, k, in)
    let ncWeightMlx = ncWeight.transpose([0, 2, 1])
    # Compute stride: prod(upsample_rates[i+1:]) for non-last, 1 for last
    var noiseStride = 1
    if i + 1 < cfg.nUpsamples:
      noiseStride = 1
      for j in (i+1)..<cfg.nUpsamples:
        noiseStride *= cfg.upsampleRates[j]
    # Padding: (stride + 1) / 2 for non-last, 0 for last
    let noisePad = if i + 1 < cfg.nUpsamples: (noiseStride + 1) div 2 else: 0

    let harNlc = har.transpose([0, 2, 1])  # (B, frames, n_fft+2)
    var xSource = conv1d(harNlc, ncWeightMlx, stride = noiseStride, padding = noisePad)
    # Add bias (plain Conv1d has bias)
    if model.hasW(ncPrefix & "bias"):
      xSource = xSource + model.w(ncPrefix & "bias")
    xSource = xSource.transpose([0, 2, 1])  # back to NCL

    # noise_res block
    let nrPrefix = prefix & "noise_res." & $i & "."
    let noiseKernel = if i + 1 < cfg.nUpsamples: 7 else: 11
    xSource = adainResBlock1(xSource, style, model, nrPrefix, @[1, 3, 5], noiseKernel)

    # Transposed conv upsample
    cur = wnConvTranspose1d(cur, model, prefix & "ups." & $i & ".",
      stride = cfg.upsampleRates[i],
      padding = (cfg.upsampleKernelSizes[i] - cfg.upsampleRates[i]) div 2)

    # Reflection pad (1, 0) on last upsample
    if i == cfg.nUpsamples - 1:
      cur = pad(cur, axes = [2], lowPad = [1], highPad = [0])

    # Trim to match time dims
    if xSource.dim(-1) != cur.dim(-1):
      let minT = min(xSource.dim(-1), cur.dim(-1))
      cur = cur.slice([0, 0, 0], [cur.dim(0), cur.dim(1), minT])
      xSource = xSource.slice([0, 0, 0], [xSource.dim(0), xSource.dim(1), minT])

    cur = cur + xSource

    # Res blocks: average of nKernels blocks
    var resAccum: Tensor
    for ii in 0..<cfg.nKernels:
      let rbIdx = i * cfg.nKernels + ii
      let rbPrefix = prefix & "resblocks." & $rbIdx & "."
      let rb = adainResBlock1(cur, style, model, rbPrefix,
        cfg.resblockDilationSizes[ii], cfg.resblockKernelSizes[ii])
      if ii == 0:
        resAccum = rb
      else:
        resAccum = resAccum + rb
    cur = resAccum / scalar(float32(cfg.nKernels))

  # ── Output conv → spec/phase → iSTFT ──
  cur = leakyRelu(cur, 0.01)
  cur = wnConv1d(cur, model, prefix & "conv_post.", padding = 3).astype(MLX_FLOAT32)

  let postNBins = cfg.genIstftNFft div 2 + 1
  let spec = exp(cur.slice([0, 0, 0], [cur.dim(0), postNBins, cur.dim(2)]))
  let phase = sin(cur.slice([0, postNBins, 0], [cur.dim(0), cur.dim(1), cur.dim(2)]))

  istftInverse(spec, phase, cosMat, sinMat, window, cfg.genIstftNFft, cfg.genIstftHop)

# ── Decoder ──────────────────────────────────────────────────────

proc decoderForward(model: KokoroModel, asr, f0Pred, nPred, style: Tensor): Tensor =
  ## Full decoder: encode → decode blocks → generator.
  ## asr: (B, d_hidden, T'), f0Pred: (B, T'), nPred: (B, T'), style: (B, style_dim/2)
  let cfg = model.config
  let prefix = "decoder."

  # F0 and N conv (downsample by 2)
  let f0Conv = wnConv1d(f0Pred.expandDims(1), model, prefix & "F0_conv.",
    stride = 2, padding = 1)  # (B, channels, T'/2)

  let nConv = wnConv1d(nPred.expandDims(1), model, prefix & "N_conv.",
    stride = 2, padding = 1)

  # Concat asr + f0 + n
  var cur = concatenate([asr, f0Conv, nConv], axis = 1)  # (B, d_hidden+2, T'/2)

  # Encode block
  cur = adainResBlk1d(cur, style, model, prefix & "encode.", upsample = false)

  # ASR residual projection
  let asrRes = wnConv1d(asr, model, prefix & "asr_res.0.")

  # Decode blocks
  for i in 0..<cfg.nDecoderBlocks:
    let isLast = (i == cfg.nDecoderBlocks - 1)
    # Concat: [cur, asrRes, f0Conv, nConv]
    cur = concatenate([cur, asrRes, f0Conv, nConv], axis = 1)
    cur = adainResBlk1d(cur, style, model, prefix & "decode." & $i & ".",
                         upsample = isLast)

  # Generator (vocoder)
  generatorForward(model, cur, style, f0Pred)

# ── Full Forward Pass ────────────────────────────────────────────

proc forward*(model: KokoroModel, phonemes: string, voice: Tensor,
              speed: float32 = 1.0): Tensor =
  ## Full Kokoro forward pass: phonemes → audio samples.
  ## voice: (510, 1, 256) or (N, 256) voice embedding tensor.
  let cfg = model.config

  # Tokenize (iterate over Unicode codepoints, not bytes)
  var inputIdsList = newSeq[int32]()
  inputIdsList.add(0)  # BOS
  for r in phonemes.runes:
    let cs = $r
    if cs in model.vocab:
      inputIdsList.add(int32(model.vocab[cs]))
  inputIdsList.add(0)  # EOS

  let nTokens = inputIdsList.len
  var inputIds = fromSeq(inputIdsList, [1, nTokens])

  # Select voice style by token count
  let voiceIdx = min(nTokens - 1, voice.dim(0) - 1)
  let refS = voice.slice([voiceIdx, 0, 0], [voiceIdx + 1, 1, cfg.styleDim]).reshape([1, cfg.styleDim])
  let sProsody = refS.slice([0, cfg.styleDim div 2], [1, cfg.styleDim])  # (1, 128)
  let sDecoder = refS.slice([0, 0], [1, cfg.styleDim div 2])              # (1, 128)

  # Text mask (all ones for full attention)
  let textMask = ones([1, nTokens])
  let inputLengths = scalar(nTokens)

  when defined(profile):
    var t0 = cpuTime()
    template lap(name: string) =
      eval(cur)  # force sync for accurate timing
      let t1 = cpuTime()
      echo "  ", name, ": ", ((t1 - t0) * 1000).formatFloat(ffDecimal, 1), "ms"
      t0 = t1
    var cur: Tensor

  # ALBERT encoding
  let bertOut = albertForward(model, inputIds, textMask)  # (1, T, 768)
  let dEn = (model.linear(bertOut, "bert_encoder.weight") +
             model.w("bert_encoder.bias")).transpose([0, 2, 1])  # (1, 512, T)
  when defined(profile):
    cur = dEn; lap("ALBERT + bert_encoder")

  # Duration prediction
  let (predDur, hiddenStates) = prosodyForward(model, dEn, sProsody,
                                                 inputLengths, textMask, speed)
  eval(predDur)
  when defined(profile):
    cur = predDur; lap("prosody (dur encoder + LSTM + dur proj)")

  # Build alignment matrix from predicted durations
  let durData = predDur.squeeze(0).dataInt32  # (T,)
  var totalFrames = 0
  for i in 0..<nTokens:
    totalFrames += durData[i].int

  if totalFrames == 0:
    return zeros([0])

  # Create alignment matrix (T, T')
  var alnData = newSeq[float32](nTokens * totalFrames)
  var col = 0
  for i in 0..<nTokens:
    let d = durData[i].int
    for j in 0..<d:
      if col + j < totalFrames:
        alnData[i * totalFrames + col + j] = 1.0
    col += d
  var predAln = fromSeq(alnData, [1, nTokens, totalFrames])

  # Expand hidden states: (1, d_hidden+style, T) @ (1, T, T') → (1, d_hidden+style, T')
  let en = hiddenStates.transpose([0, 2, 1]) @ predAln  # NCL matmul

  # F0 and N prediction
  let (f0Pred, nPred) = f0NForward(model, en, sProsody)
  when defined(profile):
    cur = nPred; lap("F0/N prediction")

  # Text encoder
  let tEn = textEncoderForward(model, inputIds, inputLengths, textMask)  # (1, 512, T)
  let asr = tEn @ predAln  # (1, 512, T')
  when defined(profile):
    cur = asr; lap("text encoder + alignment")

  # Decoder + Generator
  let audio = decoderForward(model, asr, f0Pred, nPred, sDecoder)

  eval(audio)
  when defined(profile):
    cur = audio; lap("decoder + generator")

  audio

# ── Model Loading ────────────────────────────────────────────────

proc loadKokoroMlx*(modelDir: string, voice: string = "af_heart"): KokoroModel =
  ## Load a Kokoro MLX model from a directory containing:
  ##   - model.safetensors (or kokoro-v0.19.safetensors)
  ##   - config.json
  ##   - voices/*.safetensors
  initMlx()
  initDefaultStream()

  var model = KokoroModel(
    config: defaultConfig(),
    phmzr: newPhonemizer(voice),
  )

  # Load model weights
  let weightsPath = if fileExists(modelDir / "model.safetensors"):
    modelDir / "model.safetensors"
  elif fileExists(modelDir / "kokoro-v1_0.safetensors"):
    modelDir / "kokoro-v1_0.safetensors"
  else:
    # Try to find any safetensors file in the top-level directory
    var found = ""
    for f in walkDir(modelDir, relative = false):
      if f.kind in {pcFile, pcLinkToFile} and f.path.endsWith(".safetensors"):
        found = f.path
        break
    found

  if weightsPath.len > 0:
    echo "Loading weights: ", weightsPath
    model.weights = loadSafetensors(weightsPath)
    # Detect quantized model (has .scales companion tensors)
    let qcfgPath = modelDir / "quantize.json"
    if fileExists(qcfgPath):
      let qj = parseJson(readFile(qcfgPath))
      model.quantCfg = QuantConfig(
        bits: qj.getOrDefault("bits").getInt(4),
        groupSize: qj.getOrDefault("group_size").getInt(64),
        quantized: true,
      )
      var nQuant = 0
      for k in model.weights.keys:
        if k.endsWith(".scales"): inc nQuant
      echo "Loaded ", model.weights.len, " tensors (", nQuant, " quantized, ",
           model.quantCfg.bits, "-bit)"
    else:
      model.quantCfg = QuantConfig(bits: 4, groupSize: 64, quantized: false)
      echo "Loaded ", model.weights.len, " weight tensors"

  # Load voices
  let voicesDir = modelDir / "voices"
  if dirExists(voicesDir):
    for f in walkDir(voicesDir):
      if f.path.endsWith(".safetensors"):
        let vName = f.path.splitFile().name
        let vWeights = loadSafetensors(f.path)
        if "voice" in vWeights:
          model.voices[vName] = vWeights["voice"]
    echo "Loaded ", model.voices.len, " voices"

  # Load vocab from config.json
  let configPath = modelDir / "config.json"
  if fileExists(configPath):
    let cfg = parseJson(readFile(configPath))
    if cfg.hasKey("vocab"):
      for key, val in cfg["vocab"]:
        model.vocab[key] = val.getInt()
      echo "Loaded vocab: ", model.vocab.len, " tokens"
  else:
    echo "Warning: config.json not found in ", modelDir

  # Precompute fused weight-normed weights (eliminates norm at inference)
  model.precomputeWeights()

  # Cache STFT basis matrices
  let (fb, cm, sm, w) = buildStftBasis(model.config.genIstftNFft, model.config.genIstftNFft)
  model.stftBasis = StftBasis(forwardBasis: fb, cosMat: cm, sinMat: sm, window: w, valid: true)

  return model

# ── High-level API ───────────────────────────────────────────────

proc synthesize*(model: KokoroModel, text: string,
                 voice: string = "af_heart", speed: float32 = 1.0,
                 callback: SynthCallback = nil): AudioOutput =
  ## Full Kokoro TTS pipeline.
  let cfg = model.config
  if voice notin model.voices:
    raise newException(ValueError, "voice not found: " & voice)

  let voiceTensor = model.voices[voice]

  # Phonemize
  let voiceLang = if voice.len > 0: voice[0] else: 'a'
  let phmzr = if voiceLang != model.phmzr.langCode:
    newPhonemizer(voice)
  else:
    model.phmzr

  let phonemes = phmzr.phonemize(normalizeForKokoro(text))
  if phonemes.len == 0:
    return AudioOutput(samples: @[], sampleRate: int32(cfg.samplingRate), channels: 1)

  let audio = model.forward(phonemes, voiceTensor, speed)
  eval(audio)

  result = AudioOutput(
    sampleRate: int32(cfg.samplingRate),
    channels: 1,
  )
  let n = audio.size
  result.samples = audio.toSeqF32()

  if callback != nil:
    callback(result, 0, 1)

proc mixVoice*(model: var KokoroModel, name1, name2: string,
               weight: float32 = 0.5, mixName: string = ""): string =
  ## Blend two voices via linear interpolation of their tensors.
  ## weight=0.0 → pure voice1, weight=1.0 → pure voice2.
  ## Returns the name under which the mixed voice is registered.
  if name1 notin model.voices:
    raise newException(ValueError, "voice not found: " & name1)
  if name2 notin model.voices:
    raise newException(ValueError, "voice not found: " & name2)
  let v1 = model.voices[name1]
  let v2 = model.voices[name2]
  result = if mixName.len > 0: mixName else: name1 & "+" & name2
  let w1 = scalar(1.0'f32 - weight)
  let w2 = scalar(weight)
  let mixed = v1 * w1 + v2 * w2
  eval(mixed)
  model.voices[result] = mixed

proc listVoices*(model: KokoroModel): seq[string] =
  for name in model.voices.keys:
    result.add(name)

proc close*(model: var KokoroModel) =
  model.weights.clear()
  model.voices.clear()

# ── Smoke test ───────────────────────────────────────────────────

when isMainModule:
  import std/[os, times, stats]

  if paramCount() < 1:
    echo "Usage: kokoro_mlx <model_dir> [voice] [text]"
    echo "       kokoro_mlx <model_dir> --bench"
    quit(1)

  let modelDir = paramStr(1)
  let doBench = paramCount() >= 2 and paramStr(2) == "--bench"
  let voice = if not doBench and paramCount() >= 2: paramStr(2) else: "af_heart"

  var model = loadKokoroMlx(modelDir, voice)

  if doBench:
    const Warmup = 1
    const Runs = 5
    let texts = [
      ("short", "ðə kwˈɪk bɹˈaʊn fˈɑːks dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ."),
      ("medium", "ðə kwˈɪk bɹˈaʊn fˈɑːks dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ. ɪt wʌz ðə bˈɛst ʌv tˈaɪmz, ɪt wʌz ðə wˈɜːst ʌv tˈaɪmz. ðə sˈʌn ʃˈoʊn bɹˈaɪtli ˌoʊvɚ ðə kwˈaɪət vˈɪlɪdʒ."),
      ("long", "ðə kwˈɪk bɹˈaʊn fˈɑːks dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ. ɪt wʌz ðə bˈɛst ʌv tˈaɪmz, ɪt wʌz ðə wˈɜːst ʌv tˈaɪmz. ðə sˈʌn ʃˈoʊn bɹˈaɪtli ˌoʊvɚ ðə kwˈaɪət vˈɪlɪdʒ. ʃiː wˈɔːkt slˈoʊli θɹuː ðə ɡˈɑːɹdən, ɛndʒˈɔɪɪŋ ðə fɹˈeɪɡɹəns ʌv fɹˈɛʃli kˈʌt flˈaʊɚz. ðə tʃˈɪldɹən plˈeɪd hˈæpɪli ɪn ðə pˈɑːɹk wˈaɪl ðɛɹ pˈɛɹənts wˈɑːtʃt fɹʌm ə bˈɛntʃ nˈɪɹbaɪ."),
    ]

    let voiceTensor = model.voices[voice]

    echo "Benchmark: Nim MLX (", Runs, " runs, ", Warmup, " warmup)\n"
    for (label, phonemes) in texts:
      # Warmup
      for w in 0..<Warmup:
        let a = model.forward(phonemes, voiceTensor, 1.0)
        eval(a)

      var timesMs: seq[float64]
      var samples = 0
      for r in 0..<Runs:
        let t0 = cpuTime()
        let audio = model.forward(phonemes, voiceTensor, 1.0)
        eval(audio)
        let elapsed = (cpuTime() - t0) * 1000.0
        timesMs.add(elapsed)
        samples = audio.size

      let duration = samples.float / 24000.0
      var rs: RunningStat
      for t in timesMs: rs.push(t)
      let avg = rs.mean
      let best = timesMs.min
      echo "--- ", label, " (", phonemes.runeLen, " phoneme chars) ---"
      echo "  nim-mlx:   ", avg.formatFloat(ffDecimal, 1), "ms avg  ",
           best.formatFloat(ffDecimal, 1), "ms min  ",
           samples, " samples  ", duration.formatFloat(ffDecimal, 2), "s audio  ",
           (duration / (avg / 1000.0)).formatFloat(ffDecimal, 1), "x RT"
      echo ""

  elif paramCount() >= 3:
    let text = paramStr(3)
    echo "Synthesizing: '", text, "' with voice '", voice, "'"
    let audio = model.synthesize(text, voice)
    echo "Audio: ", audio.samples.len, " samples @ ", audio.sampleRate, " Hz"
    if audio.samples.len > 0:
      audio.writeWav("kokoro_mlx_output.wav")
      echo "Written to: kokoro_mlx_output.wav"
