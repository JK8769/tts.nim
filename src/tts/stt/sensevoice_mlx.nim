## SenseVoice speech-to-text using Apple MLX backend.
## Non-autoregressive CTC model: audio → Kaldi fbank → SANM encoder → CTC decode.
## Port from mlx-audio's SenseVoice implementation to Nim, using mlx-c bindings.
## Supports: zh, en, ja, ko, yue (Cantonese). ~234M params, ~70ms for 10s audio.

import std/[tables, strutils, math, os, json, sequtils]
import ../mlx/mlx

const SENSEVOICE_SAMPLE_RATE* = 16000

# ── Configuration ────────────────────────────────────────────────

type
  SenseVoiceEncoderConfig* = object
    outputSize*: int          # 512
    attentionHeads*: int      # 4
    linearUnits*: int         # 2048
    numBlocks*: int           # 50
    tpBlocks*: int            # 20
    kernelSize*: int          # 11
    sanmShift*: int           # 0

  SenseVoiceFrontendConfig* = object
    fs*: int                  # 16000
    nMels*: int               # 80
    frameLengthMs*: int       # 25
    frameShiftMs*: int        # 10
    lfrM*: int                # 7  (stack M frames)
    lfrN*: int                # 6  (skip every N)

  SenseVoiceConfig* = object
    vocabSize*: int           # 25055
    inputSize*: int           # 560 (= 80 * 7 after LFR)
    encoder*: SenseVoiceEncoderConfig
    frontend*: SenseVoiceFrontendConfig

proc defaultConfig*(): SenseVoiceConfig =
  SenseVoiceConfig(
    vocabSize: 25055,
    inputSize: 560,
    encoder: SenseVoiceEncoderConfig(
      outputSize: 512,
      attentionHeads: 4,
      linearUnits: 2048,
      numBlocks: 50,
      tpBlocks: 20,
      kernelSize: 11,
      sanmShift: 0,
    ),
    frontend: SenseVoiceFrontendConfig(
      fs: 16000,
      nMels: 80,
      frameLengthMs: 25,
      frameShiftMs: 10,
      lfrM: 7,
      lfrN: 6,
    ),
  )

proc configFromJson*(data: JsonNode): SenseVoiceConfig =
  result = defaultConfig()
  if data.hasKey("vocab_size"):
    result.vocabSize = data["vocab_size"].getInt()
  if data.hasKey("input_size"):
    result.inputSize = data["input_size"].getInt()
  if data.hasKey("encoder_conf"):
    let ec = data["encoder_conf"]
    if ec.hasKey("output_size"): result.encoder.outputSize = ec["output_size"].getInt()
    if ec.hasKey("attention_heads"): result.encoder.attentionHeads = ec["attention_heads"].getInt()
    if ec.hasKey("linear_units"): result.encoder.linearUnits = ec["linear_units"].getInt()
    if ec.hasKey("num_blocks"): result.encoder.numBlocks = ec["num_blocks"].getInt()
    if ec.hasKey("tp_blocks"): result.encoder.tpBlocks = ec["tp_blocks"].getInt()
    if ec.hasKey("kernel_size"): result.encoder.kernelSize = ec["kernel_size"].getInt()
    # Handle typo in upstream config
    if ec.hasKey("sanm_shift"): result.encoder.sanmShift = ec["sanm_shift"].getInt()
    elif ec.hasKey("sanm_shfit"): result.encoder.sanmShift = ec["sanm_shfit"].getInt()
  if data.hasKey("frontend_conf"):
    let fc = data["frontend_conf"]
    if fc.hasKey("fs"): result.frontend.fs = fc["fs"].getInt()
    if fc.hasKey("n_mels"): result.frontend.nMels = fc["n_mels"].getInt()
    if fc.hasKey("frame_length"): result.frontend.frameLengthMs = fc["frame_length"].getInt()
    if fc.hasKey("frame_shift"): result.frontend.frameShiftMs = fc["frame_shift"].getInt()
    if fc.hasKey("lfr_m"): result.frontend.lfrM = fc["lfr_m"].getInt()
    if fc.hasKey("lfr_n"): result.frontend.lfrN = fc["lfr_n"].getInt()

# ── Model store ──────────────────────────────────────────────────

type
  SenseVoiceModel* = object
    config*: SenseVoiceConfig
    weights*: Table[string, Tensor]
    cmvnMeans*: Tensor       # (input_size,) from am.mvn
    cmvnIstd*: Tensor        # (input_size,) from am.mvn
    tokenList*: seq[string]  # token_id → string (from tokens.json)

proc w*(m: SenseVoiceModel, key: string): Tensor {.inline.} =
  if key notin m.weights:
    raise newException(KeyError, "Missing weight: " & key)
  m.weights[key]

# ── Kaldi mel filterbank ─────────────────────────────────────────

proc nextPow2(n: int): int =
  result = 1
  while result < n: result = result shl 1

proc buildMelBanksKaldi(nMels, nFft, sampleRate: int,
                        lowFreq, highFreq: float64): Tensor =
  ## Kaldi-style mel filterbank: (nMels, nFft/2+1)
  let nFreqs = nFft div 2 + 1
  let fMax = if highFreq <= 0.0: float64(sampleRate) / 2.0 else: highFreq

  # HTK mel scale
  proc hzToMel(f: float64): float64 = 1127.0 * ln(1.0 + f / 700.0)
  proc melToHz(m: float64): float64 = 700.0 * (exp(m / 1127.0) - 1.0)

  let melLow = hzToMel(lowFreq)
  let melHigh = hzToMel(fMax)

  # Mel center frequencies
  var melPts = newSeq[float64](nMels + 2)
  for i in 0..nMels+1:
    melPts[i] = melLow + float64(i) * (melHigh - melLow) / float64(nMels + 1)

  var fPts = newSeq[float64](nMels + 2)
  for i in 0..nMels+1:
    fPts[i] = melToHz(melPts[i])

  # Build triangular filterbank (nMels × nFreqs)
  var fb = newSeq[float32](nMels * nFreqs)
  for m in 0..<nMels:
    for f in 0..<nFreqs:
      let freq = float64(f) * float64(sampleRate) / float64(nFft)
      if freq >= fPts[m] and freq <= fPts[m+1] and fPts[m+1] > fPts[m]:
        fb[m * nFreqs + f] = float32((freq - fPts[m]) / (fPts[m+1] - fPts[m]))
      elif freq > fPts[m+1] and freq <= fPts[m+2] and fPts[m+2] > fPts[m+1]:
        fb[m * nFreqs + f] = float32((fPts[m+2] - freq) / (fPts[m+2] - fPts[m+1]))

  fromSeq(fb, [nMels, nFreqs])

proc computeKaldiFbank(audio: Tensor, config: SenseVoiceFrontendConfig): Tensor =
  ## Compute Kaldi-compatible log mel-filterbank features.
  ## Input: (N,) float32 PCM at 16kHz. Output: (T, nMels).
  let sampleRate = config.fs
  let winLen = sampleRate * config.frameLengthMs div 1000  # 400
  let winInc = sampleRate * config.frameShiftMs div 1000   # 160
  let paddedWinSize = nextPow2(winLen)

  # Scale to int16 range (Kaldi convention)
  let scaled = audio * scalar(float32(1 shl 15))

  # Frame the audio (snip_edges=true: only complete frames)
  let totalLen = scaled.dim(0)
  let numFrames = (totalLen - winLen) div winInc + 1
  if numFrames <= 0:
    return zeros([0, config.nMels])

  # Build frame indices
  var offsets = newSeq[int32](numFrames)
  for i in 0..<numFrames:
    offsets[i] = int32(i * winInc)
  var offsetT = fromSeq(offsets, [numFrames])
  let sampleIdx = arange(winLen, MLX_INT32)
  let indices = offsetT.expandDims(1) + sampleIdx.expandDims(0)
  var frames = scaled.take(indices.flatten(), axis = 0).reshape([numFrames, winLen])

  # Remove DC offset per frame
  let rowMeans = mean(frames, axis = 1, keepdims = true)
  frames = frames - rowMeans

  # Preemphasis (0.97)
  let firstCol = frames.slice([0, 0], [numFrames, 1])
  let shifted = frames.slice([0, 0], [numFrames, winLen - 1])
  let rest = frames.slice([0, 1], [numFrames, winLen])
  let preemph = rest - shifted * scalar(0.97'f32)
  frames = concatenate([firstCol, preemph], axis = 1)

  # Hamming window
  var winData = newSeq[float32](winLen)
  for i in 0..<winLen:
    winData[i] = float32(0.54 - 0.46 * cos(2.0 * PI * float64(i) / float64(winLen - 1)))
  var winT = fromSeq(winData, [1, winLen])
  frames = frames * winT

  # Pad to power of 2
  if paddedWinSize > winLen:
    let padLen = paddedWinSize - winLen
    frames = pad(frames, [1], [0], [padLen])

  # FFT → power spectrum
  let fftResult = rfft(frames, paddedWinSize, axis = 1)
  let spectrum = square(absVal(fftResult))

  # Mel filterbank: spectrum (numFrames, nFreqs) @ melBanks.T (nFreqs, nMels) → (numFrames, nMels)
  let melBanks = buildMelBanksKaldi(config.nMels, paddedWinSize, sampleRate, 20.0, 0.0)
  # melBanks is (nMels, nFreqs), need (nFreqs, nMels)
  let melFeatures = spectrum @ transpose(melBanks)

  # Log
  log(maximum(melFeatures, scalar(1e-8'f32)))

# ── Low Frame Rate (LFR) ────────────────────────────────────────

proc applyLfr(feats: Tensor, lfrM, lfrN: int): Tensor =
  ## Stack lfrM consecutive frames, stepping by lfrN.
  ## Input: (T, D). Output: (ceil(T/lfrN), D*lfrM).
  let T = feats.dim(0)
  let D = feats.dim(1)
  let tLfr = (T + lfrN - 1) div lfrN

  # Left-pad with copies of first frame
  let leftPad = (lfrM - 1) div 2
  var padded = feats
  if leftPad > 0:
    let firstFrame = feats.slice([0, 0], [1, D])
    let padFrames = tile(firstFrame, [leftPad, 1])
    padded = concatenate([padFrames, feats], axis = 0)
  let tPadded = padded.dim(0)

  var frames: seq[Tensor]
  for i in 0..<tLfr:
    let start = i * lfrN
    let stop = start + lfrM
    if stop <= tPadded:
      frames.add padded.slice([start, 0], [stop, D]).reshape([1, lfrM * D])
    else:
      # Right-pad with copies of last frame
      let available = padded.slice([start, 0], [tPadded, D])
      let padCount = lfrM - (tPadded - start)
      let lastFrame = padded.slice([tPadded - 1, 0], [tPadded, D])
      let padFrames = tile(lastFrame, [padCount, 1])
      let combined = concatenate([available, padFrames], axis = 0)
      frames.add combined.reshape([1, lfrM * D])

  concatenate(frames, axis = 0)

# ── CMVN normalization ───────────────────────────────────────────

proc applyCmvn(feats, means, istd: Tensor): Tensor =
  (feats + means) * istd

proc parseAmMvn*(path: string): tuple[means, istd: seq[float32]] =
  ## Parse Kaldi am.mvn file to extract AddShift (means) and Rescale (istd) values.
  let text = readFile(path)
  # Extract AddShift values
  let shiftStart = text.find("[", text.find("<AddShift>"))
  let shiftEnd = text.find("]", shiftStart)
  if shiftStart < 0 or shiftEnd < 0:
    raise newException(ValueError, "Could not parse AddShift from am.mvn")
  let meansStr = text[shiftStart+1..<shiftEnd].strip()
  for s in meansStr.splitWhitespace():
    result.means.add parseFloat(s).float32

  # Extract Rescale values
  let rescaleStart = text.find("[", text.find("<Rescale>"))
  let rescaleEnd = text.find("]", rescaleStart)
  if rescaleStart < 0 or rescaleEnd < 0:
    raise newException(ValueError, "Could not parse Rescale from am.mvn")
  let istdStr = text[rescaleStart+1..<rescaleEnd].strip()
  for s in istdStr.splitWhitespace():
    result.istd.add parseFloat(s).float32

# ── Sinusoidal position encoding ─────────────────────────────────

proc sinusoidalPosEnc(x: Tensor): Tensor =
  ## Add sinusoidal positional encoding. x: (B, T, D) → (B, T, D).
  let T = x.dim(1)
  let D = x.dim(2)
  let halfD = D div 2
  let logInc = ln(10000.0) / float64(halfD - 1)

  var data = newSeq[float32](T * D)
  for t in 0..<T:
    let pos = float64(t + 1)  # SenseVoice uses 1-based positions
    for c in 0..<halfD:
      let angle = pos * exp(-logInc * float64(c))
      data[t * D + c] = float32(sin(angle))
      data[t * D + halfD + c] = float32(cos(angle))

  var posEnc = fromSeq(data, [1, T, D])
  x + posEnc

# ── SANM Attention (Self-Attention with Normalized Memory) ───────

proc sanmAttention(model: SenseVoiceModel, x: Tensor,
                   prefix: string, nHead, inFeat, nFeat: int,
                   kernelSize, sanmShift: int): Tensor =
  ## Multi-headed attention with FSMN (Feedforward Sequential Memory Network).
  let B = x.dim(0)
  let T = x.dim(1)
  let dK = nFeat div nHead

  # Q, K, V projection
  let qkvW = model.w(prefix & "linear_q_k_v.weight")
  let qkvB = model.w(prefix & "linear_q_k_v.bias")
  let qkv = (x @ transpose(qkvW)) + qkvB
  let parts = split(qkv, 3, axis = 2)
  let q = parts[0]
  let k = parts[1]
  let v = parts[2]

  # FSMN block: depthwise Conv1d on V
  var leftPad = (kernelSize - 1) div 2
  if sanmShift > 0:
    leftPad = leftPad + sanmShift
  let rightPad = kernelSize - 1 - leftPad
  let vPadded = pad(v, [1], [leftPad], [rightPad])

  let fsmnW = model.w(prefix & "fsmn_block.weight")
  let fsmnOut = conv1d(vPadded, fsmnW, groups = nFeat)
  let fsmnMemory = fsmnOut + v

  # Multi-head attention
  let qr = q.reshape([B, T, nHead, dK]).transpose([0, 2, 1, 3])
  let kr = k.reshape([B, T, nHead, dK]).transpose([0, 2, 1, 3])
  let vr = v.reshape([B, T, nHead, dK]).transpose([0, 2, 1, 3])

  let scale = pow(float32(dK), -0.5'f32)
  let scores = (qr * scalar(scale)) @ kr.transpose([0, 1, 3, 2])
  let attn = softmax(scores, axis = -1)
  var attOut = (attn @ vr).transpose([0, 2, 1, 3]).reshape([B, T, nFeat])

  # Output projection
  let outW = model.w(prefix & "linear_out.weight")
  let outB = model.w(prefix & "linear_out.bias")
  attOut = (attOut @ transpose(outW)) + outB

  attOut + fsmnMemory

# ── Encoder layer ────────────────────────────────────────────────

proc encoderLayer(model: SenseVoiceModel, x: Tensor,
                  prefix: string, cfg: SenseVoiceEncoderConfig,
                  inSize, outSize: int): Tensor =
  ## One SANM encoder layer: norm → attention → residual → norm → FFN → residual.
  var cur = x

  # Pre-norm + self-attention
  let norm1W = model.w(prefix & "norm1.weight")
  let norm1B = model.w(prefix & "norm1.bias")
  let eps = 1e-5'f32
  let m1 = mean(cur, axis = -1, keepdims = true)
  let v1 = variance(cur, axis = -1, keepdims = true)
  let normed1 = (cur - m1) / sqrt(v1 + scalar(eps)) * norm1W + norm1B

  let attnOut = sanmAttention(model, normed1,
    prefix & "self_attn.", cfg.attentionHeads, inSize, outSize,
    cfg.kernelSize, cfg.sanmShift)

  # Residual — if sizes match, add; otherwise replace
  if inSize == outSize:
    cur = cur + attnOut
  else:
    cur = attnOut

  # Pre-norm + FFN
  let norm2W = model.w(prefix & "norm2.weight")
  let norm2B = model.w(prefix & "norm2.bias")
  let m2 = mean(cur, axis = -1, keepdims = true)
  let v2 = variance(cur, axis = -1, keepdims = true)
  let normed2 = (cur - m2) / sqrt(v2 + scalar(eps)) * norm2W + norm2B

  let ffW1 = model.w(prefix & "feed_forward.w_1.weight")
  let ffB1 = model.w(prefix & "feed_forward.w_1.bias")
  let ffW2 = model.w(prefix & "feed_forward.w_2.weight")
  let ffB2 = model.w(prefix & "feed_forward.w_2.bias")
  # ReLU activation
  let h = maximum((normed2 @ transpose(ffW1)) + ffB1, scalar(0.0'f32))
  cur + (h @ transpose(ffW2)) + ffB2

# ── Full encoder ─────────────────────────────────────────────────

proc encode(model: SenseVoiceModel, x: Tensor): Tensor =
  ## Run the SANM encoder stack: encoders0 + encoders + after_norm + tp_encoders + tp_norm.
  let cfg = model.config.encoder
  let inputSize = model.config.inputSize
  let outputSize = cfg.outputSize

  # Scale input
  var cur = x * scalar(sqrt(float32(outputSize)))

  # Sinusoidal position encoding
  cur = sinusoidalPosEnc(cur)

  # encoders0: single block, input_size → output_size
  cur = encoderLayer(model, cur, "encoder.encoders0.0.", cfg, inputSize, outputSize)

  # encoders: numBlocks-1 blocks, output_size → output_size
  for i in 0..<cfg.numBlocks - 1:
    cur = encoderLayer(model, cur, "encoder.encoders." & $i & ".", cfg, outputSize, outputSize)

  # after_norm
  let anW = model.w("encoder.after_norm.weight")
  let anB = model.w("encoder.after_norm.bias")
  let eps = 1e-5'f32
  let m3 = mean(cur, axis = -1, keepdims = true)
  let v3 = variance(cur, axis = -1, keepdims = true)
  cur = (cur - m3) / sqrt(v3 + scalar(eps)) * anW + anB

  # tp_encoders: task-specific blocks
  for i in 0..<cfg.tpBlocks:
    cur = encoderLayer(model, cur, "encoder.tp_encoders." & $i & ".", cfg, outputSize, outputSize)

  # tp_norm
  let tnW = model.w("encoder.tp_norm.weight")
  let tnB = model.w("encoder.tp_norm.bias")
  let m4 = mean(cur, axis = -1, keepdims = true)
  let v4 = variance(cur, axis = -1, keepdims = true)
  (cur - m4) / sqrt(v4 + scalar(eps)) * tnW + tnB

# ── Query embedding (language, textnorm, event/emotion) ──────────

const
  LidDict* = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}.toTable
  TextnormDict = {"withitn": 14, "woitn": 15}.toTable
  LidTokenMap* = {24884: "zh", 24885: "en", 24888: "yue", 24892: "ja",
                  24896: "ko", 24992: "nospeech"}.toTable
  EmoTokenMap* = {25001: "happy", 25002: "sad", 25003: "angry", 25004: "neutral",
                  25005: "fearful", 25006: "disgusted", 25007: "surprised",
                  25008: "other", 25009: "unk"}.toTable
  EventTokenMap* = {24993: "Speech", 24995: "BGM", 24997: "Laughter",
                    24999: "Applause"}.toTable

proc buildQuery(model: SenseVoiceModel, language: string, useItn: bool): tuple[textnormQ, inputQ: Tensor] =
  let embedW = model.w("embed.weight")
  let inputSize = model.config.inputSize

  let lid = LidDict.getOrDefault(language, 0)
  var lidIdx = @[int32(lid)]
  let langQ = take(embedW, fromSeq(lidIdx, [1]), axis = 0).reshape([1, 1, inputSize])

  let tnId = if useItn: TextnormDict["withitn"] else: TextnormDict["woitn"]
  var tnIdx = @[int32(tnId)]
  let tnQ = take(embedW, fromSeq(tnIdx, [1]), axis = 0).reshape([1, 1, inputSize])

  var emoEvtIdx = @[int32(1), int32(2)]
  let emoEvtQ = take(embedW, fromSeq(emoEvtIdx, [2]), axis = 0).reshape([1, 2, inputSize])

  let inputQ = concatenate([langQ, emoEvtQ], axis = 1)  # (1, 3, inputSize)
  (tnQ, inputQ)

# ── CTC decode ───────────────────────────────────────────────────

proc toSeqI32(t: Tensor): seq[int32] =
  let n = t.size
  result = newSeq[int32](n)
  let flat = t.flatten()
  eval(flat)
  let p = flat.dataInt32()
  for i in 0..<n:
    result[i] = p[i]

proc greedyCtcDecode(logProbs: Tensor): seq[int32] =
  ## Greedy CTC decode: argmax → deduplicate → remove blanks (id=0).
  let pred = argmax(logProbs, axis = -1)
  eval(pred)
  let predList = pred.toSeqI32()

  var prev: int32 = -1
  for t in predList:
    if t != prev:
      if t != 0:  # skip blank
        result.add t
      prev = t

proc greedyCtcDecodeWithConf(logProbs: Tensor): tuple[tokens: seq[int32], avgLogProb: float32] =
  ## Greedy CTC decode with confidence score (average log probability of non-blank tokens).
  let pred = argmax(logProbs, axis = -1)
  let maxProbs = maxVal(logProbs, axis = -1)
  eval(pred)
  eval(maxProbs)
  let predList = pred.toSeqI32()
  let probList = maxProbs.toSeqF32()

  var prev: int32 = -1
  var logProbSum: float32 = 0
  var tokenCount = 0
  for i, t in predList:
    if t != prev:
      if t != 0:
        result.tokens.add t
        logProbSum += probList[i]
        inc tokenCount
      prev = t
  result.avgLogProb = if tokenCount > 0: logProbSum / tokenCount.float32 else: -100.0

proc decodeTokens(model: SenseVoiceModel, tokenIds: seq[int32]): string =
  if model.tokenList.len > 0:
    var pieces: seq[string]
    for t in tokenIds:
      if t >= 0 and t < int32(model.tokenList.len):
        pieces.add model.tokenList[t]
    result = pieces.join("").replace("\xe2\x96\x81", " ").strip()  # ▁ → space
  else:
    result = tokenIds.mapIt($it).join(" ")

# ── Rich info extraction ─────────────────────────────────────────

type
  SenseVoiceRichInfo* = object
    language*: string
    emotion*: string
    event*: string

proc extractRichInfo(logProbs: Tensor): SenseVoiceRichInfo =
  ## Extract language, emotion, event from the first 3 output positions.
  # Position 0: language
  let lidLogProbs = logProbs.slice([0, 0], [1, logProbs.dim(1)])
  let lidPred = argmax(lidLogProbs, axis = -1)
  eval(lidPred)
  let lidToken = lidPred.itemInt32()
  result.language = LidTokenMap.getOrDefault(lidToken, "unknown")

  # Position 1: emotion
  let emoLogProbs = logProbs.slice([1, 0], [2, logProbs.dim(1)])
  let emoPred = argmax(emoLogProbs, axis = -1)
  eval(emoPred)
  let emoToken = emoPred.itemInt32()
  result.emotion = EmoTokenMap.getOrDefault(emoToken, "unk")

  # Position 2: event
  let evtLogProbs = logProbs.slice([2, 0], [3, logProbs.dim(1)])
  let evtPred = argmax(evtLogProbs, axis = -1)
  eval(evtPred)
  let evtToken = evtPred.itemInt32()
  result.event = EventTokenMap.getOrDefault(evtToken, "unknown")

# ── Forward pass ─────────────────────────────────────────────────

proc forward*(model: SenseVoiceModel, feats: Tensor,
              language: string = "auto", useItn: bool = false): Tensor =
  ## Full forward: prepend query tokens + features → encoder → CTC logits.
  ## feats: (1, T, inputSize). Returns log_probs: (1, T', vocabSize).
  let (textnormQ, inputQ) = model.buildQuery(language, useItn)

  # Prepend: [textnormQ, feats] then [inputQ, ...]
  var speech = concatenate([textnormQ, feats], axis = 1)
  speech = concatenate([inputQ, speech], axis = 1)

  let encoderOut = model.encode(speech)

  # CTC head: Linear → log_softmax
  let ctcW = model.w("ctc_lo.weight")
  let ctcB = model.w("ctc_lo.bias")
  let logits = (encoderOut @ transpose(ctcW)) + ctcB

  # log_softmax
  let maxLogits = maxVal(logits, axis = -1, keepdims = true)
  let shifted = logits - maxLogits
  let expShifted = exp(shifted)
  let logSumExp = log(sum(expShifted, axis = -1, keepdims = true))
  shifted - logSumExp

# ── Weight sanitization ──────────────────────────────────────────

proc sanitizeWeights*(weights: Table[string, Tensor]): Table[string, Tensor] =
  ## Remap upstream weight keys: "ctc.ctc_lo." → "ctc_lo."
  ## Transpose FSMN conv weights: (out, in, kernel) → (out, kernel, in) for MLX conv1d.
  result = initTable[string, Tensor]()
  for k, v in weights:
    var newKey = k.replace("ctc.ctc_lo.", "ctc_lo.")
    var newVal = v
    if "fsmn_block.weight" in newKey and v.ndim == 3:
      newVal = v.transpose([0, 2, 1])
    result[newKey] = newVal

# ── Model loading ────────────────────────────────────────────────

proc loadSenseVoiceMlx*(modelDir: string): SenseVoiceModel =
  ## Load SenseVoice model from directory containing config.json, *.safetensors, am.mvn, tokens.json.
  initMlx()
  initDefaultStream()

  # Load config
  let configPath = modelDir / "config.json"
  if not fileExists(configPath):
    raise newException(IOError, "config.json not found in: " & modelDir)
  result.config = configFromJson(parseJson(readFile(configPath)))
  let cfg = result.config

  stderr.writeLine "SenseVoice config: ", cfg.encoder.outputSize, "d, ",
    cfg.encoder.numBlocks, "+", cfg.encoder.tpBlocks, " blocks, vocab=", cfg.vocabSize

  # Load weights
  result.weights = initTable[string, Tensor]()
  for f in walkDir(modelDir):
    if f.path.endsWith(".safetensors"):
      let tensors = loadSafetensors(f.path)
      for k, v in tensors:
        result.weights[k] = v
      stderr.writeLine "Loaded ", tensors.len, " tensors from ", extractFilename(f.path)

  result.weights = sanitizeWeights(result.weights)

  # Load CMVN stats from am.mvn
  let mvnPath = modelDir / "am.mvn"
  if fileExists(mvnPath):
    let (means, istd) = parseAmMvn(mvnPath)
    var m = means
    var i = istd
    result.cmvnMeans = fromSeq(m, [means.len])
    result.cmvnIstd = fromSeq(i, [istd.len])
    stderr.writeLine "Loaded CMVN: ", means.len, " dims"

  # Load tokenizer (tokens.json fallback — no SentencePiece dependency)
  let tokensPath = modelDir / "tokens.json"
  if fileExists(tokensPath):
    let tokData = parseJson(readFile(tokensPath))
    result.tokenList = newSeq[string](tokData.len)
    for i in 0..<tokData.len:
      result.tokenList[i] = tokData[i].getStr()
    stderr.writeLine "Loaded vocab: ", result.tokenList.len, " tokens"

  stderr.writeLine "SenseVoice model loaded"

# ── Public API (compatible with whisper_mlx SpeechRecognizer) ────

type
  SpeechRecognizer* = ref object
    model*: SenseVoiceModel
    language*: string

proc newSpeechRecognizer*(modelPath: string, language: string = "auto"): SpeechRecognizer =
  if not dirExists(modelPath):
    raise newException(IOError, "SenseVoice model directory not found: " & modelPath)
  var rec = SpeechRecognizer(language: language)
  rec.model = loadSenseVoiceMlx(modelPath)
  rec

proc transcribe*(r: SpeechRecognizer, samples: openArray[float32]): string =
  ## Transcribe float32 PCM audio (16kHz mono). No fixed-length constraint (unlike Whisper).
  if samples.len == 0: return ""

  var audio = @samples
  var audioTensor = fromSeq(audio, [audio.len])

  # Frontend: fbank → LFR → CMVN
  let fc = r.model.config.frontend
  var feats = computeKaldiFbank(audioTensor, fc)
  feats = applyLfr(feats, fc.lfrM, fc.lfrN)
  if not r.model.cmvnMeans.isNil:
    feats = applyCmvn(feats, r.model.cmvnMeans, r.model.cmvnIstd)

  # Batch dimension
  let featsBatched = feats.expandDims(0)  # (1, T, inputSize)

  # Forward pass with ITN enabled for better text normalization
  let logProbs = r.model.forward(featsBatched, r.language, useItn = true)
  eval(logProbs)

  # Extract text (skip first 4 positions: language, emotion, event, textnorm)
  let textLogProbs = logProbs.slice([0, 4, 0],
    [1, logProbs.dim(1), logProbs.dim(2)]).squeeze(0)
  let (tokenIds, avgLogProb) = greedyCtcDecodeWithConf(textLogProbs)
  # Reject very low confidence transcriptions (likely noise/garbage)
  if avgLogProb < -1.0:
    return ""
  r.model.decodeTokens(tokenIds)

proc transcribeRich*(r: SpeechRecognizer, samples: openArray[float32]):
    tuple[text, language, emotion, event: string, confidence: float32] =
  ## Transcribe with rich info (language detection, emotion, audio event).
  if samples.len == 0: return ("", "unknown", "unk", "unknown", 0.0)

  var audio = @samples
  var audioTensor = fromSeq(audio, [audio.len])

  let fc = r.model.config.frontend
  var feats = computeKaldiFbank(audioTensor, fc)
  feats = applyLfr(feats, fc.lfrM, fc.lfrN)
  if not r.model.cmvnMeans.isNil:
    feats = applyCmvn(feats, r.model.cmvnMeans, r.model.cmvnIstd)

  let featsBatched = feats.expandDims(0)
  let logProbs = r.model.forward(featsBatched, r.language, useItn = true)
  eval(logProbs)

  let lp0 = logProbs.squeeze(0)

  let richInfo = extractRichInfo(lp0)
  let textLogProbs = lp0.slice([4, 0], [lp0.dim(0), lp0.dim(1)])
  let (tokenIds, avgLogProb) = greedyCtcDecodeWithConf(textLogProbs)
  let text = r.model.decodeTokens(tokenIds)

  (text, richInfo.language, richInfo.emotion, richInfo.event, avgLogProb)

proc transcribeAs*(r: SpeechRecognizer, samples: openArray[float32],
                   language: string): string =
  ## Transcribe with a forced language override (e.g. "en", "zh").
  if samples.len == 0: return ""
  var audio = @samples
  var audioTensor = fromSeq(audio, [audio.len])
  let fc = r.model.config.frontend
  var feats = computeKaldiFbank(audioTensor, fc)
  feats = applyLfr(feats, fc.lfrM, fc.lfrN)
  if not r.model.cmvnMeans.isNil:
    feats = applyCmvn(feats, r.model.cmvnMeans, r.model.cmvnIstd)
  let featsBatched = feats.expandDims(0)
  let logProbs = r.model.forward(featsBatched, language, useItn = true)
  eval(logProbs)
  let textLogProbs = logProbs.slice([0, 4, 0],
    [1, logProbs.dim(1), logProbs.dim(2)]).squeeze(0)
  let tokenIds = greedyCtcDecode(textLogProbs)
  r.model.decodeTokens(tokenIds)

proc close*(r: SpeechRecognizer) =
  r.model.weights.clear()
