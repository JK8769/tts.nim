## Silero VAD v5 — neural voice-activity detection using MLX backend.
## Architecture: STFT → 4×Conv1D encoder → LSTM → Linear → Sigmoid.
## Processes 512-sample chunks (32ms at 16kHz).
## Converted from snakers4/silero-vad ONNX model.

import std/[tables, os]
import ../mlx/mlx

const
  SILERO_CHUNK_SIZE* = 512       ## Samples per chunk (32ms at 16kHz)
  SILERO_SAMPLE_RATE* = 16000
  STFT_N_FFT = 256
  STFT_HOP = 128
  STFT_BINS = STFT_N_FFT div 2 + 1  # 129
  LSTM_HIDDEN = 128

type
  VadState* = enum
    vsSilence   ## No speech detected
    vsSpeech    ## Speech in progress
    vsTrailing  ## Speech ended, waiting holdoff

  VadEvent* = enum
    veNone          ## No state change
    veSpeechStart   ## Transitioned to speech
    veSpeechEnd     ## Transitioned to silence

  SileroVad* = ref object
    weights*: Table[string, Tensor]
    h*: Tensor            ## LSTM hidden state (1, 128)
    c*: Tensor            ## LSTM cell state (1, 128)
    threshold*: float32   ## Speech probability threshold
    state*: VadState
    holdoffCounter: int
    holdoffChunks: int    ## Chunks of silence before confirming speech end
    padBuf: seq[float32]  ## Ring buffer for pre-speech audio
    padPos: int
    padFull: bool
    padChunks: int        ## How many chunks to keep as pre-speech padding
    # Encoder conv weight/bias cached in channels-last format for MLX
    stftKernel: Tensor    ## (258, 256, 1) transposed for MLX conv1d
    encW: array[4, Tensor]
    encB: array[4, Tensor]
    decW: Tensor          ## (1, 1, 128) transposed for MLX conv1d
    decB: Tensor          ## (1,)
    lstmWih: Tensor       ## (512, 128)
    lstmWhh: Tensor       ## (512, 128)
    lstmBias: Tensor      ## (512,) combined ih+hh bias

proc w(v: SileroVad, key: string): Tensor {.inline.} =
  v.weights[key]

# ── STFT via Conv1D ─────────────────────────────────────────────

proc stft(v: SileroVad, audio: Tensor): Tensor =
  ## Compute STFT magnitude spectrogram.
  ## audio: (1, 512) → output: (1, T, 129) channels-last for MLX.
  # Input for MLX conv1d: (batch, seq, channels) = (1, 512, 1)
  let x = audio.reshape([1, SILERO_CHUNK_SIZE, 1])
  # Conv1D: kernel (258, 256, 1), stride=128 → (1, T, 258)
  let raw = conv1d(x, v.stftKernel, stride = STFT_HOP)
  # Split real/imag and compute magnitude
  let real = raw.slice([0, 0, 0], [1, raw.dim(1), STFT_BINS])
  let imag = raw.slice([0, 0, STFT_BINS], [1, raw.dim(1), 2 * STFT_BINS])
  sqrt(square(real) + square(imag))

# ── Encoder (4× Conv1D + ReLU) ─────────────────────────────────

proc encode(v: SileroVad, mag: Tensor): Tensor =
  ## 4 conv1d blocks with ReLU.
  ## Input: (1, T, 129) → Output: (1, T', 128)
  const strides = [1, 2, 2, 1]
  var x = mag
  for i in 0..3:
    x = conv1d(x, v.encW[i], stride = strides[i], padding = 1) + v.encB[i]
    x = maximum(x, scalar(0.0'f32))  # ReLU
  x

# ── LSTM cell ───────────────────────────────────────────────────

proc lstmStep(v: SileroVad, input: Tensor) =
  ## One LSTM step. Updates v.h and v.c in-place.
  ## input: (1, 128)
  let gates = (input @ transpose(v.lstmWih)) +
              (v.h @ transpose(v.lstmWhh)) + v.lstmBias  # (1, 512)
  let ig = sigmoid(gates.slice([0, 0], [1, LSTM_HIDDEN]))
  let fg = sigmoid(gates.slice([0, LSTM_HIDDEN], [1, 2 * LSTM_HIDDEN]))
  let gg = tanh(gates.slice([0, 2 * LSTM_HIDDEN], [1, 3 * LSTM_HIDDEN]))
  let og = sigmoid(gates.slice([0, 3 * LSTM_HIDDEN], [1, 4 * LSTM_HIDDEN]))
  v.c = fg * v.c + ig * gg
  v.h = og * tanh(v.c)

# ── Forward pass ────────────────────────────────────────────────

proc forward*(v: SileroVad, audio: Tensor): float32 =
  ## Process one 512-sample chunk, return speech probability.
  ## audio: (1, 512) float32 at 16kHz.
  let mag = v.stft(audio)       # (1, T, 129)
  let enc = v.encode(mag)       # (1, T', 128)
  eval(enc)

  # LSTM: process each time step
  let T = enc.dim(1)
  for t in 0..<T:
    let xt = enc.slice([0, t, 0], [1, t + 1, LSTM_HIDDEN]).reshape([1, LSTM_HIDDEN])
    v.lstmStep(xt)
  eval(v.h)

  # Output: ReLU → Linear → Sigmoid
  let hRelu = maximum(v.h, scalar(0.0'f32))
  # Conv1d kernel=1 is a linear layer: (1, 1, 128) input, (1, 1, 128) weight → (1, 1, 1)
  let hExp = hRelu.expandDims(1)  # (1, 1, 128)
  let logits = conv1d(hExp, v.decW) + v.decB  # (1, 1, 1)
  let prob = sigmoid(logits)
  eval(prob)
  prob.itemFloat32()

proc processChunk*(v: SileroVad, samples: openArray[float32]): float32 =
  ## Convenience: process raw float32 samples, return speech probability.
  assert samples.len == SILERO_CHUNK_SIZE,
    "Silero VAD requires exactly 512 samples per chunk"
  var buf = @samples
  let audio = fromSeq(buf, [1, SILERO_CHUNK_SIZE])
  v.forward(audio)

# ── Model loading ───────────────────────────────────────────────

proc loadSileroVad*(modelDir: string, threshold: float32 = 0.5,
                    holdoffChunks: int = 15,
                    padChunks: int = 3): SileroVad =
  ## Load Silero VAD from a directory containing model.safetensors.
  ## holdoffChunks: chunks of silence before confirming speech end (~480ms at 15)
  ## padChunks: chunks of pre-speech audio to keep (~96ms at 3)
  initMlx()
  initDefaultStream()

  let stPath = modelDir / "model.safetensors"
  if not fileExists(stPath):
    raise newException(IOError, "model.safetensors not found in: " & modelDir)

  let padSize = padChunks * SILERO_CHUNK_SIZE
  result = SileroVad(
    threshold: threshold,
    state: vsSilence,
    holdoffChunks: holdoffChunks,
    padBuf: newSeq[float32](padSize),
    padChunks: padChunks,
  )
  result.weights = loadSafetensors(stPath)

  # Transpose ONNX Conv1D weights from (out, in, kernel) to MLX (out, kernel, in)
  result.stftKernel = result.w("stft.forward_basis_buffer").transpose([0, 2, 1])
  for i in 0..3:
    result.encW[i] = result.w("encoder." & $i & ".reparam_conv.weight").transpose([0, 2, 1])
    result.encB[i] = result.w("encoder." & $i & ".reparam_conv.bias")
  result.decW = result.w("decoder.decoder.2.weight").transpose([0, 2, 1])
  result.decB = result.w("decoder.decoder.2.bias")

  # LSTM weights
  result.lstmWih = result.w("decoder.rnn.weight_ih")
  result.lstmWhh = result.w("decoder.rnn.weight_hh")
  # Pre-combine biases
  result.lstmBias = result.w("decoder.rnn.bias_ih") + result.w("decoder.rnn.bias_hh")
  eval(result.lstmBias)

  # Init LSTM state to zeros
  result.h = zeros([1, LSTM_HIDDEN])
  result.c = zeros([1, LSTM_HIDDEN])

  echo "Silero VAD loaded (threshold=", threshold, ")"

proc reset*(v: SileroVad) =
  ## Reset LSTM state and VAD state machine.
  v.h = zeros([1, LSTM_HIDDEN])
  v.c = zeros([1, LSTM_HIDDEN])
  v.state = vsSilence
  v.holdoffCounter = 0
  v.padPos = 0
  v.padFull = false

proc pushPad(v: SileroVad, samples: openArray[float32]) =
  let fs = min(samples.len, v.padBuf.len)
  for i in 0..<fs:
    v.padBuf[v.padPos] = samples[i]
    v.padPos = (v.padPos + 1) mod v.padBuf.len
  if v.padPos == 0 or fs >= v.padBuf.len:
    v.padFull = true

proc drainPad*(v: SileroVad): seq[float32] =
  ## Return pre-speech padding audio and clear the buffer.
  if v.padBuf.len == 0: return @[]
  if not v.padFull:
    result = v.padBuf[0..<v.padPos]
  else:
    result = newSeq[float32](v.padBuf.len)
    let start = v.padPos
    for i in 0..<v.padBuf.len:
      result[i] = v.padBuf[(start + i) mod v.padBuf.len]
  v.padPos = 0
  v.padFull = false

proc processFrame*(v: SileroVad, frame: openArray[float32]): VadEvent =
  ## Process a 512-sample frame through the neural VAD and state machine.
  ## Returns event if state changed.
  let prob = v.processChunk(frame)
  let isSpeech = prob > v.threshold

  case v.state
  of vsSilence:
    if isSpeech:
      v.state = vsSpeech
      v.holdoffCounter = 0
      return veSpeechStart
    else:
      v.pushPad(frame)
      return veNone

  of vsSpeech:
    if not isSpeech:
      v.state = vsTrailing
      v.holdoffCounter = 1
    return veNone

  of vsTrailing:
    if isSpeech:
      v.state = vsSpeech
      v.holdoffCounter = 0
      return veNone
    else:
      inc v.holdoffCounter
      if v.holdoffCounter >= v.holdoffChunks:
        v.state = vsSilence
        v.holdoffCounter = 0
        return veSpeechEnd
      return veNone

proc close*(v: SileroVad) =
  v.weights.clear()

# ── Smoke test ──────────────────────────────────────────────────

when isMainModule:
  import std/[times, strformat]

  if paramCount() < 1:
    echo "Usage: silero_vad <model_dir> [audio.wav]"
    quit(1)

  let modelDir = paramStr(1)
  var vad = loadSileroVad(modelDir)

  if paramCount() >= 2:
    let wavPath = paramStr(2)
    let data = readFile(wavPath)
    var samples: seq[float32]
    var dataOffset = -1
    for i in 0..<data.len - 8:
      if data[i] == 'd' and data[i+1] == 'a' and data[i+2] == 't' and data[i+3] == 'a':
        dataOffset = i + 8
        break
    if dataOffset < 0:
      echo "Invalid WAV: no data chunk"
      quit(1)
    let numSamples = (data.len - dataOffset) div 2
    samples = newSeq[float32](numSamples)
    for i in 0..<numSamples:
      let lo = uint8(data[dataOffset + i * 2])
      let hi = uint8(data[dataOffset + i * 2 + 1])
      let sample = cast[int16](lo.uint16 or (hi.uint16 shl 8))
      samples[i] = float32(sample) / 32768.0
    echo "Audio: ", numSamples, " samples (", (numSamples.float / 16000.0), "s)"

    let t0 = cpuTime()
    var nChunks = 0
    var i = 0
    while i + SILERO_CHUNK_SIZE <= samples.len:
      let prob = vad.processChunk(samples[i ..< i + SILERO_CHUNK_SIZE])
      let label = if prob > vad.threshold: "SPEECH" else: "silence"
      if nChunks < 20 or prob > vad.threshold:
        echo &"  {(i.float / 16000.0 * 1000).int}ms: prob={prob:.6f} {label}"
      i += SILERO_CHUNK_SIZE
      inc nChunks
    let elapsed = cpuTime() - t0
    echo nChunks, " chunks in ", (elapsed * 1000).int, "ms (",
         (elapsed / nChunks.float * 1000000).int, "us/chunk)"
  else:
    echo "Model loaded. Pass an audio file to test."

  vad.close()
