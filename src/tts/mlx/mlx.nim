## High-level MLX wrapper for Nim.
## Provides ergonomic tensor operations on top of mlx_capi.

import std/[os, tables]
import mlx_capi
export mlx_capi.MlxDtype, mlx_capi.MlxDeviceType

# ── Runtime init ─────────────────────────────────────────────────

const mlxSrcRoot = currentSourcePath().parentDir().parentDir().parentDir()
const mlxMetalLib* = mlxSrcRoot / "vendor" / "mlx" / "lib" / "mlx.metallib"

proc initMlx*() =
  ## Initialize MLX runtime. Must be called before any MLX operations.
  ## Sets MLX_METAL_PATH so the Metal shader library is found.
  putEnv("MLX_METAL_PATH", mlxMetalLib)

# ── Tensor type ──────────────────────────────────────────────────

type
  Tensor* = object
    arr*: MlxArray

  Stream* = object
    s*: MlxStream

proc gpu*(): Stream =
  Stream(s: mlx_default_gpu_stream_new())

proc cpu*(): Stream =
  Stream(s: mlx_default_cpu_stream_new())

# Default stream (GPU on Apple Silicon)
var defaultStream*: Stream

proc initDefaultStream*() =
  defaultStream = gpu()

proc s(): MlxStream {.inline.} = defaultStream.s

# ── Tensor lifecycle ─────────────────────────────────────────────

proc `=destroy`(t: var Tensor) =
  if t.arr.ctx != nil:
    discard mlx_array_free(t.arr)

proc `=copy`(dst: var Tensor, src: Tensor) =
  if dst.arr.ctx != nil:
    discard mlx_array_free(dst.arr)
  if src.arr.ctx == nil:
    dst.arr = MlxArray(ctx: nil)
  else:
    dst.arr = mlx_array_new()
    discard mlx_array_set(addr dst.arr, src.arr)

proc `=sink`(dst: var Tensor, src: Tensor) =
  if dst.arr.ctx != nil:
    discard mlx_array_free(dst.arr)
  dst.arr = src.arr

proc wrap(a: MlxArray): Tensor {.inline.} =
  Tensor(arr: a)

proc empty*(): Tensor =
  wrap(mlx_array_new())

proc isNil*(t: Tensor): bool {.inline.} =
  ## Check if this tensor has no underlying MLX array.
  t.arr.ctx == nil

# ── Tensor creation ──────────────────────────────────────────────

proc scalar*(val: float32): Tensor =
  wrap(mlx_array_new_float(val))

proc scalar*(val: int): Tensor =
  wrap(mlx_array_new_int(val.cint))

proc scalar*(val: bool): Tensor =
  wrap(mlx_array_new_bool(val))

proc fromData*(data: pointer, shape: openArray[int], dtype: MlxDtype): Tensor =
  var cshape = newSeq[cint](shape.len)
  for i, v in shape: cshape[i] = v.cint
  wrap(mlx_array_new_data(data, addr cshape[0], shape.len.cint, dtype))

proc fromSeq*(data: var seq[float32], shape: openArray[int]): Tensor =
  fromData(addr data[0], shape, MLX_FLOAT32)

proc fromSeq*(data: var seq[int32], shape: openArray[int]): Tensor =
  fromData(addr data[0], shape, MLX_INT32)

proc zeros*(shape: openArray[int], dtype: MlxDtype = MLX_FLOAT32): Tensor =
  var res = mlx_array_new()
  var cshape = newSeq[cint](shape.len)
  for i, v in shape: cshape[i] = v.cint
  discard mlx_zeros(addr res, addr cshape[0], shape.len.csize_t, dtype, s())
  wrap(res)

proc ones*(shape: openArray[int], dtype: MlxDtype = MLX_FLOAT32): Tensor =
  var res = mlx_array_new()
  var cshape = newSeq[cint](shape.len)
  for i, v in shape: cshape[i] = v.cint
  discard mlx_ones(addr res, addr cshape[0], shape.len.csize_t, dtype, s())
  wrap(res)

proc full*(shape: openArray[int], val: float32, dtype: MlxDtype = MLX_FLOAT32): Tensor =
  var res = mlx_array_new()
  var cshape = newSeq[cint](shape.len)
  for i, v in shape: cshape[i] = v.cint
  let v = mlx_array_new_float(val)
  discard mlx_full(addr res, addr cshape[0], shape.len.csize_t, v, dtype, s())
  discard mlx_array_free(v)
  wrap(res)

proc arange*(start, stop, step: float64, dtype: MlxDtype = MLX_FLOAT32): Tensor =
  var res = mlx_array_new()
  discard mlx_arange(addr res, start, stop, step, dtype, s())
  wrap(res)

proc arange*(stop: float64, dtype: MlxDtype = MLX_FLOAT32): Tensor =
  arange(0.0, stop, 1.0, dtype)

proc arange*(stop: int, dtype: MlxDtype = MLX_INT32): Tensor =
  arange(0.0, stop.float64, 1.0, dtype)

proc linspace*(start, stop: float64, num: int, dtype: MlxDtype = MLX_FLOAT32): Tensor =
  var res = mlx_array_new()
  discard mlx_linspace(addr res, start, stop, num.cint, dtype, s())
  wrap(res)

proc eye*(n: int, dtype: MlxDtype = MLX_FLOAT32): Tensor =
  var res = mlx_array_new()
  discard mlx_eye(addr res, n.cint, n.cint, 0.cint, dtype, s())
  wrap(res)

proc hanning*(n: int): Tensor =
  var res = mlx_array_new()
  discard mlx_hanning(addr res, n.cint, s())
  wrap(res)

proc tril*(a: Tensor, k: int = 0): Tensor =
  var res = mlx_array_new()
  discard mlx_tril(addr res, a.arr, k.cint, s())
  wrap(res)

proc triu*(a: Tensor, k: int = 0): Tensor =
  var res = mlx_array_new()
  discard mlx_triu(addr res, a.arr, k.cint, s())
  wrap(res)

# ── Tensor info ──────────────────────────────────────────────────

proc ndim*(t: Tensor): int = mlx_array_ndim(t.arr).int
proc size*(t: Tensor): int = mlx_array_size(t.arr).int
proc nbytes*(t: Tensor): int = mlx_array_nbytes(t.arr).int
proc dtype*(t: Tensor): MlxDtype = mlx_array_dtype(t.arr)

proc shape*(t: Tensor): seq[int] =
  let nd = t.ndim
  let p = mlx_array_shape(t.arr)
  result = newSeq[int](nd)
  for i in 0..<nd:
    result[i] = cast[ptr UncheckedArray[cint]](p)[i].int

proc dim*(t: Tensor, axis: int): int =
  mlx_array_dim(t.arr, axis.cint).int

proc `$`*(t: Tensor): string =
  var ms = mlx_string_new()
  discard mlx_array_tostring(addr ms, t.arr)
  let p = mlx_string_data(ms)
  if p != nil: result = $p
  else: result = "<empty tensor>"
  discard mlx_string_free(ms)

# ── Evaluation ───────────────────────────────────────────────────

proc eval*(t: Tensor) =
  discard mlx_array_eval(t.arr)

proc eval*(tensors: varargs[Tensor]) =
  for t in tensors:
    discard mlx_array_eval(t.arr)

# ── Data access ──────────────────────────────────────────────────

proc dataFloat32*(t: Tensor): ptr UncheckedArray[float32] =
  t.eval()
  cast[ptr UncheckedArray[float32]](mlx_array_data_float32(t.arr))

proc dataInt32*(t: Tensor): ptr UncheckedArray[int32] =
  t.eval()
  cast[ptr UncheckedArray[int32]](mlx_array_data_int32(t.arr))

proc toSeqF32*(t: Tensor): seq[float32] =
  t.eval()
  let n = t.size
  result = newSeq[float32](n)
  let p = t.dataFloat32
  if p != nil:
    copyMem(addr result[0], p, n * sizeof(float32))

proc itemFloat32*(t: Tensor): float32 =
  t.eval()
  discard mlx_array_item_float32(addr result, t.arr)

proc itemInt32*(t: Tensor): int32 =
  t.eval()
  discard mlx_array_item_int32(addr result, t.arr)

# ── Unary op helper ──────────────────────────────────────────────

template unaryOp(name, cfn: untyped): untyped =
  proc name*(a: Tensor): Tensor =
    var res = mlx_array_new()
    discard cfn(addr res, a.arr, s())
    wrap(res)

# ── Binary op helper ─────────────────────────────────────────────

template binaryOp(name, cfn: untyped): untyped =
  proc name*(a, b: Tensor): Tensor =
    var res = mlx_array_new()
    discard cfn(addr res, a.arr, b.arr, s())
    wrap(res)

# ── Arithmetic ───────────────────────────────────────────────────

binaryOp(`+`, mlx_add)
binaryOp(`-`, mlx_subtract)
binaryOp(`*`, mlx_multiply)
binaryOp(`/`, mlx_divide)
unaryOp(negative, mlx_negative)
binaryOp(pow, mlx_power)
unaryOp(square, mlx_square)
unaryOp(sqrt, mlx_sqrt)
unaryOp(rsqrt, mlx_rsqrt)
unaryOp(reciprocal, mlx_reciprocal)
unaryOp(abs, mlx_abs)
binaryOp(maximum, mlx_maximum)
binaryOp(minimum, mlx_minimum)
unaryOp(floor, mlx_floor)
unaryOp(ceil, mlx_ceil)
proc round*(a: Tensor, decimals: int = 0): Tensor =
  var res = mlx_array_new()
  discard mlx_round(addr res, a.arr, decimals.cint, s())
  wrap(res)
binaryOp(remainder, mlx_remainder)

proc clip*(a, lo, hi: Tensor): Tensor =
  var res = mlx_array_new()
  discard mlx_clip(addr res, a.arr, lo.arr, hi.arr, s())
  wrap(res)

# ── Math ─────────────────────────────────────────────────────────

unaryOp(exp, mlx_exp)
unaryOp(log, mlx_log)
unaryOp(log10, mlx_log10)
unaryOp(log2, mlx_log2)
unaryOp(sin, mlx_sin)
unaryOp(cos, mlx_cos)
unaryOp(tan, mlx_tan)
unaryOp(tanh, mlx_tanh)
binaryOp(arctan2, mlx_arctan2)
unaryOp(erf, mlx_erf)
unaryOp(sigmoid, mlx_sigmoid)
unaryOp(absVal, mlx_abs)  # abs conflicts with system.abs

# ── Softmax ──────────────────────────────────────────────────────

proc softmax*(a: Tensor, axis: int, precise: bool = false): Tensor =
  var res = mlx_array_new()
  discard mlx_softmax_axis(addr res, a.arr, axis.cint, precise, s())
  wrap(res)

# ── Reductions ───────────────────────────────────────────────────

proc sum*(a: Tensor, axis: int, keepdims: bool = false): Tensor =
  var res = mlx_array_new()
  discard mlx_sum_axis(addr res, a.arr, axis.cint, keepdims, s())
  wrap(res)

proc mean*(a: Tensor, axis: int, keepdims: bool = false): Tensor =
  var res = mlx_array_new()
  discard mlx_mean_axis(addr res, a.arr, axis.cint, keepdims, s())
  wrap(res)

proc variance*(a: Tensor, axis: int, keepdims: bool = false, ddof: int = 0): Tensor =
  var res = mlx_array_new()
  discard mlx_var_axis(addr res, a.arr, axis.cint, keepdims, ddof.cint, s())
  wrap(res)

proc cumsum*(a: Tensor, axis: int, reverse: bool = false, inclusive: bool = true): Tensor =
  var res = mlx_array_new()
  discard mlx_cumsum(addr res, a.arr, axis.cint, reverse, inclusive, s())
  wrap(res)

proc argmax*(a: Tensor, axis: int, keepdims: bool = false): Tensor =
  var res = mlx_array_new()
  discard mlx_argmax_axis(addr res, a.arr, axis.cint, keepdims, s())
  wrap(res)

proc argmin*(a: Tensor, axis: int, keepdims: bool = false): Tensor =
  var res = mlx_array_new()
  discard mlx_argmin_axis(addr res, a.arr, axis.cint, keepdims, s())
  wrap(res)

proc maxVal*(a: Tensor, axis: int, keepdims: bool = false): Tensor =
  var res = mlx_array_new()
  discard mlx_max_axis(addr res, a.arr, axis.cint, keepdims, s())
  wrap(res)

proc minVal*(a: Tensor, axis: int, keepdims: bool = false): Tensor =
  var res = mlx_array_new()
  discard mlx_min_axis(addr res, a.arr, axis.cint, keepdims, s())
  wrap(res)

# ── Matrix ops ───────────────────────────────────────────────────

binaryOp(matmul, mlx_matmul)
proc `@`*(a, b: Tensor): Tensor = matmul(a, b)

# ── Quantization ────────────────────────────────────────────────

proc quantize*(w: Tensor, groupSize: int = 64, bits: int = 4): tuple[qw, scales, biases: Tensor] =
  ## Quantize a weight tensor. Returns (quantized_weights, scales, biases).
  var res = mlx_vector_array_new()
  let nullArr = MlxArray(ctx: nil)  # no global_scale
  discard mlx_quantize(addr res, w.arr, someInt(groupSize.cint), someInt(bits.cint),
                       "affine", nullArr, s())
  var qw, sc, bi: MlxArray
  qw = mlx_array_new(); discard mlx_vector_array_get(addr qw, res, 0)
  sc = mlx_array_new(); discard mlx_vector_array_get(addr sc, res, 1)
  bi = mlx_array_new(); discard mlx_vector_array_get(addr bi, res, 2)
  discard mlx_vector_array_free(res)
  (wrap(qw), wrap(sc), wrap(bi))

proc dequantize*(w, scales, biases: Tensor, groupSize: int = 64, bits: int = 4): Tensor =
  ## Dequantize a quantized weight tensor back to float.
  var res = mlx_array_new()
  let nullArr = MlxArray(ctx: nil)
  discard mlx_dequantize(addr res, w.arr, scales.arr, biases.arr,
                         someInt(groupSize.cint), someInt(bits.cint),
                         "affine", nullArr, noneDtype(), s())
  wrap(res)

proc quantizedMatmul*(x, w, scales, biases: Tensor, transpose: bool = true,
                      groupSize: int = 64, bits: int = 4): Tensor =
  ## Matrix multiply x with quantized weights. Much faster than dequantize + matmul.
  var res = mlx_array_new()
  discard mlx_quantized_matmul(addr res, x.arr, w.arr, scales.arr, biases.arr,
                               transpose, someInt(groupSize.cint), someInt(bits.cint),
                               "affine", s())
  wrap(res)

# ── Convolution ──────────────────────────────────────────────────

proc conv1d*(input, weight: Tensor, stride: int = 1, padding: int = 0,
             dilation: int = 1, groups: int = 1): Tensor =
  var res = mlx_array_new()
  discard mlx_conv1d(addr res, input.arr, weight.arr,
                      stride.cint, padding.cint, dilation.cint, groups.cint, s())
  wrap(res)

proc conv2d*(input, weight: Tensor, stride: array[2, int] = [1, 1],
             padding: array[2, int] = [0, 0], dilation: array[2, int] = [1, 1],
             groups: int = 1): Tensor =
  ## 2D convolution. Input: (N, H, W, C_in), Weight: (C_out, kH, kW, C_in/groups).
  var res = mlx_array_new()
  discard mlx_conv2d(addr res, input.arr, weight.arr,
                      stride[0].cint, stride[1].cint,
                      padding[0].cint, padding[1].cint,
                      dilation[0].cint, dilation[1].cint,
                      groups.cint, s())
  wrap(res)

proc convTranspose1d*(input, weight: Tensor, stride: int = 1, padding: int = 0,
                       dilation: int = 1, outputPadding: int = 0, groups: int = 1): Tensor =
  var res = mlx_array_new()
  discard mlx_conv_transpose1d(addr res, input.arr, weight.arr,
                                stride.cint, padding.cint, dilation.cint,
                                outputPadding.cint, groups.cint, s())
  wrap(res)

# ── Shape manipulation ───────────────────────────────────────────

proc reshape*(a: Tensor, shape: openArray[int]): Tensor =
  var res = mlx_array_new()
  var cshape = newSeq[cint](shape.len)
  for i, v in shape: cshape[i] = v.cint
  discard mlx_reshape(addr res, a.arr, addr cshape[0], shape.len.csize_t, s())
  wrap(res)

proc transpose*(a: Tensor, axes: openArray[int]): Tensor =
  var res = mlx_array_new()
  var caxes = newSeq[cint](axes.len)
  for i, v in axes: caxes[i] = v.cint
  discard mlx_transpose_axes(addr res, a.arr, addr caxes[0], axes.len.csize_t, s())
  wrap(res)

proc transpose*(a: Tensor): Tensor =
  var res = mlx_array_new()
  discard mlx_transpose(addr res, a.arr, s())
  wrap(res)

proc swapaxes*(a: Tensor, axis1, axis2: int): Tensor =
  var res = mlx_array_new()
  discard mlx_swapaxes(addr res, a.arr, axis1.cint, axis2.cint, s())
  wrap(res)

proc expandDims*(a: Tensor, axis: int): Tensor =
  var res = mlx_array_new()
  discard mlx_expand_dims(addr res, a.arr, axis.cint, s())
  wrap(res)

proc squeeze*(a: Tensor, axis: int): Tensor =
  var res = mlx_array_new()
  discard mlx_squeeze_axis(addr res, a.arr, axis.cint, s())
  wrap(res)

proc flatten*(a: Tensor, startAxis: int = 0, endAxis: int = -1): Tensor =
  var res = mlx_array_new()
  discard mlx_flatten(addr res, a.arr, startAxis.cint, endAxis.cint, s())
  wrap(res)

# ── Indexing / slicing ───────────────────────────────────────────

proc slice*(a: Tensor, start, stop: openArray[int], strides: openArray[int] = @[]): Tensor =
  var res = mlx_array_new()
  var cstart = newSeq[cint](start.len)
  var cstop = newSeq[cint](stop.len)
  for i, v in start: cstart[i] = v.cint
  for i, v in stop: cstop[i] = v.cint
  if strides.len == 0:
    var cstrides = newSeq[cint](start.len)
    for i in 0..<start.len: cstrides[i] = 1.cint
    discard mlx_slice(addr res, a.arr,
                       addr cstart[0], start.len.csize_t,
                       addr cstop[0], stop.len.csize_t,
                       addr cstrides[0], start.len.csize_t, s())
  else:
    var cstrides = newSeq[cint](strides.len)
    for i, v in strides: cstrides[i] = v.cint
    discard mlx_slice(addr res, a.arr,
                       addr cstart[0], start.len.csize_t,
                       addr cstop[0], stop.len.csize_t,
                       addr cstrides[0], strides.len.csize_t, s())
  wrap(res)

proc take*(a, indices: Tensor, axis: int): Tensor =
  var res = mlx_array_new()
  discard mlx_take_axis(addr res, a.arr, indices.arr, axis.cint, s())
  wrap(res)

proc takeAlongAxis*(a, indices: Tensor, axis: int): Tensor =
  var res = mlx_array_new()
  discard mlx_take_along_axis(addr res, a.arr, indices.arr, axis.cint, s())
  wrap(res)

proc scatterAdd*(a, indices, values: Tensor, axis: int): Tensor =
  var res = mlx_array_new()
  discard mlx_scatter_add_axis(addr res, a.arr, indices.arr, values.arr, axis.cint, s())
  wrap(res)

# ── Concatenation / stacking ─────────────────────────────────────

proc makeVec(tensors: openArray[Tensor]): MlxVectorArray =
  result = mlx_vector_array_new()
  for t in tensors:
    discard mlx_vector_array_append_value(result, t.arr)

proc concatenate*(tensors: openArray[Tensor], axis: int): Tensor =
  var res = mlx_array_new()
  let vec = makeVec(tensors)
  discard mlx_concatenate_axis(addr res, vec, axis.cint, s())
  discard mlx_vector_array_free(vec)
  wrap(res)

proc stack*(tensors: openArray[Tensor], axis: int = 0): Tensor =
  var res = mlx_array_new()
  let vec = makeVec(tensors)
  discard mlx_stack_axis(addr res, vec, axis.cint, s())
  discard mlx_vector_array_free(vec)
  wrap(res)

proc split*(a: Tensor, numSplits: int, axis: int): seq[Tensor] =
  var vecRes = mlx_vector_array_new()
  discard mlx_split(addr vecRes, a.arr, numSplits.cint, axis.cint, s())
  let n = mlx_vector_array_size(vecRes).int
  result = newSeq[Tensor](n)
  for i in 0..<n:
    var elem = mlx_array_new()
    discard mlx_vector_array_get(addr elem, vecRes, i.csize_t)
    result[i] = wrap(elem)
  discard mlx_vector_array_free(vecRes)

# ── Repeat / tile / pad ─────────────────────────────────────────

proc repeat*(a: Tensor, repeats: int, axis: int): Tensor =
  var res = mlx_array_new()
  discard mlx_repeat_axis(addr res, a.arr, repeats.cint, axis.cint, s())
  wrap(res)

proc tile*(a: Tensor, reps: openArray[int]): Tensor =
  var res = mlx_array_new()
  var creps = newSeq[cint](reps.len)
  for i, v in reps: creps[i] = v.cint
  discard mlx_tile(addr res, a.arr, addr creps[0], reps.len.csize_t, s())
  wrap(res)

proc pad*(a: Tensor, axes: openArray[int], lowPad, highPad: openArray[int],
          val: float32 = 0.0): Tensor =
  var res = mlx_array_new()
  var caxes = newSeq[cint](axes.len)
  var clow = newSeq[cint](lowPad.len)
  var chigh = newSeq[cint](highPad.len)
  for i, v in axes: caxes[i] = v.cint
  for i, v in lowPad: clow[i] = v.cint
  for i, v in highPad: chigh[i] = v.cint
  let padVal = mlx_array_new_float(val)
  discard mlx_pad(addr res, a.arr,
                   addr caxes[0], axes.len.csize_t,
                   addr clow[0], lowPad.len.csize_t,
                   addr chigh[0], highPad.len.csize_t,
                   padVal, "constant", s())
  discard mlx_array_free(padVal)
  wrap(res)

# ── Broadcast / where ────────────────────────────────────────────

proc broadcastTo*(a: Tensor, shape: openArray[int]): Tensor =
  var res = mlx_array_new()
  var cshape = newSeq[cint](shape.len)
  for i, v in shape: cshape[i] = v.cint
  discard mlx_broadcast_to(addr res, a.arr, addr cshape[0], shape.len.csize_t, s())
  wrap(res)

proc where*(condition, x, y: Tensor): Tensor =
  var res = mlx_array_new()
  discard mlx_where(addr res, condition.arr, x.arr, y.arr, s())
  wrap(res)

# ── Comparison ───────────────────────────────────────────────────

binaryOp(`>`, mlx_greater)
binaryOp(`>=`, mlx_greater_equal)
binaryOp(`<`, mlx_less)
binaryOp(`<=`, mlx_less_equal)
binaryOp(equal, mlx_equal)

# ── Type conversion ──────────────────────────────────────────────

proc astype*(a: Tensor, dtype: MlxDtype): Tensor =
  var res = mlx_array_new()
  discard mlx_astype(addr res, a.arr, dtype, s())
  wrap(res)

# ── FFT ──────────────────────────────────────────────────────────

proc rfft*(a: Tensor, n: int = -1, axis: int = -1): Tensor =
  var res = mlx_array_new()
  discard mlx_fft_rfft(addr res, a.arr, n.cint, axis.cint, s())
  wrap(res)

proc irfft*(a: Tensor, n: int = -1, axis: int = -1): Tensor =
  var res = mlx_array_new()
  discard mlx_fft_irfft(addr res, a.arr, n.cint, axis.cint, s())
  wrap(res)

# ── Random ──────────────────────────────────────────────────────

proc randomNormal*(shape: openArray[int], dtype: MlxDtype = MLX_FLOAT32): Tensor =
  var res = mlx_array_new()
  var cshape = newSeq[cint](shape.len)
  for i, v in shape: cshape[i] = v.cint
  let nullKey = mlx_array_new()  # null key → use default PRNG state
  discard mlx_random_normal(addr res, addr cshape[0], shape.len.csize_t,
                             dtype, 0.0, 1.0, nullKey, s())
  wrap(res)

proc randomUniform*(lo, hi: Tensor, shape: openArray[int],
                     dtype: MlxDtype = MLX_FLOAT32): Tensor =
  var res = mlx_array_new()
  var cshape = newSeq[cint](shape.len)
  for i, v in shape: cshape[i] = v.cint
  let nullKey = mlx_array_new()
  discard mlx_random_uniform(addr res, lo.arr, hi.arr, addr cshape[0],
                              shape.len.csize_t, dtype, nullKey, s())
  wrap(res)

# ── Linalg ───────────────────────────────────────────────────────

proc norm*(a: Tensor, ord: float64 = 2.0, axis: int = -1, keepdims: bool = false): Tensor =
  var res = mlx_array_new()
  var ax = axis.cint
  discard mlx_linalg_norm(addr res, a.arr, ord, addr ax, 1, keepdims, s())
  wrap(res)

# ── Safetensors I/O ──────────────────────────────────────────────

proc loadSafetensors*(path: string): Table[string, Tensor] =
  ## Load a safetensors file. Returns a table of name → Tensor.
  ## Uses CPU stream for loading (Load primitive is CPU-only).
  var tensorsMap = mlx_map_string_to_array_new()
  var metaMap = mlx_map_string_to_string_new()
  let cpuStream = mlx_default_cpu_stream_new()
  let rc = mlx_load_safetensors(addr tensorsMap, addr metaMap, path.cstring, cpuStream)
  if rc != 0:
    discard mlx_map_string_to_array_free(tensorsMap)
    discard mlx_map_string_to_string_free(metaMap)
    raise newException(IOError, "Failed to load safetensors: " & path)

  result = initTable[string, Tensor]()
  var it = mlx_map_string_to_array_iterator_new(tensorsMap)
  var key: cstring
  var value: MlxArray
  while true:
    let r = mlx_map_string_to_array_iterator_next(addr key, addr value, it)
    if r != 0: break
    if key == nil: break
    # Copy the array handle — the iterator reuses its internal handle
    var copy = mlx_array_new()
    discard mlx_array_set(addr copy, value)
    result[$key] = wrap(copy)

  discard mlx_map_string_to_array_iterator_free(it)
  discard mlx_map_string_to_array_free(tensorsMap)
  discard mlx_map_string_to_string_free(metaMap)

# ── GGUF I/O ────────────────────────────────────────────────────

type GgufData* = object
  weights*: Table[string, Tensor]
  vocab*: seq[string]
  metadata*: Table[string, string]

proc loadGguf*(path: string): GgufData =
  ## Load a GGUF file via MLX. Returns weights, vocab, and string metadata.
  var gguf = mlx_io_gguf_new()
  let rc = mlx_load_gguf(addr gguf, path.cstring, s())
  if rc != 0:
    discard mlx_io_gguf_free(gguf)
    raise newException(IOError, "Failed to load GGUF: " & path)

  # Get all tensor keys
  var keys = mlx_vector_string_new()
  discard mlx_io_gguf_get_keys(addr keys, gguf)
  let nKeys = mlx_vector_string_size(keys)

  result.weights = initTable[string, Tensor]()
  for i in 0..<nKeys:
    var keyStr: cstring
    discard mlx_vector_string_get(addr keyStr, keys, i.csize_t)
    if keyStr != nil:
      var arr = mlx_array_new()
      if mlx_io_gguf_get_array(addr arr, gguf, keyStr) == 0:
        result.weights[$keyStr] = wrap(arr)
      else:
        discard mlx_array_free(arr)
  discard mlx_vector_string_free(keys)

  # Load vocab from metadata
  var hasVocab: bool
  discard mlx_io_gguf_has_metadata_vector_string(addr hasVocab, gguf, "tokenizer.ggml.tokens")
  if hasVocab:
    var vocabVec = mlx_vector_string_new()
    discard mlx_io_gguf_get_metadata_vector_string(addr vocabVec, gguf, "tokenizer.ggml.tokens")
    let nVocab = mlx_vector_string_size(vocabVec)
    for i in 0..<nVocab:
      var tok: cstring
      discard mlx_vector_string_get(addr tok, vocabVec, i.csize_t)
      result.vocab.add(if tok != nil: $tok else: "")
    discard mlx_vector_string_free(vocabVec)

  # Load common string metadata
  result.metadata = initTable[string, string]()
  for key in ["general.architecture", "general.name", "tokenizer.ggml.model"]:
    var hasKey: bool
    discard mlx_io_gguf_has_metadata_string(addr hasKey, gguf, key.cstring)
    if hasKey:
      var val = mlx_string_new()
      discard mlx_io_gguf_get_metadata_string(addr val, gguf, key.cstring)
      result.metadata[key] = $mlx_string_data(val)
      discard mlx_string_free(val)

  discard mlx_io_gguf_free(gguf)

# ── Snake activation (TTS-specific) ─────────────────────────────

proc snake*(x, alpha: Tensor): Tensor =
  ## Snake activation: x + (1/alpha) * sin(alpha * x)^2
  let ax = alpha * x
  let sinPart = square(sin(ax))
  result = x + reciprocal(alpha) * sinPart

# ── Smoke test ───────────────────────────────────────────────────

when isMainModule:
  initMlx()
  initDefaultStream()

  echo "Testing MLX bindings..."

  # Scalar
  let a = scalar(3.14'f32)
  echo "  scalar: ", a

  # Zeros
  let z = zeros([2, 3])
  echo "  zeros [2,3]: ", z

  # Arithmetic
  let x = scalar(2.0'f32)
  let y = scalar(3.0'f32)
  let added = x + y
  eval(added)
  echo "  2 + 3 = ", added.itemFloat32

  # Arange
  let r = arange(5)
  echo "  arange(5): ", r

  echo "OK: MLX bindings work"
