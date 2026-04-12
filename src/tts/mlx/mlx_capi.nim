## Low-level Nim FFI bindings for mlx-c (Apple MLX C API).
## These map 1:1 to the C functions. Higher-level wrappers are in mlx.nim.
##
## Build: vendor/mlx/ must contain lib/{libmlxc.a, libmlx.a, mlx.metallib}
##        and include/mlx/c/*.h

import std/os

const mlxSrcRoot = currentSourcePath().parentDir().parentDir().parentDir().parentDir()
const mlxInclude = mlxSrcRoot / "vendor" / "mlx" / "include"
const mlxLib = mlxSrcRoot / "vendor" / "mlx" / "lib"

{.passC: "-I" & mlxInclude.}
{.passL: "-L" & mlxLib & " -lmlxc -lmlx -lgguflib -lfmt".}
{.passL: "-lc++ -framework Metal -framework Foundation -framework Accelerate".}

# MLX_METAL_PATH must point to the metallib at runtime
# We set it relative to the lib dir

const
  arrH = "mlx/c/array.h"
  opsH = "mlx/c/ops.h"
  streamH = "mlx/c/stream.h"
  deviceH = "mlx/c/device.h"
  fftH = "mlx/c/fft.h"
  ioH = "mlx/c/io.h"
  mapH = "mlx/c/map.h"
  vecH = "mlx/c/vector.h"
  strH = "mlx/c/string.h"
  randomH = "mlx/c/random.h"
  transformsH = "mlx/c/transforms.h"
  metalH = "mlx/c/metal.h"
  linalgH = "mlx/c/linalg.h"
  ioTypesH = "mlx/c/io_types.h"
  optionalH = "mlx/c/optional.h"

# ── Optional types ──────────────────────────────────────────────

type
  MlxOptionalInt* {.importc: "mlx_optional_int", header: optionalH, bycopy.} = object
    value*: cint
    has_value*: bool

  MlxOptionalFloat* {.importc: "mlx_optional_float", header: optionalH, bycopy.} = object
    value*: cfloat
    has_value*: bool

  MlxOptionalDtype* {.importc: "mlx_optional_dtype", header: optionalH, bycopy.} = object
    value*: cint  # MlxDtype
    has_value*: bool

proc someInt*(v: cint): MlxOptionalInt =
  MlxOptionalInt(value: v, has_value: true)

proc noneInt*(): MlxOptionalInt =
  MlxOptionalInt(value: 0, has_value: false)

proc noneDtype*(): MlxOptionalDtype =
  MlxOptionalDtype(value: 0, has_value: false)

# ── Opaque handle types ──────────────────────────────────────────

type
  MlxArray* {.importc: "mlx_array", header: arrH, bycopy.} = object
    ctx*: pointer

  MlxStream* {.importc: "mlx_stream", header: streamH, bycopy.} = object
    ctx*: pointer

  MlxDevice* {.importc: "mlx_device", header: deviceH, bycopy.} = object
    ctx*: pointer

  MlxString* {.importc: "mlx_string", header: strH, bycopy.} = object
    ctx*: pointer

  MlxVectorArray* {.importc: "mlx_vector_array", header: vecH, bycopy.} = object
    ctx*: pointer

  MlxVectorInt* {.importc: "mlx_vector_int", header: vecH, bycopy.} = object
    ctx*: pointer

  MlxMapStringToArray* {.importc: "mlx_map_string_to_array", header: mapH, bycopy.} = object
    ctx*: pointer

  MlxMapStringToString* {.importc: "mlx_map_string_to_string", header: mapH, bycopy.} = object
    ctx*: pointer

  MlxVectorString* {.importc: "mlx_vector_string", header: vecH, bycopy.} = object
    ctx*: pointer

  MlxIoGguf* {.importc: "mlx_io_gguf", header: ioTypesH, bycopy.} = object
    ctx*: pointer

  MlxMapStringToArrayIterator* {.importc: "mlx_map_string_to_array_iterator", header: mapH, bycopy.} = object
    ctx*: pointer
    map_ctx*: pointer

  MlxDtype* {.importc: "mlx_dtype", header: arrH, size: sizeof(cint).} = enum
    MLX_BOOL = 0
    MLX_UINT8 = 1
    MLX_UINT16 = 2
    MLX_UINT32 = 3
    MLX_UINT64 = 4
    MLX_INT8 = 5
    MLX_INT16 = 6
    MLX_INT32 = 7
    MLX_INT64 = 8
    MLX_FLOAT16 = 9
    MLX_FLOAT32 = 10
    MLX_FLOAT64 = 11
    MLX_BFLOAT16 = 12
    MLX_COMPLEX64 = 13

  MlxDeviceType* {.importc: "mlx_device_type", header: deviceH, size: sizeof(cint).} = enum
    MLX_CPU = 0
    MLX_GPU = 1

# ── Array lifecycle ──────────────────────────────────────────────

proc mlx_array_new*(): MlxArray {.importc, header: arrH.}
proc mlx_array_free*(arr: MlxArray): cint {.importc, header: arrH.}
proc mlx_array_set*(arr: ptr MlxArray, src: MlxArray): cint {.importc, header: arrH.}

proc mlx_array_new_bool*(val: bool): MlxArray {.importc, header: arrH.}
proc mlx_array_new_int*(val: cint): MlxArray {.importc, header: arrH.}
proc mlx_array_new_float*(val: cfloat): MlxArray {.importc, header: arrH.}
proc mlx_array_new_data*(data: pointer, shape: ptr cint, dim: cint,
                          dtype: MlxDtype): MlxArray {.importc, header: arrH.}

# ── Array info ───────────────────────────────────────────────────

proc mlx_array_itemsize*(arr: MlxArray): csize_t {.importc, header: arrH.}
proc mlx_array_size*(arr: MlxArray): csize_t {.importc, header: arrH.}
proc mlx_array_nbytes*(arr: MlxArray): csize_t {.importc, header: arrH.}
proc mlx_array_ndim*(arr: MlxArray): csize_t {.importc, header: arrH.}
proc mlx_array_shape*(arr: MlxArray): ptr cint {.importc, header: arrH.}
proc mlx_array_strides*(arr: MlxArray): ptr csize_t {.importc, header: arrH.}
proc mlx_array_dim*(arr: MlxArray, dim: cint): cint {.importc, header: arrH.}
proc mlx_array_dtype*(arr: MlxArray): MlxDtype {.importc, header: arrH.}
proc mlx_array_eval*(arr: MlxArray): cint {.importc, header: arrH.}

# ── Array data access ────────────────────────────────────────────

proc mlx_array_data_float32*(arr: MlxArray): ptr cfloat {.importc, header: arrH.}
proc mlx_array_data_int32*(arr: MlxArray): ptr int32 {.importc, header: arrH.}
proc mlx_array_data_int64*(arr: MlxArray): ptr int64 {.importc, header: arrH.}
proc mlx_array_data_bool*(arr: MlxArray): ptr bool {.importc, header: arrH.}
proc mlx_array_data_uint8*(arr: MlxArray): ptr uint8 {.importc, header: arrH.}

proc mlx_array_item_float32*(res: ptr cfloat, arr: MlxArray): cint {.importc, header: arrH.}
proc mlx_array_item_int32*(res: ptr int32, arr: MlxArray): cint {.importc, header: arrH.}

# ── String ───────────────────────────────────────────────────────

proc mlx_string_new*(): MlxString {.importc, header: strH.}
proc mlx_string_free*(s: MlxString): cint {.importc, header: strH.}
proc mlx_string_data*(s: MlxString): cstring {.importc, header: strH.}
proc mlx_array_tostring*(s: ptr MlxString, arr: MlxArray): cint {.importc, header: arrH.}

# ── Device ───────────────────────────────────────────────────────

proc mlx_device_new*(): MlxDevice {.importc, header: deviceH.}
proc mlx_device_new_type*(dtype: MlxDeviceType, index: cint): MlxDevice {.importc, header: deviceH.}
proc mlx_device_free*(dev: MlxDevice): cint {.importc, header: deviceH.}
proc mlx_set_default_device*(dev: MlxDevice): cint {.importc, header: deviceH.}
proc mlx_get_default_device*(dev: ptr MlxDevice): cint {.importc, header: deviceH.}

# ── Stream ───────────────────────────────────────────────────────

proc mlx_stream_new*(): MlxStream {.importc, header: streamH.}
proc mlx_stream_new_device*(dev: MlxDevice): MlxStream {.importc, header: streamH.}
proc mlx_stream_free*(s: MlxStream): cint {.importc, header: streamH.}
proc mlx_default_cpu_stream_new*(): MlxStream {.importc, header: streamH.}
proc mlx_default_gpu_stream_new*(): MlxStream {.importc, header: streamH.}
proc mlx_synchronize*(s: MlxStream): cint {.importc, header: streamH.}

# ── Vector ───────────────────────────────────────────────────────

proc mlx_vector_array_new*(): MlxVectorArray {.importc, header: vecH.}
proc mlx_vector_array_free*(vec: MlxVectorArray): cint {.importc, header: vecH.}
proc mlx_vector_array_new_data*(data: ptr MlxArray, size: csize_t): MlxVectorArray {.importc, header: vecH.}
proc mlx_vector_array_append_value*(vec: MlxVectorArray, val: MlxArray): cint {.importc, header: vecH.}
proc mlx_vector_array_size*(vec: MlxVectorArray): csize_t {.importc, header: vecH.}
proc mlx_vector_array_get*(res: ptr MlxArray, vec: MlxVectorArray, idx: csize_t): cint {.importc, header: vecH.}

proc mlx_vector_int_new*(): MlxVectorInt {.importc, header: vecH.}
proc mlx_vector_int_free*(vec: MlxVectorInt): cint {.importc, header: vecH.}
proc mlx_vector_int_new_data*(data: ptr cint, size: csize_t): MlxVectorInt {.importc, header: vecH.}

# ── Map ──────────────────────────────────────────────────────────

proc mlx_map_string_to_array_new*(): MlxMapStringToArray {.importc, header: mapH.}
proc mlx_map_string_to_array_free*(m: MlxMapStringToArray): cint {.importc, header: mapH.}
proc mlx_map_string_to_array_get*(value: ptr MlxArray, m: MlxMapStringToArray,
                                   key: cstring): cint {.importc, header: mapH.}
proc mlx_map_string_to_array_insert*(m: MlxMapStringToArray, key: cstring,
                                      value: MlxArray): cint {.importc, header: mapH.}
proc mlx_map_string_to_array_iterator_new*(m: MlxMapStringToArray): MlxMapStringToArrayIterator {.importc, header: mapH.}
proc mlx_map_string_to_array_iterator_free*(it: MlxMapStringToArrayIterator): cint {.importc, header: mapH.}
proc mlx_map_string_to_array_iterator_next*(key: ptr cstring, value: ptr MlxArray,
                                             it: MlxMapStringToArrayIterator): cint {.importc, header: mapH.}

proc mlx_map_string_to_string_new*(): MlxMapStringToString {.importc, header: mapH.}
proc mlx_map_string_to_string_free*(m: MlxMapStringToString): cint {.importc, header: mapH.}

# ── IO (safetensors, gguf) ───────────────────────────────────────

proc mlx_load_safetensors*(res_0: ptr MlxMapStringToArray,
                            res_1: ptr MlxMapStringToString,
                            file: cstring, s: MlxStream): cint {.importc, header: ioH.}
proc mlx_save_safetensors*(file: cstring, param: MlxMapStringToArray,
                            metadata: MlxMapStringToString): cint {.importc, header: ioH.}
proc mlx_load*(res: ptr MlxArray, file: cstring, s: MlxStream): cint {.importc, header: ioH.}

# ── GGUF IO ─────────────────────────────────────────────────────

proc mlx_io_gguf_new*(): MlxIoGguf {.importc, header: ioTypesH.}
proc mlx_io_gguf_free*(io: MlxIoGguf): cint {.importc, header: ioTypesH.}
proc mlx_load_gguf*(gguf: ptr MlxIoGguf, file: cstring, s: MlxStream): cint {.importc, header: ioH.}
proc mlx_io_gguf_get_keys*(keys: ptr MlxVectorString, io: MlxIoGguf): cint {.importc, header: ioTypesH.}
proc mlx_io_gguf_get_array*(arr: ptr MlxArray, io: MlxIoGguf, key: cstring): cint {.importc, header: ioTypesH.}
proc mlx_io_gguf_get_metadata_string*(str: ptr MlxString, io: MlxIoGguf, key: cstring): cint {.importc, header: ioTypesH.}
proc mlx_io_gguf_get_metadata_vector_string*(vstr: ptr MlxVectorString, io: MlxIoGguf, key: cstring): cint {.importc, header: ioTypesH.}
proc mlx_io_gguf_has_metadata_string*(flag: ptr bool, io: MlxIoGguf, key: cstring): cint {.importc, header: ioTypesH.}
proc mlx_io_gguf_has_metadata_vector_string*(flag: ptr bool, io: MlxIoGguf, key: cstring): cint {.importc, header: ioTypesH.}

# ── Vector string ───────────────────────────────────────────────

proc mlx_vector_string_new*(): MlxVectorString {.importc, header: vecH.}
proc mlx_vector_string_free*(v: MlxVectorString): cint {.importc, header: vecH.}
proc mlx_vector_string_size*(v: MlxVectorString): csize_t {.importc, header: vecH.}
proc mlx_vector_string_get*(res: ptr cstring, v: MlxVectorString, idx: csize_t): cint {.importc, header: vecH.}

# ── Metal ────────────────────────────────────────────────────────

proc mlx_metal_set_memory_limit*(limit: csize_t): csize_t {.importc, header: metalH.}
proc mlx_metal_set_cache_limit*(limit: csize_t): csize_t {.importc, header: metalH.}

# ── Arithmetic ops ───────────────────────────────────────────────

proc mlx_add*(res: ptr MlxArray, a, b: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_subtract*(res: ptr MlxArray, a, b: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_multiply*(res: ptr MlxArray, a, b: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_divide*(res: ptr MlxArray, a, b: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_negative*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_power*(res: ptr MlxArray, a, b: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_square*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_sqrt*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_rsqrt*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_reciprocal*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_abs*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_maximum*(res: ptr MlxArray, a, b: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_minimum*(res: ptr MlxArray, a, b: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_clip*(res: ptr MlxArray, a, lo, hi: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_floor*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_ceil*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_round*(res: ptr MlxArray, a: MlxArray, decimals: cint, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_remainder*(res: ptr MlxArray, a, b: MlxArray, s: MlxStream): cint {.importc, header: opsH.}

# ── Math ops ─────────────────────────────────────────────────────

proc mlx_exp*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_log*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_log10*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_log2*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_sin*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_cos*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_tan*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_tanh*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_arctan2*(res: ptr MlxArray, a, b: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_erf*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_sigmoid*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}

# ── Activations ──────────────────────────────────────────────────

proc mlx_softmax_axis*(res: ptr MlxArray, a: MlxArray, axis: cint,
                        precise: bool, s: MlxStream): cint {.importc, header: opsH.}

# ── Reduction ops ────────────────────────────────────────────────

proc mlx_sum_axis*(res: ptr MlxArray, a: MlxArray, axis: cint,
                    keepdims: bool, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_sum_axes*(res: ptr MlxArray, a: MlxArray, axes: ptr cint,
                    axes_num: csize_t, keepdims: bool, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_mean_axis*(res: ptr MlxArray, a: MlxArray, axis: cint,
                     keepdims: bool, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_mean_axes*(res: ptr MlxArray, a: MlxArray, axes: ptr cint,
                     axes_num: csize_t, keepdims: bool, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_var_axis*(res: ptr MlxArray, a: MlxArray, axis: cint,
                    keepdims: bool, ddof: cint, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_max_axis*(res: ptr MlxArray, a: MlxArray, axis: cint,
                    keepdims: bool, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_min_axis*(res: ptr MlxArray, a: MlxArray, axis: cint,
                    keepdims: bool, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_cumsum*(res: ptr MlxArray, a: MlxArray, axis: cint, reverse: bool,
                  inclusive: bool, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_argmax_axis*(res: ptr MlxArray, a: MlxArray, axis: cint,
                       keepdims: bool, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_argmin_axis*(res: ptr MlxArray, a: MlxArray, axis: cint,
                       keepdims: bool, s: MlxStream): cint {.importc, header: opsH.}

# ── Matrix ops ───────────────────────────────────────────────────

proc mlx_matmul*(res: ptr MlxArray, a, b: MlxArray, s: MlxStream): cint {.importc, header: opsH.}

# ── Convolution ──────────────────────────────────────────────────

proc mlx_conv1d*(res: ptr MlxArray, input, weight: MlxArray,
                  stride, padding, dilation, groups: cint,
                  s: MlxStream): cint {.importc, header: opsH.}
proc mlx_conv_transpose1d*(res: ptr MlxArray, input, weight: MlxArray,
                            stride, padding, dilation, output_padding, groups: cint,
                            s: MlxStream): cint {.importc, header: opsH.}
proc mlx_conv2d*(res: ptr MlxArray, input, weight: MlxArray,
                  stride_0, stride_1, padding_0, padding_1,
                  dilation_0, dilation_1, groups: cint,
                  s: MlxStream): cint {.importc, header: opsH.}

# ── Shape manipulation ───────────────────────────────────────────

proc mlx_reshape*(res: ptr MlxArray, a: MlxArray, shape: ptr cint,
                   shape_num: csize_t, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_transpose_axes*(res: ptr MlxArray, a: MlxArray, axes: ptr cint,
                          axes_num: csize_t, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_transpose*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_swapaxes*(res: ptr MlxArray, a: MlxArray, axis1, axis2: cint,
                    s: MlxStream): cint {.importc, header: opsH.}
proc mlx_expand_dims*(res: ptr MlxArray, a: MlxArray, axis: cint,
                       s: MlxStream): cint {.importc, header: opsH.}
proc mlx_squeeze_axis*(res: ptr MlxArray, a: MlxArray, axis: cint,
                        s: MlxStream): cint {.importc, header: opsH.}
proc mlx_flatten*(res: ptr MlxArray, a: MlxArray, start_axis, end_axis: cint,
                   s: MlxStream): cint {.importc, header: opsH.}
proc mlx_contiguous*(res: ptr MlxArray, a: MlxArray, s: MlxStream): cint {.importc, header: opsH, discardable.}

# ── Indexing / slicing ───────────────────────────────────────────

proc mlx_slice*(res: ptr MlxArray, a: MlxArray,
                 start: ptr cint, start_num: csize_t,
                 stop: ptr cint, stop_num: csize_t,
                 strides: ptr cint, strides_num: csize_t,
                 s: MlxStream): cint {.importc, header: opsH.}
proc mlx_take_axis*(res: ptr MlxArray, a, indices: MlxArray, axis: cint,
                     s: MlxStream): cint {.importc, header: opsH.}
proc mlx_take_along_axis*(res: ptr MlxArray, a, indices: MlxArray, axis: cint,
                           s: MlxStream): cint {.importc, header: opsH.}
proc mlx_scatter_add_axis*(res: ptr MlxArray, a, indices, values: MlxArray,
                            axis: cint, s: MlxStream): cint {.importc, header: opsH.}

# ── Concatenation / stacking ─────────────────────────────────────

proc mlx_concatenate_axis*(res: ptr MlxArray, arrays: MlxVectorArray,
                            axis: cint, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_stack_axis*(res: ptr MlxArray, arrays: MlxVectorArray,
                      axis: cint, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_split*(res: ptr MlxVectorArray, a: MlxArray, num_splits: cint,
                 axis: cint, s: MlxStream): cint {.importc, header: opsH.}

# ── Repeat / tile / pad ─────────────────────────────────────────

proc mlx_repeat_axis*(res: ptr MlxArray, arr: MlxArray, repeats: cint,
                       axis: cint, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_tile*(res: ptr MlxArray, a: MlxArray, reps: ptr cint,
                reps_num: csize_t, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_pad*(res: ptr MlxArray, a: MlxArray,
               axes: ptr cint, axes_num: csize_t,
               low_pad_size: ptr cint, low_pad_size_num: csize_t,
               high_pad_size: ptr cint, high_pad_size_num: csize_t,
               val: MlxArray, mode: cstring, s: MlxStream): cint {.importc, header: opsH.}

# ── Broadcast / where ────────────────────────────────────────────

proc mlx_broadcast_to*(res: ptr MlxArray, a: MlxArray, shape: ptr cint,
                        shape_num: csize_t, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_where*(res: ptr MlxArray, condition, x, y: MlxArray,
                 s: MlxStream): cint {.importc, header: opsH.}

# ── Comparison ───────────────────────────────────────────────────

proc mlx_greater*(res: ptr MlxArray, a, b: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_greater_equal*(res: ptr MlxArray, a, b: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_less*(res: ptr MlxArray, a, b: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_less_equal*(res: ptr MlxArray, a, b: MlxArray, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_equal*(res: ptr MlxArray, a, b: MlxArray, s: MlxStream): cint {.importc, header: opsH.}

# ── Creation ops ─────────────────────────────────────────────────

proc mlx_zeros*(res: ptr MlxArray, shape: ptr cint, shape_num: csize_t,
                 dtype: MlxDtype, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_ones*(res: ptr MlxArray, shape: ptr cint, shape_num: csize_t,
                dtype: MlxDtype, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_full*(res: ptr MlxArray, shape: ptr cint, shape_num: csize_t,
                vals: MlxArray, dtype: MlxDtype, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_arange*(res: ptr MlxArray, start, stop, step: cdouble,
                  dtype: MlxDtype, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_linspace*(res: ptr MlxArray, start, stop: cdouble, num: cint,
                    dtype: MlxDtype, s: MlxStream): cint {.importc, header: opsH.}
proc mlx_eye*(res: ptr MlxArray, n, m, k: cint, dtype: MlxDtype,
               s: MlxStream): cint {.importc, header: opsH.}
proc mlx_hanning*(res: ptr MlxArray, n: cint,
                   s: MlxStream): cint {.importc, header: opsH.}
proc mlx_tril*(res: ptr MlxArray, x: MlxArray, k: cint,
                s: MlxStream): cint {.importc, header: opsH.}
proc mlx_triu*(res: ptr MlxArray, x: MlxArray, k: cint,
                s: MlxStream): cint {.importc, header: opsH.}

# ── Type conversion ──────────────────────────────────────────────

proc mlx_astype*(res: ptr MlxArray, a: MlxArray, dtype: MlxDtype,
                  s: MlxStream): cint {.importc, header: opsH.}

# ── FFT ──────────────────────────────────────────────────────────

proc mlx_fft_rfft*(res: ptr MlxArray, a: MlxArray, n: cint, axis: cint,
                    s: MlxStream): cint {.importc, header: fftH.}
proc mlx_fft_irfft*(res: ptr MlxArray, a: MlxArray, n: cint, axis: cint,
                     s: MlxStream): cint {.importc, header: fftH.}
proc mlx_fft_fft*(res: ptr MlxArray, a: MlxArray, n: cint, axis: cint,
                   s: MlxStream): cint {.importc, header: fftH.}
proc mlx_fft_ifft*(res: ptr MlxArray, a: MlxArray, n: cint, axis: cint,
                    s: MlxStream): cint {.importc, header: fftH.}

# ── Random ───────────────────────────────────────────────────────

proc mlx_random_normal*(res: ptr MlxArray, shape: ptr cint, shape_num: csize_t,
                         dtype: MlxDtype, loc, scale: cfloat,
                         key: MlxArray, s: MlxStream): cint {.importc, header: randomH.}
proc mlx_random_uniform*(res: ptr MlxArray, lo, hi: MlxArray,
                          shape: ptr cint, shape_num: csize_t,
                          dtype: MlxDtype, key: MlxArray,
                          s: MlxStream): cint {.importc, header: randomH.}

# ── Linalg ───────────────────────────────────────────────────────

proc mlx_linalg_norm*(res: ptr MlxArray, a: MlxArray,
                       ord: cdouble, axis: ptr cint, axisNum: csize_t,
                       keepdims: bool, s: MlxStream): cint {.importc, header: linalgH.}

# ── Transforms ───────────────────────────────────────────────────

proc mlx_eval*(arr: MlxArray): cint {.importc: "mlx_array_eval", header: arrH.}

# ── Convenience: eval multiple ───────────────────────────────────

proc mlx_eval_many*(arrs: ptr MlxArray, num: csize_t) =
  ## Evaluate multiple arrays
  for i in 0..<num.int:
    discard mlx_eval(cast[ptr UncheckedArray[MlxArray]](arrs)[i])

# ── Quantization ────────────────────────────────────────────────

proc mlx_quantize*(res: ptr MlxVectorArray, w: MlxArray,
                   group_size: MlxOptionalInt, bits: MlxOptionalInt,
                   mode: cstring, global_scale: MlxArray,
                   s: MlxStream): cint {.importc, header: opsH.}

proc mlx_dequantize*(res: ptr MlxArray, w: MlxArray,
                     scales: MlxArray, biases: MlxArray,
                     group_size: MlxOptionalInt, bits: MlxOptionalInt,
                     mode: cstring, global_scale: MlxArray,
                     dtype: MlxOptionalDtype,
                     s: MlxStream): cint {.importc, header: opsH.}

proc mlx_quantized_matmul*(res: ptr MlxArray, x: MlxArray,
                           w: MlxArray, scales: MlxArray, biases: MlxArray,
                           doTranspose: bool,
                           group_size: MlxOptionalInt, bits: MlxOptionalInt,
                           mode: cstring,
                           s: MlxStream): cint {.importc, header: opsH.}
