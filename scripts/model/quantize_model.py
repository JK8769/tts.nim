"""Quantize a Kokoro MLX model to 4-bit using MLX.

Usage:
  python scripts/quantize_model.py src/res/models/kokoro-mlx src/res/models/kokoro-mlx-q4
  python scripts/quantize_model.py src/res/models/kokoro-zh-mlx src/res/models/kokoro-zh-mlx-q4

Quantizes all weight tensors (>= 64 elements along last dim) to 4-bit.
Stores: key.weight -> key (packed uint32), key.scales (float16), key.biases (float16)
Small tensors and biases are kept as float16.
Voices are kept as float16 (small already).
"""
import os
import sys
import shutil
import mlx.core as mx
from safetensors.torch import load_file
from safetensors import safe_open
import numpy as np

GROUP_SIZE = 64
BITS = 4
MIN_DIM_FOR_QUANT = 64  # Don't quantize if last dim < this


def quantize_weights(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "voices"), exist_ok=True)

    # Copy config.json
    shutil.copy(os.path.join(input_dir, "config.json"),
                os.path.join(output_dir, "config.json"))

    # Find main safetensors file
    st_file = None
    for f in os.listdir(input_dir):
        if f.endswith(".safetensors") and "voice" not in f.lower():
            st_file = os.path.join(input_dir, f)
            break
    if not st_file:
        print("No safetensors file found in", input_dir)
        sys.exit(1)

    print(f"Loading {st_file}...")
    # Use safe_open to get tensor info without loading all to memory
    tensors = {}
    with safe_open(st_file, framework="numpy") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    quantized = {}
    kept_float = 0
    quantized_count = 0
    orig_bytes = 0
    quant_bytes = 0

    for key, np_tensor in sorted(tensors.items()):
        orig_bytes += np_tensor.nbytes
        shape = np_tensor.shape

        # Decide whether to quantize this tensor
        can_quantize = (
            len(shape) >= 2 and
            shape[-1] >= MIN_DIM_FOR_QUANT and
            shape[-1] % GROUP_SIZE == 0 and
            "bias" not in key.split(".")[-1] and
            "gamma" not in key and
            "beta" not in key and
            "LayerNorm" not in key
        )

        if can_quantize:
            w = mx.array(np_tensor.astype(np.float32))
            qw, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
            mx.eval(qw, scales, biases)
            quantized[key] = np.array(qw, copy=False)
            quantized[key + ".scales"] = np.array(scales, copy=False)
            quantized[key + ".biases"] = np.array(biases, copy=False)
            qsize = quantized[key].nbytes + quantized[key + ".scales"].nbytes + quantized[key + ".biases"].nbytes
            quant_bytes += qsize
            quantized_count += 1
        else:
            # Keep as float16 to save space
            quantized[key] = np_tensor.astype(np.float16)
            quant_bytes += quantized[key].nbytes
            kept_float += 1

    # Save quantized model
    out_name = os.path.basename(st_file)
    out_path = os.path.join(output_dir, out_name)

    # Convert to torch tensors for safetensors saving
    import torch
    torch_tensors = {}
    for k, v in quantized.items():
        torch_tensors[k] = torch.from_numpy(v.copy())

    from safetensors.torch import save_file
    save_file(torch_tensors, out_path)

    print(f"Quantized {quantized_count} tensors, kept {kept_float} as float16")
    print(f"Original:  {orig_bytes / 1e6:.1f} MB")
    print(f"Quantized: {quant_bytes / 1e6:.1f} MB ({quant_bytes * 100 / orig_bytes:.0f}%)")
    print(f"Saved: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

    # Copy voices as float16
    voices_dir = os.path.join(input_dir, "voices")
    if os.path.isdir(voices_dir):
        count = 0
        for vf in os.listdir(voices_dir):
            if vf.endswith(".safetensors"):
                src = os.path.join(voices_dir, vf)
                dst = os.path.join(output_dir, "voices", vf)
                with safe_open(src, framework="numpy") as f:
                    vt = {}
                    for key in f.keys():
                        vt[key] = torch.from_numpy(f.get_tensor(key).astype(np.float16).copy())
                save_file(vt, dst)
                count += 1
        print(f"Copied {count} voices as float16")

    # Write quantization config
    import json
    qcfg = {"group_size": GROUP_SIZE, "bits": BITS}
    with open(os.path.join(output_dir, "quantize.json"), "w") as f:
        json.dump(qcfg, f)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python quantize_model.py <input_dir> <output_dir>")
        sys.exit(1)
    quantize_weights(sys.argv[1], sys.argv[2])
