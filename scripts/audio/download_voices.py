"""Download all Kokoro voice .pt files from HuggingFace and convert to .safetensors.

Usage:
  python scripts/download_voices.py src/res/models/kokoro-mlx
  python scripts/download_voices.py src/res/models/kokoro-mlx-q4

Downloads voices not already present in the model's voices/ directory.
Converts from PyTorch .pt format to safetensors with key "voice" as float16.
"""
import os
import sys
import torch
import numpy as np
from safetensors.torch import save_file

HF_REPO = "hexgrad/Kokoro-82M"
VOICES_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/voices"

# All known voices from the upstream repo
ALL_VOICES = [
    # American English female
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    # American English male
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck", "am_santa",
    # British English female
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    # British English male
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]


def download_voice(name: str, voices_dir: str) -> bool:
    """Download a single voice .pt file and convert to .safetensors."""
    out_path = os.path.join(voices_dir, f"{name}.safetensors")
    if os.path.exists(out_path):
        return False  # already exists

    url = f"{VOICES_URL}/{name}.pt"
    pt_path = os.path.join(voices_dir, f"{name}.pt")

    # Download
    import urllib.request
    print(f"  Downloading {name}...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, pt_path)
    except Exception as e:
        print(f"FAILED: {e}")
        if os.path.exists(pt_path):
            os.remove(pt_path)
        return False

    # Convert .pt -> .safetensors
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            voice_tensor = data.get("voice", data.get("embedding", list(data.values())[0]))
        else:
            voice_tensor = data
        voice_tensor = voice_tensor.to(torch.float16)
        save_file({"voice": voice_tensor}, out_path)
        os.remove(pt_path)
        print(f"OK ({list(voice_tensor.shape)})")
        return True
    except Exception as e:
        print(f"CONVERT FAILED: {e}")
        if os.path.exists(pt_path):
            os.remove(pt_path)
        if os.path.exists(out_path):
            os.remove(out_path)
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python download_voices.py <model_dir> [--english-only]")
        sys.exit(1)

    model_dir = sys.argv[1]
    english_only = "--english-only" in sys.argv

    voices_dir = os.path.join(model_dir, "voices")
    os.makedirs(voices_dir, exist_ok=True)

    # Filter to English voices only if requested
    voices = [v for v in ALL_VOICES if v[0] in ("a", "b")] if english_only else ALL_VOICES

    existing = set(f.replace(".safetensors", "") for f in os.listdir(voices_dir)
                   if f.endswith(".safetensors"))
    missing = [v for v in voices if v not in existing]

    if not missing:
        print(f"All {len(voices)} voices already present in {voices_dir}")
        return

    print(f"Downloading {len(missing)} missing voices to {voices_dir}...")
    downloaded = 0
    for name in missing:
        if download_voice(name, voices_dir):
            downloaded += 1

    total = len(existing) + downloaded
    print(f"\nDone: {downloaded} new voices downloaded, {total} total in {voices_dir}")


if __name__ == "__main__":
    main()
