#!/usr/bin/env python3
"""Analyze all voice samples and generate voices.json for tts.nim."""

import json, sys, os
import numpy as np
import parselmouth
from parselmouth.praat import call

def analyze_voice(wav_path):
    """Extract acoustic features from a WAV file."""
    snd = parselmouth.Sound(wav_path)
    duration = snd.get_total_duration()

    # Pitch (F0)
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
    f0_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    f0_min = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    f0_max = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")

    # Harmonics-to-Noise Ratio (breathiness)
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)

    # Jitter & Shimmer
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    # Spectral center of gravity (brightness)
    spectrum = call(snd, "To Spectrum", "yes")
    cog = call(spectrum, "Get centre of gravity", 2)

    # RMS energy
    rms = call(snd, "Get root-mean-square", 0, 0)

    return {
        "f0_mean": round(f0_mean, 1),
        "f0_std": round(f0_std, 1),
        "f0_range": round(f0_max - f0_min, 1),
        "hnr": round(hnr, 1),
        "jitter_pct": round(jitter * 100, 2),
        "shimmer_pct": round(shimmer * 100, 1),
        "spectral_cog": round(cog, 0),
        "rms": round(rms, 4),
        "duration": round(duration, 2),
    }


def classify_voice(name, f):
    """Map acoustic features to descriptive tags."""
    gender = "female" if name[1] == "f" else "male"
    lang = {"a": "en-US", "b": "en-GB", "z": "zh", "e": "es", "f": "fr",
            "h": "hi", "i": "it", "j": "ja", "p": "pt"}.get(name[0], "en")

    # -- Pitch --
    if gender == "female":
        if f["f0_mean"] < 170:
            pitch_tag = "low-pitched"
        elif f["f0_mean"] < 210:
            pitch_tag = "moderate-pitched"
        elif f["f0_mean"] < 250:
            pitch_tag = "high-pitched"
        else:
            pitch_tag = "very high-pitched"
    else:
        if f["f0_mean"] < 110:
            pitch_tag = "very deep"
        elif f["f0_mean"] < 130:
            pitch_tag = "deep"
        elif f["f0_mean"] < 160:
            pitch_tag = "moderate-pitched"
        else:
            pitch_tag = "high-pitched"

    # -- Expressiveness (F0 variability) --
    if f["f0_std"] < 18:
        expr_tag = "monotone"
    elif f["f0_std"] < 30:
        expr_tag = "steady"
    elif f["f0_std"] < 45:
        expr_tag = "expressive"
    else:
        expr_tag = "very expressive"

    # -- Breathiness (HNR: lower = more breathy) --
    # TTS voices generally have lower HNR than natural speech
    if f["hnr"] < 8:
        breath_tag = "breathy"
    elif f["hnr"] < 12:
        breath_tag = "slightly breathy"
    elif f["hnr"] < 18:
        breath_tag = "smooth"
    else:
        breath_tag = "clear"

    # -- Brightness (spectral center of gravity) --
    if f["spectral_cog"] < 450:
        bright_tag = "dark"
    elif f["spectral_cog"] < 650:
        bright_tag = "warm"
    elif f["spectral_cog"] < 900:
        bright_tag = "balanced"
    else:
        bright_tag = "bright"

    # -- Texture (jitter + shimmer combined) --
    roughness = f["jitter_pct"] + f["shimmer_pct"] / 3
    if roughness > 5:
        texture_tag = "rough"
    elif roughness > 3.5:
        texture_tag = "textured"
    elif roughness > 2:
        texture_tag = "natural"
    else:
        texture_tag = "polished"

    # -- Energy --
    if f["rms"] < 0.025:
        energy_tag = "soft"
    elif f["rms"] < 0.045:
        energy_tag = "moderate"
    elif f["rms"] < 0.065:
        energy_tag = "energetic"
    else:
        energy_tag = "powerful"

    # -- Speaking pace (duration relative to fixed text) --
    # EN text ~17 words, ZH text ~40 chars, normalize differently
    if lang == "zh":
        # Chinese: shorter = faster
        if f["duration"] < 10.5:
            pace_tag = "fast"
        elif f["duration"] < 12.0:
            pace_tag = "moderate"
        else:
            pace_tag = "slow"
    else:
        if f["duration"] < 7.5:
            pace_tag = "fast"
        elif f["duration"] < 8.5:
            pace_tag = "moderate"
        else:
            pace_tag = "slow"

    tags = {
        "pitch": pitch_tag,
        "expressiveness": expr_tag,
        "breathiness": breath_tag,
        "brightness": bright_tag,
        "texture": texture_tag,
        "energy": energy_tag,
        "pace": pace_tag,
    }

    # Build natural language description
    desc_parts = []
    # Start with pitch + gender
    if pitch_tag in ("very deep", "deep", "low-pitched"):
        desc_parts.append(f"{pitch_tag} {gender} voice")
    elif pitch_tag in ("high-pitched", "very high-pitched"):
        desc_parts.append(f"{pitch_tag} {gender} voice")
    else:
        desc_parts.append(f"{gender} voice")

    # Brightness/warmth
    if bright_tag in ("dark", "warm"):
        desc_parts.append(f"with {bright_tag} tone")
    elif bright_tag == "bright":
        desc_parts.append("with bright tone")

    # Breathiness
    if breath_tag == "breathy":
        desc_parts.append("breathy")
    elif breath_tag == "clear":
        desc_parts.append("clear")

    # Expressiveness
    if expr_tag == "monotone":
        desc_parts.append("steady delivery")
    elif expr_tag == "very expressive":
        desc_parts.append("very animated")
    elif expr_tag == "expressive":
        desc_parts.append("expressive")

    # Energy + pace
    ep = []
    if energy_tag in ("energetic", "powerful"):
        ep.append(energy_tag)
    elif energy_tag == "soft":
        ep.append("soft-spoken")
    if pace_tag == "fast":
        ep.append("fast-paced")
    elif pace_tag == "slow":
        ep.append("unhurried")
    if ep:
        desc_parts.append(", ".join(ep))

    description = ", ".join(desc_parts)
    # Capitalize first letter
    description = description[0].upper() + description[1:]

    return {
        "gender": gender,
        "language": lang,
        "tags": tags,
        "description": description,
        "f0_hz": f["f0_mean"],
        "hnr_db": f["hnr"],
        "spectral_cog_hz": f["spectral_cog"],
    }


# Process all voices
all_results = {}

for subdir, label in [("/tmp/voice_samples/en", "EN"), ("/tmp/voice_samples/zh", "ZH")]:
    voices = sorted(f.replace(".wav", "") for f in os.listdir(subdir) if f.endswith(".wav"))
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  {label}: {len(voices)} voices", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    for voice in voices:
        wav_path = os.path.join(subdir, f"{voice}.wav")
        try:
            feat = analyze_voice(wav_path)
            result = classify_voice(voice, feat)
            all_results[voice] = result
            t = result["tags"]
            print(f"  {voice:12s}  {result['f0_hz']:6.1f}Hz  {t['pitch']:16s} {t['brightness']:9s} {t['breathiness']:16s} {t['expressiveness']:15s} {t['energy']:10s} {t['pace']:8s}  | {result['description']}", file=sys.stderr)
        except Exception as e:
            print(f"  {voice}: ERROR {e}", file=sys.stderr)

print(json.dumps(all_results, indent=2))
