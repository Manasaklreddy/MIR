import os
from typing import Dict
import numpy as np
import soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torch

def separate_stems(y: np.ndarray, sr: int, stems: int = 4) -> Dict[str, np.ndarray]:
    """
    Separate audio into stems using Demucs.

    Args:
        y: Mono audio array
        sr: Sample rate
        stems: 2 or 4 (2: vocals/accompaniment, 4: vocals/drums/bass/other)
    Returns:
        Dict mapping stem name to mono audio array at original sr.
    """
    if stems not in (2, 4):
        raise ValueError("stems must be 2 or 4")

    # Demucs expects stereo, 32-bit float, 44.1kHz
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)  # duplicate mono to stereo
    elif y.shape[0] != 2:
        raise ValueError("Input audio must be mono or stereo.")

    # Resample if needed
    if sr != 44100:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=44100)
        sr = 44100

    # Demucs expects shape (channels, samples)
    model = get_model("htdemucs")
    model.eval()
    with torch.no_grad():
        sources = apply_model(model, torch.tensor(y).unsqueeze(0))[0].cpu().numpy()

    # sources shape: (stems, channels, samples)
    stem_names = model.sources
    result: Dict[str, np.ndarray] = {}
    for i, name in enumerate(stem_names):
        audio = sources[i]
        # Convert to mono by averaging channels
        mono = audio.mean(axis=0)
        result[name] = mono.astype(np.float32)

    # For 2 stems, combine all non-vocals as "accompaniment"
    if stems == 2:
        vocals = result.get("vocals", np.zeros_like(list(result.values())[0]))
        accompaniment = sum([v for k, v in result.items() if k != "vocals"])
        return {"vocals": vocals, "accompaniment": accompaniment}
    return result

def save_stems(stems_audio: Dict[str, np.ndarray], sr: int, out_dir: str) -> Dict[str, str]:
    """Save stems as WAV files in `out_dir`. Returns map of stem->path"""
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    for name, audio in stems_audio.items():
        out_path = os.path.join(out_dir, f"{name}.wav")
        sf.write(out_path, audio, sr, subtype="PCM_16")
        paths[name] = out_path
    return paths