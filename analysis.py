from typing import Dict, Any, Optional, Tuple

import numpy as np
import librosa


def compute_stft(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512, to_db: bool = True) -> Dict[str, Any]:
	"""Compute STFT magnitude and optional dB-scaled spectrogram."""
	S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
	if to_db:
		S_db = librosa.amplitude_to_db(S, ref=np.max)
		return {"S": S, "S_db": S_db, "sr": sr, "hop_length": hop_length}
	return {"S": S, "sr": sr, "hop_length": hop_length}


def detect_pitch_pyinn(y: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 2000.0, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
	"""Estimate fundamental frequency using librosa.pyin.

	Returns f0 array (Hz) with NaNs for unvoiced frames and voiced_prob.
	"""
	f0, voiced_flag, voiced_prob = librosa.pyin(
		y,
		sr=sr,
		fmin=fmin,
		fmax=fmax,
		hop_length=hop_length,
	)
	return f0, voiced_prob


def compute_chromagram(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
	"""Compute a chromagram (CQT-based)."""
	chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
	return chroma 