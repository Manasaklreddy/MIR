import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import librosa


def extract_features(y: np.ndarray, sr: int, hop_length: int = 512, n_mfcc: int = 13) -> Dict[str, Any]:
	"""Extract core MIR features from an audio signal.

	Args:
		y: Audio time-series (mono float32)
		sr: Sample rate
		hop_length: Hop length for STFT-derived features
		n_mfcc: Number of MFCCs

	Returns:
		Dictionary of features with per-frame arrays and global summaries.
	"""
	# Frame-wise features
	mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
	spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
	zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
	rms = librosa.feature.rms(y=y, hop_length=hop_length)

	# Tempo and beats
	tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
	beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

	# Aggregate statistics per feature (global)
	def summarize(x: np.ndarray) -> Dict[str, float]:
		return {
			"mean": float(np.nanmean(x)),
			"std": float(np.nanstd(x)),
			"min": float(np.nanmin(x)),
			"max": float(np.nanmax(x)),
		}

	features: Dict[str, Any] = {
		"mfcc": mfcc,  # shape (n_mfcc, frames)
		"spectral_centroid": spectral_centroid,  # (1, frames)
		"zcr": zcr,  # (1, frames)
		"rms": rms,  # (1, frames)
		"tempo": float(tempo),
		"beat_frames": beat_frames,
		"beat_times": beat_times,
		"summary": {
			"mfcc": summarize(mfcc),
			"spectral_centroid": summarize(spectral_centroid),
			"zcr": summarize(zcr),
			"rms": summarize(rms),
		},
		"meta": {
			"sr": sr,
			"hop_length": hop_length,
			"n_mfcc": n_mfcc,
		},
	}
	return features


def features_to_dataframe(features: Dict[str, Any]) -> pd.DataFrame:
	"""Flatten per-frame features into a tabular DataFrame for CSV export.

	Each row corresponds to a frame index with MFCCs, spectral centroid, ZCR, RMS.
	"""
	mfcc = features["mfcc"]  # (n_mfcc, frames)
	spectral_centroid = features["spectral_centroid"][0]
	zcr = features["zcr"][0]
	rms = features["rms"][0]
	frames = mfcc.shape[1]
	n_mfcc = mfcc.shape[0]

	data = {f"mfcc_{i+1}": mfcc[i] for i in range(n_mfcc)}
	data.update({
		"spectral_centroid": spectral_centroid[:frames],
		"zcr": zcr[:frames],
		"rms": rms[:frames],
		"frame": np.arange(frames),
	})
	return pd.DataFrame(data)


def save_features_csv(features: Dict[str, Any], out_csv_path: str) -> str:
	"""Save features to CSV at `out_csv_path`. Returns the path."""
	df = features_to_dataframe(features)
	os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
	df.to_csv(out_csv_path, index=False)
	return out_csv_path


def load_audio(path: str, sr: Optional[int] = None) -> tuple[np.ndarray, int]:
	"""Load an audio file as mono float32 using librosa.load.

	Args:
		path: Path to audio file
		sr: Target sample rate; if None, uses file's native rate
	"""
	y, sr_ret = librosa.load(path, sr=sr, mono=True)
	return y, sr_ret 