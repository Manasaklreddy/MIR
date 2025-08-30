from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display

sns.set(style="whitegrid")


def plot_waveform(y: np.ndarray, sr: int, ax: Optional[plt.Axes] = None):
    """Plot the waveform."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.figure
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return fig


def plot_spectrogram(S_db: np.ndarray, sr: int, hop_length: int, ax: Optional[plt.Axes] = None):
    """Plot a dB-scaled spectrogram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log", ax=ax)
    ax.set_title("Spectrogram (dB)")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    return fig


def plot_chromagram(chroma: np.ndarray, sr: int, hop_length: int, ax: Optional[plt.Axes] = None):
    """Plot a chromagram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.figure
    img = librosa.display.specshow(chroma, x_axis="time", y_axis="chroma", hop_length=hop_length, sr=sr, ax=ax)
    ax.set_title("Chromagram")
    plt.colorbar(img, ax=ax)
    return fig


def plot_beats(y: np.ndarray, sr: int, beat_times: np.ndarray, ax: Optional[plt.Axes] = None):
    """Overlay beat positions on waveform."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.figure
    librosa.display.waveshow(y, sr=sr, alpha=0.7, ax=ax)
    for bt in beat_times:
        ax.axvline(x=bt, color="r", linestyle="--", alpha=0.6)
    ax.set_title("Beats over Waveform")
    ax.set_xlabel("Time (s)")
    return fig