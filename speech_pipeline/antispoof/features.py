"""LFCC feature extraction."""

from __future__ import annotations

import numpy as np
from scipy.fft import dct
import librosa


def _linear_filterbank(sr: int, n_fft: int, n_filters: int, fmin: float, fmax: float) -> np.ndarray:
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    edges = np.linspace(fmin, fmax, n_filters + 2)
    filters = np.zeros((n_filters, len(freqs)), dtype=np.float32)
    for idx in range(n_filters):
        left, center, right = edges[idx : idx + 3]
        left_slope = (freqs - left) / max(center - left, 1e-6)
        right_slope = (right - freqs) / max(right - center, 1e-6)
        filters[idx] = np.maximum(0.0, np.minimum(left_slope, right_slope))
    return filters


def extract_lfcc(
    audio: np.ndarray,
    sr: int = 16_000,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_filters: int = 40,
    n_coeffs: int = 20,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Compute LFCC features for anti-spoofing."""

    audio = np.asarray(audio, dtype=np.float32)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    power = np.abs(stft) ** 2
    filterbank = _linear_filterbank(sr, n_fft, n_filters=n_filters, fmin=fmin, fmax=fmax or sr / 2.0)
    filtered = np.dot(filterbank, power)
    log_filtered = np.log(np.maximum(filtered, 1e-10))
    cepstra = dct(log_filtered, type=2, axis=0, norm="ortho")[:n_coeffs]
    return cepstra.astype(np.float32)

