"""Prosody transfer based on F0, energy, and DTW alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.interpolate import interp1d

from ..utils.audio import extract_f0_energy, load_audio, normalize_audio


@dataclass(slots=True)
class ProsodyTransferResult:
    """Result of prosody transfer."""

    audio: np.ndarray
    sample_rate: int


class ProsodyTransfer:
    """Match lecture prosody to synthesized audio using DTW and contour scaling."""

    def __init__(self, sample_rate: int = 16_000, hop_length: int = 320, max_dtw_cells: int = 20_000_000) -> None:
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_dtw_cells = max_dtw_cells

    def _contour_stack(self, contours) -> np.ndarray:
        f0 = np.log1p(np.maximum(contours.f0, 0.0))
        energy = np.log1p(np.maximum(contours.energy, 0.0))
        return np.stack([f0, energy], axis=1)

    @staticmethod
    def _linear_warp_path(source_len: int, target_len: int) -> np.ndarray:
        """Approximate monotonic alignment without allocating an O(N*M) matrix."""

        if source_len <= 0 or target_len <= 0:
            return np.zeros((0, 2), dtype=np.int64)
        src_idx = np.linspace(0, source_len - 1, num=target_len).astype(np.int64)
        tgt_idx = np.arange(target_len, dtype=np.int64)
        return np.stack([src_idx, tgt_idx], axis=1)

    def _compute_warp_path(self, source_stack: np.ndarray, target_stack: np.ndarray) -> np.ndarray:
        """Compute DTW path when feasible; otherwise use a linear fallback path."""

        source_len = int(len(source_stack))
        target_len = int(len(target_stack))
        if source_len <= 0 or target_len <= 0:
            return np.zeros((0, 2), dtype=np.int64)

        # librosa DTW internally forms a pairwise distance matrix. Guard against
        # extreme allocations on long recordings.
        if source_len * target_len > self.max_dtw_cells:
            return self._linear_warp_path(source_len, target_len)

        try:
            _distance, paths = librosa.sequence.dtw(X=source_stack.T, Y=target_stack.T, metric="euclidean")
            wp = np.asarray(paths[::-1], dtype=np.int64)
            if len(wp) == 0:
                return self._linear_warp_path(source_len, target_len)
            return wp
        except (MemoryError, np.core._exceptions._ArrayMemoryError):
            return self._linear_warp_path(source_len, target_len)

    def transfer(self, source_audio: str | Path | np.ndarray, target_audio: str | Path | np.ndarray) -> ProsodyTransferResult:
        """Apply a source lecture prosody profile to synthesized speech."""

        source, sr = load_audio(source_audio, sr=self.sample_rate)
        target, _ = load_audio(target_audio, sr=self.sample_rate)
        source_contours = extract_f0_energy(source, sr=sr, hop_length=self.hop_length)
        target_contours = extract_f0_energy(target, sr=sr, hop_length=self.hop_length)
        source_stack = self._contour_stack(source_contours)
        target_stack = self._contour_stack(target_contours)

        if len(source_stack) == 0 or len(target_stack) == 0:
            return ProsodyTransferResult(audio=normalize_audio(target), sample_rate=sr)

        wp = self._compute_warp_path(source_stack, target_stack)

        if len(wp) == 0:
            return ProsodyTransferResult(audio=normalize_audio(target), sample_rate=sr)

        src_idx = wp[:, 0]
        tgt_idx = wp[:, 1]
        aligned_src_energy = source_contours.energy[np.clip(src_idx, 0, len(source_contours.energy) - 1)]
        aligned_tgt_energy = target_contours.energy[np.clip(tgt_idx, 0, len(target_contours.energy) - 1)]
        aligned_src_f0 = source_contours.f0[np.clip(src_idx, 0, len(source_contours.f0) - 1)]
        aligned_tgt_f0 = target_contours.f0[np.clip(tgt_idx, 0, len(target_contours.f0) - 1)]

        voiced_src = aligned_src_f0 > 0
        voiced_tgt = aligned_tgt_f0 > 0
        if np.any(voiced_src) and np.any(voiced_tgt):
            pitch_ratio = np.median(aligned_src_f0[voiced_src]) / max(1e-6, np.median(aligned_tgt_f0[voiced_tgt]))
            semitone_shift = float(12.0 * np.log2(np.clip(pitch_ratio, 0.5, 2.0)))
        else:
            semitone_shift = 0.0

        target_adjusted = target
        if abs(semitone_shift) > 0.25:
            target_adjusted = librosa.effects.pitch_shift(target_adjusted, sr=sr, n_steps=semitone_shift)

        source_duration = len(source) / sr
        target_duration = len(target_adjusted) / sr
        if source_duration > 0 and target_duration > 0:
            rate = target_duration / source_duration
            if 0.5 <= rate <= 2.0 and abs(rate - 1.0) > 0.05:
                target_adjusted = librosa.effects.time_stretch(target_adjusted, rate=rate)

        src_energy_interp = interp1d(
            np.linspace(0.0, 1.0, num=len(aligned_src_energy)),
            aligned_src_energy,
            fill_value="extrapolate",
        )
        tgt_energy_interp = interp1d(
            np.linspace(0.0, 1.0, num=len(aligned_tgt_energy)),
            aligned_tgt_energy,
            fill_value="extrapolate",
        )
        src_energy_curve = src_energy_interp(np.linspace(0.0, 1.0, num=len(target_adjusted)))
        tgt_energy_curve = tgt_energy_interp(np.linspace(0.0, 1.0, num=len(target_adjusted)))
        energy_gain = np.exp(np.clip(src_energy_curve - tgt_energy_curve, -1.5, 1.5))
        adjusted = target_adjusted * energy_gain.astype(np.float32)
        return ProsodyTransferResult(audio=normalize_audio(adjusted), sample_rate=sr)

