"""Audio loading, framing, feature extraction, and contour utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf
from scipy.interpolate import interp1d


@dataclass(slots=True)
class ProsodyContours:
    """Pitch and energy contours sampled per frame."""

    f0: np.ndarray
    energy: np.ndarray
    times: np.ndarray


def load_audio(path_or_array: str | Path | np.ndarray, sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Load mono audio from disk or pass through an existing waveform."""

    if isinstance(path_or_array, (str, Path)):
        audio, file_sr = librosa.load(str(path_or_array), sr=sr, mono=True)
        return audio.astype(np.float32), file_sr
    audio = np.asarray(path_or_array, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return audio, sr


def save_audio(path: str | Path, audio: np.ndarray, sr: int = 16_000) -> None:
    """Write a waveform to disk."""

    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr)


def normalize_audio(audio: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Peak-normalize a waveform without changing its shape."""

    audio = np.asarray(audio, dtype=np.float32)
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs <= 0:
        return audio
    return np.clip(audio / max_abs * peak, -1.0, 1.0)


def chunk_audio(audio: np.ndarray, sr: int, chunk_seconds: float, overlap_seconds: float = 0.0) -> list[np.ndarray]:
    """Split a waveform into overlapping chunks."""

    chunk_samples = max(1, int(chunk_seconds * sr))
    overlap_samples = max(0, int(overlap_seconds * sr))
    hop = max(1, chunk_samples - overlap_samples)
    chunks = []
    for start in range(0, len(audio), hop):
        end = min(len(audio), start + chunk_samples)
        chunk = audio[start:end]
        if len(chunk) > 0:
            chunks.append(chunk)
        if end >= len(audio):
            break
    return chunks


def log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16_000,
    n_fft: int = 400,
    hop_length: int = 320,
    win_length: int = 400,
    n_mels: int = 80,
    fmin: int = 20,
    fmax: int | None = None,
) -> np.ndarray:
    """Compute a log-mel spectrogram using librosa."""

    audio = np.asarray(audio, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax or sr // 2,
        power=2.0,
    )
    return np.log(np.maximum(mel, 1e-10)).astype(np.float32)


def frame_signal(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Frame a waveform into overlapping windows."""

    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < frame_length:
        pad = np.zeros(frame_length - len(audio), dtype=np.float32)
        audio = np.concatenate([audio, pad])
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    return frames.T.copy()


def rms_energy(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Compute per-frame RMS energy."""

    frames = frame_signal(audio, frame_length, hop_length)
    return np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12).astype(np.float32)


def extract_f0_energy(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 320,
    fmin: float = 50.0,
    fmax: float = 500.0,
    max_chunk_seconds: float = 30.0,
) -> ProsodyContours:
    """Extract F0 and energy contours for prosody transfer."""

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        empty = np.zeros(0, dtype=np.float32)
        return ProsodyContours(f0=empty, energy=empty, times=empty)

    max_chunk_samples = max(hop_length * 4, int(max_chunk_seconds * sr))
    f0_chunks: list[np.ndarray] = []
    energy_chunks: list[np.ndarray] = []
    times_chunks: list[np.ndarray] = []

    overlap = max(hop_length * 8, int(0.5 * sr))
    step = max(1, max_chunk_samples - overlap)
    for start in range(0, len(audio), step):
        end = min(len(audio), start + max_chunk_samples)
        chunk = audio[start:end]
        if chunk.size == 0:
            continue
        chunk_f0 = librosa.yin(chunk, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
        chunk_f0 = np.nan_to_num(chunk_f0, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        chunk_energy = librosa.feature.rms(y=chunk, hop_length=hop_length)[0].astype(np.float32)
        chunk_times = librosa.frames_to_time(np.arange(len(chunk_f0)), sr=sr, hop_length=hop_length).astype(np.float32)

        if start > 0 and len(chunk_f0) > 0:
            trim = min(len(chunk_f0), max(1, int(overlap / hop_length)))
            chunk_f0 = chunk_f0[trim:]
            chunk_energy = chunk_energy[trim:]
            chunk_times = chunk_times[trim:]

        if len(chunk_f0) > 0:
            chunk_times = chunk_times + (start / sr)
            f0_chunks.append(chunk_f0)
            energy_chunks.append(chunk_energy)
            times_chunks.append(chunk_times)

        if end >= len(audio):
            break

    f0 = np.concatenate(f0_chunks) if f0_chunks else np.zeros(0, dtype=np.float32)
    energy = np.concatenate(energy_chunks) if energy_chunks else np.zeros(0, dtype=np.float32)
    times = np.concatenate(times_chunks) if times_chunks else np.zeros(0, dtype=np.float32)
    return ProsodyContours(f0=f0, energy=energy, times=times)


def interpolate_contour(values: np.ndarray, target_length: int) -> np.ndarray:
    """Resize a 1-D contour to a requested length."""

    if len(values) == target_length:
        return values.astype(np.float32)
    if len(values) == 0:
        return np.zeros(target_length, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(values))
    x_new = np.linspace(0.0, 1.0, num=target_length)
    return interp1d(x_old, values, kind="linear", fill_value="extrapolate")(x_new).astype(np.float32)


def estimate_snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    """Estimate SNR in dB from clean and noise signals."""

    clean_power = float(np.mean(np.square(clean)) + 1e-12)
    noise_power = float(np.mean(np.square(noise)) + 1e-12)
    return float(10.0 * np.log10(clean_power / noise_power))


def simple_spectral_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Fallback denoiser using a light spectral subtraction scheme."""

    audio = np.asarray(audio, dtype=np.float32)
    n_fft = 512
    hop_length = 128
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(stft), np.angle(stft)
    noise_frames = max(1, int(0.5 * sr / hop_length))
    noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    reduced = np.maximum(magnitude - 1.5 * noise_profile, 0.1 * magnitude)
    reconstructed = librosa.istft(reduced * np.exp(1j * phase), hop_length=hop_length, length=len(audio))
    return normalize_audio(reconstructed.astype(np.float32))
