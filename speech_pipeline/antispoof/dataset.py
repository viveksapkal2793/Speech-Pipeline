"""Datasets for anti-spoof training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from ..config import AudioConfig
from .features import extract_lfcc


@dataclass(slots=True)
class AntiSpoofExample:
    """A single anti-spoof training example."""

    features: torch.Tensor
    label: torch.Tensor


class AntiSpoofManifestDataset(Dataset[AntiSpoofExample]):
    """CSV-based anti-spoof dataset."""

    def __init__(
        self,
        manifest_path: str | Path,
        audio_config: AudioConfig | None = None,
        chunk_seconds: float | None = None,
        chunk_overlap_seconds: float = 0.0,
        max_chunks_per_file: int | None = None,
    ) -> None:
        self.manifest = pd.read_csv(manifest_path)
        self.audio_config = audio_config or AudioConfig()
        self.chunk_seconds = chunk_seconds
        self.chunk_overlap_seconds = max(0.0, chunk_overlap_seconds)
        self.max_chunks_per_file = max_chunks_per_file
        self.rows: list[dict[str, object]] = self._build_rows()

    def _build_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        has_segments = {'start_sec', 'end_sec'}.issubset(set(self.manifest.columns))
        if has_segments:
            for _, row in self.manifest.iterrows():
                rows.append(
                    {
                        'audio_path': str(row['audio_path']),
                        'label': int(row['label']),
                        'start_sec': float(row['start_sec']),
                        'end_sec': float(row['end_sec']),
                    }
                )
            return rows

        if self.chunk_seconds is None or self.chunk_seconds <= 0:
            for _, row in self.manifest.iterrows():
                rows.append({'audio_path': str(row['audio_path']), 'label': int(row['label'])})
            return rows

        hop = max(1e-3, self.chunk_seconds - self.chunk_overlap_seconds)
        for _, row in self.manifest.iterrows():
            audio_path = Path(str(row['audio_path']))
            info = sf.info(str(audio_path))
            duration = float(info.frames) / float(info.samplerate)
            if duration <= 0:
                continue
            starts = np.arange(0.0, duration, hop, dtype=np.float64)
            if self.max_chunks_per_file is not None and self.max_chunks_per_file > 0:
                starts = starts[: self.max_chunks_per_file]
            label = int(row['label'])
            for start in starts:
                end = min(duration, float(start + self.chunk_seconds))
                if (end - start) < 0.2:
                    continue
                rows.append(
                    {
                        'audio_path': str(audio_path),
                        'label': label,
                        'start_sec': float(start),
                        'end_sec': float(end),
                    }
                )
        return rows

    def _load_audio_segment(self, row: dict[str, object]) -> tuple[np.ndarray, int]:
        path = str(row['audio_path'])
        target_sr = self.audio_config.sample_rate
        if 'start_sec' not in row or 'end_sec' not in row:
            audio, sr = sf.read(path, dtype='float32', always_2d=False)
        else:
            info = sf.info(path)
            start_frame = max(0, int(float(row['start_sec']) * info.samplerate))
            end_frame = min(info.frames, int(float(row['end_sec']) * info.samplerate))
            num_frames = max(0, end_frame - start_frame)
            if num_frames == 0:
                return np.zeros(1, dtype=np.float32), target_sr
            audio, sr = sf.read(path, start=start_frame, frames=num_frames, dtype='float32', always_2d=False)

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
            sr = target_sr
        return audio, sr

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> AntiSpoofExample:
        row = self.rows[idx]
        audio, sr = self._load_audio_segment(row)
        lfcc = extract_lfcc(audio, sr=sr)
        features = torch.from_numpy(lfcc).unsqueeze(0)
        label = torch.tensor(int(row['label']), dtype=torch.long)
        return AntiSpoofExample(features=features, label=label)


def collate_antispoof_batch(batch: list[AntiSpoofExample]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad LFCC features to a common time dimension."""

    features = [item.features.squeeze(0).transpose(0, 1) for item in batch]
    labels = torch.stack([item.label for item in batch])
    padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    padded = padded.transpose(1, 2).unsqueeze(1)
    return padded, labels

