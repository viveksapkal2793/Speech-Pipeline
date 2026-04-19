"""Datasets for frame-level language identification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import AudioConfig
from ..utils.audio import load_audio, log_mel_spectrogram


@dataclass(slots=True)
class LIDExample:
    """A single LID training example."""

    features: torch.Tensor
    labels: torch.Tensor


def _parse_frame_labels(value: str, label_map: dict[str, int]) -> np.ndarray | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.exists() and path.suffix.lower() == ".npy":
        loaded = np.load(path, allow_pickle=True)
        return loaded.astype(np.int64)
    if path.exists() and path.suffix.lower() == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return np.asarray([label_map.get(str(item), int(item)) for item in loaded], dtype=np.int64)
    delimiter = ";" if ";" in value else "," if "," in value else " "
    parts = [part.strip() for part in value.split(delimiter) if part.strip()]
    if not parts:
        return None
    parsed: list[int] = []
    for part in parts:
        if part in label_map:
            parsed.append(label_map[part])
        else:
            parsed.append(int(part))
    return np.asarray(parsed, dtype=np.int64)


def _resize_discrete_labels(labels: np.ndarray, target_length: int) -> np.ndarray:
    """Resize a discrete label sequence without inventing intermediate classes."""

    if len(labels) == target_length:
        return labels.astype(np.int64)
    if len(labels) == 0:
        return np.full(target_length, -100, dtype=np.int64)
    if target_length <= 0:
        return np.asarray([], dtype=np.int64)
    source_idx = np.linspace(0, len(labels) - 1, num=target_length)
    source_idx = np.rint(source_idx).astype(np.int64)
    source_idx = np.clip(source_idx, 0, len(labels) - 1)
    return labels[source_idx].astype(np.int64)


class LIDManifestDataset(Dataset[LIDExample]):
    """CSV-based dataset for LID training."""

    def __init__(
        self,
        manifest_path: str | Path,
        label_map: dict[str, int],
        audio_config: AudioConfig | None = None,
    ) -> None:
        self.manifest = pd.read_csv(manifest_path)
        self.label_map = label_map
        self.audio_config = audio_config or AudioConfig()

    def __len__(self) -> int:
        return len(self.manifest)

    def _encode_label(self, value: object) -> int:
        if isinstance(value, str) and value in self.label_map:
            return self.label_map[value]
        return int(value)

    def __getitem__(self, idx: int) -> LIDExample:
        row = self.manifest.iloc[idx]
        audio, sr = load_audio(row["audio_path"], sr=self.audio_config.sample_rate)
        mel = log_mel_spectrogram(
            audio,
            sr=sr,
            n_fft=self.audio_config.n_fft,
            hop_length=self.audio_config.hop_length,
            win_length=self.audio_config.win_length,
            n_mels=self.audio_config.n_mels,
            fmin=self.audio_config.fmin,
            fmax=self.audio_config.fmax,
        )
        mel = torch.from_numpy(mel.T.copy())  # [frames, mels]

        frame_labels = None
        if "frame_labels" in self.manifest.columns and pd.notna(row.get("frame_labels", None)):
            frame_labels = _parse_frame_labels(str(row["frame_labels"]), self.label_map)

        if frame_labels is None:
            label_id = self._encode_label(row["label"])
            labels = torch.full((mel.shape[0],), label_id, dtype=torch.long)
        else:
            labels_np = np.asarray(frame_labels, dtype=np.int64)
            if len(labels_np) != mel.shape[0]:
                labels_np = _resize_discrete_labels(labels_np, mel.shape[0])
            labels = torch.from_numpy(labels_np)
        return LIDExample(features=mel, labels=labels)


def collate_lid_batch(batch: list[LIDExample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length LID examples."""

    features = [item.features for item in batch]
    labels = [item.labels for item in batch]
    lengths = torch.tensor([feat.shape[0] for feat in features], dtype=torch.long)
    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return padded_features, padded_labels, lengths
