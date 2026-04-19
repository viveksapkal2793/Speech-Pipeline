"""Inference helpers for the LID model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import AudioConfig
from ..utils.audio import load_audio, log_mel_spectrogram
from ..utils.device import resolve_device
from .model import FrameLIDModel


@dataclass(slots=True)
class LIDPrediction:
    """Frame-level language prediction output."""

    frame_labels: list[str]
    frame_probabilities: np.ndarray


class LIDInferencer:
    """Run the frame-level language identification model on audio."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.device = torch.device(device) if device is not None else resolve_device()
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        self.label_map: dict[str, int] = checkpoint["label_map"]
        self.id_to_label = {idx: label for label, idx in self.label_map.items()}
        audio_cfg = checkpoint.get("audio_config", {})
        lid_cfg = checkpoint.get("lid_config", {})
        self.audio_config = AudioConfig(**audio_cfg)
        self.model = FrameLIDModel(
            n_mels=self.audio_config.n_mels,
            num_classes=len(self.label_map),
            hidden_size=lid_cfg.get("hidden_size", 128),
            dropout=lid_cfg.get("dropout", 0.2),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, audio_or_path: str | Path | np.ndarray) -> LIDPrediction:
        """Predict a label for each audio frame."""

        audio, sr = load_audio(audio_or_path, sr=self.audio_config.sample_rate)
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
        features = torch.from_numpy(mel.T.copy()).unsqueeze(0).to(self.device)
        logits = self.model(features)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        labels = [self.id_to_label[int(idx)] for idx in np.argmax(probs, axis=-1)]
        return LIDPrediction(frame_labels=labels, frame_probabilities=probs)

