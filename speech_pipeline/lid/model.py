"""Frame-level CNN + BiLSTM language identification model."""

from __future__ import annotations

import torch
from torch import nn


class FrameLIDModel(nn.Module):
    """Predict Hindi vs English labels for each short-time frame."""

    def __init__(self, n_mels: int = 80, num_classes: int = 2, hidden_size: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.temporal = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [batch, frames, classes]."""

        if features.dim() != 3:
            raise ValueError(f"Expected [batch, time, mels], got {tuple(features.shape)}")
        x = features.transpose(1, 2).unsqueeze(1)
        x = self.frontend(x).squeeze(2).transpose(1, 2)
        x, _ = self.temporal(x)
        return self.head(x)

