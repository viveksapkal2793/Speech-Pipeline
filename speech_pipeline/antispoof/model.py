"""CNN classifier for bona fide vs spoof detection."""

from __future__ import annotations

import torch
from torch import nn


class LFCCSpoofModel(nn.Module):
    """Predict whether an utterance is bona fide or spoofed."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return class logits for LFCC features."""

        if features.dim() != 4:
            raise ValueError(f"Expected [batch, channels, coeffs, frames], got {tuple(features.shape)}")
        return self.classifier(self.net(features))

