"""Device helpers."""

from __future__ import annotations

import torch


def resolve_device(prefer_gpu: bool = True) -> torch.device:
    """Return the best available torch device."""

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and torch.backends.mps.is_available():  # pragma: no cover - macOS only
        return torch.device("mps")
    return torch.device("cpu")


def amp_enabled(device: torch.device) -> bool:
    """Whether mixed precision is worth enabling on the current device."""

    return device.type == "cuda"

