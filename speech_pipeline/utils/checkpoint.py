"""Checkpoint helpers for torch modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist a checkpoint dictionary with torch.save."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(checkpoint_path))


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a torch checkpoint dictionary."""

    return torch.load(str(path), map_location=map_location)
