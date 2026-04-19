"""Train the frame-level language identification model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import AudioConfig, LIDConfig
from ..utils.checkpoint import save_checkpoint
from ..utils.device import amp_enabled, resolve_device
from .dataset import LIDManifestDataset, collate_lid_batch
from .model import FrameLIDModel


def build_label_map(manifest_path: str | Path) -> dict[str, int]:
    """Infer a stable label map from the manifest."""

    df = pd.read_csv(manifest_path)
    labels = sorted({str(label) for label in df["label"].dropna().tolist()})
    return {label: idx for idx, label in enumerate(labels)}


def train_model(
    manifest: str | Path,
    output_path: str | Path,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str | None = None,
    num_workers: int = 0,
) -> None:
    """Train and save the LID model."""

    device_t = resolve_device(prefer_gpu=device is None or device != "cpu")
    if device is not None:
        device_t = torch.device(device)
    label_map = build_label_map(manifest)
    dataset = LIDManifestDataset(manifest, label_map=label_map, audio_config=AudioConfig())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_lid_batch)

    model = FrameLIDModel(n_mels=AudioConfig().n_mels, num_classes=len(label_map), hidden_size=LIDConfig().hidden_size)
    model.to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled(device_t))

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for features, labels, _lengths in progress:
            features = features.to(device_t)
            labels = labels.to(device_t)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled(device_t)):
                logits = model(features)
                loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")
        print(f"epoch {epoch + 1}: loss={epoch_loss / max(1, len(loader)):.4f}")

    save_checkpoint(
        output_path,
        {
            "model_state_dict": model.state_dict(),
            "label_map": label_map,
            "audio_config": asdict(AudioConfig()),
            "lid_config": asdict(LIDConfig()),
        },
    )


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Train the frame-level LID model.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    train_model(
        manifest=args.manifest,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
