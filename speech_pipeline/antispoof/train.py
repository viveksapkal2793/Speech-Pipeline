"""Train the anti-spoof CNN classifier."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import AudioConfig
from ..utils.checkpoint import save_checkpoint
from ..utils.device import amp_enabled, resolve_device
from .dataset import AntiSpoofManifestDataset, collate_antispoof_batch
from .model import LFCCSpoofModel


def train_model(
    manifest: str | Path,
    output_path: str | Path,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str | None = None,
    num_workers: int = 0,
    chunk_seconds: float | None = None,
    chunk_overlap_seconds: float = 0.0,
    max_chunks_per_file: int | None = None,
) -> None:
    """Train and save the spoof detector."""

    device_t = torch.device(device) if device is not None else resolve_device()
    dataset = AntiSpoofManifestDataset(
        manifest,
        audio_config=AudioConfig(),
        chunk_seconds=chunk_seconds,
        chunk_overlap_seconds=chunk_overlap_seconds,
        max_chunks_per_file=max_chunks_per_file,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_antispoof_batch)

    model = LFCCSpoofModel().to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled(device_t))

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        progress = tqdm(loader, desc=f'epoch {epoch + 1}/{epochs}', leave=False)
        for features, labels in progress:
            features = features.to(device_t)
            labels = labels.to(device_t)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled(device_t)):
                logits = model(features)
                loss = nn.functional.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            progress.set_postfix(loss=f'{loss.item():.4f}')
        print(f'epoch {epoch + 1}: loss={running_loss / max(1, len(loader)):.4f}')

    save_checkpoint(
        output_path,
        {
            'model_state_dict': model.state_dict(),
            'audio_config': asdict(AudioConfig()),
        },
    )


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description='Train the anti-spoof classifier.')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--chunk-seconds', type=float, default=None, help='Optional chunk duration in seconds for memory-safe training.')
    parser.add_argument('--chunk-overlap-seconds', type=float, default=0.0, help='Overlap between consecutive training chunks.')
    parser.add_argument('--max-chunks-per-file', type=int, default=None, help='Optional cap on chunks sampled from each file.')
    args = parser.parse_args()
    train_model(
        manifest=args.manifest,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
        chunk_seconds=args.chunk_seconds,
        chunk_overlap_seconds=args.chunk_overlap_seconds,
        max_chunks_per_file=args.max_chunks_per_file,
    )


if __name__ == '__main__':
    main()
