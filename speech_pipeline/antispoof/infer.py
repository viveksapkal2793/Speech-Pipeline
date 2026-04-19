"""Inference for the anti-spoof classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import AudioConfig
from ..utils.audio import chunk_audio, load_audio
from ..utils.device import resolve_device
from .features import extract_lfcc
from .model import LFCCSpoofModel


@dataclass(slots=True)
class SpoofPrediction:
    """Classifier output."""

    label: str
    probabilities: np.ndarray


class AntiSpoofInferencer:
    """Run the spoof classifier on a waveform."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        chunk_seconds: float | None = 8.0,
        chunk_overlap_seconds: float = 1.0,
        max_chunks: int | None = None,
    ) -> None:
        self.device = torch.device(device) if device is not None else resolve_device()
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        self.audio_config = AudioConfig(**checkpoint.get('audio_config', {}))
        self.model = LFCCSpoofModel().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.chunk_seconds = chunk_seconds
        self.chunk_overlap_seconds = max(0.0, chunk_overlap_seconds)
        self.max_chunks = max_chunks

    @torch.no_grad()
    def predict(self, audio_or_path: str | Path | np.ndarray) -> SpoofPrediction:
        """Predict bona fide vs spoof."""

        audio, sr = load_audio(audio_or_path, sr=self.audio_config.sample_rate)
        chunks: list[np.ndarray]
        if self.chunk_seconds is None or self.chunk_seconds <= 0:
            chunks = [audio]
        else:
            chunks = chunk_audio(audio, sr=sr, chunk_seconds=self.chunk_seconds, overlap_seconds=self.chunk_overlap_seconds)

        per_chunk_probs: list[np.ndarray] = []
        for idx, chunk in enumerate(chunks):
            if self.max_chunks is not None and idx >= self.max_chunks:
                break
            if len(chunk) < int(0.2 * sr):
                continue
            lfcc = extract_lfcc(chunk, sr=sr)
            features = torch.from_numpy(lfcc).unsqueeze(0).unsqueeze(0).to(self.device)
            logits = self.model(features)
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
            per_chunk_probs.append(probs)

        if per_chunk_probs:
            probs = np.mean(np.stack(per_chunk_probs, axis=0), axis=0)
        else:
            lfcc = extract_lfcc(audio, sr=sr)
            features = torch.from_numpy(lfcc).unsqueeze(0).unsqueeze(0).to(self.device)
            logits = self.model(features)
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

        label = 'bona_fide' if int(np.argmax(probs)) == 0 else 'spoof'
        return SpoofPrediction(label=label, probabilities=probs)
