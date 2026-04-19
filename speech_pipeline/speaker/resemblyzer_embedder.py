"""Speaker embeddings with Resemblyzer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils.audio import load_audio, normalize_audio


class ResemblyzerEmbedder:
    """Compute a fixed-dimensional speaker embedding from a reference recording."""

    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate
        from resemblyzer import VoiceEncoder, preprocess_wav

        self._VoiceEncoder = VoiceEncoder
        self._preprocess_wav = preprocess_wav
        self.encoder = VoiceEncoder()

    def embed(self, audio_or_path: str | Path | np.ndarray) -> np.ndarray:
        """Return a speaker embedding vector."""

        if isinstance(audio_or_path, (str, Path)):
            wav = self._preprocess_wav(str(audio_or_path))
        else:
            audio, _ = load_audio(audio_or_path, sr=self.sample_rate)
            wav = self._preprocess_wav(normalize_audio(audio))
        embedding = self.encoder.embed_utterance(wav)
        return np.asarray(embedding, dtype=np.float32)

