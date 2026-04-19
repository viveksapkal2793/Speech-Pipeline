"""DeepFilterNet wrapper with a safe spectral fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.audio import load_audio, normalize_audio, simple_spectral_denoise


@dataclass(slots=True)
class DenoiseResult:
    """Result returned by the denoiser."""

    audio: np.ndarray
    sample_rate: int
    backend: str


class DeepFilterNetDenoiser:
    """Denoise audio with DeepFilterNet when available, otherwise fallback."""

    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate
        self._backend = "fallback"
        self._model: Any | None = None
        self._state: Any | None = None
        self._enhance_fn = None
        self._load_backend()

    def _load_backend(self) -> None:
        try:
            from df.enhance import enhance, init_df  # type: ignore

            loaded = init_df()
            if isinstance(loaded, tuple):
                if len(loaded) == 3:
                    self._model, self._state, _ = loaded
                elif len(loaded) == 2:
                    self._model, self._state = loaded
                else:
                    self._model = loaded[0]
                    self._state = loaded[1] if len(loaded) > 1 else None
            else:
                self._model = loaded
            self._enhance_fn = enhance
            self._backend = "deepfilternet"
        except Exception:
            self._backend = "fallback"

    def denoise(self, audio_or_path: str | Path | np.ndarray) -> DenoiseResult:
        """Return a denoised waveform."""

        audio, sr = load_audio(audio_or_path, sr=self.sample_rate)
        if self._backend == "deepfilternet" and self._enhance_fn is not None and self._model is not None:
            try:
                clean = self._enhance_fn(self._model, self._state, audio)
                clean = np.asarray(clean, dtype=np.float32)
                return DenoiseResult(audio=normalize_audio(clean), sample_rate=sr, backend=self._backend)
            except Exception:
                pass
        clean = simple_spectral_denoise(audio, sr)
        return DenoiseResult(audio=clean, sample_rate=sr, backend="fallback")

