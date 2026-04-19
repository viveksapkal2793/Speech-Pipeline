"""IndicF5-based multilingual speech synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile

import numpy as np
import torch

from ..utils.audio import save_audio


@dataclass(slots=True)
class SynthesizedSpeech:
    """Synthesized waveform and output path."""

    path: Path
    audio: np.ndarray
    sample_rate: int


class IndicF5Synthesizer:
    """Generate Marathi/Indic speech using IndicF5 with a reference prompt audio."""

    @contextmanager
    def _disable_torch_compile(self):
        """Temporarily disable torch.compile for IndicF5 model initialization."""

        original_compile = getattr(torch, "compile", None)

        def _identity_compile(model, *args, **kwargs):
            return model

        if original_compile is not None:
            torch.compile = _identity_compile  # type: ignore[assignment]
        try:
            yield
        finally:
            if original_compile is not None:
                torch.compile = original_compile  # type: ignore[assignment]

    @contextmanager
    def _force_non_meta_loading(self):
        """Force non-meta tensor initialization while loading IndicF5.

        Transformers can sometimes instantiate remote-code models under a
        meta-tensor / low-memory loading context. That breaks vocoder setup in
        IndicF5 on Windows. We temporarily force a CPU default device and turn
        off low-memory loading so normal tensors are created during model init.
        """

        set_default_device = getattr(torch, "set_default_device", None)
        restore_device = None
        if set_default_device is not None:
            try:
                restore_device = torch.get_default_device()
            except Exception:
                restore_device = None
            set_default_device("cpu")
        try:
            yield
        finally:
            if set_default_device is not None and restore_device is not None:
                set_default_device(restore_device)

    def __init__(self, model_name_or_path: str = "ai4bharat/IndicF5", device: str | None = None) -> None:
        from transformers import AutoConfig, AutoModel

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        with self._disable_torch_compile(), self._force_non_meta_loading():
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            try:
                config.name_or_path = model_name_or_path
            except Exception:
                pass
            try:
                self.model = AutoModel.from_config(config, trust_remote_code=True)
            except Exception:
                self.model = AutoModel.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    low_cpu_mem_usage=False,
                    device_map=None,
                    torch_dtype=torch.float32,
                )
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def synthesize(
        self,
        text: str,
        speaker_embedding: np.ndarray | None = None,
        speaker_reference: str | Path | None = None,
        reference_text: str | None = None,
        output_path: str | Path | None = None,
    ) -> SynthesizedSpeech:
        """Synthesize speech using IndicF5.

        IndicF5 requires a prompt/reference audio and the transcript spoken in that
        reference audio.
        """

        if speaker_reference is None:
            raise ValueError("IndicF5 requires a reference prompt audio.")
        if not reference_text or not reference_text.strip():
            raise ValueError("IndicF5 requires reference_text for the prompt audio.")

        if output_path is not None:
            output = Path(output_path)
        else:
            fd, temp_name = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            output = Path(temp_name)

        with torch.no_grad():
            generated = self.model(
                text,
                ref_audio_path=str(speaker_reference),
                ref_text=reference_text.strip(),
            )

        waveform = generated
        if isinstance(generated, dict):
            waveform = generated.get("audio", generated.get("waveform", generated))
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        waveform = np.asarray(waveform)
        if waveform.dtype == np.int16:
            waveform = waveform.astype(np.float32) / 32768.0
        else:
            waveform = waveform.astype(np.float32)

        sample_rate = 24_000
        save_audio(output, waveform, sample_rate)
        return SynthesizedSpeech(path=output, audio=waveform, sample_rate=sample_rate)
