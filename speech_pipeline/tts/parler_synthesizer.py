"""Indic Parler-TTS synthesis backend.

This backend avoids the IndicF5/Vocos path entirely and uses Parler-TTS's
prompt+description interface. It does not require a reference speaker
embedding, but it can still fit into the pipeline as the synthesis stage for
Marathi and other supported Indic languages.
"""

from __future__ import annotations

from dataclasses import dataclass
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


class ParlerSynthesizer:
    """Generate speech with Indic Parler-TTS.

    The model takes two text inputs:
    - prompt_text: the text to synthesize
    - description: a style/speaker description

    The backend does not use a speaker embedding or speaker reference audio.
    """

    def __init__(
        self,
        model_name_or_path: str = "ai4bharat/indic-parler-tts",
        device: str | None = None,
        description: str | None = None,
    ) -> None:
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
        except Exception as exc:  # pragma: no cover - import-time dependency failure
            raise ImportError(
                "ParlerSynthesizer requires the `parler-tts` package. "
                "Install it with: pip install git+https://github.com/huggingface/parler-tts.git"
            ) from exc
        from transformers import AutoTokenizer

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.description_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.sample_rate = int(getattr(self.model.config, "sampling_rate", 24000))
        self.description = description
        config_limit = int(getattr(self.model.config, "max_position_embeddings", 256) or 256)
        tokenizer_limit = int(getattr(self.prompt_tokenizer, "model_max_length", config_limit) or config_limit)
        if tokenizer_limit > 100_000:
            tokenizer_limit = config_limit
        self.max_prompt_tokens = max(32, min(config_limit, tokenizer_limit, 256))

    def _default_description(self, language: str | None = None) -> str:
        """Build a reasonable prompt description for Indic Parler-TTS."""

        lang = str(language or "").strip().lower()
        if lang.startswith("mar"):
            return (
                "A clear Marathi speaker delivers a calm, natural lecture in a close-sounding studio recording "
                "with very high quality and no background noise."
            )
        if lang.startswith("hin"):
            return (
                "A clear Hindi speaker delivers a calm, natural lecture in a close-sounding studio recording "
                "with very high quality and no background noise."
            )
        return (
            "A clear Indian speaker delivers a calm, natural lecture in a close-sounding studio recording "
            "with very high quality and no background noise."
        )

    def synthesize(
        self,
        text: str,
        speaker_embedding: np.ndarray | None = None,
        speaker_reference: str | Path | None = None,
        reference_text: str | None = None,
        output_path: str | Path | None = None,
        language: str | None = None,
    ) -> SynthesizedSpeech:
        """Synthesize speech using Indic Parler-TTS."""

        if output_path is not None:
            output = Path(output_path)
        else:
            fd, temp_name = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            output = Path(temp_name)

        description = self.description or self._default_description(language)
        prompt = text.strip()
        if not prompt:
            raise ValueError("Indic Parler-TTS requires non-empty text.")

        description_inputs = self.description_tokenizer(description, return_tensors="pt").to(self.device)
        prompt_inputs = self.prompt_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_tokens,
        ).to(self.device)

        with torch.inference_mode():
            generation = self.model.generate(
                input_ids=description_inputs.input_ids,
                attention_mask=description_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
            )

        waveform = generation.detach().cpu().numpy().squeeze()
        waveform = np.asarray(waveform, dtype=np.float32)
        save_audio(output, waveform, self.sample_rate)
        return SynthesizedSpeech(path=output, audio=waveform, sample_rate=self.sample_rate)
