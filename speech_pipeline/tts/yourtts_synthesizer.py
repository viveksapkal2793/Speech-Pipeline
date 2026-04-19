"""Voice cloning synthesis using Coqui YourTTS."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from pathlib import Path
import os
import tempfile

import numpy as np

from ..utils.audio import load_audio, save_audio


@dataclass(slots=True)
class SynthesizedSpeech:
    """Synthesized waveform and output path."""

    path: Path
    audio: np.ndarray
    sample_rate: int


class YourTTSSynthesizer:
    """Generate speech in the target language while cloning a reference voice."""

    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/your_tts", device: str | None = None) -> None:
        from TTS.api import TTS

        self.tts = TTS(model_name=model_name, progress_bar=False, gpu=device == "cuda")
        self.device = device
        self.supported_languages = self._get_supported_languages()

    def _get_supported_languages(self) -> set[str]:
        """Return the language tags supported by the loaded TTS model, if available."""

        candidates = [
            getattr(self.tts, "languages", None),
            getattr(getattr(self.tts, "synthesizer", None), "languages", None),
            getattr(getattr(getattr(self.tts, "synthesizer", None), "tts_model", None), "languages", None),
            getattr(getattr(getattr(self.tts, "synthesizer", None), "tts_model", None), "language_manager", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            if isinstance(candidate, dict):
                return {str(key).lower() for key in candidate.keys()}
            if hasattr(candidate, "name_to_id"):
                return {str(key).lower() for key in getattr(candidate, "name_to_id", {}).keys()}
            if isinstance(candidate, (list, tuple, set)):
                return {str(item).lower() for item in candidate}
        return set()

    def _normalize_language(self, language: str | None) -> str | None:
        """Only return a language code if the model advertises support for it."""

        if language is None:
            return None
        language = str(language).strip().lower()
        if not language:
            return None
        if self.supported_languages and language not in self.supported_languages:
            return None
        return language

    def synthesize(
        self,
        text: str,
        speaker_embedding: np.ndarray | None = None,
        speaker_reference: str | Path | None = None,
        language: str = "en",
        output_path: str | Path | None = None,
    ) -> SynthesizedSpeech:
        """Synthesize speech using the closest supported conditioning path."""

        if output_path is not None:
            output = Path(output_path)
        else:
            fd, temp_name = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            output = Path(temp_name)
        kwargs = {"text": text}

        if speaker_reference is not None:
            kwargs["speaker_wav"] = str(speaker_reference)
        elif speaker_embedding is not None and "speaker_embedding" in signature(self.tts.tts).parameters:
            kwargs["speaker_embedding"] = speaker_embedding
        else:
            raise ValueError(
                "YourTTS needs either a reference speaker wav or a backend that accepts speaker_embedding."
            )

        normalized_language = self._normalize_language(language)
        if "language" in signature(self.tts.tts).parameters and normalized_language is not None:
            kwargs["language"] = normalized_language

        try:
            audio = self.tts.tts(**kwargs)
        except Exception:
            if speaker_reference is None:
                raise
            fallback_kwargs = {"text": text, "speaker_wav": str(speaker_reference), "file_path": str(output)}
            if normalized_language is not None:
                fallback_kwargs["language"] = normalized_language
            self.tts.tts_to_file(**fallback_kwargs)
            loaded_audio, sr = load_audio(output, sr=22_050)
            return SynthesizedSpeech(path=output, audio=loaded_audio, sample_rate=sr)

        if isinstance(audio, tuple):
            waveform, sample_rate = audio
        else:
            waveform = audio
            sample_rate = getattr(self.tts.synthesizer, "output_sample_rate", 22_050)
        waveform = np.asarray(waveform, dtype=np.float32)
        save_audio(output, waveform, int(sample_rate))
        return SynthesizedSpeech(path=output, audio=waveform, sample_rate=int(sample_rate))

