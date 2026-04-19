"""Whisper small ASR with custom beam search and n-gram LM rescoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as F

from ..config import ASRConfig
from ..utils.audio import chunk_audio, load_audio, normalize_audio
from ..utils.device import resolve_device
from .ngram_lm import NGramLanguageModel


@dataclass(slots=True)
class BeamState:
    """State for one decoding beam."""

    tokens: list[int]
    acoustic_score: float
    total_score: float
    text: str
    ended: bool = False


class ConstrainedWhisperASR:
    """Transcribe audio using Whisper with custom beam search and LM scoring."""

    def __init__(
        self,
        model_name: str = "small",
        device: str | None = None,
        lm_corpus: str | Path | None = None,
        lm_order: int = 3,
        lm_weight: float = 0.25,
        length_penalty: float = 0.0,
    ) -> None:
        self.device = torch.device(device) if device is not None else resolve_device()
        import whisper

        self.whisper = whisper
        self.model = whisper.load_model(model_name, device=str(self.device))
        self.model.eval()
        self.lm_weight = lm_weight
        self.length_penalty = length_penalty
        self.lm = NGramLanguageModel.from_corpus(lm_corpus, order=lm_order) if lm_corpus else None
        self.config = ASRConfig(whisper_model=model_name, lm_order=lm_order, lm_weight=lm_weight, length_penalty=length_penalty)

    def _tokenizer(self, language: str | None):
        return self.whisper.tokenizer.get_tokenizer(self.model.is_multilingual, language=language, task="transcribe")

    def _initial_tokens(self, language: str | None) -> list[int]:
        tokenizer = self._tokenizer(language)
        if hasattr(tokenizer, "sot_sequence_including_notimestamps"):
            return list(tokenizer.sot_sequence_including_notimestamps)
        return list(tokenizer.sot_sequence)

    def _detect_language(self, mel: torch.Tensor) -> str:
        """Detect a language token for a chunk when none is provided."""

        try:
            if hasattr(self.whisper, "detect_language"):
                detected = self.whisper.detect_language(self.model, mel.unsqueeze(0))
                if isinstance(detected, tuple):
                    for item in detected:
                        if isinstance(item, dict) and item:
                            return max(item, key=item.get)
                        if isinstance(item, str) and item:
                            return item
            if hasattr(self.model, "detect_language"):
                detected = self.model.detect_language(mel.unsqueeze(0))
                if isinstance(detected, tuple):
                    for item in detected:
                        if isinstance(item, dict) and item:
                            return max(item, key=item.get)
                        if isinstance(item, str) and item:
                            return item
        except Exception:
            pass
        return "en"

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove Whisper control tokens and normalize whitespace."""

        cleaned = re.sub(r"<\|[^>]+\|>", " ", text)
        cleaned = cleaned.replace("\u200b", " ")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _decode_beam(self, mel: torch.Tensor, language: str | None, beam_size: int, max_steps: int = 96) -> str:
        tokenizer = self._tokenizer(language)
        audio_features = self.model.embed_audio(mel.unsqueeze(0).to(self.device))
        initial_tokens = self._initial_tokens(language)
        beams = [BeamState(tokens=initial_tokens, acoustic_score=0.0, total_score=0.0, text="", ended=False)]
        eot = tokenizer.eot

        for _step in range(max_steps):
            candidates: list[BeamState] = []
            active = [beam for beam in beams if not beam.ended]
            if not active:
                break
            for beam in active:
                tokens = torch.tensor([beam.tokens], device=self.device, dtype=torch.long)
                logits = self.model.logits(tokens, audio_features)
                next_logprobs = F.log_softmax(logits[:, -1], dim=-1)[0]
                top_logprobs, top_ids = torch.topk(next_logprobs, k=min(beam_size, next_logprobs.shape[-1]))
                for token_id, logprob in zip(top_ids.tolist(), top_logprobs.tolist()):
                    new_tokens = beam.tokens + [token_id]
                    decoded = tokenizer.decode(new_tokens)
                    decoded = self._clean_text(decoded)
                    acoustic_score = beam.acoustic_score + float(logprob)
                    lm_score = self.lm.score_text(decoded) if self.lm is not None else 0.0
                    total_score = acoustic_score + self.lm_weight * lm_score - self.length_penalty * len(new_tokens)
                    candidates.append(
                        BeamState(
                            tokens=new_tokens,
                            acoustic_score=acoustic_score,
                            total_score=total_score,
                            text=decoded,
                            ended=token_id == eot,
                        )
                    )
            if not candidates:
                break
            candidates.sort(key=lambda item: item.total_score, reverse=True)
            beams = candidates[:beam_size]
            if all(beam.ended for beam in beams):
                break

        best = max(beams, key=lambda item: item.total_score)
        return self._clean_text(tokenizer.decode(best.tokens))

    def transcribe(
        self,
        audio_or_path: str | Path | np.ndarray,
        language: str | None = None,
        beam_size: int | None = None,
        chunk_seconds: float | None = None,
        overlap_seconds: float | None = None,
    ) -> str:
        """Transcribe an audio file or waveform into text."""

        config = self.config
        beam_size = beam_size or config.beam_size
        chunk_seconds = chunk_seconds or config.chunk_seconds
        overlap_seconds = overlap_seconds if overlap_seconds is not None else config.overlap_seconds
        audio, sr = load_audio(audio_or_path, sr=16_000)
        chunks = chunk_audio(audio, sr, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds)
        transcripts: list[str] = []
        for chunk in chunks:
            chunk = normalize_audio(chunk)
            padded = self.whisper.pad_or_trim(chunk)
            mel = self.whisper.log_mel_spectrogram(padded).to(self.device)
            chunk_language = language or self._detect_language(mel)
            transcripts.append(self._decode_beam(mel, language=chunk_language, beam_size=beam_size))
        return self._clean_text(" ".join(part.strip() for part in transcripts if part.strip()))

