"""Project-wide configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AudioConfig:
    """Audio processing defaults."""

    sample_rate: int = 16_000
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 320
    n_mels: int = 80
    fmin: int = 20
    fmax: int = 8_000


@dataclass(slots=True)
class LIDConfig:
    """Language identification defaults."""

    num_classes: int = 2
    frame_hop_ms: int = 20
    hidden_size: int = 128
    dropout: float = 0.2


@dataclass(slots=True)
class ASRConfig:
    """ASR and decoding defaults."""

    whisper_model: str = "small"
    beam_size: int = 5
    lm_order: int = 3
    lm_weight: float = 0.25
    length_penalty: float = 0.0
    chunk_seconds: float = 28.0
    overlap_seconds: float = 1.0


@dataclass(slots=True)
class TranslationConfig:
    """Translation defaults."""

    target_lang: str = "hin_Deva"
    source_lang: str = "auto"
    lexicon_path: str | None = None


@dataclass(slots=True)
class TTSConfig:
    """Voice cloning defaults."""

    model_name: str = "tts_models/multilingual/multi-dataset/your_tts"
    language: str = "hi"


@dataclass(slots=True)
class PipelineConfig:
    """Top-level pipeline configuration."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    lid: LIDConfig = field(default_factory=LIDConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)

