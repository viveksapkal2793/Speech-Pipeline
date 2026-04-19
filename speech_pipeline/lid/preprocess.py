"""Preprocessing utilities for pseudo-labeling and chunking LID training data."""

from __future__ import annotations

import argparse
import difflib
from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import unicodedata
from typing import Literal

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from ..config import AudioConfig
from ..utils.audio import load_audio
from ..utils.text import is_devanagari

try:
    import whisper as openai_whisper
except Exception:  # pragma: no cover - optional backend
    openai_whisper = None


@dataclass(slots=True)
class WordTimestamp:
    """A timestamped word returned by Whisper."""

    word: str
    start: float
    end: float
    language: str
    probability: float


DEFAULT_WHISPER_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "whisper-small"

SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


HINDI_ROMAN_HINTS = {
    "hai",
    "hain",
    "tha",
    "thi",
    "the",
    "tha",
    "nahi",
    "nahin",
    "aur",
    "mein",
    "main",
    "tum",
    "aap",
    "hum",
    "ye",
    "yeh",
    "wo",
    "woh",
    "ka",
    "ki",
    "ke",
    "ko",
    "se",
    "par",
    "toh",
    "bhi",
    "bahut",
    "kyun",
    "kyunki",
    "kya",
    "kaise",
    "sab",
    "isko",
    "usko",
    "ismein",
    "usmein",
    "wala",
    "wali",
    "wale",
    "ji",
    "men",
    "me",
    "raha",
    "rahi",
    "rahe",
    "hota",
    "hoti",
    "hote",
    "kar",
    "karta",
    "karti",
    "karte",
    "karna",
    "kiye",
    "kiya",
    "humein",
    "mujhe",
    "mujhse",
    "ham",
    "hum",
    "aap",
    "tum",
    "tumhe",
    "yeh",
    "ye",
    "woh",
    "wo",
    "is",
    "us",
    "se",
    "ko",
    "ka",
    "ki",
    "ke",
    "par",
    "liye",
    "ya",
    "agar",
    "jab",
    "tab",
    "phir",
    "kyun",
    "kyunki",
    "iska",
    "uska",
    "inke",
    "unke",
    "unki",
    "unko",
    "jise",
    "jaisa",
    "sab",
    "kuch",
    "koi",
    "kisi",
    "bahut",
    "nahin",
    "nahi",
    "toh",
    "to",
}

ENGLISH_HINTS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "this",
    "that",
    "these",
    "those",
    "we",
    "you",
    "they",
    "he",
    "she",
    "it",
    "me",
    "my",
    "mine",
    "your",
    "yours",
    "our",
    "ours",
    "their",
    "theirs",
    "him",
    "her",
    "them",
    "be",
    "am",
    "been",
    "being",
    "not",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "will",
    "would",
    "can",
    "could",
    "should",
    "lecture",
    "interview",
    "question",
    "answer",
    "today",
    "because",
    "however",
    "also",
    "important",
    "please",
    "from",
    "into",
    "over",
    "under",
    "before",
    "after",
    "during",
    "because",
    "while",
    "between",
    "through",
    "without",
    "within",
    "about",
    "against",
    "toward",
    "towards",
    "since",
    "until",
}

DEFAULT_DEVANAGARI_ENGLISH_LOANWORDS = {
    "सिस्टम",
    "मॉडल",
    "प्रोसेस",
    "पॉलिसी",
    "क्वेश्चन",
    "इम्पॉर्टेंट",
    "लेक्चर",
    "क्लास",
    "कोर्स",
    "इंटरव्यू",
    "पब्लिक",
    "डेमोक्रेसी",
    "टेक्नोलॉजी",
    "कंप्यूटर",
    "इकोनॉमी",
    "एजुकेशन",
    "नेशन",
    "विकास",
    "साइंस",
    "मैनेजमेंट",
    "इंडस्ट्री",
}


def _clean_token(token: str) -> str:
    """Strip punctuation around a token."""

    return re.sub(r"^[^\w\u0900-\u097F]+|[^\w\u0900-\u097F]+$", "", token.strip())


def _tokenize_transcript(text: str) -> list[str]:
    """Split a transcript into word-like tokens while preserving Hinglish tokens."""

    return [token for token in re.split(r"\s+", text.strip()) if token]


def _load_lexicon(path: str | Path | None) -> set[str]:
    """Load a whitespace-separated or line-separated word lexicon."""

    if path is None:
        return set()
    lexicon_path = Path(path)
    if not lexicon_path.exists():
        raise FileNotFoundError(f"Lexicon file not found: {lexicon_path}")
    text = lexicon_path.read_text(encoding="utf-8", errors="ignore")
    words = set()
    for token in _tokenize_transcript(text):
        cleaned = _clean_token(token)
        if cleaned:
            words.add(cleaned.lower())
    return words


def _load_subtitle_lexicon(path: str | Path | None, romanize_devanagari: bool = False) -> set[str]:
    """Load a lexicon from an SRT or subtitle-style transcript file."""

    if path is None:
        return set()
    subtitle_path = Path(path)
    if not subtitle_path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")
    words: set[str] = set()
    for raw_line in subtitle_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        for token in _tokenize_transcript(line):
            cleaned = _clean_token(token)
            if not cleaned:
                continue
            if romanize_devanagari and is_devanagari(cleaned):
                cleaned = _approx_romanize_devanagari(cleaned)
            words.add(cleaned.lower())
    return words


def _approx_romanize_devanagari(text: str) -> str:
    """Lightweight Romanization for common Devanagari words.

    This is intentionally approximate: it only needs to get close enough to
    compare against a lexicon or detect familiar English loanwords.
    """

    base_map = {
        "अ": "a",
        "आ": "aa",
        "इ": "i",
        "ई": "ii",
        "उ": "u",
        "ऊ": "uu",
        "ए": "e",
        "ऐ": "ai",
        "ओ": "o",
        "औ": "au",
        "ऋ": "ri",
        "क": "k",
        "ख": "kh",
        "ग": "g",
        "घ": "gh",
        "ङ": "n",
        "च": "ch",
        "छ": "chh",
        "ज": "j",
        "झ": "jh",
        "ञ": "n",
        "ट": "t",
        "ठ": "th",
        "ड": "d",
        "ढ": "dh",
        "ण": "n",
        "त": "t",
        "थ": "th",
        "द": "d",
        "ध": "dh",
        "न": "n",
        "प": "p",
        "फ": "ph",
        "ब": "b",
        "भ": "bh",
        "म": "m",
        "य": "y",
        "र": "r",
        "ल": "l",
        "व": "v",
        "श": "sh",
        "ष": "sh",
        "स": "s",
        "ह": "h",
        "क़": "q",
        "ख़": "kh",
        "ग़": "gh",
        "ज़": "z",
        "ड़": "d",
        "ढ़": "dh",
        "फ़": "f",
        "य़": "y",
        "ँ": "",
        "ं": "n",
        "ः": "h",
        "़": "",
        "ॉ": "o",
        "ो": "o",
        "ौ": "au",
        "ा": "a",
        "ि": "i",
        "ी": "i",
        "ु": "u",
        "ू": "u",
        "े": "e",
        "ै": "ai",
        "ृ": "ri",
        "्": "",
        "़": "",
        " ": "",
    }
    output: list[str] = []
    for char in unicodedata.normalize("NFC", text):
        output.append(base_map.get(char, ""))
    roman = "".join(output).lower()
    roman = re.sub(r"(.)\1{2,}", r"\1\1", roman)
    return roman


def _looks_like_english_devanagari(token: str, english_lexicon: set[str] | None = None) -> bool:
    """Detect Hindi-script tokens that are actually English loanwords or transliterations."""

    lexicon = english_lexicon or set()
    devanagari = token.strip()
    if not devanagari or not is_devanagari(devanagari):
        return False

    roman = _approx_romanize_devanagari(devanagari)
    if not roman:
        return False

    candidates = set(ENGLISH_HINTS) | DEFAULT_DEVANAGARI_ENGLISH_LOANWORDS | lexicon
    if roman in candidates:
        return True

    return False


def classify_word_language(
    word: str,
    chunk_language: str | None = None,
    english_lexicon: set[str] | None = None,
    hindi_lexicon: set[str] | None = None,
) -> str | None:
    """Assign a coarse Hindi/English label to one Whisper word."""

    token = _clean_token(word)
    if not token:
        return None
    lower = token.lower()

    if is_devanagari(token):
        if _looks_like_english_devanagari(token, english_lexicon=english_lexicon):
            return "en"
        return "hi"

    if english_lexicon and lower in english_lexicon:
        return "en"
    if hindi_lexicon and lower in hindi_lexicon:
        return "hi"
    if lower in HINDI_ROMAN_HINTS:
        return "hi"
    if lower in ENGLISH_HINTS:
        return "en"

    if re.fullmatch(r"[A-Z]{2,}", token):
        return "en"
    if re.fullmatch(r"[0-9]+([.,][0-9]+)?", token):
        return None
    if re.fullmatch(r"[A-Za-z']+", token):
        if english_lexicon and lower in english_lexicon:
            return "en"
        if hindi_lexicon and lower in hindi_lexicon:
            return "hi"
        if lower.endswith(("wala", "wali", "wale", "ji", "se", "ko", "ka", "ki", "ke")) and lower not in ENGLISH_HINTS:
            return "hi"
        if len(lower) <= 2 and lower not in ENGLISH_HINTS and lower not in HINDI_ROMAN_HINTS:
            return chunk_language
        return "en"

    return chunk_language


def _duration_seconds(audio_path: str | Path) -> float:
    """Get the duration of an audio file without loading it fully."""

    info = sf.info(str(audio_path))
    return float(info.frames / info.samplerate)


def _list_audio_files(folder: str | Path) -> list[Path]:
    """List audio files in a folder in deterministic order."""

    directory = Path(folder)
    if not directory.exists():
        raise FileNotFoundError(f"Chunks directory not found: {directory}")
    files = [path for path in sorted(directory.iterdir()) if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES]
    if not files:
        raise ValueError(f"No audio files found in chunks directory: {directory}")
    return files


def _load_audio_window(audio_path: str | Path, start_sec: float, duration_sec: float, sample_rate: int) -> np.ndarray:
    """Load a single time window from a long audio file."""

    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True, offset=float(start_sec), duration=float(duration_sec))
    return audio.astype(np.float32)


def _load_whisper_asr(model_path: str | Path, sample_rate: int) -> object:
    """Load a local Whisper checkpoint as a Transformers ASR pipeline."""

    model_root = Path(model_path)
    if not model_root.exists():
        raise FileNotFoundError(f"Whisper model directory not found: {model_root}")

    device_index = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(str(model_root), local_files_only=True, torch_dtype=torch_dtype)
    processor = AutoProcessor.from_pretrained(str(model_root), local_files_only=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device_index,
    )
    return asr


def _load_openai_whisper(model_name: str, device: str | None = None, download_root: str | Path | None = None) -> object:
    """Load the official OpenAI Whisper model."""

    if openai_whisper is None:
        raise ImportError("openai-whisper is not installed.")
    whisper_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return openai_whisper.load_model(
        model_name,
        device=whisper_device,
        download_root=str(download_root) if download_root else None,
    )


def _normalize_whisper_chunks(result: object) -> tuple[list[dict[str, object]], str | None, str]:
    """Normalize a Whisper pipeline result into chunk dictionaries."""

    chunk_language = None
    transcript = ""
    raw_chunks: list[object] = []

    if isinstance(result, dict):
        chunk_language = result.get("language")
        transcript = str(result.get("text", "")).strip()
        raw_chunks = list(result.get("chunks", []) or [])
        if not raw_chunks and "segments" in result:
            raw_chunks = list(result.get("segments", []) or [])
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                text = str(item.get("text", item.get("word", ""))).strip()
                timestamp = item.get("timestamp", None)
                if timestamp is None and "start" in item and "end" in item:
                    timestamp = (item.get("start"), item.get("end"))
                probability = item.get("probability", item.get("avg_logprob", 1.0))
                raw_chunks.append({"text": text, "timestamp": timestamp, "probability": probability})
        transcript = " ".join(chunk["text"] for chunk in raw_chunks if chunk.get("text")).strip()

    chunks: list[dict[str, object]] = []
    for item in raw_chunks:
        if isinstance(item, dict):
            words = item.get("words", None)
            if isinstance(words, list) and words:
                for word_item in words:
                    if not isinstance(word_item, dict):
                        continue
                    word_text = str(word_item.get("word", word_item.get("text", ""))).strip()
                    if not word_text:
                        continue
                    timestamp = None
                    if "timestamp" in word_item:
                        timestamp = word_item.get("timestamp", None)
                    elif "start" in word_item and "end" in word_item:
                        timestamp = (word_item.get("start"), word_item.get("end"))
                    probability = word_item.get("probability", word_item.get("avg_logprob", 1.0))
                    chunks.append({"text": word_text, "timestamp": timestamp, "probability": probability})
                continue
            text = str(item.get("text", item.get("word", ""))).strip()
            timestamp = item.get("timestamp", None)
            if timestamp is None and "start" in item and "end" in item:
                timestamp = (item.get("start"), item.get("end"))
            probability = item.get("probability", item.get("avg_logprob", 1.0))
            chunks.append({"text": text, "timestamp": timestamp, "probability": probability})
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            chunks.append({"text": str(item[0]).strip(), "timestamp": item[1], "probability": 1.0})

    return chunks, chunk_language, transcript


def _run_whisper_variant(
    asr_pipe: object,
    audio: np.ndarray,
    *,
    language: str | None,
    return_timestamps: str | bool,
) -> tuple[list[dict[str, object]], str | None, str, object]:
    """Run Whisper with a specific language hint."""

    if hasattr(asr_pipe, "transcribe"):
        transcribe_kwargs: dict[str, object] = {
            "word_timestamps": bool(return_timestamps),
            "verbose": False,
            "fp16": False,
            "temperature": 0.0,
            "best_of": 1,
            "beam_size": 5,
            "condition_on_previous_text": False,
        }
        if language is not None:
            transcribe_kwargs["language"] = language
        try:
            result = asr_pipe.transcribe(audio, **transcribe_kwargs)
        except (ValueError, RuntimeError) as exc:
            message = str(exc).lower()
            if "nan" in message or "invalid values" in message or "categorical" in message:
                try:
                    asr_pipe.to("cpu")
                    transcribe_kwargs["fp16"] = False
                    result = asr_pipe.transcribe(audio, **transcribe_kwargs)
                except Exception:
                    raise exc
            else:
                raise
    else:
        generate_kwargs: dict[str, object] = {"task": "transcribe"}
        if language is not None:
            generate_kwargs["language"] = language
        result = asr_pipe(audio, return_timestamps=return_timestamps, generate_kwargs=generate_kwargs)
    chunks, chunk_language, transcript = _normalize_whisper_chunks(result)
    return chunks, chunk_language, transcript, result


def _json_safe(value: object) -> object:
    """Convert common Whisper return values into JSON-safe structures."""

    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_debug_json(
    debug_dir: str | Path | None,
    chunk_name: str,
    *,
    audio_path: str | Path,
    variant: str,
    duration_sec: float,
    chunk_language: str | None,
    transcript: str,
    normalized_chunks: list[dict[str, object]],
    raw_result: object,
) -> None:
    """Write a Whisper debug dump for inspection."""

    if debug_dir is None:
        return
    debug_root = Path(debug_dir)
    debug_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "audio_path": str(audio_path),
        "variant": variant,
        "duration_sec": float(duration_sec),
        "chunk_language": chunk_language,
        "transcript": transcript,
        "normalized_chunks": _json_safe(normalized_chunks),
        "raw_result": _json_safe(raw_result),
    }
    (debug_root / f"{chunk_name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _distribute_words_over_span(words: list[str], start_sec: float, end_sec: float) -> list[WordTimestamp]:
    """Assign approximate timestamps to words across a detected span."""

    if not words:
        return []
    span = max(0.0, end_sec - start_sec)
    if span <= 0.0:
        return []

    step = span / len(words)
    output: list[WordTimestamp] = []
    for index, word in enumerate(words):
        word_start = start_sec + index * step
        word_end = start_sec + (index + 1) * step
        output.append(
            WordTimestamp(
                word=word,
                start=float(word_start),
                end=float(word_end),
                language="",
                probability=1.0,
            )
        )
    return output


def _infer_chunk_language(
    words: list[str],
    fallback: str | None = None,
    english_lexicon: set[str] | None = None,
    hindi_lexicon: set[str] | None = None,
) -> str | None:
    """Infer a coarse language tag from a token list."""

    hi_score = 0
    en_score = 0
    for token in words:
        guessed = classify_word_language(
            token,
            chunk_language=fallback,
            english_lexicon=english_lexicon,
            hindi_lexicon=hindi_lexicon,
        )
        if guessed == "hi":
            hi_score += 1
        elif guessed == "en":
            en_score += 1
    if hi_score == en_score == 0:
        return fallback
    if hi_score >= en_score:
        return "hi"
    return "en"


def _whisper_word_timestamps(
    asr_pipe: object,
    audio: np.ndarray,
    duration_sec: float,
    english_lexicon: set[str] | None = None,
    hindi_lexicon: set[str] | None = None,
) -> tuple[list[WordTimestamp], str | None, str, str, list[dict[str, object]], object]:
    """Run Whisper and extract pseudo word timestamps.

    The Transformers Whisper pipeline does not always return word-level timestamps
    in every environment/version combination. We therefore:
    1. Prefer direct word chunks when they are available.
    2. Fall back to segment-level chunks and distribute words across the segment.
    3. Fall back again to a uniform distribution across the full clip.
    """

    tried_variants = [
        (None, "word"),
        ("en", "word"),
        ("hi", "word"),
        (None, True),
        ("en", True),
        ("hi", True),
    ]
    chunks: list[dict[str, object]] = []
    chunk_language: str | None = None
    transcript = ""
    selected_variant = "none"
    raw_result: object = None
    for language, timestamps in tried_variants:
        try:
            chunks, chunk_language, transcript, raw_result = _run_whisper_variant(
                asr_pipe,
                audio,
                language=language,
                return_timestamps=timestamps,
            )
        except Exception as exc:
            selected_variant = f"language={language!r},timestamps={timestamps!r},error={type(exc).__name__}"
            continue
        selected_variant = f"language={language!r},timestamps={timestamps!r}"
        if transcript or chunks:
            break

    words: list[WordTimestamp] = []
    if chunks:
        for chunk in chunks:
            chunk_text = str(chunk.get("text", "")).strip()
            timestamp = chunk.get("timestamp", None)
            probability = float(chunk.get("probability", 1.0) or 1.0)

            if isinstance(timestamp, (tuple, list)) and len(timestamp) == 2:
                start, end = float(timestamp[0]), float(timestamp[1])
            else:
                start, end = 0.0, 0.0

            if not chunk_text:
                continue

            chunk_words = _tokenize_transcript(chunk_text)
            if not chunk_words:
                continue

            if len(chunk_words) == 1 and end > start:
                words.append(
                    WordTimestamp(
                        word=chunk_words[0],
                        start=start,
                        end=end,
                        language="",
                        probability=probability,
                    )
                )
                continue

            if end > start:
                distributed = _distribute_words_over_span(chunk_words, start, end)
            else:
                distributed = []
            if distributed:
                words.extend(distributed)

    if not words and transcript:
        transcript_words = _tokenize_transcript(transcript)
        words = _distribute_words_over_span(transcript_words, 0.0, duration_sec)

    if not chunk_language:
        chunk_language = _infer_chunk_language(
            [word.word for word in words],
            fallback=None,
            english_lexicon=english_lexicon,
            hindi_lexicon=hindi_lexicon,
        )

    return words, chunk_language, transcript, selected_variant, chunks, raw_result


def _assign_word_languages(
    words: list[WordTimestamp],
    chunk_language: str | None,
    min_probability: float,
    english_lexicon: set[str] | None = None,
    hindi_lexicon: set[str] | None = None,
) -> list[WordTimestamp]:
    """Attach a coarse language label to each Whisper word."""

    labeled: list[WordTimestamp] = []
    for word in words:
        if word.probability < min_probability:
            continue
        language = classify_word_language(
            word.word,
            chunk_language=chunk_language,
            english_lexicon=english_lexicon,
            hindi_lexicon=hindi_lexicon,
        )
        labeled.append(
            WordTimestamp(
                word=word.word,
                start=word.start,
                end=word.end,
                language=language or (chunk_language or "en"),
                probability=word.probability,
            )
        )
    return labeled


def _words_to_frame_labels(
    words: list[WordTimestamp],
    num_frames: int,
    sample_rate: int,
    hop_length: int,
    silence_threshold_sec: float = 0.15,
) -> np.ndarray:
    """Convert timestamped words into frame-level labels."""

    labels = np.full(num_frames, -100, dtype=np.int64)
    if not words:
        return labels

    last_end = 0.0
    for word in words:
        if word.end <= word.start:
            continue
        if word.start - last_end > silence_threshold_sec:
            last_end = word.start
        frame_start = max(0, int(math.floor(word.start * sample_rate / hop_length)))
        frame_end = min(num_frames, int(math.ceil(word.end * sample_rate / hop_length)))
        if frame_end <= frame_start:
            continue
        value = 1 if word.language == "hi" else 0
        labels[frame_start:frame_end] = value
        last_end = word.end
    return labels


def _majority_label(frame_labels: np.ndarray) -> str:
    """Convert frame labels to a single clip label."""

    valid = frame_labels[frame_labels >= 0]
    if valid.size == 0:
        return "en"
    hi = int(np.sum(valid == 1))
    en = int(np.sum(valid == 0))
    return "hi" if hi > en else "en"


def preprocess_lid_audio(
    audio_path: str | Path,
    output_dir: str | Path,
    whisper_model_path: str | Path = DEFAULT_WHISPER_MODEL_DIR,
    asr_backend: str = "auto",
    openai_whisper_model: str = "small",
    openai_whisper_download_root: str | Path | None = None,
    sample_rate: int = 16_000,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 0.0,
    min_word_probability: float = 0.45,
    debug_dir: str | Path | None = None,
    debug_limit: int = 1,
    start_sec: float | None = None,
    end_sec: float | None = None,
    exclude_start_sec: float | None = None,
    exclude_end_sec: float | None = None,
) -> pd.DataFrame:
    """Chunk a long audio file and generate pseudo frame labels with Whisper."""

    output_root = Path(output_dir)
    chunk_audio_dir = output_root / "chunks"
    frame_label_dir = output_root / "frame_labels"
    chunk_audio_dir.mkdir(parents=True, exist_ok=True)
    frame_label_dir.mkdir(parents=True, exist_ok=True)

    if asr_backend in {"auto", "openai"}:
        try:
            asr_pipe = _load_openai_whisper(openai_whisper_model, download_root=openai_whisper_download_root)
        except Exception as exc:
            if asr_backend == "openai":
                raise RuntimeError(f"Failed to load openai-whisper backend: {exc}") from exc
            asr_pipe = _load_whisper_asr(whisper_model_path, sample_rate=sample_rate)
    else:
        asr_pipe = _load_whisper_asr(whisper_model_path, sample_rate=sample_rate)
    audio_duration = _duration_seconds(audio_path)
    range_start = 0.0 if start_sec is None else max(0.0, float(start_sec))
    range_end = audio_duration if end_sec is None else min(audio_duration, float(end_sec))
    step = max(1e-3, chunk_seconds - overlap_seconds)

    audio_cfg = AudioConfig(sample_rate=sample_rate)
    rows: list[dict[str, object]] = []
    chunk_index = 0
    current = range_start
    while current < range_end:
        window_end = min(range_end, current + chunk_seconds)
        if exclude_start_sec is not None and exclude_end_sec is not None:
            if not (window_end <= exclude_start_sec or current >= exclude_end_sec):
                current += step
                continue
        duration = max(0.0, window_end - current)
        if duration <= 0.0:
            current += step
            continue

        audio = _load_audio_window(audio_path, current, duration, sample_rate)
        if audio.size == 0:
            current += step
            continue

        words, chunk_language, transcript, selected_variant, normalized_chunks, raw_result = _whisper_word_timestamps(
            asr_pipe,
            audio,
            duration_sec=duration,
        )
        labeled_words = _assign_word_languages(
            words,
            chunk_language=chunk_language,
            min_probability=min_word_probability,
        )
        frame_count = int(math.ceil(len(audio) / audio_cfg.hop_length))
        frame_labels = _words_to_frame_labels(labeled_words, frame_count, sample_rate=sample_rate, hop_length=audio_cfg.hop_length)
        clip_label = _majority_label(frame_labels)
        transcript = transcript or " ".join(word.word.strip() for word in labeled_words).strip()

        chunk_name = f"chunk_{chunk_index:06d}"
        audio_out = chunk_audio_dir / f"{chunk_name}.wav"
        labels_out = frame_label_dir / f"{chunk_name}.npy"

        sf.write(str(audio_out), audio, sample_rate)
        np.save(labels_out, frame_labels)
        if chunk_index < debug_limit:
            _write_debug_json(
                debug_dir,
                chunk_name,
                audio_path=audio_path,
                variant=selected_variant,
                duration_sec=duration,
                chunk_language=chunk_language,
                transcript=transcript,
                normalized_chunks=normalized_chunks,
                raw_result=raw_result,
            )
        rows.append(
            {
                "audio_path": str(audio_out),
                "label": clip_label,
                "frame_labels": str(labels_out),
                "start_sec": float(current),
                "end_sec": float(window_end),
                "transcript": transcript,
                "chunk_language": chunk_language or "",
            }
        )

        chunk_index += 1
        current += step

    manifest = pd.DataFrame(rows)
    manifest_path = output_root / "lid_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest


def preprocess_lid_chunks(
    chunks_dir: str | Path,
    output_dir: str | Path,
    whisper_model_path: str | Path = DEFAULT_WHISPER_MODEL_DIR,
    asr_backend: str = "auto",
    openai_whisper_model: str = "small",
    openai_whisper_download_root: str | Path | None = None,
    sample_rate: int = 16_000,
    min_word_probability: float = 0.45,
    debug_dir: str | Path | None = None,
    debug_limit: int = 1,
) -> pd.DataFrame:
    """Generate pseudo frame labels for already-split audio chunks."""

    output_root = Path(output_dir)
    frame_label_dir = output_root / "frame_labels"
    frame_label_dir.mkdir(parents=True, exist_ok=True)

    if asr_backend in {"auto", "openai"}:
        try:
            asr_pipe = _load_openai_whisper(openai_whisper_model, download_root=openai_whisper_download_root)
        except Exception as exc:
            if asr_backend == "openai":
                raise RuntimeError(f"Failed to load openai-whisper backend: {exc}") from exc
            asr_pipe = _load_whisper_asr(whisper_model_path, sample_rate=sample_rate)
    else:
        asr_pipe = _load_whisper_asr(whisper_model_path, sample_rate=sample_rate)
    audio_cfg = AudioConfig(sample_rate=sample_rate)

    rows: list[dict[str, object]] = []
    for chunk_index, audio_path in enumerate(_list_audio_files(chunks_dir)):
        audio, _ = load_audio(str(audio_path), sr=sample_rate)
        duration = float(len(audio) / sample_rate) if len(audio) else 0.0
        if audio.size == 0 or duration <= 0.0:
            continue

        words, chunk_language, transcript, selected_variant, normalized_chunks, raw_result = _whisper_word_timestamps(
            asr_pipe, audio, duration_sec=duration
        )
        labeled_words = _assign_word_languages(
            words,
            chunk_language=chunk_language,
            min_probability=min_word_probability,
        )
        frame_count = int(math.ceil(len(audio) / audio_cfg.hop_length))
        frame_labels = _words_to_frame_labels(labeled_words, frame_count, sample_rate=sample_rate, hop_length=audio_cfg.hop_length)
        clip_label = _majority_label(frame_labels)
        transcript = transcript or " ".join(word.word.strip() for word in labeled_words).strip()

        chunk_name = audio_path.stem
        labels_out = frame_label_dir / f"{chunk_name}.npy"
        np.save(labels_out, frame_labels)
        if chunk_index < debug_limit:
            _write_debug_json(
                debug_dir,
                chunk_name,
                audio_path=audio_path,
                variant=selected_variant,
                duration_sec=duration,
                chunk_language=chunk_language,
                transcript=transcript,
                normalized_chunks=normalized_chunks,
                raw_result=raw_result,
            )
        rows.append(
            {
                "audio_path": str(audio_path),
                "label": clip_label,
                "frame_labels": str(labels_out),
                "start_sec": "",
                "end_sec": "",
                "transcript": transcript,
                "chunk_language": chunk_language or "",
            }
        )

    manifest = pd.DataFrame(rows)
    manifest_path = output_root / "lid_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest


def main() -> None:
    """CLI entry point for LID preprocessing."""

    parser = argparse.ArgumentParser(description="Generate pseudo frame labels for LID training.")
    parser.add_argument("--audio", help="Path to the long WAV file.")
    parser.add_argument("--chunks-dir", help="Directory containing already-split audio chunks.")
    parser.add_argument("--output-dir", required=True, help="Directory where chunks, labels, and manifest will be saved.")
    parser.add_argument(
        "--whisper-model-path",
        default=str(DEFAULT_WHISPER_MODEL_DIR),
        help="Path to the local Hugging Face Whisper checkpoint directory.",
    )
    parser.add_argument(
        "--asr-backend",
        choices=["auto", "openai", "transformers"],
        default="auto",
        help="ASR backend to try first. auto tries openai-whisper first, then transformers.",
    )
    parser.add_argument(
        "--openai-whisper-model",
        default="small",
        help="OpenAI Whisper model name to load when using the openai backend.",
    )
    parser.add_argument(
        "--openai-whisper-download-root",
        default=None,
        help="Optional download root for openai-whisper weights.",
    )
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--overlap-seconds", type=float, default=0.0)
    parser.add_argument("--min-word-probability", type=float, default=0.45)
    parser.add_argument("--debug-dir", default=None, help="Optional directory to write raw Whisper debug JSON files.")
    parser.add_argument("--debug-limit", type=int, default=1, help="Maximum number of chunks to dump in debug mode.")
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--exclude-start-sec", type=float, default=None)
    parser.add_argument("--exclude-end-sec", type=float, default=None)
    args = parser.parse_args()

    if args.chunks_dir:
        if args.audio:
            print("Both --audio and --chunks-dir were provided; using --chunks-dir and skipping long-audio chunking.")
        manifest = preprocess_lid_chunks(
            chunks_dir=args.chunks_dir,
            output_dir=args.output_dir,
            whisper_model_path=args.whisper_model_path,
            asr_backend=args.asr_backend,
            openai_whisper_model=args.openai_whisper_model,
            openai_whisper_download_root=args.openai_whisper_download_root,
            sample_rate=args.sample_rate,
            min_word_probability=args.min_word_probability,
            debug_dir=args.debug_dir,
            debug_limit=args.debug_limit,
        )
    else:
        if not args.audio:
            raise SystemExit("Provide either --audio for long-audio chunking or --chunks-dir for existing chunks.")
        manifest = preprocess_lid_audio(
            audio_path=args.audio,
            output_dir=args.output_dir,
            whisper_model_path=args.whisper_model_path,
            asr_backend=args.asr_backend,
            openai_whisper_model=args.openai_whisper_model,
            openai_whisper_download_root=args.openai_whisper_download_root,
            sample_rate=args.sample_rate,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            min_word_probability=args.min_word_probability,
            debug_dir=args.debug_dir,
            debug_limit=args.debug_limit,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
            exclude_start_sec=args.exclude_start_sec,
            exclude_end_sec=args.exclude_end_sec,
        )
    print(f"Saved {len(manifest)} training chunks to {args.output_dir}")
    print(f"Manifest: {Path(args.output_dir) / 'lid_manifest.csv'}")


if __name__ == "__main__":
    main()
