"""Utility helpers for n-gram language models."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from pathlib import Path

from .text import tokenize_words, normalize_whitespace


@dataclass(slots=True)
class NGramStats:
    """Container for n-gram statistics."""

    order: int
    vocab_size: int


def load_corpus_text(path: str | Path) -> str:
    """Load plain text from a corpus file."""

    return Path(path).read_text(encoding="utf-8", errors="ignore")


def corpus_to_sentences(text: str) -> list[list[str]]:
    """Split a corpus into tokenized sentences."""

    raw_sentences = re.split(r"[.!?\n]+", text)
    sentences = []
    for sentence in raw_sentences:
        tokens = [tok.lower() for tok in tokenize_words(normalize_whitespace(sentence)) if tok.strip()]
        if tokens:
            sentences.append(tokens)
    return sentences

