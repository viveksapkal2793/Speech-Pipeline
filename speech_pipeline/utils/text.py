"""Text normalization and tokenization helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable

_WORD_RE = re.compile(r"[A-Za-z\u0900-\u097F']+|[0-9]+|[^\w\s]", re.UNICODE)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim the input."""

    return re.sub(r"\s+", " ", text).strip()


def tokenize_words(text: str) -> list[str]:
    """Tokenize mixed-script text into words and punctuation."""

    return _WORD_RE.findall(text)


def detokenize(tokens: Iterable[str]) -> str:
    """Rebuild readable text from token strings."""

    out: list[str] = []
    for token in tokens:
        if not out:
            out.append(token)
            continue
        if re.fullmatch(r"[^\w\s]", token):
            out[-1] = out[-1] + token
        elif out[-1].endswith(" "):
            out.append(token)
        else:
            out.append(" " + token)
    return normalize_whitespace("".join(out))


def is_devanagari(text: str) -> bool:
    """Detect whether a string contains Devanagari script."""

    return bool(re.search(r"[\u0900-\u097F]", text))


def is_english_word(word: str) -> bool:
    """Heuristically detect a Latin-script token."""

    return bool(re.fullmatch(r"[A-Za-z']+", word))

