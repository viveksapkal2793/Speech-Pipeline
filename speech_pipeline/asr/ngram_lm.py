"""N-gram language model used to constrain Whisper decoding."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path

from ..utils.lm import corpus_to_sentences, load_corpus_text


@dataclass(slots=True)
class NGramLanguageModel:
    """A smoothed word-level n-gram model."""

    order: int
    vocab: set[str]
    ngram_counts: dict[tuple[str, ...], int]
    context_counts: dict[tuple[str, ...], int]

    @classmethod
    def from_corpus(cls, corpus_path: str | Path, order: int = 3) -> "NGramLanguageModel":
        """Construct an n-gram model from raw text."""

        text = load_corpus_text(corpus_path)
        sentences = corpus_to_sentences(text)
        vocab = {"<s>", "</s>", "<unk>"}
        ngram_counts: Counter[tuple[str, ...]] = Counter()
        context_counts: Counter[tuple[str, ...]] = Counter()
        for sentence in sentences:
            padded = ["<s>"] * (order - 1) + sentence + ["</s>"]
            vocab.update(sentence)
            for idx in range(len(padded) - order + 1):
                ngram = tuple(padded[idx : idx + order])
                context = ngram[:-1]
                ngram_counts[ngram] += 1
                context_counts[context] += 1
        return cls(order=order, vocab=vocab, ngram_counts=dict(ngram_counts), context_counts=dict(context_counts))

    def _count(self, ngram: tuple[str, ...]) -> int:
        return self.ngram_counts.get(ngram, 0)

    def _context_count(self, context: tuple[str, ...]) -> int:
        return self.context_counts.get(context, 0)

    def score_tokens(self, words: list[str]) -> float:
        """Return the log probability of a word sequence."""

        tokens = ["<s>"] * (self.order - 1) + [word.lower() for word in words] + ["</s>"]
        vocab_size = max(1, len(self.vocab))
        score = 0.0
        for idx in range(self.order - 1, len(tokens)):
            ngram = tuple(tokens[idx - self.order + 1 : idx + 1])
            context = ngram[:-1]
            count = self._count(ngram)
            context_count = self._context_count(context)
            prob = (count + 1.0) / (context_count + vocab_size)
            score += math.log(prob)
        return score

    def score_text(self, text: str) -> float:
        """Score text after word tokenization."""

        words = [word for word in text.strip().split() if word]
        if not words:
            return 0.0
        return self.score_tokens(words)

