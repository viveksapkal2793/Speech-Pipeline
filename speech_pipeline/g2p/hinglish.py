"""Hinglish to IPA grapheme-to-phoneme conversion."""

from __future__ import annotations

import re

from ..utils.text import detokenize, is_devanagari, is_english_word, tokenize_words


class HinglishG2P:
    """Convert Hinglish text to IPA using Epitran and rule-based fallbacks."""

    def __init__(self) -> None:
        self.hi_epi = self._load_epitran("hin-Deva")
        self.en_epi = self._load_epitran("eng-Latn")
        self.rule_exceptions = {
            "ai": "aɪ",
            "aur": "aʊr",
            "hai": "hɛː",
            "haan": "haːn",
            "nahi": "nəˈhiː",
            "ka": "ka",
            "ki": "ki",
            "ko": "ko",
            "mein": "meːn",
            "main": "meːn",
            "tum": "t̪um",
            "aap": "aːp",
            "lecture": "lɛk.t͡ʃər",
            "class": "klɑːs",
        }

    def _load_epitran(self, code: str):
        try:
            import epitran

            return epitran.Epitran(code)
        except Exception:
            return None

    def _roman_to_ipa(self, word: str) -> str:
        """Apply a deterministic romanized Hinglish-to-IPA mapping."""

        word_lower = word.lower()
        if word_lower in self.rule_exceptions:
            return self.rule_exceptions[word_lower]

        replacements = [
            ("chh", "t͡ʃʰ"),
            ("sh", "ʃ"),
            ("ch", "t͡ʃ"),
            ("jh", "d͡ʒʱ"),
            ("kh", "kʰ"),
            ("gh", "gʱ"),
            ("ph", "pʰ"),
            ("bh", "bʱ"),
            ("th", "t̪ʰ"),
            ("dh", "d̪ʱ"),
            ("aa", "aː"),
            ("ii", "iː"),
            ("ee", "iː"),
            ("oo", "uː"),
            ("uu", "uː"),
            ("ai", "aɪ"),
            ("au", "aʊ"),
            ("ng", "ŋ"),
            ("ny", "ɲ"),
            ("qu", "kw"),
        ]
        consonants = {
            "a": "ə",
            "b": "b",
            "c": "k",
            "d": "d̪",
            "e": "e",
            "f": "f",
            "g": "g",
            "h": "h",
            "i": "i",
            "j": "d͡ʒ",
            "k": "k",
            "l": "l",
            "m": "m",
            "n": "n",
            "o": "o",
            "p": "p",
            "q": "k",
            "r": "r",
            "s": "s",
            "t": "t̪",
            "u": "u",
            "v": "ʋ",
            "w": "ʋ",
            "x": "ks",
            "y": "j",
            "z": "z",
        }
        working = word_lower
        for src, dst in replacements:
            working = working.replace(src, dst)
        result: list[str] = []
        for char in working:
            if char in consonants:
                result.append(consonants[char])
            elif char in {"ʃ", "ʒ", "ŋ", "ɲ"}:
                result.append(char)
            elif char.isdigit():
                result.append(char)
            elif char in {"'", "-"}:
                continue
            else:
                result.append(char)
        ipa = "".join(result)
        ipa = re.sub(r"(.)\1+", r"\1", ipa)
        return ipa or word

    def _word_to_ipa(self, word: str) -> str:
        if not word:
            return word
        if is_devanagari(word) and self.hi_epi is not None:
            try:
                return self.hi_epi.transliterate(word)
            except Exception:
                return self._roman_to_ipa(word)
        if is_english_word(word) and self.en_epi is not None and word.lower() in self.rule_exceptions:
            try:
                return self.en_epi.transliterate(word)
            except Exception:
                return self._roman_to_ipa(word)
        return self._roman_to_ipa(word)

    def text_to_ipa(self, text: str) -> str:
        """Convert mixed Hinglish text to an IPA string."""

        tokens = tokenize_words(text)
        ipa_tokens = []
        for token in tokens:
            if re.fullmatch(r"[A-Za-z\u0900-\u097F']+", token):
                ipa_tokens.append(self._word_to_ipa(token))
            else:
                ipa_tokens.append(token)
        return detokenize(ipa_tokens)

