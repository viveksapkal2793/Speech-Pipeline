"""IndicTrans2-backed translation with dictionary fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re

from ..utils.text import detokenize, tokenize_words


DEFAULT_LEXICON = {
    "the": "the",
    "is": "hai",
    "are": "hain",
    "and": "aur",
    "to": "ko",
    "of": "ka",
    "in": "mein",
    "for": "ke liye",
    "lecture": "vyakhyan",
    "class": "kaksha",
    "student": "chhatra",
    "teacher": "shikshak",
    "audio": "audiyo",
    "speech": "speech",
}


@dataclass(slots=True)
class TranslationResult:
    """Translation output payload."""

    text: str
    backend: str


class HinglishTranslator:
    """Translate Hinglish text into a target low-resource language."""

    def __init__(
        self,
        target_lang: str = "hin_Deva",
        lexicon_path: str | Path | None = None,
        model_name: str | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        self.target_lang = target_lang
        self.lexicon = self._load_lexicon(lexicon_path)
        self.backend = "dictionary"
        self.model = None
        self.tokenizer = None
        self.model_path = Path(model_path) if model_path is not None else None
        self.model_name = str(self.model_path) if self.model_path is not None else (model_name or self._default_model_name(target_lang))
        self._load_indictrans2()

    def _default_model_name(self, target_lang: str) -> str:
        if target_lang.startswith("hin") or target_lang.startswith("mar") or target_lang.startswith("pan"):
            return "ai4bharat/indictrans2-en-indic-dist-200M"
        return "ai4bharat/indictrans2-en-indic-dist-200M"

    def _load_lexicon(self, lexicon_path: str | Path | None) -> dict[str, str]:
        if lexicon_path is None:
            return dict(DEFAULT_LEXICON)
        path = Path(lexicon_path)
        if path.exists():
            return {str(k).lower(): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
        return dict(DEFAULT_LEXICON)

    def _load_indictrans2(self) -> None:
        try:
            from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

            load_kwargs = {"local_files_only": True} if self.model_path is not None else {}
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True, **load_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, **load_kwargs)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                config=config,
                trust_remote_code=True,
                **load_kwargs,
            )
            self.backend = "indictrans2"
        except Exception:
            self.backend = "dictionary"
            self.model = None
            self.tokenizer = None

    def _dictionary_translate(self, text: str) -> str:
        tokens = tokenize_words(text)
        out: list[str] = []
        for token in tokens:
            if re.fullmatch(r"[A-Za-z']+", token):
                out.append(self.lexicon.get(token.lower(), token))
            else:
                out.append(token)
        return detokenize(out)

    @staticmethod
    def _clean_input_text(text: str) -> str:
        cleaned = re.sub(r"<\|[^>]+\|>", " ", text)
        cleaned = cleaned.replace("\u200b", " ")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def translate(self, text: str, src_lang: str = "eng_Latn", tgt_lang: str | None = None) -> TranslationResult:
        """Translate text using IndicTrans2 when possible."""

        cleaned = self._clean_input_text(text)
        target = tgt_lang or self.target_lang
        if not cleaned:
            return TranslationResult(text="", backend="dictionary")

        if self.backend == "indictrans2" and self.model is not None and self.tokenizer is not None:
            try:
                self.tokenizer._switch_to_input_mode()
            except Exception:
                pass
            prompt = f"{src_lang} {target} {cleaned}"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            generated = self.model.generate(**inputs, max_new_tokens=128, num_beams=4)
            translated = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            return TranslationResult(text=translated.strip(), backend="indictrans2")
        return TranslationResult(text=self._dictionary_translate(cleaned), backend="dictionary")
