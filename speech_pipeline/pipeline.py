"""End-to-end speech processing pipeline CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

import numpy as np
import soundfile as sf
import torch

from .asr.whisper_constrained import ConstrainedWhisperASR
from .config import PipelineConfig
from .denoise.deepfilternet_denoiser import DeepFilterNetDenoiser
from .g2p.hinglish import HinglishG2P
from .lid.infer import LIDInferencer
from .prosody.transfer import ProsodyTransfer
from .speaker.resemblyzer_embedder import ResemblyzerEmbedder
from .translation.translator import HinglishTranslator
from .tts.indicf5_synthesizer import IndicF5Synthesizer
from .tts.parler_synthesizer import ParlerSynthesizer
from .tts.yourtts_synthesizer import YourTTSSynthesizer
from .antispoof.infer import AntiSpoofInferencer
from .utils.audio import load_audio, save_audio
from .utils.device import resolve_device


_NON_SPEECH_MARKER_RE = re.compile(r"[\[(]\s*([^\])]+)\s*[\])]", re.IGNORECASE)


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())


def _remove_non_speech_markers(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        content = match.group(1).strip().lower()
        noise_markers = (
            "speaks in",
            "non-english",
            "music",
            "applause",
            "laughter",
            "noise",
            "silence",
            "inaudible",
            "speech",
        )
        if any(marker in content for marker in noise_markers):
            return " "
        return match.group(0)

    cleaned = re.sub(r"<\|[^>]+\|>", " ", text)
    cleaned = _NON_SPEECH_MARKER_RE.sub(_replace, cleaned)
    return cleaned


def _collapse_repeated_words(text: str, max_run: int = 2) -> str:
    tokens = text.split()
    if not tokens:
        return ""
    out: list[str] = []
    prev_norm = ""
    run = 0
    for token in tokens:
        norm = re.sub(r"[^\w\u0900-\u097F']+", "", token).lower()
        if norm and norm == prev_norm:
            run += 1
        else:
            run = 1
            prev_norm = norm
        if run <= max_run or not norm:
            out.append(token)
    return " ".join(out)


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?\u0964])\s+|\n+", text)
    return [part.strip(" \t,;:") for part in parts if part and part.strip()]


def _split_words_to_max_chars(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        extra = 1 if current else 0
        if current and (current_len + extra + len(word)) > max_chars:
            chunks.append(" ".join(current).strip())
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra + len(word)
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _chunk_text_for_tts(text: str, max_chars: int = 220, max_sentences_per_chunk: int = 2) -> list[str]:
    sentences = _sentence_split(text)
    if not sentences:
        return _split_words_to_max_chars(text, max_chars)

    normalized: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            normalized.append(sentence)
            continue
        clauses = re.split(r"(?<=[,;:])\s+", sentence)
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            if len(clause) <= max_chars:
                normalized.append(clause)
            else:
                normalized.extend(_split_words_to_max_chars(clause, max_chars))

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in normalized:
        extra = 1 if current else 0
        if current and (
            (current_len + extra + len(sentence)) > max_chars
            or len(current) >= max_sentences_per_chunk
        ):
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += extra + len(sentence)
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _sanitize_transcript(text: str) -> str:
    cleaned = _remove_non_speech_markers(text)
    cleaned = _collapse_repeated_words(cleaned, max_run=2)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""

    filtered_sentences: list[str] = []
    prev_norm = ""
    for sentence in _sentence_split(cleaned):
        words = re.findall(r"[A-Za-z\u0900-\u097F']+", sentence)
        if len(words) >= 8:
            unique_ratio = len({word.lower() for word in words}) / max(1, len(words))
            if unique_ratio < 0.22:
                continue
        norm = _normalize_for_compare(sentence)
        if norm and norm == prev_norm:
            continue
        filtered_sentences.append(sentence)
        prev_norm = norm

    if not filtered_sentences:
        return cleaned
    rebuilt = " ".join(filtered_sentences)
    rebuilt = re.sub(r"\s+([,.;!?])", r"\1", rebuilt)
    return re.sub(r"\s+", " ", rebuilt).strip()


def _concat_audio_chunks(audio_chunks: list[np.ndarray], sample_rate: int, pause_seconds: float = 0.12) -> np.ndarray:
    if not audio_chunks:
        return np.zeros(0, dtype=np.float32)
    pause = np.zeros(int(sample_rate * pause_seconds), dtype=np.float32)
    parts: list[np.ndarray] = []
    for idx, audio in enumerate(audio_chunks):
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            continue
        parts.append(chunk)
        if idx < len(audio_chunks) - 1 and pause.size > 0:
            parts.append(pause)
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def run_pipeline(
    input_audio: str | Path,
    reference_audio: str | Path,
    output_dir: str | Path,
    clean_audio_path: str | Path | None = None,
    lm_corpus: str | Path | None = None,
    translation_target: str = 'hin_Deva',
    translation_lexicon: str | Path | None = None,
    translation_model_path: str | Path | None = None,
    tts_backend: str = "parler",
    tts_model_path: str | Path | None = None,
    reference_text: str | None = None,
    reference_text_file: str | Path | None = None,
    whisper_model: str = 'small',
    lid_checkpoint: str | Path | None = None,
    antispoof_checkpoint: str | Path | None = None,
    device: str | None = None,
    combine_existing_tts_chunks_only: bool = False,
) -> dict[str, object]:
    """Run the full speech processing chain and write artifacts to disk."""

    config = PipelineConfig()
    device_t = torch.device(device) if device is not None else resolve_device()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clean_path = output_path / 'clean.wav'
    if clean_audio_path is not None:
        existing_clean = Path(clean_audio_path)
        if not existing_clean.exists():
            raise FileNotFoundError(f"clean_audio_path does not exist: {existing_clean}")
        if existing_clean.resolve() != clean_path.resolve():
            shutil.copyfile(existing_clean, clean_path)
        else:
            clean_path = existing_clean
    else:
        raw_audio, sr = load_audio(input_audio, sr=config.audio.sample_rate)
        denoiser = DeepFilterNetDenoiser(sample_rate=config.audio.sample_rate)
        denoised = denoiser.denoise(raw_audio).audio
        save_audio(clean_path, denoised, sr=config.audio.sample_rate)

    lid_output = None
    if lid_checkpoint is not None:
        lid = LIDInferencer(lid_checkpoint, device=device)
        lid_output = lid.predict(clean_path)

    if combine_existing_tts_chunks_only:
        chunk_dir = output_path / 'tts_chunks'
        existing_chunk_paths = sorted(chunk_dir.glob('chunk_*.wav'))
        if not existing_chunk_paths:
            raise FileNotFoundError(
                f"No existing TTS chunks found in {chunk_dir}. Disable --combine-existing-tts-chunks-only to generate chunks."
            )

        synthesized_chunks: list[np.ndarray] = []
        synthesized_sr: int | None = None
        for chunk_path in existing_chunk_paths:
            audio_chunk, chunk_sr = sf.read(str(chunk_path), dtype='float32', always_2d=False)
            audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
            if audio_chunk.ndim > 1:
                audio_chunk = np.mean(audio_chunk, axis=-1)
            if synthesized_sr is None:
                synthesized_sr = int(chunk_sr)
            if int(chunk_sr) != synthesized_sr:
                audio_chunk, _ = load_audio(chunk_path, sr=synthesized_sr)
            synthesized_chunks.append(audio_chunk)

        if synthesized_sr is None:
            raise RuntimeError("Failed to load existing TTS chunks.")

        synth_path = output_path / 'synth.wav'
        synth_audio = _concat_audio_chunks(synthesized_chunks, sample_rate=synthesized_sr)
        save_audio(synth_path, synth_audio, sr=synthesized_sr)

        prosody = ProsodyTransfer(sample_rate=synthesized_sr)
        prosody_result = prosody.transfer(clean_path, synth_path)
        prosody_path = output_path / 'prosody_matched.wav'
        save_audio(prosody_path, prosody_result.audio, prosody_result.sample_rate)

        spoof_output = None
        if antispoof_checkpoint is not None:
            spoof = AntiSpoofInferencer(antispoof_checkpoint, device=device)
            spoof_output = spoof.predict(prosody_path)

        prior_result_path = output_path / 'result.json'
        prior_result: dict[str, object] = {}
        if prior_result_path.exists():
            try:
                loaded = json.loads(prior_result_path.read_text(encoding='utf-8'))
                if isinstance(loaded, dict):
                    prior_result = loaded
            except Exception:
                prior_result = {}

        transcript = str(prior_result.get('transcript', ''))
        transcript_raw = str(prior_result.get('transcript_raw', transcript))
        transcript_chunks = prior_result.get('transcript_chunks', [])
        if not isinstance(transcript_chunks, list):
            transcript_chunks = []
        translated_chunks = prior_result.get('translated_chunks', [])
        if not isinstance(translated_chunks, list):
            translated_chunks = []
        if translated_chunks:
            translated_chunks = translated_chunks[:len(existing_chunk_paths)]
        translated_text = str(prior_result.get('translated_text', " ".join(translated_chunks).strip()))
        ipa = str(prior_result.get('ipa', ''))
        reference_text_out = prior_result.get('reference_text', reference_text or "")
        translation_backend = str(prior_result.get('translation_backend', 'unknown'))
        speaker_embedding_shape = prior_result.get('speaker_embedding_shape', [])
        if not isinstance(speaker_embedding_shape, list):
            speaker_embedding_shape = []

        result = {
            'denoised_audio': str(clean_path),
            'transcript': transcript,
            'transcript_raw': transcript_raw,
            'transcript_chunks': transcript_chunks,
            'ipa': ipa,
            'translated_text': translated_text,
            'translated_chunks': translated_chunks,
            'translation_backend': translation_backend,
            'reference_text': reference_text_out,
            'tts_backend': tts_backend,
            'num_tts_chunks': len(existing_chunk_paths),
            'speaker_embedding_shape': speaker_embedding_shape,
            'synthesized_audio': str(synth_path),
            'prosody_matched_audio': str(prosody_path),
            'lid': None if lid_output is None else {
                'frame_labels': lid_output.frame_labels,
                'frame_probabilities_shape': list(lid_output.frame_probabilities.shape),
            },
            'spoof': None if spoof_output is None else {
                'label': spoof_output.label,
                'probabilities': spoof_output.probabilities.tolist(),
            },
        }

        prior_result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result

    asr = ConstrainedWhisperASR(
        model_name=whisper_model,
        device=device,
        lm_corpus=lm_corpus,
        lm_order=config.asr.lm_order,
        lm_weight=config.asr.lm_weight,
        length_penalty=config.asr.length_penalty,
    )
    transcript_raw = asr.transcribe(clean_path, language=None, beam_size=config.asr.beam_size)
    transcript = _sanitize_transcript(transcript_raw)
    if reference_text_file is not None:
        reference_text = Path(reference_text_file).read_text(encoding='utf-8', errors='ignore').strip()
    if reference_text is None:
        reference_text = asr.transcribe(reference_audio, language=None, beam_size=config.asr.beam_size)

    g2p = HinglishG2P()
    ipa = g2p.text_to_ipa(transcript)

    translator = HinglishTranslator(
        target_lang=translation_target,
        lexicon_path=translation_lexicon,
        model_path=translation_model_path,
    )
    transcript_chunks = _chunk_text_for_tts(transcript)
    translated_chunks: list[str] = []
    translation_backend = "dictionary"
    for chunk in transcript_chunks:
        translated_chunk = translator.translate(chunk, src_lang='eng_Latn', tgt_lang=translation_target)
        translation_backend = translated_chunk.backend
        text = translated_chunk.text.strip()
        if text:
            translated_chunks.append(text)

    if not translated_chunks and transcript:
        translated = translator.translate(transcript, src_lang='eng_Latn', tgt_lang=translation_target)
        translation_backend = translated.backend
        if translated.text.strip():
            translated_chunks = [translated.text.strip()]
            transcript_chunks = [transcript]

    translated_text = " ".join(translated_chunks).strip()

    chunk_dir = output_path / 'tts_chunks'
    chunk_dir.mkdir(parents=True, exist_ok=True)
    synthesized_chunks: list[np.ndarray] = []
    synthesized_sr: int | None = None
    embedder = ResemblyzerEmbedder(sample_rate=config.audio.sample_rate)
    speaker_embedding = embedder.embed(reference_audio)

    if not translated_chunks:
        raise ValueError("No translated text available for TTS after transcript cleanup.")

    if tts_backend == "indicf5":
        tts = IndicF5Synthesizer(model_name_or_path=tts_model_path or "ai4bharat/IndicF5", device=str(device_t))
    elif tts_backend == "parler":
        tts = ParlerSynthesizer(model_name_or_path=tts_model_path or "ai4bharat/indic-parler-tts", device=str(device_t))
    elif tts_backend == "yourtts":
        tts = YourTTSSynthesizer(device=str(device_t))
    else:
        raise ValueError(f"Unknown TTS backend: {tts_backend}")

    for idx, translated_chunk in enumerate(translated_chunks):
        chunk_output = chunk_dir / f'chunk_{idx:04d}.wav'
        if tts_backend == "indicf5":
            chunk_synth = tts.synthesize(
                text=translated_chunk,
                speaker_embedding=speaker_embedding,
                speaker_reference=reference_audio,
                reference_text=reference_text,
                output_path=chunk_output,
            )
        elif tts_backend == "parler":
            chunk_synth = tts.synthesize(
                text=translated_chunk,
                speaker_embedding=speaker_embedding,
                speaker_reference=reference_audio,
                reference_text=reference_text,
                output_path=chunk_output,
                language=translation_target,
            )
        else:  # yourtts
            chunk_synth = tts.synthesize(
                text=translated_chunk,
                speaker_embedding=speaker_embedding,
                speaker_reference=reference_audio,
                language=translation_target,
                output_path=chunk_output,
            )

        if synthesized_sr is None:
            synthesized_sr = int(chunk_synth.sample_rate)

        if int(chunk_synth.sample_rate) != synthesized_sr:
            audio_chunk, _ = load_audio(chunk_synth.path, sr=synthesized_sr)
        else:
            audio_chunk = np.asarray(chunk_synth.audio, dtype=np.float32)
        synthesized_chunks.append(audio_chunk)

    if synthesized_sr is None:
        raise RuntimeError("TTS did not return any synthesized chunks.")

    synth_path = output_path / 'synth.wav'
    synth_audio = _concat_audio_chunks(synthesized_chunks, sample_rate=synthesized_sr)
    save_audio(synth_path, synth_audio, sr=synthesized_sr)

    class _SynthesisResult:
        def __init__(self, path: Path, audio: np.ndarray, sample_rate: int) -> None:
            self.path = path
            self.audio = audio
            self.sample_rate = sample_rate

    synthesized = _SynthesisResult(path=synth_path, audio=synth_audio, sample_rate=synthesized_sr)

    prosody = ProsodyTransfer(sample_rate=synthesized.sample_rate)
    prosody_result = prosody.transfer(clean_path, synthesized.path)
    prosody_path = output_path / 'prosody_matched.wav'
    save_audio(prosody_path, prosody_result.audio, prosody_result.sample_rate)

    spoof_output = None
    if antispoof_checkpoint is not None:
        spoof = AntiSpoofInferencer(antispoof_checkpoint, device=device)
        spoof_output = spoof.predict(prosody_path)

    result = {
        'denoised_audio': str(clean_path),
        'transcript': transcript,
        'transcript_raw': transcript_raw,
        'transcript_chunks': transcript_chunks,
        'ipa': ipa,
        'translated_text': translated_text,
        'translated_chunks': translated_chunks,
        'translation_backend': translation_backend,
        'reference_text': reference_text,
        'tts_backend': tts_backend,
        'num_tts_chunks': len(translated_chunks),
        'speaker_embedding_shape': list(speaker_embedding.shape),
        'synthesized_audio': str(synthesized.path),
        'prosody_matched_audio': str(prosody_path),
        'lid': None if lid_output is None else {
            'frame_labels': lid_output.frame_labels,
            'frame_probabilities_shape': list(lid_output.frame_probabilities.shape),
        },
        'spoof': None if spoof_output is None else {
            'label': spoof_output.label,
            'probabilities': spoof_output.probabilities.tolist(),
        },
    }

    (output_path / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main() -> None:
    """CLI entry point for the full pipeline."""

    parser = argparse.ArgumentParser(description='Run the end-to-end Hinglish speech pipeline.')
    parser.add_argument('--input-audio', required=True)
    parser.add_argument('--reference-audio', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--clean-audio-path', default=None, help='Optional existing clean.wav to resume from. If provided, denoising is skipped.')
    parser.add_argument('--lm-corpus', default=None)
    parser.add_argument('--translation-target', default='hin_Deva')
    parser.add_argument('--translation-lexicon', default=None)
    parser.add_argument('--translation-model-path', default=None)
    parser.add_argument('--tts-backend', choices=['indicf5', 'parler', 'yourtts'], default='parler')
    parser.add_argument('--tts-model-path', default=None)
    parser.add_argument('--reference-text', default=None, help='Optional reference transcript for the 60-second prompt audio. If omitted, the pipeline transcribes it automatically.')
    parser.add_argument('--reference-text-file', default=None, help='Path to a UTF-8 text file containing the reference prompt transcript.')
    parser.add_argument('--whisper-model', default='small')
    parser.add_argument('--lid-checkpoint', default=None)
    parser.add_argument('--antispoof-checkpoint', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument(
        '--combine-existing-tts-chunks-only',
        action='store_true',
        help='Skip new TTS generation and continue pipeline by combining already generated WAV chunks from output_dir/tts_chunks.',
    )
    args = parser.parse_args()

    result = run_pipeline(
        input_audio=args.input_audio,
        reference_audio=args.reference_audio,
        output_dir=args.output_dir,
        clean_audio_path=args.clean_audio_path,
        lm_corpus=args.lm_corpus,
        translation_target=args.translation_target,
        translation_lexicon=args.translation_lexicon,
        translation_model_path=args.translation_model_path,
        tts_backend=args.tts_backend,
        tts_model_path=args.tts_model_path,
        reference_text=args.reference_text,
        reference_text_file=args.reference_text_file,
        whisper_model=args.whisper_model,
        lid_checkpoint=args.lid_checkpoint,
        antispoof_checkpoint=args.antispoof_checkpoint,
        device=args.device,
        combine_existing_tts_chunks_only=args.combine_existing_tts_chunks_only,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
