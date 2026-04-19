"""Extract and preprocess a held-out segment for LID evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf

from .preprocess import preprocess_lid_audio


def _duration_seconds(audio_path: str | Path) -> float:
    """Return an audio file duration in seconds."""

    info = sf.info(str(audio_path))
    return float(info.frames / info.samplerate)


def extract_segment(
    source_audio: str | Path,
    output_audio: str | Path,
    start_sec: float,
    duration_sec: float,
    sample_rate: int = 16_000,
) -> Path:
    """Extract a single audio segment and save it as a standalone WAV file."""

    audio, sr = librosa.load(
        str(source_audio),
        sr=sample_rate,
        mono=True,
        offset=float(start_sec),
        duration=float(duration_sec),
    )
    output_path = Path(output_audio)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio.astype("float32"), sr)
    return output_path


def prepare_eval_segment(
    source_audio: str | Path,
    output_dir: str | Path,
    segment_start_sec: float,
    segment_duration_sec: float = 600.0,
    sample_rate: int = 16_000,
    whisper_model_path: str | Path | None = None,
    asr_backend: str = "auto",
    openai_whisper_model: str = "small",
    openai_whisper_download_root: str | Path | None = None,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 0.0,
    min_word_probability: float = 0.45,
    debug_dir: str | Path | None = None,
    debug_limit: int = 1,
) -> pd.DataFrame:
    """Extract a held-out segment and preprocess it into LID-ready chunks."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    source_duration = _duration_seconds(source_audio)
    if segment_start_sec < 0:
        raise ValueError(f"segment_start_sec must be non-negative, got {segment_start_sec}")
    if segment_duration_sec <= 0:
        raise ValueError(f"segment_duration_sec must be positive, got {segment_duration_sec}")
    if segment_start_sec + segment_duration_sec > source_duration + 1e-6:
        raise ValueError(
            f"Requested segment [{segment_start_sec}, {segment_start_sec + segment_duration_sec}] exceeds "
            f"source duration {source_duration:.2f}s for {source_audio}. "
            "Pass the full lecture audio here, not the already extracted eval_segment.wav."
        )

    segment_audio = output_root / "eval_segment.wav"
    extract_segment(
        source_audio=source_audio,
        output_audio=segment_audio,
        start_sec=segment_start_sec,
        duration_sec=segment_duration_sec,
        sample_rate=sample_rate,
    )
    if _duration_seconds(segment_audio) <= 0.0:
        raise RuntimeError(
            f"Extracted segment is empty: {segment_audio}. Check source audio, start time, and duration."
        )

    prep_dir = output_root / "lid_prep"
    manifest = preprocess_lid_audio(
        audio_path=segment_audio,
        output_dir=prep_dir,
        whisper_model_path=whisper_model_path or Path(__file__).resolve().parents[2] / "models" / "whisper-small",
        asr_backend=asr_backend,
        openai_whisper_model=openai_whisper_model,
        openai_whisper_download_root=openai_whisper_download_root,
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        min_word_probability=min_word_probability,
        debug_dir=debug_dir,
        debug_limit=debug_limit,
        start_sec=0.0,
        end_sec=segment_duration_sec,
    )

    manifest_path = prep_dir / "lid_manifest.csv"
    df = pd.read_csv(manifest_path)
    if "start_sec" in df.columns:
        df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce") + float(segment_start_sec)
    if "end_sec" in df.columns:
        df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce") + float(segment_start_sec)
    df.to_csv(manifest_path, index=False)

    metadata = {
        "source_audio": str(source_audio),
        "segment_audio": str(segment_audio),
        "segment_start_sec": float(segment_start_sec),
        "segment_duration_sec": float(segment_duration_sec),
        "preprocess_dir": str(prep_dir),
        "manifest_path": str(manifest_path),
    }
    (output_root / "eval_segment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return df


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Extract and preprocess a held-out LID evaluation segment.")
    parser.add_argument("--source-audio", required=True, help="Path to the full source WAV.")
    parser.add_argument("--output-dir", required=True, help="Directory where the eval segment and preprocessed chunks will be written.")
    parser.add_argument("--segment-start-sec", type=float, required=True, help="Start time of the 10-minute segment in the source audio.")
    parser.add_argument("--segment-duration-sec", type=float, default=600.0, help="Duration of the held-out segment in seconds.")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--whisper-model-path", default=None, help="Local Whisper HF model directory for preprocessing.")
    parser.add_argument(
        "--asr-backend",
        choices=["auto", "openai", "transformers"],
        default="auto",
        help="ASR backend for preprocessing.",
    )
    parser.add_argument("--openai-whisper-model", default="small")
    parser.add_argument("--openai-whisper-download-root", default=None)
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--overlap-seconds", type=float, default=0.0)
    parser.add_argument("--min-word-probability", type=float, default=0.45)
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--debug-limit", type=int, default=1)
    args = parser.parse_args()

    df = prepare_eval_segment(
        source_audio=args.source_audio,
        output_dir=args.output_dir,
        segment_start_sec=args.segment_start_sec,
        segment_duration_sec=args.segment_duration_sec,
        sample_rate=args.sample_rate,
        whisper_model_path=args.whisper_model_path,
        asr_backend=args.asr_backend,
        openai_whisper_model=args.openai_whisper_model,
        openai_whisper_download_root=args.openai_whisper_download_root,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        min_word_probability=args.min_word_probability,
        debug_dir=args.debug_dir,
        debug_limit=args.debug_limit,
    )
    print(f"Prepared {len(df)} chunks for LID evaluation in {Path(args.output_dir) / 'lid_prep'}")
    print(f"Segment: {Path(args.output_dir) / 'eval_segment.wav'}")


if __name__ == "__main__":
    main()
