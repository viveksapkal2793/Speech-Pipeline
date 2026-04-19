"""Build chunk-based anti-spoof train/eval manifests from source audio files."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import pandas as pd
import soundfile as sf


def _segment_rows(
    audio_paths: list[str],
    label: int,
    chunk_seconds: float,
    chunk_overlap_seconds: float,
    max_chunks_per_file: int | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    hop = max(1e-3, chunk_seconds - chunk_overlap_seconds)
    for path in audio_paths:
        audio_path = Path(path)
        info = sf.info(str(audio_path))
        duration = float(info.frames) / float(info.samplerate)
        if duration <= 0:
            continue

        starts: list[float] = []
        current = 0.0
        while current < duration:
            starts.append(current)
            if max_chunks_per_file is not None and len(starts) >= max_chunks_per_file:
                break
            current += hop

        for start in starts:
            end = min(duration, start + chunk_seconds)
            if (end - start) < 0.2:
                continue
            rows.append(
                {
                    "audio_path": str(audio_path),
                    "label": int(label),
                    "start_sec": round(float(start), 4),
                    "end_sec": round(float(end), 4),
                }
            )
    return rows


def _split_rows(rows: list[dict[str, object]], train_ratio: float, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not rows:
        return [], []
    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    if n_total >= 2:
        n_train = min(max(1, n_train), n_total - 1)
    else:
        n_train = 1
    return shuffled[:n_train], shuffled[n_train:]


def build_manifests(
    bona_fide_audio: list[str],
    spoof_audio: list[str],
    output_dir: str | Path,
    chunk_seconds: float = 8.0,
    chunk_overlap_seconds: float = 1.0,
    train_ratio: float = 0.8,
    max_chunks_per_file: int | None = None,
    seed: int = 13,
) -> tuple[Path, Path]:
    """Create chunk-based train/eval manifests for anti-spoof modeling."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    bona_rows = _segment_rows(
        audio_paths=bona_fide_audio,
        label=0,
        chunk_seconds=chunk_seconds,
        chunk_overlap_seconds=chunk_overlap_seconds,
        max_chunks_per_file=max_chunks_per_file,
    )
    spoof_rows = _segment_rows(
        audio_paths=spoof_audio,
        label=1,
        chunk_seconds=chunk_seconds,
        chunk_overlap_seconds=chunk_overlap_seconds,
        max_chunks_per_file=max_chunks_per_file,
    )

    if not bona_rows or not spoof_rows:
        raise ValueError("Need at least one chunk for both bona fide and spoof classes.")

    bona_train, bona_eval = _split_rows(bona_rows, train_ratio=train_ratio, seed=seed)
    spoof_train, spoof_eval = _split_rows(spoof_rows, train_ratio=train_ratio, seed=seed + 1)

    train_rows = bona_train + spoof_train
    eval_rows = bona_eval + spoof_eval

    if not eval_rows:
        eval_rows = train_rows[-2:] if len(train_rows) >= 2 else list(train_rows)
        train_rows = train_rows[:-2] if len(train_rows) >= 2 else list(train_rows)

    train_df = pd.DataFrame(train_rows).sort_values(["label", "audio_path", "start_sec"]).reset_index(drop=True)
    eval_df = pd.DataFrame(eval_rows).sort_values(["label", "audio_path", "start_sec"]).reset_index(drop=True)

    train_manifest = output / "antispoof_train_manifest.csv"
    eval_manifest = output / "antispoof_eval_manifest.csv"
    train_df.to_csv(train_manifest, index=False)
    eval_df.to_csv(eval_manifest, index=False)
    return train_manifest, eval_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create chunk-based anti-spoof train/eval manifests.")
    parser.add_argument("--bona-fide-audio", nargs="+", required=True, help="One or more original/human audio paths.")
    parser.add_argument("--spoof-audio", nargs="+", required=True, help="One or more synthesized/spoof audio paths.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-seconds", type=float, default=8.0)
    parser.add_argument("--chunk-overlap-seconds", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--max-chunks-per-file", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    train_manifest, eval_manifest = build_manifests(
        bona_fide_audio=args.bona_fide_audio,
        spoof_audio=args.spoof_audio,
        output_dir=args.output_dir,
        chunk_seconds=args.chunk_seconds,
        chunk_overlap_seconds=args.chunk_overlap_seconds,
        train_ratio=args.train_ratio,
        max_chunks_per_file=args.max_chunks_per_file,
        seed=args.seed,
    )
    print(f"Train manifest: {train_manifest}")
    print(f"Eval manifest: {eval_manifest}")


if __name__ == "__main__":
    main()
