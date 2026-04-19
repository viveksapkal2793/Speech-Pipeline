"""Fill missing manifest timestamps for chunked LID data."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
import soundfile as sf


DEFAULT_EXCLUDE_START_SEC = 2 * 60 * 60 + 20 * 60
DEFAULT_EXCLUDE_END_SEC = 2 * 60 * 60 + 54 * 60


def _chunk_sort_key(audio_path: str | Path) -> tuple[int, str]:
    """Sort chunk files by their numeric suffix when available."""

    stem = Path(audio_path).stem
    match = re.search(r"(\d+)$", stem)
    if match is None:
        return (math.inf, stem.lower())
    return (int(match.group(1)), stem.lower())


def _audio_duration_seconds(audio_path: str | Path) -> float:
    """Read the duration of an audio file from its header."""

    info = sf.info(str(audio_path))
    return float(info.frames / info.samplerate)


def fill_manifest_timestamps(
    manifest_path: str | Path,
    *,
    output_path: str | Path | None = None,
    exclude_start_sec: float = DEFAULT_EXCLUDE_START_SEC,
    exclude_end_sec: float = DEFAULT_EXCLUDE_END_SEC,
) -> pd.DataFrame:
    """Populate missing start/end timestamps in a manifest.

    The rows are sorted by chunk filename suffix before timestamps are assigned.
    Timestamps are reconstructed in original video time by skipping the excluded
    interval after the last chunk that ends before the gap.
    """

    manifest_file = Path(manifest_path)
    if output_path is None:
        output_file = manifest_file
    else:
        output_file = Path(output_path)

    df = pd.read_csv(manifest_file)
    if "audio_path" not in df.columns:
        raise ValueError("Manifest must contain an 'audio_path' column.")

    if "start_sec" not in df.columns:
        df["start_sec"] = pd.NA
    if "end_sec" not in df.columns:
        df["end_sec"] = pd.NA

    ordered_indices = sorted(df.index.tolist(), key=lambda idx: _chunk_sort_key(df.at[idx, "audio_path"]))
    gap_start = float(exclude_start_sec)
    gap_end = float(exclude_end_sec)
    gap_duration = max(0.0, gap_end - gap_start)

    current_time = 0.0
    for idx in ordered_indices:
        audio_path = df.at[idx, "audio_path"]
        chunk_duration = _audio_duration_seconds(audio_path)

        if current_time >= gap_start:
            current_time += gap_duration

        start_sec = float(current_time)
        end_sec = float(current_time + chunk_duration)

        if start_sec < gap_start < end_sec:
            raise ValueError(
                f"Chunk {audio_path} overlaps the excluded interval "
                f"{gap_start:.3f}..{gap_end:.3f}. Rebuild the chunks with a boundary-safe split."
            )

        df.at[idx, "start_sec"] = start_sec
        df.at[idx, "end_sec"] = end_sec
        current_time = end_sec

    df.to_csv(output_file, index=False)
    return df


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Fill missing start/end times in a chunk manifest.")
    parser.add_argument("--manifest", required=True, help="Path to the manifest CSV to update.")
    parser.add_argument(
        "--output-manifest",
        default=None,
        help="Optional output path. If omitted, the input manifest is updated in place.",
    )
    parser.add_argument(
        "--exclude-start-sec",
        type=float,
        default=DEFAULT_EXCLUDE_START_SEC,
        help="Start time of the skipped interval in seconds.",
    )
    parser.add_argument(
        "--exclude-end-sec",
        type=float,
        default=DEFAULT_EXCLUDE_END_SEC,
        help="End time of the skipped interval in seconds.",
    )
    args = parser.parse_args()

    df = fill_manifest_timestamps(
        args.manifest,
        output_path=args.output_manifest,
        exclude_start_sec=args.exclude_start_sec,
        exclude_end_sec=args.exclude_end_sec,
    )
    output_path = args.output_manifest or args.manifest
    print(f"Updated {len(df)} rows in {output_path}")


if __name__ == "__main__":
    main()
