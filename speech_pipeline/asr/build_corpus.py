"""Build an ASR language-model corpus from subtitle or transcript files."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


def _read_text(path: str | Path) -> str:
    """Read a UTF-8 text file."""

    return Path(path).read_text(encoding="utf-8", errors="ignore")


def _extract_srt_text(path: str | Path) -> list[str]:
    """Extract subtitle text lines from an SRT file."""

    lines: list[str] = []
    for raw_line in _read_text(path).splitlines():
        line = raw_line.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        cleaned = re.sub(r"\s+", " ", line)
        if cleaned:
            lines.append(cleaned)
    return lines


def _normalize_text_line(text: str) -> str:
    """Normalize a free-form transcript line for LM training."""

    text = re.sub(r"\s+", " ", text.strip())
    text = text.replace("\ufeff", "")
    return text


def build_corpus(
    inputs: list[str | Path],
    output_path: str | Path,
    dedupe: bool = True,
) -> Path:
    """Build a plain-text corpus file from one or more transcript files."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    corpus_lines: list[str] = []
    seen: set[str] = set()

    for input_path in inputs:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {path}")

        if path.suffix.lower() == ".srt":
            source_lines = _extract_srt_text(path)
        else:
            source_lines = _read_text(path).splitlines()

        for line in source_lines:
            normalized = _normalize_text_line(line)
            if not normalized:
                continue
            if dedupe:
                key = normalized.lower()
                if key in seen:
                    continue
                seen.add(key)
            corpus_lines.append(normalized)

    output.write_text("\n".join(corpus_lines) + ("\n" if corpus_lines else ""), encoding="utf-8")
    return output


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Build an ASR LM corpus from subtitle or transcript files.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more transcript or subtitle files (.srt or .txt).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output plain-text corpus file.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep repeated lines instead of deduplicating them.",
    )
    args = parser.parse_args()

    output = build_corpus(args.input, args.output, dedupe=not args.no_dedupe)
    print(f"Saved LM corpus to {output}")


if __name__ == "__main__":
    main()
