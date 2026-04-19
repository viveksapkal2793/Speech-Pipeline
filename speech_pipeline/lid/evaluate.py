"""Evaluate a trained LID model on frame and switch metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .infer import LIDInferencer


@dataclass(slots=True)
class FrameMetrics:
    """Frame-level binary classification metrics."""

    accuracy: float | None = None
    precision_en: float | None = None
    recall_en: float | None = None
    f1_en: float | None = None
    precision_hi: float | None = None
    recall_hi: float | None = None
    f1_hi: float | None = None
    macro_f1: float | None = None
    num_frames: int = 0


@dataclass(slots=True)
class SwitchMetrics:
    """Language switch timestamp metrics."""

    switch_within_200ms: float | None = None
    switch_precision: float | None = None
    switch_recall: float | None = None
    switch_f1: float | None = None
    switch_mae_ms: float | None = None
    num_segments: int = 0


def _parse_frame_labels(value: object) -> np.ndarray | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    path = Path(str(value))
    if path.exists() and path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=True).astype(np.int64)
    if path.exists() and path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return np.asarray([int(x) for x in data], dtype=np.int64)
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("["):
        try:
            return np.asarray([int(x) for x in json.loads(text)], dtype=np.int64)
        except Exception:
            pass
    delimiter = ";" if ";" in text else "," if "," in text else " "
    parts = [part.strip() for part in text.split(delimiter) if part.strip()]
    if not parts:
        return None
    return np.asarray([int(part) for part in parts], dtype=np.int64)


def _resize_discrete(labels: np.ndarray, target_length: int) -> np.ndarray:
    if len(labels) == target_length:
        return labels.astype(np.int64)
    if target_length <= 0:
        return np.asarray([], dtype=np.int64)
    if len(labels) == 0:
        return np.full(target_length, -100, dtype=np.int64)
    idx = np.linspace(0, len(labels) - 1, num=target_length)
    idx = np.rint(idx).astype(np.int64)
    idx = np.clip(idx, 0, len(labels) - 1)
    return labels[idx].astype(np.int64)


def _fill_ignored(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).copy()
    valid = np.flatnonzero(labels >= 0)
    if valid.size == 0:
        return labels
    first = int(valid[0])
    labels[:first] = labels[first]
    last = labels[first]
    for idx in range(first + 1, len(labels)):
        if labels[idx] < 0:
            labels[idx] = last
        else:
            last = labels[idx]
    return labels


def _labels_to_switch_times(labels: np.ndarray, hop_seconds: float) -> list[float]:
    labels = _fill_ignored(labels)
    if len(labels) == 0:
        return []
    switches: list[float] = []
    for idx in range(1, len(labels)):
        if labels[idx] != labels[idx - 1]:
            switches.append(idx * hop_seconds)
    return switches


def _match_switches(reference: list[float], hypothesis: list[float], tolerance: float = 0.2) -> tuple[int, int, int, list[float]]:
    used = set()
    matched_errors: list[float] = []
    tp = 0
    for ref_t in reference:
        best_idx = None
        best_err = tolerance
        for idx, hyp_t in enumerate(hypothesis):
            if idx in used:
                continue
            err = abs(hyp_t - ref_t)
            if err <= best_err:
                best_err = err
                best_idx = idx
        if best_idx is not None:
            used.add(best_idx)
            tp += 1
            matched_errors.append(best_err)
    fp = len(hypothesis) - tp
    fn = len(reference) - tp
    return tp, fp, fn, matched_errors


def _class_metrics(ref: np.ndarray, hyp: np.ndarray, cls: int) -> tuple[float | None, float | None, float | None]:
    tp = int(np.sum((ref == cls) & (hyp == cls)))
    fp = int(np.sum((ref != cls) & (hyp == cls)))
    fn = int(np.sum((ref == cls) & (hyp != cls)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2.0 * precision * recall / max(1e-12, precision + recall)) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def evaluate_lid(manifest_path: str | Path, checkpoint_path: str | Path, device: str | None = None, tolerance_ms: float = 200.0) -> dict[str, object]:
    """Evaluate frame-level LID and switch accuracy on a manifest."""

    df = pd.read_csv(manifest_path)
    inferencer = LIDInferencer(checkpoint_path, device=device)
    hop_seconds = inferencer.audio_config.hop_length / inferencer.audio_config.sample_rate
    tolerance = tolerance_ms / 1000.0

    frame_ref_all: list[int] = []
    frame_hyp_all: list[int] = []
    switch_tp = switch_fp = switch_fn = 0
    switch_errors: list[float] = []
    num_segments = 0

    for _, row in df.iterrows():
        ref_labels = _parse_frame_labels(row.get("frame_labels", None))
        if ref_labels is None:
            continue
        pred = inferencer.predict(row["audio_path"])
        pred_labels = np.asarray([inferencer.label_map.get(label, -1) for label in pred.frame_labels], dtype=np.int64)
        ref_labels = np.asarray(ref_labels, dtype=np.int64)
        if len(pred_labels) != len(ref_labels):
            pred_labels = _resize_discrete(pred_labels, len(ref_labels))

        valid_mask = ref_labels >= 0
        if not np.any(valid_mask):
            continue

        ref_valid = ref_labels[valid_mask]
        pred_valid = pred_labels[valid_mask]

        frame_ref_all.extend(ref_valid.tolist())
        frame_hyp_all.extend(pred_valid.tolist())

        ref_switches = parse_switch_times(row.get("switch_times", None))
        if not ref_switches:
            ref_switches = _labels_to_switch_times(ref_labels, hop_seconds)
        pred_switches = _labels_to_switch_times(pred_labels, hop_seconds)
        tp, fp, fn, errors = _match_switches(ref_switches, pred_switches, tolerance=tolerance)
        switch_tp += tp
        switch_fp += fp
        switch_fn += fn
        switch_errors.extend(errors)
        num_segments += 1

    if not frame_ref_all:
        return {
            "frame": asdict(FrameMetrics(num_frames=0)),
            "switch": asdict(SwitchMetrics(num_segments=0)),
        }

    ref_arr = np.asarray(frame_ref_all, dtype=np.int64)
    hyp_arr = np.asarray(frame_hyp_all, dtype=np.int64)
    accuracy = float(np.mean(ref_arr == hyp_arr))
    p_en, r_en, f1_en = _class_metrics(ref_arr, hyp_arr, 0)
    p_hi, r_hi, f1_hi = _class_metrics(ref_arr, hyp_arr, 1)
    macro_f1 = float(np.mean([f1_en, f1_hi]))

    frame_metrics = FrameMetrics(
        accuracy=accuracy,
        precision_en=p_en,
        recall_en=r_en,
        f1_en=f1_en,
        precision_hi=p_hi,
        recall_hi=r_hi,
        f1_hi=f1_hi,
        macro_f1=macro_f1,
        num_frames=int(len(ref_arr)),
    )

    switch_precision = switch_tp / max(1, switch_tp + switch_fp)
    switch_recall = switch_tp / max(1, switch_tp + switch_fn)
    switch_f1 = (2.0 * switch_precision * switch_recall / max(1e-12, switch_precision + switch_recall)) if (switch_precision + switch_recall) > 0 else 0.0
    switch_within = switch_tp / max(1, switch_tp + switch_fn)
    switch_mae_ms = float(np.mean(switch_errors) * 1000.0) if switch_errors else None
    switch_metrics = SwitchMetrics(
        switch_within_200ms=float(switch_within),
        switch_precision=float(switch_precision),
        switch_recall=float(switch_recall),
        switch_f1=float(switch_f1),
        switch_mae_ms=switch_mae_ms,
        num_segments=num_segments,
    )

    return {"frame": asdict(frame_metrics), "switch": asdict(switch_metrics)}


def parse_switch_times(value: object) -> list[float]:
    """Parse a switch-times field from the manifest."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [float(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            return [float(v) for v in json.loads(text)]
        except Exception:
            pass
    delimiter = ";" if ";" in text else "," if "," in text else " "
    parts = [part.strip() for part in text.split(delimiter) if part.strip()]
    return [float(part) for part in parts]


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Evaluate a trained LID model.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with audio_path and frame_labels.")
    parser.add_argument("--checkpoint", required=True, help="Trained LID checkpoint.")
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--tolerance-ms", type=float, default=200.0, help="Switch matching tolerance in milliseconds.")
    parser.add_argument("--output", default="lid_evaluation_report.json", help="Where to write the evaluation JSON.")
    args = parser.parse_args()

    report = evaluate_lid(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        device=args.device,
        tolerance_ms=args.tolerance_ms,
    )
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
