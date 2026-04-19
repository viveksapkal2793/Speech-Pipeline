"""Evaluation CLI for ASR, LID switching, MCD, spoof EER, and LID attack robustness."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import re

import librosa
import numpy as np
import soundfile as sf

from .antispoof.infer import AntiSpoofInferencer
from .asr.whisper_constrained import ConstrainedWhisperASR
from .lid.infer import LIDInferencer
from .utils.audio import load_audio, estimate_snr_db
from .utils.device import resolve_device
import pandas as pd


@dataclass(slots=True)
class ASRMetrics:
    """Word error rate metrics."""

    wer_overall: float | None = None
    wer_en: float | None = None
    wer_hi: float | None = None
    num_samples: int = 0


@dataclass(slots=True)
class LIDSwitchMetrics:
    """LID switch timestamp metrics."""

    switch_within_200ms: float | None = None
    switch_mae_ms: float | None = None
    switch_precision: float | None = None
    switch_recall: float | None = None
    switch_f1: float | None = None
    num_samples: int = 0


@dataclass(slots=True)
class MCDMetrics:
    """Mel-cepstral distortion metrics."""

    mcd_mean: float | None = None
    num_pairs: int = 0


@dataclass(slots=True)
class SpoofMetrics:
    """Spoof detection metrics."""

    eer: float | None = None
    eer_threshold: float | None = None
    auc_roc: float | None = None
    threshold: float = 0.5
    accuracy_at_0_5: float | None = None
    precision_at_0_5: float | None = None
    recall_at_0_5: float | None = None
    f1_at_0_5: float | None = None
    fpr_at_0_5: float | None = None
    fnr_at_0_5: float | None = None
    apcer_at_0_5: float | None = None
    bpcer_at_0_5: float | None = None
    tp: int | None = None
    fp: int | None = None
    tn: int | None = None
    fn: int | None = None
    num_bona_fide: int = 0
    num_spoof: int = 0
    num_samples: int = 0


@dataclass(slots=True)
class AttackMetrics:
    """Adversarial robustness metrics."""

    min_epsilon_mean: float | None = None
    min_epsilon_median: float | None = None
    snr_db_mean: float | None = None
    num_samples: int = 0


def _normalize_label(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"en", "eng", "english", "english_segment"}:
        return "en"
    if text in {"hi", "hin", "hindi", "hindi_segment"}:
        return "hi"
    return text


def _split_tokens(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[A-Za-z0-9']+", text.lower()) if tok]


def _levenshtein_distance(ref: list[str], hyp: list[str]) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, ref_tok in enumerate(ref, start=1):
        curr = [i]
        for j, hyp_tok in enumerate(hyp, start=1):
            cost = 0 if ref_tok == hyp_tok else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute word error rate."""

    ref_tokens = _split_tokens(reference)
    hyp_tokens = _split_tokens(hypothesis)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return _levenshtein_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def parse_time_list(value: object) -> list[float]:
    """Parse switch timestamps from a JSON list or delimiter-separated string."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [float(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            loaded = json.loads(text)
            return [float(v) for v in loaded]
        except Exception:
            pass
    delimiter = ";" if ";" in text else "," if "," in text else " "
    parts = [part.strip() for part in text.split(delimiter) if part.strip()]
    return [float(part) for part in parts]


def predicted_switch_times(frame_labels: list[str], frame_hop_seconds: float) -> list[float]:
    """Convert per-frame labels into switch timestamps."""

    if not frame_labels:
        return []
    switch_times: list[float] = []
    for idx in range(1, len(frame_labels)):
        if frame_labels[idx] != frame_labels[idx - 1]:
            switch_times.append(idx * frame_hop_seconds)
    return switch_times


def match_switches(reference: list[float], hypothesis: list[float], tolerance: float = 0.2) -> tuple[int, int, int, list[float]]:
    """Greedy matching between reference and predicted switch timestamps."""

    if not reference and not hypothesis:
        return 0, 0, 0, []
    used_h = set()
    matched_errors: list[float] = []
    tp = 0
    for ref_t in reference:
        best_idx = None
        best_err = tolerance
        for idx, hyp_t in enumerate(hypothesis):
            if idx in used_h:
                continue
            err = abs(hyp_t - ref_t)
            if err <= best_err:
                best_err = err
                best_idx = idx
        if best_idx is not None:
            used_h.add(best_idx)
            tp += 1
            matched_errors.append(best_err)
    fp = len(hypothesis) - tp
    fn = len(reference) - tp
    return tp, fp, fn, matched_errors


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute the equal error rate from binary labels and scores."""

    eer, _threshold = compute_eer_details(labels, scores)
    return eer


def compute_eer_details(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Compute EER and the threshold where FPR and FNR are closest."""

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    positives = labels == 1
    negatives = labels == 0
    if not np.any(positives) or not np.any(negatives):
        return float("nan"), float("nan")

    thresholds = np.unique(scores)
    thresholds = np.concatenate(([scores.min() - 1e-6], thresholds, [scores.max() + 1e-6]))
    best = None
    for threshold in thresholds:
        predicted_positive = scores >= threshold
        fp = np.sum(predicted_positive & negatives)
        fn = np.sum(~predicted_positive & positives)
        fpr = fp / max(1, np.sum(negatives))
        fnr = fn / max(1, np.sum(positives))
        gap = abs(fpr - fnr)
        eer = 0.5 * (fpr + fnr)
        if best is None or gap < best[0]:
            best = (gap, eer, float(threshold))
    if best is None:
        return float("nan"), float("nan")
    return float(best[1]), float(best[2])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Compute average ranks (1-indexed) with tie handling."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.zeros(values.shape[0], dtype=np.float64)
    i = 0
    n = values.shape[0]
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def compute_auc_roc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC using rank statistics (no external dependencies)."""

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    rank_sum_pos = float(np.sum(ranks[positives]))
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def compute_threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    """Compute confusion-matrix metrics at a fixed spoof threshold."""

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    predicted_spoof = scores >= threshold
    is_spoof = labels == 1
    is_bona_fide = labels == 0

    tp = int(np.sum(predicted_spoof & is_spoof))
    fp = int(np.sum(predicted_spoof & is_bona_fide))
    tn = int(np.sum(~predicted_spoof & is_bona_fide))
    fn = int(np.sum(~predicted_spoof & is_spoof))

    n_pos = int(np.sum(is_spoof))
    n_neg = int(np.sum(is_bona_fide))
    n_total = max(1, labels.shape[0])

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, n_pos)
    f1 = (2.0 * precision * recall / max(1e-12, precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n_total
    fpr = fp / max(1, n_neg)
    fnr = fn / max(1, n_pos)

    # PAD convention under binary setup used here:
    # spoof=1 (attack), bona_fide=0 (bonafide).
    # APCER: attack accepted as bona fide -> FN / #attack.
    # BPCER: bona fide rejected as attack -> FP / #bonafide.
    apcer = fnr
    bpcer = fpr

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "apcer": float(apcer),
        "bpcer": float(bpcer),
        "num_bona_fide": n_neg,
        "num_spoof": n_pos,
    }


def compute_mcd(reference_audio: str | Path, synthesized_audio: str | Path, sr: int = 16_000) -> float:
    """Compute Mel-Cepstral Distortion between two audio files."""

    ref_wav, ref_sr = load_audio(reference_audio, sr=sr)
    syn_wav, syn_sr = load_audio(synthesized_audio, sr=sr)
    if ref_sr != syn_sr:
        raise ValueError("Reference and synthesized audio must use the same sample rate after resampling.")

    ref_mfcc = librosa.feature.mfcc(y=ref_wav, sr=ref_sr, n_mfcc=24, n_fft=512, hop_length=160)
    syn_mfcc = librosa.feature.mfcc(y=syn_wav, sr=syn_sr, n_mfcc=24, n_fft=512, hop_length=160)

    if ref_mfcc.shape[1] == 0 or syn_mfcc.shape[1] == 0:
        return float("nan")

    _, wp = librosa.sequence.dtw(X=ref_mfcc[1:], Y=syn_mfcc[1:], metric="euclidean")
    path = np.asarray(wp[::-1], dtype=np.int64)
    if len(path) == 0:
        return float("nan")

    diffs = ref_mfcc[1:, path[:, 0]] - syn_mfcc[1:, path[:, 1]]
    frame_cost = np.sqrt(2.0 * np.sum(np.square(diffs), axis=0))
    return float((10.0 / math.log(10.0)) * np.mean(frame_cost))


def evaluate_asr(manifest_path: str | Path, model_name: str, lm_corpus: str | Path | None, device: str | None) -> ASRMetrics:
    """Evaluate WER on a manifest with audio_path, reference_text, and optional language."""

    df = pd.read_csv(manifest_path)
    asr = ConstrainedWhisperASR(model_name=model_name, device=device, lm_corpus=lm_corpus)
    rows = []
    for _, row in df.iterrows():
        hyp = asr.transcribe(row["audio_path"], language=row.get("language", None), beam_size=5)
        rows.append(
            {
                "language": _normalize_label(row.get("language", "")),
                "wer": word_error_rate(str(row["reference_text"]), hyp),
            }
        )
    if not rows:
        return ASRMetrics(num_samples=0)

    result = ASRMetrics(num_samples=len(rows))
    result.wer_overall = float(np.mean([item["wer"] for item in rows]))
    en_rows = [item["wer"] for item in rows if item["language"] == "en"]
    hi_rows = [item["wer"] for item in rows if item["language"] == "hi"]
    result.wer_en = float(np.mean(en_rows)) if en_rows else None
    result.wer_hi = float(np.mean(hi_rows)) if hi_rows else None
    return result


def evaluate_lid_switches(manifest_path: str | Path, lid_checkpoint: str | Path, device: str | None) -> LIDSwitchMetrics:
    """Evaluate timestamp precision of language switches."""

    df = pd.read_csv(manifest_path)
    inferencer = LIDInferencer(lid_checkpoint, device=device)
    hop_seconds = inferencer.audio_config.hop_length / inferencer.audio_config.sample_rate
    tolerance = 0.2

    total_tp = total_fp = total_fn = 0
    all_errors: list[float] = []
    all_ref = []
    all_hyp = []

    for _, row in df.iterrows():
        pred = inferencer.predict(row["audio_path"])
        hyp_switches = predicted_switch_times(pred.frame_labels, hop_seconds)
        ref_switches = parse_time_list(row.get("switch_times", []))
        tp, fp, fn, errors = match_switches(ref_switches, hyp_switches, tolerance=tolerance)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_errors.extend(errors)
        all_ref.append(len(ref_switches))
        all_hyp.append(len(hyp_switches))

    if len(df) == 0:
        return LIDSwitchMetrics(num_samples=0)

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = (2.0 * precision * recall / max(1e-12, precision + recall)) if (precision + recall) > 0 else 0.0
    within = total_tp / max(1, sum(all_ref))
    mae_ms = float(np.mean(all_errors) * 1000.0) if all_errors else None

    return LIDSwitchMetrics(
        switch_within_200ms=float(within),
        switch_mae_ms=mae_ms,
        switch_precision=float(precision),
        switch_recall=float(recall),
        switch_f1=float(f1),
        num_samples=len(df),
    )


def evaluate_mcd(manifest_path: str | Path, sample_rate: int = 16_000) -> MCDMetrics:
    """Evaluate average MCD on pairs of reference and synthesized audio."""

    df = pd.read_csv(manifest_path)
    values: list[float] = []
    for _, row in df.iterrows():
        values.append(compute_mcd(row["reference_audio"], row["synthesized_audio"], sr=sample_rate))
    values = [value for value in values if np.isfinite(value)]
    if not values:
        return MCDMetrics(num_pairs=0)
    return MCDMetrics(mcd_mean=float(np.mean(values)), num_pairs=len(values))


def evaluate_spoof(
    manifest_path: str | Path,
    antispoof_checkpoint: str | Path,
    device: str | None,
    chunk_seconds: float | None = 8.0,
    chunk_overlap_seconds: float = 1.0,
    max_chunks: int | None = None,
    threshold: float = 0.5,
) -> SpoofMetrics:
    """Evaluate spoof detection EER."""

    df = pd.read_csv(manifest_path)
    inferencer = AntiSpoofInferencer(
        antispoof_checkpoint,
        device=device,
        chunk_seconds=chunk_seconds,
        chunk_overlap_seconds=chunk_overlap_seconds,
        max_chunks=max_chunks,
    )
    has_segments = {'start_sec', 'end_sec'}.issubset(set(df.columns))
    labels = []
    scores = []
    for _, row in df.iterrows():
        if has_segments and pd.notna(row.get('start_sec')) and pd.notna(row.get('end_sec')):
            path = str(row['audio_path'])
            info = sf.info(path)
            start_frame = max(0, int(float(row['start_sec']) * info.samplerate))
            end_frame = min(info.frames, int(float(row['end_sec']) * info.samplerate))
            frames = max(0, end_frame - start_frame)
            if frames == 0:
                continue
            audio, sr = sf.read(path, start=start_frame, frames=frames, dtype='float32', always_2d=False)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
            if sr != inferencer.audio_config.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=inferencer.audio_config.sample_rate).astype(np.float32)
            prediction = inferencer.predict(audio)
        else:
            prediction = inferencer.predict(row["audio_path"])
        labels.append(int(row["label"]))
        scores.append(float(prediction.probabilities[1]))
    if not labels:
        return SpoofMetrics(num_samples=0)

    labels_np = np.asarray(labels)
    scores_np = np.asarray(scores)
    eer, eer_threshold = compute_eer_details(labels_np, scores_np)
    auc_roc = compute_auc_roc(labels_np, scores_np)
    threshold_metrics = compute_threshold_metrics(labels_np, scores_np, threshold=threshold)

    return SpoofMetrics(
        eer=float(eer),
        eer_threshold=float(eer_threshold),
        auc_roc=float(auc_roc),
        threshold=float(threshold),
        accuracy_at_0_5=float(threshold_metrics["accuracy"]),
        precision_at_0_5=float(threshold_metrics["precision"]),
        recall_at_0_5=float(threshold_metrics["recall"]),
        f1_at_0_5=float(threshold_metrics["f1"]),
        fpr_at_0_5=float(threshold_metrics["fpr"]),
        fnr_at_0_5=float(threshold_metrics["fnr"]),
        apcer_at_0_5=float(threshold_metrics["apcer"]),
        bpcer_at_0_5=float(threshold_metrics["bpcer"]),
        tp=int(threshold_metrics["tp"]),
        fp=int(threshold_metrics["fp"]),
        tn=int(threshold_metrics["tn"]),
        fn=int(threshold_metrics["fn"]),
        num_bona_fide=int(threshold_metrics["num_bona_fide"]),
        num_spoof=int(threshold_metrics["num_spoof"]),
        num_samples=len(labels),
    )


def _majority_label(frame_labels: list[str]) -> str:
    if not frame_labels:
        return ""
    counts: dict[str, int] = {}
    for label in frame_labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)


def _predict_label_with_epsilon(inferencer: LIDInferencer, audio: np.ndarray, epsilon: float, seed: int = 13) -> str:
    rng = np.random.default_rng(seed)
    noise = rng.choice([-1.0, 1.0], size=audio.shape[0]).astype(np.float32)
    perturbed = np.clip(audio + epsilon * noise, -1.0, 1.0)
    prediction = inferencer.predict(perturbed)
    return _majority_label(prediction.frame_labels)


def estimate_min_epsilon_to_flip_lid(
    inferencer: LIDInferencer,
    audio_path: str | Path,
    eps_max: float = 0.1,
    eps_min: float = 1e-5,
    search_steps: int = 16,
) -> tuple[float | None, float | None]:
    """Estimate the minimum perturbation epsilon that flips the LID prediction."""

    audio, _sr = load_audio(audio_path, sr=inferencer.audio_config.sample_rate)
    base_prediction = inferencer.predict(audio)
    base_label = _majority_label(base_prediction.frame_labels)
    if not base_label:
        return None, None

    upper = eps_min
    upper_label = _predict_label_with_epsilon(inferencer, audio, upper)
    while upper <= eps_max and upper_label == base_label:
        upper *= 2.0
        upper_label = _predict_label_with_epsilon(inferencer, audio, upper)
    if upper > eps_max and upper_label == base_label:
        return None, None

    lower = upper / 2.0
    for _ in range(search_steps):
        mid = 0.5 * (lower + upper)
        mid_label = _predict_label_with_epsilon(inferencer, audio, mid)
        if mid_label == base_label:
            lower = mid
        else:
            upper = mid

    final_audio, _ = load_audio(audio_path, sr=inferencer.audio_config.sample_rate)
    rng = np.random.default_rng(13)
    noise = rng.choice([-1.0, 1.0], size=final_audio.shape[0]).astype(np.float32)
    perturbed = np.clip(final_audio + upper * noise, -1.0, 1.0)
    snr_db = estimate_snr_db(final_audio, perturbed - final_audio)
    return float(upper), float(snr_db)


def evaluate_attack(manifest_path: str | Path, lid_checkpoint: str | Path, device: str | None) -> AttackMetrics:
    """Evaluate minimum epsilon required to flip LID predictions."""

    df = pd.read_csv(manifest_path)
    inferencer = LIDInferencer(lid_checkpoint, device=device)
    epsilons: list[float] = []
    snrs: list[float] = []
    for _, row in df.iterrows():
        epsilon, snr_db = estimate_min_epsilon_to_flip_lid(inferencer, row["audio_path"])
        if epsilon is not None:
            epsilons.append(epsilon)
        if snr_db is not None:
            snrs.append(snr_db)
    if not epsilons:
        return AttackMetrics(num_samples=len(df))
    return AttackMetrics(
        min_epsilon_mean=float(np.mean(epsilons)),
        min_epsilon_median=float(np.median(epsilons)),
        snr_db_mean=float(np.mean(snrs)) if snrs else None,
        num_samples=len(df),
    )


def build_report(
    asr_manifest: str | Path | None,
    lid_manifest: str | Path | None,
    mcd_manifest: str | Path | None,
    spoof_manifest: str | Path | None,
    attack_manifest: str | Path | None,
    lid_checkpoint: str | Path | None,
    antispoof_checkpoint: str | Path | None,
    whisper_model: str,
    lm_corpus: str | Path | None,
    device: str | None,
    spoof_chunk_seconds: float | None = 8.0,
    spoof_chunk_overlap_seconds: float = 1.0,
    spoof_max_chunks: int | None = None,
    spoof_threshold: float = 0.5,
) -> dict[str, object]:
    """Run every requested metric and return a JSON-serializable report."""

    report: dict[str, object] = {}
    if asr_manifest is not None:
        report["asr"] = asdict(evaluate_asr(asr_manifest, whisper_model, lm_corpus, device))
    if lid_manifest is not None:
        if lid_checkpoint is None:
            raise ValueError("--lid-checkpoint is required for LID evaluation")
        report["lid_switch"] = asdict(evaluate_lid_switches(lid_manifest, lid_checkpoint, device))
    if mcd_manifest is not None:
        report["mcd"] = asdict(evaluate_mcd(mcd_manifest))
    if spoof_manifest is not None:
        if antispoof_checkpoint is None:
            raise ValueError("--antispoof-checkpoint is required for spoof evaluation")
        report["spoof"] = asdict(
            evaluate_spoof(
                spoof_manifest,
                antispoof_checkpoint,
                device,
                chunk_seconds=spoof_chunk_seconds,
                chunk_overlap_seconds=spoof_chunk_overlap_seconds,
                max_chunks=spoof_max_chunks,
                threshold=spoof_threshold,
            )
        )
    if attack_manifest is not None:
        if lid_checkpoint is None:
            raise ValueError("--lid-checkpoint is required for adversarial evaluation")
        report["attack"] = asdict(evaluate_attack(attack_manifest, lid_checkpoint, device))
    return report


def main() -> None:
    """CLI entry point for evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate the speech pipeline against benchmark metrics.")
    parser.add_argument("--asr-manifest", default=None, help="CSV with audio_path, reference_text, and optional language")
    parser.add_argument("--lid-manifest", default=None, help="CSV with audio_path and switch_times")
    parser.add_argument("--mcd-manifest", default=None, help="CSV with reference_audio and synthesized_audio")
    parser.add_argument("--spoof-manifest", default=None, help="CSV with audio_path and label")
    parser.add_argument("--attack-manifest", default=None, help="CSV with audio_path for LID attack robustness")
    parser.add_argument("--lid-checkpoint", default=None)
    parser.add_argument("--antispoof-checkpoint", default=None)
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--lm-corpus", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--spoof-chunk-seconds", type=float, default=8.0)
    parser.add_argument("--spoof-chunk-overlap-seconds", type=float, default=1.0)
    parser.add_argument("--spoof-max-chunks", type=int, default=None)
    parser.add_argument("--spoof-threshold", type=float, default=0.5, help="Threshold for confusion-matrix metrics on spoof probability.")
    parser.add_argument("--output", default="evaluation_report.json")
    args = parser.parse_args()

    report = build_report(
        asr_manifest=args.asr_manifest,
        lid_manifest=args.lid_manifest,
        mcd_manifest=args.mcd_manifest,
        spoof_manifest=args.spoof_manifest,
        attack_manifest=args.attack_manifest,
        lid_checkpoint=args.lid_checkpoint,
        antispoof_checkpoint=args.antispoof_checkpoint,
        whisper_model=args.whisper_model,
        lm_corpus=args.lm_corpus,
        device=args.device,
        spoof_chunk_seconds=args.spoof_chunk_seconds,
        spoof_chunk_overlap_seconds=args.spoof_chunk_overlap_seconds,
        spoof_max_chunks=args.spoof_max_chunks,
        spoof_threshold=args.spoof_threshold,
    )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

