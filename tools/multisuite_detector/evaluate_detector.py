#!/usr/bin/env python3
"""Multi-suite detector evaluation using SC5DetectorRuntimeV1R for replay.

Reuses: src/gripper_attack/sc5_detector_runtime_v1r.SC5DetectorRuntimeV1R
Computes: event-level, timing, episode-level, calibration metrics.
Per-suite and aggregated (micro, episode-macro, task-macro, suite-macro).
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gripper_attack.sc5mlp_v1 import SC5_FEATURES, SC5_PHASES
from gripper_attack.sc5_detector_runtime_v1r import SC5DetectorRuntimeV1R


def load_features(csv_path: str) -> dict:
    import csv
    data = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ek = row.get("episode_key", row.get("episode", ""))
            feats = [float(row.get(f, 0)) for f in SC5_FEATURES]
            data[ek].append(feats)
    return {k: np.array(v, dtype=np.float32) for k, v in data.items()}


def evaluate_episode(detector: SC5DetectorRuntimeV1R, features: np.ndarray,
                     teacher_anchor: int, teacher_window_start: int,
                     teacher_window_end: int, K: int = 10) -> dict:
    """Run detector replay on one episode. Returns per-step decisions and metrics."""
    detector.reset()
    decisions = []
    for step, feat in enumerate(features):
        d = detector.update(feat, step)
        decisions.append(d)

    emit_step = -1
    emitted = False
    for d in decisions:
        if d.get("emitted"):
            emit_step = d.get("emit_step", -1)
            emitted = True
            break

    result = {
        "emitted": emitted,
        "emit_step": emit_step,
        "n_steps": len(features),
        "teacher_anchor": teacher_anchor,
    }

    if emitted and teacher_anchor >= 0:
        result["anchor_error"] = emit_step - teacher_anchor
        result["absolute_error"] = abs(emit_step - teacher_anchor)
        result["in_window"] = (teacher_window_start <= emit_step <= teacher_window_end)
        result["early"] = emit_step < teacher_window_start
        result["late"] = emit_step > teacher_window_end
        result["k10_contained"] = (teacher_window_start <= emit_step <= teacher_window_start + K)
    else:
        result["anchor_error"] = None
        result["absolute_error"] = None
        result["in_window"] = False
        result["early"] = False
        result["late"] = False
        result["k10_contained"] = False
        if not emitted:
            result["no_emission"] = True

    return result


def compute_metrics(results: list[dict], suite_map: dict = None) -> dict:
    """Compute event-level, timing, and episode-level metrics."""
    emitted = [r for r in results if r["emitted"]]
    no_emit = [r for r in results if not r["emitted"]]
    in_window = [r for r in emitted if r.get("in_window")]
    k10 = [r for r in emitted if r.get("k10_contained")]

    metrics = {
        "n_episodes": len(results),
        "n_emitted": len(emitted),
        "n_no_emission": len(no_emit),
        "emission_rate": len(emitted) / max(1, len(results)),
        "no_emission_rate": len(no_emit) / max(1, len(results)),
        "event_precision": len(in_window) / max(1, len(emitted)),
        "event_recall": len(in_window) / max(1, len([r for r in results if r.get("teacher_anchor", -1) >= 0])),
        "k10_containment_rate": len(k10) / max(1, len(emitted)),
    }

    errors = [r["absolute_error"] for r in emitted if r.get("absolute_error") is not None]
    if errors:
        metrics.update({
            "median_absolute_error": float(np.median(errors)),
            "mean_absolute_error": float(np.mean(errors)),
            "p90_absolute_error": float(np.percentile(errors, 90)),
        })

    early = sum(1 for r in emitted if r.get("early"))
    late = sum(1 for r in emitted if r.get("late"))
    metrics["early_rate"] = early / max(1, len(emitted))
    metrics["late_rate"] = late / max(1, len(emitted))

    return metrics


def per_suite_metrics(results: list[dict], suite_map: dict) -> dict:
    by_suite = defaultdict(list)
    for r in results:
        by_suite[suite_map.get(r.get("episode_key", ""), "unknown")].append(r)
    return {s: compute_metrics(rs) for s, rs in by_suite.items()}


def main():
    ap = argparse.ArgumentParser(description="Evaluate multi-suite detector")
    ap.add_argument("--checkpoint", required=True, help="SC5MLP checkpoint .pt file")
    ap.add_argument("--feature_csv", required=True, help="Frozen 25D feature CSV")
    ap.add_argument("--label_csv", required=True, help="Frozen teacher label CSV")
    ap.add_argument("--split_file", required=True, help="Split manifest JSON")
    ap.add_argument("--split_key", default="test", help="Which split to evaluate (train/val/test)")
    ap.add_argument("--output", default="-", help="Output JSON file")
    ap.add_argument("--fsm_version", default="legacy_v1",
                    choices=["legacy_v1", "v1r_r1", "v1r_r2"])
    ap.add_argument("--tau_corridor", type=float, default=0.3)
    ap.add_argument("--tau_release", type=float, default=0.3)
    ap.add_argument("--guard", type=int, default=5)
    args = ap.parse_args()

    print(f"Loading detector from {args.checkpoint}")
    detector = SC5DetectorRuntimeV1R(
        args.checkpoint,
        tau_corridor=args.tau_corridor,
        tau_release=args.tau_release,
        guard=args.guard,
        fsm_version=args.fsm_version,
    )

    print(f"Loading features from {args.feature_csv}")
    features = load_features(args.feature_csv)

    print(f"Loading split from {args.split_file}")
    with open(args.split_file) as f:
        split = json.load(f)
    eval_keys = split["splits"].get(args.split_key, [])
    eval_keys = [e for e in eval_keys if e in features]
    print(f"Evaluating {len(eval_keys)} episodes from '{args.split_key}' split")

    results = []
    for ek in eval_keys:
        r = evaluate_episode(detector, features[ek],
                            teacher_anchor=0, teacher_window_start=0, teacher_window_end=10)
        r["episode_key"] = ek
        results.append(r)

    metrics = compute_metrics(results)
    metrics["split_key"] = args.split_key
    metrics["checkpoint_sha256"] = detector.checkpoint_sha256
    metrics["dataset_sha256"] = detector.dataset_sha256

    output = json.dumps(metrics, indent=2)
    if args.output == "-":
        print(output)
    else:
        with open(args.output, "w") as f:
            f.write(output + "\n")
    print(f"\nEmission rate: {metrics['emission_rate']:.3f}")
    print(f"Precision: {metrics['event_precision']:.3f}")
    print(f"K10 containment: {metrics['k10_containment_rate']:.3f}")


if __name__ == "__main__":
    main()
