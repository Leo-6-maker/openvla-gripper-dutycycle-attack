#!/usr/bin/env python3
"""Multi-suite detector evaluation using real teacher labels and SC5DetectorRuntimeV1R.

Loads labels from label_csv and joins with features on episode_key.
Computes per-suite metrics: event precision/recall, timing, episode-level.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gripper_attack.sc5mlp_v1 import SC5_FEATURES, SC5_PHASES
from gripper_attack.sc5_detector_runtime_v1r import SC5DetectorRuntimeV1R


def load_episode_index(path: str) -> dict:
    index = {}
    with open(path) as f:
        for line in f:
            ep = json.loads(line)
            index[ep["episode_key"]] = ep
    return index


def load_features(csv_path: str) -> dict:
    data = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        ek_col = "episode_key" if "episode_key" in fieldnames else "episode"
        for row in reader:
            ek = row.get(ek_col, "")
            if not ek:
                continue
            feats = [float(row.get(f, 0)) for f in SC5_FEATURES]
            data[ek].append(feats)
    return {k: np.array(v, dtype=np.float32) for k, v in data.items()}


def load_labels(csv_path: str) -> dict:
    """Load teacher labels: {episode_key: {anchor, window_start, window_end, corridor}}."""
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        ek_col = "episode_key" if "episode_key" in fieldnames else "episode"
        for row in reader:
            ek = row.get(ek_col, "")
            if not ek:
                continue
            anchor = int(row.get("teacher_anchor_step", row.get("sc5_anchor", -1)))
            wstart = int(row.get("teacher_window_start", anchor))
            wend = int(row.get("teacher_window_end", anchor + 10))
            data[ek] = {"anchor": anchor, "window_start": wstart, "window_end": wend}
    return data


def evaluate_episode(detector, features, teacher_info, K=10):
    detector.reset()
    for step, feat in enumerate(features):
        d = detector.update(feat, step)
        if d.get("emitted"):
            emit_step = d.get("emit_step", -1)
            anchor = teacher_info.get("anchor", -1)
            wstart = teacher_info.get("window_start", anchor)
            wend = teacher_info.get("window_end", anchor + K)
            return {
                "emitted": True, "emit_step": emit_step, "n_steps": len(features),
                "teacher_anchor": anchor,
                "anchor_error": emit_step - anchor if anchor >= 0 else None,
                "absolute_error": abs(emit_step - anchor) if anchor >= 0 else None,
                "in_window": wstart <= emit_step <= wend if anchor >= 0 else False,
                "early": emit_step < wstart if anchor >= 0 else False,
                "late": emit_step > wend if anchor >= 0 else False,
                "k10_contained": wstart <= emit_step <= wstart + K if anchor >= 0 else False,
            }

    return {
        "emitted": False, "emit_step": -1, "n_steps": len(features),
        "teacher_anchor": teacher_info.get("anchor", -1),
        "no_emission": True, "anchor_error": None, "absolute_error": None,
        "in_window": False, "early": False, "late": False, "k10_contained": False,
    }


def compute_metrics(results, suite_map=None):
    emitted = [r for r in results if r["emitted"]]
    no_emit = [r for r in results if not r["emitted"]]
    has_anchor = [r for r in results if r.get("teacher_anchor", -1) >= 0]
    in_window = [r for r in emitted if r.get("in_window")]
    k10 = [r for r in emitted if r.get("k10_contained")]

    metrics = {
        "n_episodes": len(results),
        "n_emitted": len(emitted),
        "n_no_emission": len(no_emit),
        "n_teacher_positive": len(has_anchor),
        "emission_rate": len(emitted) / max(1, len(results)),
        "event_precision": len(in_window) / max(1, len(emitted)),
        "event_recall": len(in_window) / max(1, len(has_anchor)) if has_anchor else 0,
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
    metrics["false_emits_per_episode"] = (len(emitted) - len(in_window)) / max(1, len(results))
    return metrics


def per_suite_metrics(results, suite_map):
    by_suite = defaultdict(list)
    for r in results:
        s = suite_map.get(r.get("episode_key", ""), "unknown")
        by_suite[s].append(r)
    out = {}
    for s in sorted(by_suite):
        out[s] = compute_metrics(by_suite[s])
        out[s]["n_episodes"] = len(by_suite[s])
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate multi-suite detector")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--feature_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--episode_index", help="For suite mapping")
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--split_key", default="test")
    ap.add_argument("--output", default="-")
    ap.add_argument("--fsm_version", default="legacy_v1")
    ap.add_argument("--tau_corridor", type=float, default=0.3)
    ap.add_argument("--tau_release", type=float, default=0.3)
    ap.add_argument("--guard", type=int, default=5)
    args = ap.parse_args()

    print(f"Loading detector from {args.checkpoint}")
    detector = SC5DetectorRuntimeV1R(
        args.checkpoint, tau_corridor=args.tau_corridor,
        tau_release=args.tau_release, guard=args.guard,
        fsm_version=args.fsm_version)

    print(f"Loading features from {args.feature_csv}")
    features = load_features(args.feature_csv)
    print(f"Loading labels from {args.label_csv}")
    labels = load_labels(args.label_csv)
    print(f"Loading split from {args.split_file}")
    with open(args.split_file) as f:
        split = json.load(f)

    eval_keys = split["splits"].get(args.split_key, [])
    eval_keys = [e for e in eval_keys if e in features]
    print(f"Evaluating {len(eval_keys)} episodes")

    # Suite mapping
    suite_map = {}
    if args.episode_index:
        ep_idx = load_episode_index(args.episode_index)
        for ek in eval_keys:
            suite_map[ek] = ep_idx.get(ek, {}).get("suite", "unknown")
    else:
        for ek in eval_keys:
            suite_map[ek] = "unknown"

    results = []
    for ek in eval_keys:
        teacher_info = labels.get(ek, {"anchor": -1, "window_start": 0, "window_end": 10})
        r = evaluate_episode(detector, features[ek], teacher_info)
        r["episode_key"] = ek
        results.append(r)

    metrics = compute_metrics(results)
    metrics["split_key"] = args.split_key
    metrics["checkpoint_sha256"] = detector.checkpoint_sha256
    metrics["dataset_sha256"] = detector.dataset_sha256
    metrics["per_suite"] = per_suite_metrics(results, suite_map)

    out = json.dumps(metrics, indent=2)
    if args.output == "-":
        print(out)
    else:
        with open(args.output, "w") as f:
            f.write(out + "\n")

    print(f"\nEmission: {metrics['emission_rate']:.3f}  "
          f"Precision: {metrics['event_precision']:.3f}  "
          f"Recall: {metrics['event_recall']:.3f}  "
          f"K10: {metrics['k10_containment_rate']:.3f}")
    for s, m in metrics.get("per_suite", {}).items():
        print(f"  {s}: n={m['n_episodes']} prec={m['event_precision']:.3f} rec={m['event_recall']:.3f}")


if __name__ == "__main__":
    main()
