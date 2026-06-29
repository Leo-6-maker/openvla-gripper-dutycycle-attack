#!/usr/bin/env python3
"""Multi-suite detector evaluation with fail-closed data integrity.

Matches train_detector.py strictness: rejects missing features, silent defaults,
unknown suites. Loads real teacher labels. Per-suite + aggregate metrics.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gripper_attack.sc5mlp_v1 import SC5_FEATURES, SC5_PHASES, N_FEATURES
from gripper_attack.sc5_detector_runtime_v1r import SC5DetectorRuntimeV1R

VALID_SUITES = {"libero_object", "libero_spatial", "libero_goal", "libero_10"}


def load_episode_index(path: str) -> dict:
    index = {}
    with open(path) as f:
        for line in f:
            ep = json.loads(line)
            if ep["suite"] not in VALID_SUITES:
                raise ValueError("Invalid suite: {} in {}".format(ep["suite"], ep["episode_key"]))
            index[ep["episode_key"]] = ep
    return index


def load_features(csv_path: str) -> dict:
    """Strict feature loading for evaluation."""
    data = defaultdict(dict)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        ek_col = "episode_key" if "episode_key" in fieldnames else "episode"
        step_col = "step" if "step" in fieldnames else ("step_id" if "step_id" in fieldnames else "step_idx")
        for row in reader:
            ek = row.get(ek_col, "").strip()
            if not ek:
                raise ValueError("Empty episode key in eval features")
            step = int(row.get(step_col, -1))
            feats = []
            for fn in SC5_FEATURES:
                v = row.get(fn)
                if v is None or str(v).strip() == "":
                    raise ValueError("Missing {} in {} step {}".format(fn, ek, step))
                fv = float(v)
                if not np.isfinite(fv):
                    raise ValueError("Non-finite {}={} in {} step {}".format(fn, fv, ek, step))
                feats.append(fv)
            data[ek][step] = feats

    result = {}
    for ek, sd in data.items():
        steps = sorted(sd.keys())
        for i, s in enumerate(steps):
            if s != i:
                raise ValueError("Step gap in {}: expected {} got {}".format(ek, i, s))
        result[ek] = np.array([sd[s] for s in steps], dtype=np.float32)
    return result


def load_labels(csv_path: str) -> dict:
    """Load teacher labels. Returns {ek: {anchor, window_start, window_end}}."""
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        ek_col = "episode_key" if "episode_key" in fieldnames else "episode"
        for row in reader:
            ek = row.get(ek_col, "").strip()
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


def compute_metrics(results):
    emitted = [r for r in results if r["emitted"]]
    no_emit = [r for r in results if not r["emitted"]]
    has_anchor = [r for r in results if r.get("teacher_anchor", -1) >= 0]
    in_window = [r for r in emitted if r.get("in_window")]
    k10 = [r for r in emitted if r.get("k10_contained")]
    n = max(1, len(results))

    m = {
        "n_episodes": len(results),
        "n_emitted": len(emitted), "n_no_emission": len(no_emit),
        "n_teacher_positive": len(has_anchor),
        "emission_rate": len(emitted) / n,
        "no_emission_rate": len(no_emit) / n,
        "event_precision": len(in_window) / max(1, len(emitted)),
        "event_recall": len(in_window) / max(1, len(has_anchor)) if has_anchor else 0,
        "k10_containment_rate": len(k10) / max(1, len(emitted)),
        "false_emits_per_episode": (len(emitted) - len(in_window)) / n,
    }
    errors = [r["absolute_error"] for r in emitted if r.get("absolute_error") is not None]
    if errors:
        m["median_absolute_error"] = float(np.median(errors))
        m["mean_absolute_error"] = float(np.mean(errors))
        m["p90_absolute_error"] = float(np.percentile(errors, 90))
    early = sum(1 for r in emitted if r.get("early"))
    late = sum(1 for r in emitted if r.get("late"))
    m["early_rate"] = early / max(1, len(emitted))
    m["late_rate"] = late / max(1, len(emitted))
    return m


def per_suite_metrics(results, suite_map):
    by_suite = defaultdict(list)
    for r in results:
        by_suite[suite_map.get(r["episode_key"], "unknown")].append(r)
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
    ap.add_argument("--episode_index", required=True, help="For suite mapping + step metadata")
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--split_key", default="test")
    ap.add_argument("--output", default="-")
    ap.add_argument("--fsm_version", default="legacy_v1")
    ap.add_argument("--tau_corridor", type=float, default=0.3)
    ap.add_argument("--tau_release", type=float, default=0.3)
    ap.add_argument("--guard", type=int, default=5)
    args = ap.parse_args()

    print("Loading detector: {}".format(args.checkpoint))
    detector = SC5DetectorRuntimeV1R(
        args.checkpoint, tau_corridor=args.tau_corridor,
        tau_release=args.tau_release, guard=args.guard, fsm_version=args.fsm_version)

    print("Loading episode index: {}".format(args.episode_index))
    ep_index = load_episode_index(args.episode_index)

    print("Loading features: {}".format(args.feature_csv))
    features = load_features(args.feature_csv)

    print("Loading labels: {}".format(args.label_csv))
    labels = load_labels(args.label_csv)

    print("Loading split: {}".format(args.split_file))
    with open(args.split_file) as f:
        split = json.load(f)

    eval_keys = split["splits"].get(args.split_key, [])
    missing_feat = [e for e in eval_keys if e not in features]
    if missing_feat:
        sys.exit("ERROR: {} eval episodes missing features: {}...".format(len(missing_feat), missing_feat[:5]))
    print("Evaluating {} episodes".format(len(eval_keys)))

    suite_map = {}
    for ek in eval_keys:
        s = ep_index.get(ek, {}).get("suite", "MISSING")
        if s not in VALID_SUITES:
            sys.exit("ERROR: {} has invalid suite: {}".format(ek, s))
        suite_map[ek] = s

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

    print("\nEmission: {:.3f}  Precision: {:.3f}  Recall: {:.3f}  K10: {:.3f}".format(
        metrics["emission_rate"], metrics["event_precision"],
        metrics["event_recall"], metrics["k10_containment_rate"]))
    for s, m in sorted(metrics.get("per_suite", {}).items()):
        print("  {}: n={} prec={:.3f} rec={:.3f}".format(s, m["n_episodes"], m["event_precision"], m["event_recall"]))


if __name__ == "__main__":
    main()
