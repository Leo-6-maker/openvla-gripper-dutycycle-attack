#!/usr/bin/env python3
"""Multi-suite detector evaluation using shared strict_loader.

Fail-closed data integrity matching train_detector.py.
Uses teacher events with event_type distinction (primary/no_event/abstain/unsupported).
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strict_loader import (
    load_episode_index, load_features, load_teacher_events, VALID_SUITES,
)
from gripper_attack.sc5_detector_runtime_v1r import SC5DetectorRuntimeV1R


def evaluate_episode(detector, features, teacher_info, K=10):
    detector.reset()
    for step, feat in enumerate(features):
        d = detector.update(feat, step)
        if d.get("emitted"):
            emit_step = d.get("emit_step", -1)
            anchor = teacher_info.get("anchor", -1)
            wstart = teacher_info.get("window_start", anchor)
            wend = teacher_info.get("window_end", anchor + K)
            has_event = teacher_info.get("has_event", anchor >= 0)
            return {
                "emitted": True, "emit_step": emit_step, "n_steps": len(features),
                "teacher_anchor": anchor,
                "anchor_error": emit_step - anchor if anchor >= 0 else None,
                "absolute_error": abs(emit_step - anchor) if anchor >= 0 else None,
                "in_window": wstart <= emit_step <= wend if anchor >= 0 else False,
                "early": emit_step < wstart if anchor >= 0 else False,
                "late": emit_step > wend if anchor >= 0 else False,
                "k10_contained": wstart <= emit_step <= wstart + K if anchor >= 0 else False,
                "has_teacher_event": has_event,
                "event_type": teacher_info.get("event_type", "unknown"),
            }
    return {
        "emitted": False, "emit_step": -1, "n_steps": len(features),
        "teacher_anchor": teacher_info.get("anchor", -1),
        "no_emission": True, "anchor_error": None, "absolute_error": None,
        "in_window": False, "early": False, "late": False, "k10_contained": False,
        "has_teacher_event": teacher_info.get("has_event", False),
        "event_type": teacher_info.get("event_type", "unknown"),
    }


def compute_metrics(results):
    emitted = [r for r in results if r["emitted"]]
    no_emit = [r for r in results if not r["emitted"]]
    has_event = [r for r in results if r.get("has_teacher_event")]
    no_event = [r for r in results if not r.get("has_teacher_event")]
    in_window = [r for r in emitted if r.get("in_window")]
    k10 = [r for r in emitted if r.get("k10_contained")]
    n = max(1, len(results))

    # Correct abstention: no emit on no-event episode
    correct_abstention = sum(1 for r in no_emit if not r.get("has_teacher_event"))

    m = {
        "n_episodes": len(results),
        "n_emitted": len(emitted), "n_no_emission": len(no_emit),
        "n_teacher_positive": len(has_event), "n_teacher_negative": len(no_event),
        "emission_rate": len(emitted) / n,
        "no_emission_rate": len(no_emit) / n,
        "event_precision": len(in_window) / max(1, len(emitted)),
        "event_recall": len(in_window) / max(1, len(has_event)) if has_event else 0,
        "k10_containment_rate": len(k10) / max(1, len(emitted)),
        "false_emits_per_episode": (len(emitted) - len(in_window)) / n,
        "correct_abstention_rate": correct_abstention / max(1, len(no_event)) if no_event else 0,
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
    ap.add_argument("--label_csv", required=True, help="Teacher event CSV (per-episode anchors)")
    ap.add_argument("--episode_index", required=True)
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

    print("Loading teacher events: {}".format(args.label_csv))
    events = load_teacher_events(args.label_csv)

    print("Loading split: {}".format(args.split_file))
    with open(args.split_file) as f:
        split = json.load(f)

    eval_keys = split["splits"].get(args.split_key, [])

    # Fail-closed: reject missing features or events
    missing_feat = sorted([e for e in eval_keys if e not in features])
    missing_event = sorted([e for e in eval_keys if e not in events])
    errors = []
    if missing_feat:
        errors.append("{} eval episodes MISSING features: {}...".format(len(missing_feat), missing_feat[:5]))
    if missing_event:
        errors.append("{} eval episodes MISSING teacher events: {}...".format(len(missing_event), missing_event[:5]))
    if errors:
        for e in errors:
            print("FAIL: {}".format(e))
        sys.exit(1)

    print("Evaluating {} episodes".format(len(eval_keys)))

    suite_map = {}
    for ek in eval_keys:
        s = ep_index[ek]["suite"]
        if s not in VALID_SUITES:
            sys.exit("Invalid suite: {} in {}".format(s, ek))
        suite_map[ek] = s

    results = []
    for ek in eval_keys:
        teacher_info = events[ek]
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
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(out + "\n")

    print("\nEpisodes: {}  Emission: {:.3f}  Precision: {:.3f}  Recall: {:.3f}  K10: {:.3f}  Abstention: {:.3f}".format(
        metrics["n_episodes"], metrics["emission_rate"], metrics["event_precision"],
        metrics["event_recall"], metrics["k10_containment_rate"], metrics["correct_abstention_rate"]))
    for s, m in sorted(metrics.get("per_suite", {}).items()):
        print("  {}: n={} prec={:.3f} rec={:.3f}".format(s, m["n_episodes"], m["event_precision"], m["event_recall"]))


if __name__ == "__main__":
    main()
