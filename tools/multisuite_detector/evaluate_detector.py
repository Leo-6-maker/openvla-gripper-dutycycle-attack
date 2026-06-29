#!/usr/bin/env python3
"""Multi-suite detector evaluation using model logits + inline FSM replay.

Shares the same score-only FSM path as train_detector F1 checkpoint selection.
No runtime dependency — avoids 3-head/4-head state_dict incompatibility.
"""
from __future__ import annotations
import argparse, json, sys, torch
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strict_loader import (
    load_episode_index, load_features, load_teacher_events, VALID_SUITES,
)
from gripper_attack.sc5mlp_v1 import SC5MLPV1, SC5_FEATURES, SC5_PHASES, N_FEATURES, N_PHASES


def run_fsm(corridor_p, release_p, phase_names, tau_c, tau_r, guard):
    """Run legacy_v1 FSM: IDLE→ARMED→EMITTED. Returns (emitted, emit_step)."""
    state = "IDLE"
    arm_step = -1
    n = len(corridor_p)
    for step in range(n):
        if state == "IDLE":
            if phase_names[step] == "stable_carry" and corridor_p[step] > tau_c:
                state = "ARMED"
                arm_step = step
        elif state == "ARMED":
            if step >= arm_step + guard and corridor_p[step] > tau_c and release_p[step] < tau_r:
                return True, step
    return False, -1


def evaluate_episode_with_model(model, features, teacher_info, tau_c, tau_r, guard, K=10):
    """Evaluate one episode using model logits + inline FSM."""
    feats = torch.from_numpy(features)
    with torch.no_grad():
        out = model(feats)
    cp = torch.sigmoid(out["corridor_logit"]).squeeze(-1).numpy()
    rp = torch.sigmoid(out["release_logit"]).squeeze(-1).numpy()
    phase_idx = out["phase_logits"].argmax(dim=-1).numpy()
    phase_names = [SC5_PHASES[p] for p in phase_idx]

    emitted, emit_step = run_fsm(cp, rp, phase_names, tau_c, tau_r, guard)

    anchor = teacher_info.get("anchor", -1)
    wstart = teacher_info.get("window_start", anchor)
    wend = teacher_info.get("window_end", anchor + K)
    has_event = teacher_info.get("has_event", anchor >= 0)

    if emitted and anchor >= 0:
        in_window = wstart <= emit_step <= wend
    else:
        in_window = False

    return {
        "emitted": emitted, "emit_step": emit_step, "n_steps": len(features),
        "teacher_anchor": anchor,
        "anchor_error": emit_step - anchor if (emitted and anchor >= 0) else None,
        "absolute_error": abs(emit_step - anchor) if (emitted and anchor >= 0) else None,
        "in_window": in_window,
        "early": (emitted and anchor >= 0 and emit_step < wstart),
        "late": (emitted and anchor >= 0 and emit_step > wend),
        "k10_contained": (emitted and anchor >= 0 and wstart <= emit_step <= wstart + K),
        "has_teacher_event": has_event,
        "event_type": teacher_info.get("event_type", "unknown"),
    }


def compute_metrics(results):
    n = max(1, len(results))
    emitted = [r for r in results if r["emitted"]]
    no_emit = [r for r in results if not r["emitted"]]
    has_event = [r for r in results if r.get("has_teacher_event")]
    no_event = [r for r in results if not r.get("has_teacher_event")]

    # Correct accounting: wrong-time emission = FP (bad detection) + missed real event = FN
    tp = sum(1 for r in results if r["has_teacher_event"] and r["emitted"] and r["in_window"])
    fp = sum(1 for r in results if r["emitted"] and not r["in_window"])
    fn = sum(1 for r in results if r["has_teacher_event"] and not r["in_window"])
    tn = sum(1 for r in results if not r["has_teacher_event"] and not r["emitted"])
    in_window = [r for r in emitted if r.get("in_window")]
    k10 = [r for r in emitted if r.get("k10_contained")]
    correct_abstention = sum(1 for r in no_emit if not r.get("has_teacher_event"))

    m = {
        "n_episodes": len(results),
        "n_emitted": len(emitted), "n_no_emission": len(no_emit),
        "n_teacher_positive": len(has_event), "n_teacher_negative": len(no_event),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "emission_rate": len(emitted) / n,
        "no_emission_rate": len(no_emit) / n,
        "event_precision": tp / max(1, tp + fp),
        "event_recall": tp / max(1, tp + fn),
        "k10_containment_rate": len(k10) / max(1, len(emitted)),
        "false_emits_per_episode": fp / n,
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
    ap.add_argument("--checkpoint", required=True, help="SC5MLPV1 checkpoint .pt")
    ap.add_argument("--feature_csv", required=True)
    ap.add_argument("--label_csv", required=True, help="Teacher event CSV (per-episode)")
    ap.add_argument("--episode_index", required=True)
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--split_key", default="test")
    ap.add_argument("--output", default="-")
    ap.add_argument("--tau_corridor", type=float, default=0.3)
    ap.add_argument("--tau_release", type=float, default=0.3)
    ap.add_argument("--guard", type=int, default=5)
    args = ap.parse_args()

    print("Loading checkpoint: {}".format(args.checkpoint))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SC5MLPV1()
    state = {k: v for k, v in ckpt["model_state"].items() if not k.startswith("confidence_head")}
    model.load_state_dict(state, strict=False)
    model.eval()

    mean = ckpt.get("mean")
    std = ckpt.get("std")

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

    # Normalize
    if mean is not None and std is not None:
        for ek in list(features.keys()):
            features[ek] = (features[ek] - mean.astype(np.float32)) / np.maximum(std.astype(np.float32), 1e-8)

    suite_map = {}
    for ek in eval_keys:
        s = ep_index[ek]["suite"]
        if s not in VALID_SUITES:
            sys.exit("Invalid suite: {} in {}".format(s, ek))
        suite_map[ek] = s

    results = []
    for ek in eval_keys:
        teacher_info = events[ek]
        r = evaluate_episode_with_model(model, features[ek], teacher_info,
                                         args.tau_corridor, args.tau_release, args.guard)
        r["episode_key"] = ek
        results.append(r)

    metrics = compute_metrics(results)
    metrics["split_key"] = args.split_key
    metrics["checkpoint_metric"] = ckpt.get("checkpoint_metric", "unknown")
    metrics["per_suite"] = per_suite_metrics(results, suite_map)

    out = json.dumps(metrics, indent=2)
    if args.output == "-":
        print(out)
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(out + "\n")

    print("\nEpisodes: {}  TP: {}  FP: {}  FN: {}  TN: {}".format(
        metrics["n_episodes"], metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]))
    print("Precision: {:.3f}  Recall: {:.3f}  K10: {:.3f}  Abstention: {:.3f}".format(
        metrics["event_precision"], metrics["event_recall"],
        metrics["k10_containment_rate"], metrics["correct_abstention_rate"]))
    for s, m in sorted(metrics.get("per_suite", {}).items()):
        print("  {}: n={} TP={} FP={} FN={} prec={:.3f} rec={:.3f}".format(
            s, m["n_episodes"], m["tp"], m["fp"], m["fn"], m["event_precision"], m["event_recall"]))


if __name__ == "__main__":
    main()
