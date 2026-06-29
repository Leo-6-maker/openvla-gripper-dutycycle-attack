#!/usr/bin/env python3
"""Multi-suite detector evaluation using shared score_fsm_legacy_v1.

Strict checkpoint loading: fails on any state_dict mismatch.
Thresholds bound from checkpoint, CLI override rejected in formal mode.
Input SHAs cross-bound in output for provenance.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, torch
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
from score_fsm_legacy_v1 import run_fsm_legacy_v1, model_to_scores


REQUIRED_CHECKPOINT_KEYS = {
    "model_state", "mean", "std", "feature_names", "phase_classes",
}
SC5MLPV1_STATE_KEYS = {"shared.0.weight", "shared.0.bias", "shared.2.weight", "shared.2.bias",
                         "phase_head.weight", "phase_head.bias",
                         "corridor_head.weight", "corridor_head.bias",
                         "release_head.weight", "release_head.bias"}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_checkpoint_strict(path: str):
    """Load checkpoint with strict validation. Fails on any mismatch."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    for key in REQUIRED_CHECKPOINT_KEYS:
        if key not in ckpt:
            raise ValueError("Checkpoint missing required key: {}".format(key))

    mean = ckpt["mean"]
    std = ckpt["std"]
    if mean is None or std is None:
        raise ValueError("Checkpoint missing mean/std")
    if hasattr(mean, 'shape'):
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
    if mean.shape != (25,) or std.shape != (25,):
        raise ValueError("mean/std shape != (25,)")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("NaN/Inf in mean/std")
    if not np.all(std > 0):
        raise ValueError("Zero or negative in std")

    if list(ckpt["feature_names"]) != list(SC5_FEATURES):
        raise ValueError("feature_names mismatch")
    if list(ckpt["phase_classes"]) != list(SC5_PHASES):
        raise ValueError("phase_classes mismatch")

    # Strict model state loading
    state = ckpt["model_state"]
    state_keys = set(state.keys())
    if state_keys != SC5MLPV1_STATE_KEYS:
        extra = state_keys - SC5MLPV1_STATE_KEYS
        missing = SC5MLPV1_STATE_KEYS - state_keys
        msg = []
        if extra:
            msg.append("extra keys: {}".format(sorted(extra)))
        if missing:
            msg.append("missing keys: {}".format(sorted(missing)))
        raise ValueError("model_state key mismatch: {}".format("; ".join(msg)))

    model = SC5MLPV1()
    model.load_state_dict(state, strict=True)
    model.eval()

    # Thresholds: use checkpoint values if present, else require CLI to match defaults
    tau_c = float(ckpt.get("tau_corridor", 0.3))
    tau_r = float(ckpt.get("tau_release", 0.3))
    guard = int(ckpt.get("guard", 5))

    return model, mean, std, tau_c, tau_r, guard, ckpt


def evaluate_episode(model, features, teacher_info, tau_c, tau_r, guard, K=10):
    feats = torch.from_numpy(features)
    cp, rp, phase_names = model_to_scores(model, feats)
    emitted, emit_step = run_fsm_legacy_v1(cp, rp, phase_names, tau_c, tau_r, guard)

    anchor = teacher_info.get("anchor", -1)
    wstart = teacher_info.get("window_start", anchor)
    wend = teacher_info.get("window_end", anchor + K)
    has_event = teacher_info.get("has_event", anchor >= 0)
    in_window = bool(emitted and anchor >= 0 and wstart <= emit_step <= wend)

    return {
        "emitted": emitted, "emit_step": emit_step, "n_steps": len(features),
        "teacher_anchor": anchor,
        "anchor_error": emit_step - anchor if (emitted and anchor >= 0) else None,
        "absolute_error": abs(emit_step - anchor) if (emitted and anchor >= 0) else None,
        "in_window": in_window,
        "early": bool(emitted and anchor >= 0 and emit_step < wstart),
        "late": bool(emitted and anchor >= 0 and emit_step > wend),
        "k10_contained": bool(emitted and anchor >= 0 and wstart <= emit_step <= wstart + K),
        "has_teacher_event": has_event,
        "event_type": teacher_info.get("event_type", "unknown"),
    }


def compute_metrics(results):
    n = max(1, len(results))
    emitted = [r for r in results if r["emitted"]]
    no_emit = [r for r in results if not r["emitted"]]
    has_event = [r for r in results if r.get("has_teacher_event")]
    no_event = [r for r in results if not r.get("has_teacher_event")]

    tp = sum(1 for r in results if r["has_teacher_event"] and r["emitted"] and r["in_window"])
    fp = sum(1 for r in results if r["emitted"] and not r["in_window"])
    fn = sum(1 for r in results if r["has_teacher_event"] and not r["in_window"])
    tn = sum(1 for r in results if not r["has_teacher_event"] and not r["emitted"])
    in_window_emit = [r for r in emitted if r.get("in_window")]
    k10 = [r for r in emitted if r.get("k10_contained")]
    correct_abstention = sum(1 for r in no_emit if not r.get("has_teacher_event"))

    m = {
        "n_episodes": len(results), "n_emitted": len(emitted),
        "n_no_emission": len(no_emit),
        "n_teacher_positive": len(has_event), "n_teacher_negative": len(no_event),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "emission_rate": len(emitted) / n,
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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--feature_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--episode_index", required=True)
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--split_key", default="test")
    ap.add_argument("--output", default="-")
    ap.add_argument("--tau_corridor", type=float, default=None,
                    help="Override (must equal checkpoint value in formal mode)")
    ap.add_argument("--tau_release", type=float, default=None)
    ap.add_argument("--guard", type=int, default=None)
    args = ap.parse_args()

    # Strict checkpoint load
    model, mean, std, tau_c, tau_r, guard, ckpt_meta = load_checkpoint_strict(args.checkpoint)

    # Thresholds: CLI override must match checkpoint, or be unset
    for cli_val, ckpt_val, name in [
        (args.tau_corridor, tau_c, "tau_corridor"),
        (args.tau_release, tau_r, "tau_release"),
        (args.guard, guard, "guard"),
    ]:
        if cli_val is not None and cli_val != ckpt_val:
            sys.exit("CLI {}={} conflicts with checkpoint value={}".format(name, cli_val, ckpt_val))

    print("Checkpoint thresholds: tau_c={} tau_r={} guard={}".format(tau_c, tau_r, guard))
    print("Checkpoint metric: {}".format(ckpt_meta.get("checkpoint_metric", "unknown")))

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
        errors.append("{} eval episodes MISSING features".format(len(missing_feat)))
    if missing_event:
        errors.append("{} eval episodes MISSING teacher events".format(len(missing_event)))
    if errors:
        for e in errors:
            print("FAIL: {}".format(e))
        sys.exit(1)

    print("Evaluating {} episodes".format(len(eval_keys)))

    # Normalize with checkpoint mean/std
    for ek in list(features.keys()):
        features[ek] = (features[ek] - mean) / np.maximum(std, 1e-8)

    suite_map = {}
    for ek in eval_keys:
        s = ep_index[ek]["suite"]
        if s not in VALID_SUITES:
            sys.exit("Invalid suite: {} in {}".format(s, ek))
        suite_map[ek] = s

    results = []
    for ek in eval_keys:
        teacher_info = events[ek]
        r = evaluate_episode(model, features[ek], teacher_info, tau_c, tau_r, guard)
        r["episode_key"] = ek
        results.append(r)

    metrics = compute_metrics(results)
    metrics["split_key"] = args.split_key
    metrics["checkpoint_sha256"] = sha256_file(args.checkpoint)
    metrics["checkpoint_metric"] = ckpt_meta.get("checkpoint_metric", "unknown")
    metrics["checkpoint_epoch"] = ckpt_meta.get("epoch", -1)
    metrics["tau_corridor"] = tau_c
    metrics["tau_release"] = tau_r
    metrics["guard"] = guard
    metrics["input_binding"] = {
        "feature_csv_sha256": sha256_file(args.feature_csv),
        "label_csv_sha256": sha256_file(args.label_csv),
        "episode_index_sha256": sha256_file(args.episode_index),
        "split_file_sha256": sha256_file(args.split_file),
    }
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
