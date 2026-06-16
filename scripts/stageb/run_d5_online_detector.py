#!/usr/bin/env python3
"""D5 online causal stopping detector — first-threshold-crossing.

Reuses:
  - CandidateRanker from train_d1b_detector.py
  - Accepted episode manifest for exact episode binding
  - Teacher-P labels v2 for supervision

Online rule:
  At each CLOSE candidate step t (in causal order):
    1. Compute D5 score using only features at step <= t
    2. If score >= frozen_tau: EMIT immediately, STOP waiting
    3. Otherwise: continue to next candidate
  After last candidate without emission: MISS

Tau is frozen on validation set only.
Test is evaluated exactly once.
"""
import csv, json, os, sys, math, argparse
from collections import defaultdict, Counter

sys.path.insert(0, "/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/scripts/stageb")

import torch
import numpy as np
from train_d1b_detector import CandidateRanker, FEATURE_NAMES, normalize_features

ROOTS = {
    "orig": "/data/liuyu/outputs/d5_120_privileged_capture",
    "gpu13": "/data/liuyu/outputs/d44d_balanced120_gpu13_r1",
    "gpu26": "/data/liuyu/outputs/d44d_balanced120_gpu26_r1",
    "gpu50": "/data/liuyu/outputs/d44d_balanced120_gpu50_r1",
}
LABELS = "/data/liuyu/outputs/d5_label_generation/d5_teacher_p_labels_v2.csv"
ACCEPTED = "/data/liuyu/outputs/d5_label_generation/d44d_accepted_episode_manifest.csv"
CKPT = "/data/liuyu/outputs/d5_training/d5_candidate_best.pt"


def load_data():
    labels = {}
    for r in csv.DictReader(open(LABELS)):
        labels[(r["task"], int(r["state_id"]))] = r

    accepted = {}
    for r in csv.DictReader(open(ACCEPTED)):
        if r.get("status") == "BOUND":
            accepted[(r["task"], int(r["state_id"]))] = r

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    means = ckpt["means"]; stdevs = ckpt["stdevs"]; impute = ckpt["impute"]
    model = CandidateRanker(n_features=16)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return labels, accepted, model, means, stdevs, impute


def online_detect(edir, model, means, stdevs, impute, tau):
    """Causal online detection: first candidate with score >= tau wins."""
    ccf = os.path.join(edir, "detector_candidates.csv")
    if not os.path.exists(ccf):
        return -1, 0.0, []

    cands = list(csv.DictReader(open(ccf)))
    if not cands:
        return -1, 0.0, []

    candidates = []
    for c in cands:
        step = int(c["step"])
        abstained = int(c.get("abstained", 0) or 0) == 1
        row = {}
        for fn in FEATURE_NAMES:
            row[fn] = c.get("feat_" + fn, c.get(fn, ""))
        candidates.append({"step": step, "features": row, "abstained": abstained})

    # Sort by step (causal order)
    candidates.sort(key=lambda x: x["step"])

    all_scores = []
    for cand in candidates:
        features = [cand["features"]]
        X = normalize_features(features, means, stdevs, impute)
        with torch.no_grad():
            score = float(model(X).item())
        all_scores.append({"step": cand["step"], "score": score, "abstained": cand["abstained"]})
        if not cand["abstained"] and score >= tau:
            return cand["step"], score, all_scores

    # No emission
    return -1, 0.0, all_scores


def evaluate_split(split_name, episodes, model, means, stdevs, impute, tau):
    results = []
    for key, acc, lp in episodes:
        task, sid = key
        rname = acc["accepted_root"]
        edir_name = acc["accepted_episode_dir"]
        rpath = ROOTS.get(rname, "")
        edir = os.path.join(rpath, edir_name) if rpath else ""
        if not os.path.isdir(edir):
            continue

        p_anchor = int(lp["anchor"])
        p_ws = int(lp["ws"])
        p_we = int(lp["we"])
        sp = acc["split"]

        emit_step, emit_score, all_scores = online_detect(
            edir, model, means, stdevs, impute, tau)

        in_window = emit_step >= p_ws and emit_step < p_we if emit_step >= 0 else False
        at_anchor = emit_step == p_anchor
        early = emit_step < p_ws if emit_step >= 0 else False
        late = emit_step >= p_we if emit_step >= 0 else False
        miss = emit_step < 0
        offset = emit_step - p_anchor if emit_step >= 0 else None

        results.append({
            "task": task, "sid": sid, "split": sp,
            "anchor": p_anchor, "ws": p_ws, "we": p_we,
            "emit_step": emit_step, "emit_score": emit_score,
            "in_window": in_window, "at_anchor": at_anchor,
            "early": early, "late": late, "miss": miss,
            "offset": offset if offset is not None else "",
            "n_candidates": len(all_scores),
        })

    return results


def compute_metrics(results):
    n = len(results)
    emitted = sum(1 for r in results if not r["miss"])
    in_win = sum(1 for r in results if r["in_window"])
    exact = sum(1 for r in results if r["at_anchor"])
    early = sum(1 for r in results if r["early"])
    late = sum(1 for r in results if r["late"])
    miss = sum(1 for r in results if r["miss"])
    offsets = [r["offset"] for r in results if isinstance(r["offset"], (int, float))]

    return {
        "n": n, "emitted": emitted, "emitted_pct": 100 * emitted / n,
        "in_window": in_win, "in_window_pct": 100 * in_win / n,
        "exact": exact, "exact_pct": 100 * exact / n,
        "early": early, "early_pct": 100 * early / n,
        "late": late, "late_pct": 100 * late / n,
        "miss": miss, "miss_pct": 100 * miss / n,
        "median_offset": sorted(offsets)[len(offsets) // 2] if offsets else None,
        "mean_abs_offset": np.mean([abs(o) for o in offsets]) if offsets else None,
    }


def main():
    labels, accepted, model, means, stdevs, impute = load_data()

    # Collect all labeled episodes
    all_eps = []
    for (task, sid), acc in sorted(accepted.items()):
        lp = labels.get((task, sid), {})
        if lp.get("status") != "VALID_LABELED":
            continue
        all_eps.append(((task, sid), acc, lp))

    # Split
    val_eps = [(k, a, l) for k, a, l in all_eps if a["split"] == "val"]
    test_eps = [(k, a, l) for k, a, l in all_eps if a["split"] == "test"]
    train_eps = [(k, a, l) for k, a, l in all_eps if a["split"] == "train"]

    print("Train: {}  Val: {}  Test: {}".format(len(train_eps), len(val_eps), len(test_eps)))

    # ── Tau selection on validation ──
    print("\n=== Tau sweep on validation ===")
    best_tau = 0.0
    best_in_win = 0.0

    for tau in [round(x * 0.05, 3) for x in range(-60, 60)]:
        results = evaluate_split("val", val_eps, model, means, stdevs, impute, tau)
        metrics = compute_metrics(results)
        if metrics["in_window_pct"] > best_in_win:
            best_in_win = metrics["in_window_pct"]
            best_tau = tau

    print("Best tau: {:.3f} (val in-window: {:.1f}%)".format(best_tau, best_in_win))

    # ── Evaluate all splits with best_tau ──
    print("\n=== Final evaluation (tau={:.3f}) ===".format(best_tau))

    for split_name, eps in [("Train", train_eps), ("Val", val_eps), ("Test", test_eps)]:
        results = evaluate_split(split_name, eps, model, means, stdevs, impute, best_tau)
        m = compute_metrics(results)

        print("\n{} (n={}):".format(split_name, m["n"]))
        print("  Emit:    {:5.1f}% ({}/{})".format(m["emitted_pct"], m["emitted"], m["n"]))
        print("  In-Win:  {:5.1f}% ({}/{})".format(m["in_window_pct"], m["in_window"], m["n"]))
        print("  Exact:   {:5.1f}% ({}/{})".format(m["exact_pct"], m["exact"], m["n"]))
        print("  Early:   {:5.1f}% ({}/{})".format(m["early_pct"], m["early"], m["n"]))
        print("  Late:    {:5.1f}% ({}/{})".format(m["late_pct"], m["late"], m["n"]))
        print("  Miss:    {:5.1f}% ({}/{})".format(m["miss_pct"], m["miss"], m["n"]))
        print("  Med off: {:+d}".format(int(m["median_offset"])) if m["median_offset"] is not None else "  N/A")
        print("  MAE:     {:.1f}".format(m["mean_abs_offset"]) if m["mean_abs_offset"] is not None else "  N/A")

    # Per-task Test
    print("\n=== Per-task Test ===")
    test_results = evaluate_split("test", test_eps, model, means, stdevs, impute, best_tau)
    task_m = defaultdict(list)
    for r in test_results:
        task_m[r["task"]].append(r)
    for tk in sorted(task_m):
        tr = task_m[tk]
        in_win = sum(1 for r in tr if r["in_window"])
        print("  {}: {}/{} ({:.0f}%)".format(tk, in_win, len(tr), 100 * in_win / len(tr)))

    # Save config
    config = {
        "tau": best_tau,
        "val_in_window_pct": best_in_win,
        "stopping_rule": "first_threshold_crossing",
        "model_sha": CKPT,
    }
    out_path = "/data/liuyu/outputs/d5_training/d5_online_config.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print("\nFrozen config: {} (tau={:.3f})".format(out_path, best_tau))

    return 0


if __name__ == "__main__":
    sys.exit(main())
