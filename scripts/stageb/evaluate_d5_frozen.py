#!/usr/bin/env python3
"""D5 formal evaluation — frozen config, per-trace readout, artifact SHAs.

Reads frozen config (tau, checkpoint path). Evaluates on all splits.
Outputs: per-trace CSV, summary JSON, artifact manifest.
"""
import argparse, csv, hashlib, json, math, os, sys, time
from collections import defaultdict, Counter

import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from train_d1b_detector import CandidateRanker, FEATURE_NAMES, normalize_features


def sha256_file(path):
    if not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s):
    return hashlib.sha256(s.encode()).hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--accepted-manifest", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--capture-roots", required=True,
                   help="JSON: {name: path}")
    p.add_argument("--config", required=True,
                   help="JSON: {tau, checkpoint_path, ...}")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="all",
                   choices=["train", "val", "test", "external", "all"])
    return p.parse_args()


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CandidateRanker(n_features=16)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    means = ckpt["means"]; stdevs = ckpt["stdevs"]; impute = ckpt["impute"]
    return model, means, stdevs, impute, ckpt


def online_detect(edir, model, means, stdevs, impute, tau):
    ccf = os.path.join(edir, "detector_candidates.csv")
    if not os.path.exists(ccf):
        return {"emit_step": -1, "emit_score": 0.0, "n_candidates": 0,
                "n_abstained": 0, "emit_abstained": False, "all_scores": []}

    cands = list(csv.DictReader(open(ccf)))
    candidates = []
    for c in cands:
        step = int(c["step"])
        abstained = int(c.get("abstained", 0) or 0) == 1
        row = {}
        for fn in FEATURE_NAMES:
            row[fn] = c.get("feat_" + fn, c.get(fn, ""))
        candidates.append({"step": step, "features": row, "abstained": abstained})
    candidates.sort(key=lambda x: x["step"])

    all_scores = []
    emit_step = -1; emit_score = 0.0; emit_abstained = False
    for cand in candidates:
        X = normalize_features([cand["features"]], means, stdevs, impute)
        with torch.no_grad():
            score = float(model(X).item())
        all_scores.append({"step": cand["step"], "score": round(score, 6),
                           "abstained": cand["abstained"]})
        if emit_step < 0:
            if not cand["abstained"] and score >= tau:
                emit_step = cand["step"]
                emit_score = score
                emit_abstained = cand["abstained"]

    return {"emit_step": emit_step, "emit_score": emit_score,
            "n_candidates": len(candidates),
            "n_abstained": sum(1 for c in candidates if c["abstained"]),
            "emit_abstained": emit_abstained,
            "all_scores": all_scores}


def classify_emit(emit_step, anchor, ws, we):
    if emit_step < 0:
        return "miss"
    if emit_step < ws:
        return "pre_window_early"
    if emit_step == anchor:
        return "exact_anchor"
    if emit_step < anchor:
        return "in_window_pre_anchor"
    if emit_step < we:
        return "in_window_post_anchor"
    return "late"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    config = json.load(open(args.config))
    tau = float(config["tau"])
    ckpt_path = config["checkpoint_path"]

    # Load data
    accepted = {}
    for r in csv.DictReader(open(args.accepted_manifest)):
        if r.get("status") == "BOUND":
            accepted[(r["task"], int(r["state_id"]))] = r

    labels = {}
    for r in csv.DictReader(open(args.labels)):
        labels[(r["task"], int(r["state_id"]))] = r

    roots = json.load(open(args.capture_roots))

    # Load model
    model, means, stdevs, impute, ckpt = load_model(ckpt_path)

    # ── Evaluate ──
    results = []
    for (task, sid), acc in sorted(accepted.items()):
        lp = labels.get((task, sid), {})
        sp = acc.get("split", "?")
        if args.split != "all" and sp != args.split:
            continue

        p_status = lp.get("status", "UNKNOWN")
        p_anchor = int(lp.get("anchor", -1))
        p_ws = int(lp.get("ws", -1))
        p_we = int(lp.get("we", -1))

        rname = acc["accepted_root"]
        edir_name = acc["accepted_episode_dir"]
        rpath = roots.get(rname, "")
        edir = os.path.join(rpath, edir_name) if rpath else ""

        det = online_detect(edir, model, means, stdevs, impute, tau)

        # Classify
        emit_cls = classify_emit(det["emit_step"], p_anchor, p_ws, p_we)
        in_window = det["emit_step"] >= p_ws and det["emit_step"] < p_we if det["emit_step"] >= 0 else False
        offset = det["emit_step"] - p_anchor if det["emit_step"] >= 0 else None
        is_labeled = p_status == "VALID_LABELED"

        results.append({
            "task": task, "state_id": sid, "split": sp,
            "teacher_p_status": p_status,
            "teacher_p_anchor": p_anchor, "ws": p_ws, "we": p_we,
            "emit_step": det["emit_step"], "emit_score": det["emit_score"],
            "emit_class": emit_cls, "in_window": int(in_window),
            "offset": offset if offset is not None else "",
            "n_candidates": det["n_candidates"],
            "n_abstained_cands": det["n_abstained"],
            "is_labeled": int(is_labeled),
        })

    # ── Per-trace output ──
    out_csv = os.path.join(args.output_dir, "d5_evaluation_readout.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print("Per-trace: {}".format(out_csv))

    # ── Summary ──
    labeled = [r for r in results if r["is_labeled"]]
    n = len(labeled)

    def pct(numer, denom):
        return round(100 * numer / denom, 1) if denom > 0 else 0.0

    in_win = sum(r["in_window"] for r in labeled)
    exact = sum(1 for r in labeled if r["emit_class"] == "exact_anchor")
    early = sum(1 for r in labeled if r["emit_class"] == "pre_window_early")
    miss = sum(1 for r in labeled if r["emit_class"] == "miss")
    late = sum(1 for r in labeled if r["emit_class"] == "late")
    emit = sum(1 for r in labeled if r["emit_step"] >= 0)
    offsets = [r["offset"] for r in labeled if isinstance(r["offset"], (int, float))]

    summary = {
        "tau": tau,
        "n_labeled": n,
        "n_total": len(results),
        "n_abstain": sum(1 for r in results if r["teacher_p_status"] == "VALID_TEACHER_P_ABSTAIN"),
        "emit_pct": pct(emit, n),
        "in_window_pct": pct(in_win, n),
        "exact_pct": pct(exact, n),
        "early_pct": pct(early, n),
        "late_pct": pct(late, n),
        "miss_pct": pct(miss, n),
        "median_offset": int(np.median(offsets)) if offsets else None,
        "mean_abs_offset": round(float(np.mean([abs(o) for o in offsets])), 1) if offsets else None,
    }

    print("\n=== D5 Frozen Evaluation (tau={:.3f}) ===".format(tau))
    print("Labeled: {} / {} (+ {} abstain)".format(n, len(results), summary["n_abstain"]))
    print("Emit:    {:.1f}% ({}/{})".format(summary["emit_pct"], emit, n))
    print("In-Win:  {:.1f}% ({}/{})".format(summary["in_window_pct"], in_win, n))
    print("Exact:   {:.1f}%".format(summary["exact_pct"]))
    print("Early:   {:.1f}%".format(summary["early_pct"]))
    print("Late:    {:.1f}%".format(summary["late_pct"]))
    print("Miss:    {:.1f}%".format(summary["miss_pct"]))
    print("Med off: {}".format(summary["median_offset"]))
    print("MAE:     {}".format(summary["mean_abs_offset"]))

    # Per-task
    print("\nPer-task (test only):")
    for tk in sorted(set(r["task"] for r in results if r["split"] == "test")):
        tr = [r for r in labeled if r["task"] == tk and r["split"] == "test"]
        if not tr: continue
        iw = sum(r["in_window"] for r in tr)
        print("  {}: {}/{} ({:.0f}%)".format(tk, iw, len(tr), pct(iw, len(tr))))

    # ── Artifact manifest ──
    artifacts = {
        "code_commit": os.popen("git rev-parse HEAD").read().strip(),
        "evaluation_script_sha256": sha256_file(__file__),
        "accepted_manifest_sha256": sha256_file(args.accepted_manifest),
        "labels_sha256": sha256_file(args.labels),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "config_sha256": sha256_file(args.config),
        "capture_roots_sha256": sha256_file(args.capture_roots),
        "tau": tau,
        "feature_schema": FEATURE_NAMES,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Save summary
    out_json = os.path.join(args.output_dir, "d5_evaluation_summary.json")
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "artifacts": artifacts}, f, indent=2)
    print("\nSummary: {}".format(out_json))

    # Save artifact manifest
    out_art = os.path.join(args.output_dir, "d5_artifact_manifest.json")
    with open(out_art, "w") as f:
        json.dump(artifacts, f, indent=2)
    print("Artifacts: {}".format(out_art))

    return 0


if __name__ == "__main__":
    sys.exit(main())
