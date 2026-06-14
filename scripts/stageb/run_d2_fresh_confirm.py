#!/usr/bin/env python3
"""D2 Phase D: Fresh confirmation — evaluate frozen seed42 checkpoint on new traces.
Loads the exact checkpoint from D1b training. Single evaluation pass.
Must match frozen checkpoint SHA: cdd3cbe4f42592dab81590d84f5a8ff67b9fc3b7
"""

import argparse, csv, hashlib, json, os, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

FROZEN_CHECKPOINT_SHA = "cdd3cbe4f42592dab81590d84f5a8ff67b9fc3b7326f691742b9a438f1174858"
TIE_TOLERANCE = 0.001

# Import model from training module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "stageb"))
# Use local imports relative to the server pipeline
PIPELINE_ROOT = "/data/liuyu/l12_e4c2_pipeline"
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "scripts", "stageb"))

from train_d1b_detector import CandidateRanker, FEATURE_NAMES, normalize_features


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def evaluate_model(model, traces, means, stdevs, impute, device):
    model.eval()
    results = []
    with torch.no_grad():
        for tid, candidates in traces.items():
            X = normalize_features(candidates, means, stdevs, impute).to(device)
            scores = model(X).cpu().numpy()
            max_score = scores.max()
            ties = [i for i, s in enumerate(scores) if abs(s - max_score) < TIE_TOLERANCE]

            tp_step = None; tp_idx = None
            for j, c in enumerate(candidates):
                if int(c.get("is_teacher_p", 0)) == 1:
                    tp_step = int(c["candidate_step"]); tp_idx = j; break

            if tp_idx is not None:
                tp_s = scores[tp_idx]
                n_higher = sum(1 for s in scores if s > tp_s + TIE_TOLERANCE)
                n_equal_tp = sum(1 for s in scores if abs(s - tp_s) < TIE_TOLERANCE)
            else:
                n_higher = -1; n_equal_tp = -1

            is_unique_top1 = (n_higher == 0 and n_equal_tp == 1)
            if len(ties) == 1:
                pred_step = int(candidates[ties[0]]["candidate_step"])
                abs_err = abs(pred_step - tp_step) if tp_step is not None else -1
            else:
                pred_step = -1; abs_err = -1

            results.append({
                "trace_id": tid,
                "task_key": candidates[0]["task_key"],
                "state_id": candidates[0]["state_id"],
                "n_candidates": len(candidates),
                "is_correct_top1": int(is_unique_top1),
                "tp_competition_rank": n_higher + 1 if n_higher >= 0 else -1,
                "n_ties_for_max": len(ties),
                "unique_decision": int(len(ties) == 1),
                "abs_error": abs_err,
            })
    return results


def evaluate_baseline(traces):
    results = []
    for tid, candidates in traces.items():
        scores = np.array([float(c.get("total_score", 0)) for c in candidates])
        max_score = scores.max()
        ties = [i for i, s in enumerate(scores) if abs(s - max_score) < TIE_TOLERANCE]
        tp_step = None; tp_idx = None
        for j, c in enumerate(candidates):
            if int(c.get("is_teacher_p", 0)) == 1:
                tp_step = int(c["candidate_step"]); tp_idx = j; break
        tp_s = scores[tp_idx]
        n_higher = sum(1 for s in scores if s > tp_s + TIE_TOLERANCE)
        n_equal_tp = sum(1 for s in scores if abs(s - tp_s) < TIE_TOLERANCE)
        is_unique_top1 = (n_higher == 0 and n_equal_tp == 1)
        if len(ties) == 1:
            pred_step = int(candidates[ties[0]]["candidate_step"])
            abs_err = abs(pred_step - tp_step) if tp_step is not None else -1
        else:
            pred_step = -1; abs_err = -1
        results.append({
            "trace_id": tid, "task_key": candidates[0]["task_key"],
            "state_id": candidates[0]["state_id"],
            "n_candidates": len(candidates),
            "is_correct_top1": int(is_unique_top1),
            "tp_competition_rank": n_higher + 1,
            "n_ties_for_max": len(ties),
            "unique_decision": int(len(ties) == 1),
            "abs_error": abs_err,
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidate-table", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # Verify checkpoint SHA
    actual = sha256_file(args.checkpoint)
    if actual != FROZEN_CHECKPOINT_SHA:
        print(f"FATAL: checkpoint SHA mismatch. Expected {FROZEN_CHECKPOINT_SHA[:16]}, got {actual[:16]}")
        sys.exit(1)
    print(f"Checkpoint SHA verified: {actual[:16]}...")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Checkpoint: epoch={ckpt['epoch']} val_acc={ckpt['val_top1_acc']:.4f}")

    candidates = list(csv.DictReader(open(args.candidate_table)))
    by_trace = defaultdict(list)
    for c in candidates:
        if int(c.get("is_teacher_p", 0)) == 1:
            by_trace[c["trace_id"]].append(c)

    # Only use ELIGIBLE_MULTI_CANDIDATE fresh traces (exactly 1 TP, >=2 candidates)
    multi_traces = {}
    for tid, cands in by_trace.items():
        n_pos = sum(1 for c in cands if int(c.get("is_teacher_p", 0)) == 1)
        if n_pos == 1 and len(cands) >= 2:
            multi_traces[tid] = cands
    print(f"Fresh eligible multi-candidate traces: {len(multi_traces)}")

    if len(multi_traces) == 0:
        print("RESULT: NO_FRESH_ELIGIBLE_MULTI_TRACES")
        return

    means = ckpt["normalization"]["means"]
    stdevs = ckpt["normalization"]["stdevs"]
    impute = ckpt["normalization"]["impute"]

    model = CandidateRanker(n_features=16).to(device)
    model.load_state_dict(ckpt["model_state"])

    model_results = evaluate_model(model, multi_traces, means, stdevs, impute, device)
    baseline_results = evaluate_baseline(multi_traces)

    n = len(model_results)
    m_correct = sum(r["is_correct_top1"] for r in model_results)
    b_correct = sum(r["is_correct_top1"] for r in baseline_results)
    m_ties = sum(1 for r in model_results if r["n_ties_for_max"] > 1)
    b_ties = sum(1 for r in baseline_results if r["n_ties_for_max"] > 1)
    m_mae = np.mean([r["abs_error"] for r in model_results if r["unique_decision"] == 1 and r["abs_error"] >= 0])
    b_mae = np.mean([r["abs_error"] for r in baseline_results if r["unique_decision"] == 1 and r["abs_error"] >= 0])

    print(f"\n=== FRESH CONFIRMATION (n={n}) ===")
    print(f"Model top-1: {m_correct}/{n} = {m_correct/n:.4f}  ties={m_ties}/{n}")
    print(f"Baseline top-1: {b_correct}/{n} = {b_correct/n:.4f}  ties={b_ties}/{n}")
    print(f"Model cond MAE: {m_mae:.1f}  Baseline cond MAE: {b_mae:.1f}")

    with open(out / "d2_fresh_confirmation.csv", "w", newline="") as f:
        fields = ["trace_id", "task_key", "state_id", "n_candidates",
                  "model_correct", "model_ties", "model_abs_error",
                  "baseline_correct", "baseline_ties", "baseline_abs_error"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for mr, br in zip(model_results, baseline_results):
            w.writerow({
                "trace_id": mr["trace_id"], "task_key": mr["task_key"],
                "state_id": mr["state_id"], "n_candidates": mr["n_candidates"],
                "model_correct": mr["is_correct_top1"], "model_ties": mr["n_ties_for_max"],
                "model_abs_error": mr["abs_error"],
                "baseline_correct": br["is_correct_top1"], "baseline_ties": br["n_ties_for_max"],
                "baseline_abs_error": br["abs_error"],
            })

    with open(out / "d2_fresh_confirmation_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "model", "baseline"]); w.writeheader()
        w.writerows([
            {"metric": "n_traces", "model": n, "baseline": n},
            {"metric": "top1", "model": f"{m_correct}/{n}", "baseline": f"{b_correct}/{n}"},
            {"metric": "top1_rate", "model": round(m_correct/n, 4), "baseline": round(b_correct/n, 4)},
            {"metric": "n_ties", "model": m_ties, "baseline": b_ties},
            {"metric": "cond_mae", "model": round(m_mae, 1), "baseline": round(b_mae, 1)},
        ])

    print(f"\nOutput: {out}")


if __name__ == "__main__":
    main()
