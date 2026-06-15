#!/usr/bin/env python3
"""D2.1a: Fresh confirmation — evaluate frozen seed42 checkpoint on new traces.
Must match frozen checkpoint SHA: cdd3cbe4f42592dab81590d84f5a8ff67b9fc3b7

D2.1a fixes:
  - select_eligible_multi_traces() is the single source of truth for grouping.
  - Sentinel prevents second evaluation.
  - tasks-represented gate for FRESH_SAMPLE_TOO_SMALL classification.
"""

import argparse, csv, hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

FROZEN_CHECKPOINT_SHA = "cdd3cbe4f42592dab81590d84f5a8ff67b9fc3b7326f691742b9a438f1174858"
TIE_TOLERANCE = 0.001
MIN_FRESH_MULTI = 20
MIN_TASKS_REPRESENTED = 8

PIPELINE_ROOT = "/data/liuyu/l12_e4c2_pipeline"
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "scripts", "stageb"))

from train_d1b_detector import CandidateRanker, FEATURE_NAMES, normalize_features


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def select_eligible_multi_traces(candidate_rows, status_rows):
    """Select traces eligible for fresh multi-candidate confirmation.

    A trace is eligible iff:
      1. Its category in status_rows is ELIGIBLE_MULTI_CANDIDATE
      2. It has >=2 total candidates
      3. Exactly 1 candidate has is_teacher_p == 1

    Args:
        candidate_rows: list of dicts from close_candidates CSV
        status_rows: list of dicts from trace_status CSV

    Returns:
        dict: {trace_id: [candidate_dicts]} for eligible traces
    """
    eligible_ids = {r["trace_id"] for r in status_rows
                    if r["category"] == "ELIGIBLE_MULTI_CANDIDATE"}

    by_trace = defaultdict(list)
    for c in candidate_rows:
        by_trace[c["trace_id"]].append(c)

    multi_traces = {}
    for tid, cands in by_trace.items():
        if tid not in eligible_ids:
            continue
        n_pos = sum(1 for c in cands if int(c.get("is_teacher_p", 0)) == 1)
        if n_pos == 1 and len(cands) >= 2:
            multi_traces[tid] = cands

    return multi_traces


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
                "trace_id": tid, "task_key": candidates[0]["task_key"],
                "state_id": candidates[0]["state_id"], "n_candidates": len(candidates),
                "is_correct_top1": int(is_unique_top1),
                "tp_competition_rank": n_higher + 1 if n_higher >= 0 else -1,
                "n_ties_for_max": len(ties), "unique_decision": int(len(ties) == 1),
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
            "state_id": candidates[0]["state_id"], "n_candidates": len(candidates),
            "is_correct_top1": int(is_unique_top1),
            "tp_competition_rank": n_higher + 1,
            "n_ties_for_max": len(ties), "unique_decision": int(len(ties) == 1),
            "abs_error": abs_err,
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidate-table", required=True)
    ap.add_argument("--trace-status", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # ── Sentinel ──
    sentinel_path = out / "fresh_confirmation_started.json"
    if sentinel_path.exists():
        print("FATAL: fresh confirmation already started (sentinel exists)")
        sys.exit(2)
    with open(sentinel_path, "w") as f:
        json.dump({
            "checkpoint_sha": sha256_file(args.checkpoint),
            "candidate_table_sha": sha256_file(args.candidate_table),
            "trace_status_sha": sha256_file(args.trace_status),
            "timestamp": str(np.datetime64('now')),
            "runner_sha": sha256_file(__file__),
        }, f, indent=2)

    # Verify checkpoint SHA
    actual = sha256_file(args.checkpoint)
    if actual != FROZEN_CHECKPOINT_SHA:
        print(f"FATAL: checkpoint SHA mismatch")
        sys.exit(1)
    print(f"Checkpoint SHA verified: {actual[:16]}...")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Checkpoint: epoch={ckpt['epoch']} val_acc={ckpt['val_top1_acc']:.4f}")

    # Load and select
    candidates = list(csv.DictReader(open(args.candidate_table)))
    status_rows = list(csv.DictReader(open(args.trace_status)))
    multi_traces = select_eligible_multi_traces(candidates, status_rows)
    n_tasks = len(set(multi_traces[tid][0]["task_key"] for tid in multi_traces))

    print(f"Candidates: {len(candidates)}  Status rows: {len(status_rows)}")
    print(f"Eligible multi-candidate traces: {len(multi_traces)}")
    print(f"Tasks represented: {n_tasks}")

    if len(multi_traces) == 0:
        print("RESULT: NO_FRESH_ELIGIBLE_MULTI_TRACES")
        return

    if len(multi_traces) < MIN_FRESH_MULTI or n_tasks < MIN_TASKS_REPRESENTED:
        result_class = "FRESH_SAMPLE_TOO_SMALL"
        print(f"RESULT_CLASS: {result_class} (need >= {MIN_FRESH_MULTI} traces and >= {MIN_TASKS_REPRESENTED} tasks)")
    else:
        result_class = None  # determined after evaluation

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
    paired = defaultdict(int)
    for mr, br in zip(model_results, baseline_results):
        key = (mr["is_correct_top1"], br["is_correct_top1"])
        paired[f"m{key[0]}_b{key[1]}"] += 1

    print(f"\n=== FRESH CONFIRMATION (n={n}, tasks={n_tasks}) ===")
    print(f"Model top-1: {m_correct}/{n} = {m_correct/n:.4f}  ties={m_ties}")
    print(f"Baseline top-1: {b_correct}/{n} = {b_correct/n:.4f}  ties={b_ties}")
    print(f"Model cond MAE: {m_mae:.1f}  Baseline: {b_mae:.1f}")
    print(f"Paired: {dict(paired)}")

    if result_class is None:
        if m_correct > b_correct:
            result_class = "FRESH_REPLICATION_GAIN"
        elif m_correct == b_correct:
            result_class = "FRESH_REPLICATION_NO_GAIN"
        else:
            result_class = "FRESH_REPLICATION_REGRESSION"

    print(f"RESULT_CLASS: {result_class}")

    # Write outputs
    with open(out / "d2_fresh_confirmation.csv", "w", newline="") as f:
        fields = ["trace_id", "task_key", "state_id", "n_candidates",
                  "model_correct", "model_ties", "model_abs_error",
                  "baseline_correct", "baseline_ties", "baseline_abs_error"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for mr, br in zip(model_results, baseline_results):
            w.writerow({**{f"model_{k}": v for k, v in mr.items()},
                        **{f"baseline_{k}": v for k, v in br.items()}})

    with open(out / "d2_fresh_confirmation_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "model", "baseline"]); w.writeheader()
        for kv in [("n", n, n), ("top1_rate", round(m_correct/n, 4), round(b_correct/n, 4)),
                    ("n_ties", m_ties, b_ties), ("cond_mae", round(m_mae, 1), round(b_mae, 1)),
                    ("n_tasks", n_tasks, "")]:
            w.writerow({"metric": kv[0], "model": kv[1], "baseline": kv[2]})

    with open(out / "d2_fresh_confirmation_run_log.txt", "w") as f:
        f.write(f"D2 FRESH CONFIRMATION LOG\n")
        f.write(f"checkpoint_sha: {actual}\n")
        f.write(f"candidate_table_sha: {sha256_file(args.candidate_table)}\n")
        f.write(f"trace_status_sha: {sha256_file(args.trace_status)}\n")
        f.write(f"n_eligible_multi: {n}\n")
        f.write(f"n_tasks_represented: {n_tasks}\n")
        f.write(f"result_class: {result_class}\n")
        f.write(f"model_top1: {m_correct}/{n}\n")
        f.write(f"baseline_top1: {b_correct}/{n}\n")

    print(f"Output: {out}")


if __name__ == "__main__":
    main()
