#!/usr/bin/env python3
"""D2.2: Fresh confirmation (REPAIRED).

Fixes:
  - CSV schema: flat trace-level fields, not prefix-duplicated keys.
  - Sentinel: created only AFTER all preflight gates pass.
  - Trace matching: model/baseline joined by explicit trace_id.
  - Zero unique-decision MAE → "NA" (not NaN/empty).
  - Frozen checkpoint SHA verified before sentinel.
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

PIPELINE_ROOT = os.environ.get("L12_PIPELINE_ROOT", "/data/liuyu/l12_e4c2_pipeline")
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "scripts", "stageb"))

from train_d1b_detector import CandidateRanker, FEATURE_NAMES, normalize_features


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def select_eligible_multi_traces(candidate_rows, status_rows):
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
            tp_s = scores[tp_idx]
            n_higher = sum(1 for s in scores if s > tp_s + TIE_TOLERANCE)
            n_equal_tp = sum(1 for s in scores if abs(s - tp_s) < TIE_TOLERANCE)
            is_unique_top1 = (n_higher == 0 and n_equal_tp == 1)
            is_top2 = (n_higher <= 1)
            if len(ties) == 1:
                pred_step = int(candidates[ties[0]]["candidate_step"])
                abs_err = abs(pred_step - tp_step) if tp_step is not None else -1
                is_near = abs_err <= 4 if abs_err >= 0 else False
            else:
                pred_step = -1; abs_err = -1; is_near = False
            results.append({
                "trace_id": tid,
                "is_correct_top1": int(is_unique_top1),
                "is_top2": int(is_top2),
                "is_near_pm4": int(is_near),
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
        is_top2 = (n_higher <= 1)
        if len(ties) == 1:
            pred_step = int(candidates[ties[0]]["candidate_step"])
            abs_err = abs(pred_step - tp_step) if tp_step is not None else -1
            is_near = abs_err <= 4 if abs_err >= 0 else False
        else:
            pred_step = -1; abs_err = -1; is_near = False
        results.append({
            "trace_id": tid,
            "is_correct_top1": int(is_unique_top1),
            "is_top2": int(is_top2),
            "is_near_pm4": int(is_near),
            "n_ties_for_max": len(ties),
            "unique_decision": int(len(ties) == 1),
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

    # ── Preflight: verify all inputs first ──
    checkpoint_sha = sha256_file(args.checkpoint)
    candidate_sha = sha256_file(args.candidate_table)
    status_sha = sha256_file(args.trace_status)
    runner_sha = sha256_file(__file__)

    print("=== PREFLIGHT ===")
    print(f"  checkpoint: {checkpoint_sha[:16]}...")
    print(f"  candidate_table: {candidate_sha[:16]}...")
    print(f"  trace_status: {status_sha[:16]}...")
    print(f"  runner: {runner_sha[:16]}...")

    if checkpoint_sha != FROZEN_CHECKPOINT_SHA:
        print(f"FATAL: checkpoint SHA mismatch")
        sys.exit(1)
    print("  checkpoint SHA: VERIFIED")

    # Load data
    candidates = list(csv.DictReader(open(args.candidate_table)))
    status_rows = list(csv.DictReader(open(args.trace_status)))

    # Verify status has 98 rows
    assert len(status_rows) == 98, f"Expected 98 status rows, got {len(status_rows)}"

    # Select eligible
    multi_traces = select_eligible_multi_traces(candidates, status_rows)
    n_tasks = len(set(multi_traces[tid][0]["task_key"] for tid in multi_traces))

    print(f"  Eligible multi: {len(multi_traces)}  Tasks: {n_tasks}")

    # Verify each eligible trace has exactly 1 positive
    for tid, cands in multi_traces.items():
        n_pos = sum(1 for c in cands if int(c.get("is_teacher_p", 0)) == 1)
        assert n_pos == 1, f"{tid}: expected 1 positive, got {n_pos}"

    # ── All preflight gates passed — now create sentinel ──
    sentinel_path = out / "fresh_confirmation_started.json"
    if sentinel_path.exists():
        print("FATAL: fresh confirmation already started")
        sys.exit(2)

    with open(sentinel_path, "w") as f:
        json.dump({
            "checkpoint_sha": checkpoint_sha,
            "candidate_table_sha": candidate_sha,
            "trace_status_sha": status_sha,
            "runner_sha": runner_sha,
            "n_eligible_multi": len(multi_traces),
            "n_tasks_represented": n_tasks,
            "timestamp": str(np.datetime64('now')),
        }, f, indent=2)
    print("  Sentinel created\n")

    if len(multi_traces) < MIN_FRESH_MULTI or n_tasks < MIN_TASKS_REPRESENTED:
        print(f"FRESH_SAMPLE_TOO_SMALL (need >= {MIN_FRESH_MULTI} traces, >= {MIN_TASKS_REPRESENTED} tasks)")
        return

    # Load checkpoint
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    means = ckpt["normalization"]["means"]
    stdevs = ckpt["normalization"]["stdevs"]
    impute = ckpt["normalization"]["impute"]

    model = CandidateRanker(n_features=16).to(device)
    model.load_state_dict(ckpt["model_state"])

    # ── Evaluate ──
    model_results = evaluate_model(model, multi_traces, means, stdevs, impute, device)
    baseline_results = evaluate_baseline(multi_traces)

    # Check trace_ids match
    m_ids = {r["trace_id"] for r in model_results}
    b_ids = {r["trace_id"] for r in baseline_results}
    assert m_ids == b_ids, f"Trace ID mismatch: model={len(m_ids)}, baseline={len(b_ids)}"

    # Join by trace_id
    b_map = {r["trace_id"]: r for r in baseline_results}
    n = len(model_results)
    m_correct = sum(r["is_correct_top1"] for r in model_results)
    b_correct = sum(b_map[r["trace_id"]]["is_correct_top1"] for r in model_results)
    m_ties = sum(1 for r in model_results if r["n_ties_for_max"] > 1)
    b_ties = sum(b_map[r["trace_id"]]["n_ties_for_max"] > 1 for r in model_results)

    m_uniq = [r["abs_error"] for r in model_results if r["unique_decision"] == 1 and r["abs_error"] >= 0]
    b_uniq = [b_map[r["trace_id"]]["abs_error"] for r in model_results if b_map[r["trace_id"]]["unique_decision"] == 1 and b_map[r["trace_id"]]["abs_error"] >= 0]
    m_mae = round(np.mean(m_uniq), 1) if m_uniq else "NA"
    b_mae = round(np.mean(b_uniq), 1) if b_uniq else "NA"

    # Paired
    both = sum(1 for r in model_results if r["is_correct_top1"] and b_map[r["trace_id"]]["is_correct_top1"])
    m_only = sum(1 for r in model_results if r["is_correct_top1"] and not b_map[r["trace_id"]]["is_correct_top1"])
    b_only = sum(1 for r in model_results if not r["is_correct_top1"] and b_map[r["trace_id"]]["is_correct_top1"])
    neither = n - both - m_only - b_only

    # Determine result class
    if m_correct > b_correct:
        result_class = "FRESH_REPLICATION_GAIN"
    elif m_correct == b_correct:
        result_class = "FRESH_REPLICATION_NO_GAIN"
    else:
        result_class = "FRESH_REPLICATION_REGRESSION"

    print(f"\n=== FRESH CONFIRMATION (n={n}, tasks={n_tasks}) ===")
    print(f"Model top-1: {m_correct}/{n} = {m_correct/n:.4f}  ties={m_ties}")
    print(f"Baseline top-1: {b_correct}/{n} = {b_correct/n:.4f}  ties={b_ties}")
    print(f"Model cond MAE: {m_mae}  Baseline cond MAE: {b_mae}")
    print(f"Paired: both={both} model-only={m_only} baseline-only={b_only} neither={neither}")
    print(f"RESULT_CLASS: {result_class}")

    # ── Write outputs (fixed CSV schema) ──
    with open(out / "d2_fresh_confirmation.csv", "w", newline="") as f:
        fields = ["trace_id", "model_correct", "model_ties", "model_abs_error",
                  "baseline_correct", "baseline_ties", "baseline_abs_error"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for mr in model_results:
            tid = mr["trace_id"]
            br = b_map[tid]
            w.writerow({
                "trace_id": tid,
                "model_correct": mr["is_correct_top1"], "model_ties": mr["n_ties_for_max"],
                "model_abs_error": mr["abs_error"],
                "baseline_correct": br["is_correct_top1"], "baseline_ties": br["n_ties_for_max"],
                "baseline_abs_error": br["abs_error"],
            })

    with open(out / "d2_fresh_confirmation_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"]); w.writeheader()
        for kv in [("n_traces", n), ("n_tasks", n_tasks),
                    ("model_top1_rate", round(m_correct/n, 4)),
                    ("baseline_top1_rate", round(b_correct/n, 4)),
                    ("model_n_ties", m_ties), ("baseline_n_ties", b_ties),
                    ("model_cond_mae", m_mae), ("baseline_cond_mae", b_mae),
                    ("paired_both_correct", both), ("paired_model_only", m_only),
                    ("paired_baseline_only", b_only), ("paired_neither", neither),
                    ("result_class", result_class)]:
            w.writerow({"metric": kv[0], "value": kv[1]})

    with open(out / "d2_fresh_confirmation_run_log.txt", "w") as f:
        f.write(f"D2 FRESH CONFIRMATION LOG\n")
        f.write(f"checkpoint_sha: {checkpoint_sha}\n")
        f.write(f"candidate_table_sha: {candidate_sha}\n")
        f.write(f"trace_status_sha: {status_sha}\n")
        f.write(f"runner_sha: {runner_sha}\n")
        f.write(f"n_eligible_multi: {n}\n")
        f.write(f"n_tasks_represented: {n_tasks}\n")
        f.write(f"result_class: {result_class}\n")

    print(f"Output: {out}")


if __name__ == "__main__":
    main()
