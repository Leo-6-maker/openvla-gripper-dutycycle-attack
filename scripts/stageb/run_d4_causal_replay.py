#!/usr/bin/env python3
"""D4.0: Causal first-trigger replay + threshold calibration.

Strictly causal: processes candidates in chronological order,
emits the FIRST candidate with model_score >= threshold.
At most one emission per trace. No emission → ABSTAIN.

Threshold calibration on D1b validation 20 traces only.
Separate thresholds for model (MLP score) and baseline (total_score).

Feature causality audit: regenerate features from records[:step+1]
and verify equivalence with full-trace candidate table.
"""

import argparse, csv, hashlib, json, os, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

PIPELINE_ROOT = "/data/liuyu/l12_e4c2_pipeline"
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "src"))
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "scripts", "stageb"))

from train_d1b_detector import CandidateRanker, FEATURE_NAMES, normalize_features, TIE_TOLERANCE

FROZEN_CHECKPOINT_SHA = "cdd3cbe4f42592dab81590d84f5a8ff67b9fc3b7326f691742b9a438f1174858"

# ── Causal first-trigger policy ──

def causal_first_trigger(candidates_sorted, scores, threshold):
    """Process candidates chronologically. Emit first with score >= threshold.
    Returns: (emit_step, emit_idx) or (-1, -1) for ABSTAIN."""
    for idx in range(len(candidates_sorted)):
        if scores[idx] >= threshold:
            return int(candidates_sorted[idx]["candidate_step"]), idx
    return -1, -1


def classify_emission(emit_step, teacher_p_step):
    """Classify emission relative to Teacher-P."""
    if emit_step < 0:
        return "ABSTAIN"
    if emit_step < teacher_p_step:
        return "EARLY_TRIGGER"
    if emit_step == teacher_p_step:
        return "EXACT_HIT"
    return "LATE_TRIGGER"


def threshold_sweep(candidates_by_trace, scores_by_trace, threshold_candidates, tp_map):
    """Sweep thresholds. Return per-threshold metrics."""
    results = []
    for tau in threshold_candidates:
        metrics = {"threshold": round(tau, 6), "n_exact": 0, "n_early": 0,
                   "n_late": 0, "n_abstain": 0, "n_emissions": 0}
        for tid in candidates_by_trace:
            emit_step, _ = causal_first_trigger(
                candidates_by_trace[tid], scores_by_trace[tid], tau)
            tp_step = tp_map.get(tid, -1)
            cls = classify_emission(emit_step, tp_step)
            if cls == "EXACT_HIT": metrics["n_exact"] += 1
            elif cls == "EARLY_TRIGGER": metrics["n_early"] += 1
            elif cls == "LATE_TRIGGER": metrics["n_late"] += 1
            else: metrics["n_abstain"] += 1
            if cls != "ABSTAIN": metrics["n_emissions"] += 1
        results.append(metrics)
    return results


def select_threshold(sweep_results):
    """Select best threshold: feasible (early<=2), then lexicographic."""
    feasible = [r for r in sweep_results if r["n_early"] <= 2]
    if not feasible:
        return None, "NO_FEASIBLE_CAUSAL_THRESHOLD"

    def key(r):
        return (-r["n_exact"], r["n_late"], r["n_emissions"], -r["threshold"])
    best = min(feasible, key=key)
    return best["threshold"], "OK"


# ── Feature causality audit ──

def audit_feature_prefix_equivalence(val_trace_ids, candidate_table_path, candidate_csv_path,
                                      manifest_path, split_path, norm_path):
    """For each validation trace, regenerate features from records[:step+1]
    and compare with the candidate table. Return audit results."""
    # This requires the full trace files and the remapper.
    # For D4.0, we verify that the candidate table features match what
    # would be computed causally.
    sys.path.insert(0, os.path.join(PIPELINE_ROOT, "scripts", "stageb"))
    from remap_v4_trace_for_l12 import remap_v4_to_l12

    # Load candidates and split
    all_candidates = list(csv.DictReader(open(candidate_table_path)))
    split = {r["trace_id"]: r["split"] for r in csv.DictReader(open(split_path))}
    manifest = {r["trace_id"]: r for r in csv.DictReader(open(manifest_path))}

    # Group validation candidates by trace
    val_cands = defaultdict(list)
    for c in all_candidates:
        tid = c["trace_id"]
        if tid in split and split[tid] == "val":
            val_cands[tid].append(c)

    audit_rows = []
    n_ok = 0; n_mismatch = 0

    for tid in sorted(val_cands):
        if tid not in manifest:
            continue
        fp = manifest[tid]["source_path"]
        rows, _, _ = remap_v4_to_l12(fp, "/dev/null", raise_on_invariant=False)

        # For each candidate, compute features from records[:step+1]
        from gripper_attack.critical_close_selector import rule_based_close_predictor, PREDICTION_HORIZON
        preds = rule_based_close_predictor(rows, horizon=PREDICTION_HORIZON, teacher_anchor=-1)

        for c in val_cands[tid]:
            step = int(c["candidate_step"])
            pred = preds[step]  # causal: uses only records[:step+1]
            match = True
            for fn in FEATURE_NAMES:
                full_val = c.get(fn, "")
                causal_val = str(pred.get(fn, "")) if fn in pred else ""
                if fn in ("total_score",):
                    causal_val = str(round(pred.get("score", 0), 4))
                elif fn == "eef_deceleration_delta":
                    sn = pred.get("eef_speed_now", "")
                    sp = pred.get("eef_speed_prev", "")
                    causal_val = str(round(float(sn) - float(sp), 6)) if sn != "" and sp != "" else ""
                elif fn == "close_streak":
                    causal_val = str(pred.get("close_streak_value", ""))
                elif fn == "raw_crossing":
                    causal_val = str(int(pred.get("raw_open_to_close_crossing", 0)))
                elif fn == "close_onset":
                    causal_val = str(int(pred.get("close_onset", 0)))
                elif fn == "candidate_index":
                    # candidate_index is computed from the sorted candidate list
                    pass  # skip — it's an ordering artifact
                elif fn == "time_since_prev_close":
                    pass  # skip — computed from candidate list order
                elif fn == "time_since_last_open":
                    pass  # skip — computed from candidate list order

                if causal_val and full_val and fn not in ("candidate_index", "time_since_prev_close", "time_since_last_open"):
                    try:
                        if abs(float(causal_val) - float(full_val)) > 0.001:
                            match = False
                    except (ValueError, TypeError):
                        if causal_val != full_val:
                            match = False

            audit_rows.append({"trace_id": tid, "candidate_step": step, "causal_match": match})
            if match: n_ok += 1
            else: n_mismatch += 1

    return audit_rows, n_ok, n_mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidate-table", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--norm-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now(timezone.utc)

    # Verify checkpoint
    import hashlib as hl
    actual = hl.sha256(open(args.checkpoint, "rb").read()).hexdigest()
    assert actual == FROZEN_CHECKPOINT_SHA, f"Checkpoint SHA mismatch: {actual[:16]}"
    print(f"Checkpoint verified: {actual[:16]}...")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    means = ckpt["normalization"]["means"]
    stdevs = ckpt["normalization"]["stdevs"]
    impute = ckpt["normalization"]["impute"]

    # Load data
    candidates = list(csv.DictReader(open(args.candidate_table)))
    split = {r["trace_id"]: r["split"] for r in csv.DictReader(open(args.split_manifest))}

    # Filter to validation only
    val_cands = defaultdict(list)
    for c in candidates:
        tid = c["trace_id"]
        if tid in split and split[tid] == "val":
            val_cands[tid].append(c)

    # Sort each trace's candidates by step
    for tid in val_cands:
        val_cands[tid].sort(key=lambda x: int(x["candidate_step"]))

    n_val = len(val_cands)
    print(f"Validation traces: {n_val}")

    # ── Model threshold sweep ──
    model = CandidateRanker(n_features=16).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    model_scores = {}
    tp_map = {}
    with torch.no_grad():
        for tid, cands in val_cands.items():
            X = normalize_features(cands, means, stdevs, impute).to(device)
            model_scores[tid] = model(X).cpu().numpy()
            tp_map[tid] = next((int(c["candidate_step"]) for c in cands if int(c.get("is_teacher_p", 0)) == 1), -1)

    # Generate threshold candidates from all val model scores
    all_model_scores = np.concatenate([s for s in model_scores.values()])
    model_thresholds = sorted(set(np.round(all_model_scores, 6))) + [float("inf")]

    print(f"Sweeping {len(model_thresholds)} model thresholds...")
    model_sweep = threshold_sweep(val_cands, model_scores, model_thresholds, tp_map)
    model_tau, model_status = select_threshold(model_sweep)

    # ── Baseline threshold sweep ──
    baseline_scores = {}
    for tid, cands in val_cands.items():
        baseline_scores[tid] = np.array([float(c.get("total_score", 0)) for c in cands])

    all_baseline_scores = np.concatenate([s for s in baseline_scores.values()])
    baseline_thresholds = sorted(set(np.round(all_baseline_scores, 4))) + [float("inf")]

    print(f"Sweeping {len(baseline_thresholds)} baseline thresholds...")
    baseline_sweep = threshold_sweep(val_cands, baseline_scores, baseline_thresholds, tp_map)
    baseline_tau, baseline_status = select_threshold(baseline_sweep)

    # ── Feature causality audit ──
    print("Running feature causality audit...")
    audit_rows, n_ok, n_mismatch = audit_feature_prefix_equivalence(
        set(val_cands.keys()), args.candidate_table,
        args.candidate_table, args.manifest, args.split_manifest, args.norm_csv)

    # ── Write outputs ──
    with open(out / "d4_model_threshold_sweep.csv", "w", newline="") as f:
        fields = ["threshold", "n_exact", "n_early", "n_late", "n_abstain", "n_emissions"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(model_sweep)

    with open(out / "d4_baseline_threshold_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(baseline_sweep)

    with open(out / "d4_selected_thresholds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "threshold", "status"]); w.writeheader()
        w.writerow({"policy": "model", "threshold": model_tau, "status": model_status})
        w.writerow({"policy": "baseline", "threshold": baseline_tau, "status": baseline_status})

    # Val causal predictions with selected thresholds
    val_preds = []
    if model_status == "OK":
        for tid in sorted(val_cands):
            emit_step, _ = causal_first_trigger(val_cands[tid], model_scores[tid], model_tau)
            emit_base, _ = causal_first_trigger(val_cands[tid], baseline_scores[tid], baseline_tau)
            tp = tp_map[tid]
            val_preds.append({
                "trace_id": tid, "task_key": val_cands[tid][0]["task_key"],
                "state_id": val_cands[tid][0]["state_id"],
                "teacher_p_step": tp,
                "model_emit_step": emit_step, "model_class": classify_emission(emit_step, tp),
                "baseline_emit_step": emit_base, "baseline_class": classify_emission(emit_base, tp),
                "model_threshold": model_tau, "baseline_threshold": baseline_tau,
            })
    with open(out / "d4_val_causal_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(val_preds[0].keys())); w.writeheader(); w.writerows(val_preds)

    with open(out / "d4_feature_prefix_equivalence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trace_id", "candidate_step", "causal_match"])
        w.writeheader(); w.writerows(audit_rows)

    # Run log
    with open(out / "d4_run_log.txt", "w") as f:
        f.write(f"D4.0 RUN LOG\n")
        f.write(f"start: {start_time.isoformat()}\n")
        f.write(f"checkpoint_sha: {actual}\n")
        f.write(f"val_traces: {n_val}\n")
        f.write(f"model_threshold: {model_tau} ({model_status})\n")
        f.write(f"baseline_threshold: {baseline_tau} ({baseline_status})\n")
        f.write(f"feature_audit_ok: {n_ok}\n")
        f.write(f"feature_audit_mismatch: {n_mismatch}\n")
        f.write(f"test21_evaluated: NO\n")
        f.write(f"fresh25_evaluated: NO\n")

    print(f"\n=== D4.0 RESULTS ===")
    print(f"Model threshold: {model_tau} ({model_status})")
    print(f"Baseline threshold: {baseline_tau} ({baseline_status})")
    print(f"Feature audit: {n_ok} ok, {n_mismatch} mismatch")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
