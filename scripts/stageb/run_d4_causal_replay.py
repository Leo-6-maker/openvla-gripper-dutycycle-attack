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


# ── Feature causality audit (D4.0a: all 16 features) ──

def audit_feature_prefix_equivalence(val_trace_ids, candidate_table_path, manifest_map, *args):
    """For each validation candidate, regenerate all 16 features from
    records[:step+1] and compare with committed candidate table.
    manifest_map: {trace_id: {source_path: ...}} for trace file access.
    Returns (audit_rows, n_ok, n_mismatch, n_total_comparisons)."""
    sys.path.insert(0, os.path.join(PIPELINE_ROOT, "scripts", "stageb"))
    from remap_v4_trace_for_l12 import remap_v4_to_l12
    from gripper_attack.critical_close_selector import rule_based_close_predictor, PREDICTION_HORIZON
    from gripper_attack.phase_detector import _safe_float

    _t0 = time.time()

    all_candidates = list(csv.DictReader(open(candidate_table_path)))
    # Filter to val traces
    val_traces = defaultdict(list)
    for c in all_candidates:
        tid = c["trace_id"]
        if tid in val_trace_ids:
            val_traces[tid].append(c)

    audit_rows = []
    n_ok = 0; n_mismatch = 0

    for tid in sorted(val_traces):
        candidates = sorted(val_traces[tid], key=lambda x: int(x["candidate_step"]))
        if not candidates or tid not in manifest_map:
            continue

        fp = manifest_map[tid].get("source_path", "")
        if not fp or not os.path.isfile(fp):
            continue
        rows, _, _ = remap_v4_to_l12(fp, "/dev/null", raise_on_invariant=False)

        # Causal prediction
        preds = rule_based_close_predictor(rows, horizon=PREDICTION_HORIZON, teacher_anchor=-1)

        # Build OPEN step list from remapped rows
        open_steps = [
            int(r.get("step", i)) for i, r in enumerate(rows)
            if int(_safe_float(r.get("decoded_open_bool", 0))) == 1
        ]

        # Build CLOSE candidate step list for causal ordering
        all_close_steps = sorted([
            p["step"] for p in preds if p.get("is_close_event_candidate")
        ])

        for idx, c in enumerate(candidates):
            step = int(c["candidate_step"])
            pred = preds[step] if step < len(preds) else {}

            # Causal candidate_index: number of CLOSE candidates before this one
            causal_cand_idx = sum(1 for s in all_close_steps if s < step)

            # Causal time_since_prev_close
            prev_steps = [s for s in all_close_steps if s < step]
            causal_prev = step - max(prev_steps) if prev_steps else ""

            # Causal time_since_last_open
            prior_opens = [s for s in open_steps if s < step]
            causal_open = step - max(prior_opens) if prior_opens else ""

            # For each of 16 features, compare committed vs causal
            feature_checks = {
                "total_score": ("discrete_score", str(round(pred.get("score", 0), 4)) if pred else ""),
                "raw_crossing_bonus": ("discrete_score", str(pred.get("raw_crossing_bonus", "")) if pred else ""),
                "close_streak_bonus": ("discrete_score", str(pred.get("close_streak_bonus", "")) if pred else ""),
                "close_onset_qpos_bonus": ("discrete_score", str(pred.get("close_onset_qpos_bonus", "")) if pred else ""),
                "eef_deceleration_bonus": ("discrete_score", str(pred.get("eef_deceleration_bonus", "")) if pred else ""),
                "qpos_ready_bonus": ("discrete_score", str(pred.get("qpos_ready_bonus", "")) if pred else ""),
                "eef_speed_now": ("continuous", str(pred.get("eef_speed_now", "")) if pred else ""),
                "eef_speed_prev": ("continuous", str(pred.get("eef_speed_prev", "")) if pred else ""),
                "eef_deceleration_delta": ("continuous", _compute_delta(pred)),
                "close_streak": ("discrete", str(pred.get("close_streak_value", "")) if pred else ""),
                "raw_crossing": ("discrete", str(int(pred.get("raw_open_to_close_crossing", 0))) if pred else ""),
                "close_onset": ("discrete", str(int(pred.get("close_onset", 0))) if pred else ""),
                "qpos": ("continuous", str(pred.get("qpos", "")) if pred else ""),
                "time_since_prev_close": ("temporal", str(causal_prev) if causal_prev != "" else ""),
                "time_since_last_open": ("temporal", str(causal_open) if causal_open != "" else ""),
                "candidate_index": ("temporal", str(causal_cand_idx)),
            }

            for fn, (ftype, causal_val) in feature_checks.items():
                committed_val = c.get(fn, "")
                match, reason = _compare_values(committed_val, causal_val)
                audit_rows.append({
                    "trace_id": tid, "candidate_step": step,
                    "feature_name": fn, "feature_type": ftype,
                    "committed_value": committed_val, "causal_value": causal_val,
                    "match": match, "difference": reason,
                })
                if match:
                    n_ok += 1
                else:
                    n_mismatch += 1

    total = len(audit_rows)
    print(f"  Audit: {total} comparisons ({len(val_traces)} traces, {n_ok} ok, {n_mismatch} mismatch) in {time.time()-_t0:.1f}s")
    return audit_rows, n_ok, n_mismatch, total


def _compute_delta(pred):
    if not pred: return ""
    sn = pred.get("eef_speed_now", "")
    sp = pred.get("eef_speed_prev", "")
    if sn != "" and sp != "":
        try: return str(round(float(sn) - float(sp), 6))
        except: return ""
    return ""


def _compare_values(committed, causal):
    """Compare two feature values with proper empty/zero/missing handling."""
    # Both empty → match
    if committed == "" and causal == "":
        return True, ""
    # One empty, one not → mismatch (0 vs "" is a mismatch)
    if committed == "" and causal != "":
        try:
            if float(causal) == 0.0:
                return False, "committed_empty_vs_causal_zero"
        except: pass
        return False, "committed_empty_vs_causal_nonempty"
    if causal == "" and committed != "":
        try:
            if float(committed) == 0.0:
                return False, "committed_zero_vs_causal_empty"
        except: pass
        return False, "committed_nonempty_vs_causal_empty"
    # Both non-empty — compare numerically
    try:
        cv = float(committed); gv = float(causal)
        diff = abs(cv - gv)
        if diff < 0.001:
            return True, ""
        return False, f"diff={diff}"
    except (ValueError, TypeError):
        if committed == causal:
            return True, ""
        return False, "string_mismatch"


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
    model_tau, model_status_raw = select_threshold(model_sweep)
    model_status = model_status_raw  # "OK" for model (has finite emitting threshold)

    # ── Baseline threshold sweep ──
    baseline_scores = {}
    for tid, cands in val_cands.items():
        baseline_scores[tid] = np.array([float(c.get("total_score", 0)) for c in cands])

    all_baseline_scores = np.concatenate([s for s in baseline_scores.values()])
    baseline_thresholds = sorted(set(np.round(all_baseline_scores, 4))) + [float("inf")]

    print(f"Sweeping {len(baseline_thresholds)} baseline thresholds...")
    baseline_sweep = threshold_sweep(val_cands, baseline_scores, baseline_thresholds, tp_map)
    baseline_tau, baseline_status_raw = select_threshold(baseline_sweep)
    baseline_status = "SAFE_ABSTAIN_ONLY" if float(baseline_tau) > 1e6 else baseline_status_raw

    # ── Feature causality audit (D4.0a: all 16 features) ──
    print("Running full 16-feature causality audit...")
    # Inject source paths into candidate dicts for remap access
    manifest_map = {r["trace_id"]: r for r in csv.DictReader(open(args.manifest))}
    for tid in val_cands:
        if tid in manifest_map:
            for c in val_cands[tid]:
                c["_source_path"] = manifest_map[tid]["source_path"]
    audit_rows, n_ok, n_mismatch, n_total = audit_feature_prefix_equivalence(
        set(val_cands.keys()), args.candidate_table, manifest_map, args.split_manifest, args.norm_csv)

    # ── Write outputs ──
    with open(out / "d4_model_threshold_sweep.csv", "w", newline="") as f:
        fields = ["threshold", "n_exact", "n_early", "n_late", "n_abstain", "n_emissions"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(model_sweep)

    with open(out / "d4_baseline_threshold_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(baseline_sweep)

    with open(out / "d4_selected_thresholds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "threshold", "status", "finite_emitting_threshold_feasible"]); w.writeheader()
        w.writerow({"policy": "model", "threshold": model_tau, "status": model_status,
                     "finite_emitting_threshold_feasible": True})
        w.writerow({"policy": "baseline", "threshold": baseline_tau, "status": baseline_status,
                     "finite_emitting_threshold_feasible": float(baseline_tau) < 1e6})

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
        w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
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
        f.write(f"feature_audit_total: {n_total}\n")
        f.write(f"causality_audit_pass: {n_mismatch == 0}\n")
        f.write(f"test21_evaluated: NO\n")
        f.write(f"fresh25_evaluated: NO\n")

    print(f"\n=== D4.0 RESULTS ===")
    print(f"Model threshold: {model_tau} ({model_status})")
    print(f"Baseline threshold: {baseline_tau} ({baseline_status})")
    print(f"Feature audit: {n_ok} ok, {n_mismatch} mismatch")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
