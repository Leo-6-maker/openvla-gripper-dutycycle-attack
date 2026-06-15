#!/usr/bin/env python3
"""D4.1: Post-hoc causal first-trigger replay on test21 + fresh25.

Uses frozen model threshold τ=0.236312 (baseline τ=inf).
Full 16-feature causality audit on ALL candidates before sentinel.
One-shot evaluation. CPU only. No retraining.
"""

import argparse, csv, hashlib, json, math, os, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

PIPELINE_ROOT = os.environ.get("L12_PIPELINE_ROOT", "/data/liuyu/l12_e4c2_pipeline")
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "src"))
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "scripts", "stageb"))

from train_d1b_detector import CandidateRanker, FEATURE_NAMES, normalize_features
# These functions are in the same file; the D4.0 functions are at module level
from run_l12_e4c2b_repair import sha256_file
from run_d2_fresh_confirm import select_eligible_multi_traces

FROZEN_CHECKPOINT_SHA = "cdd3cbe4f42592dab81590d84f5a8ff67b9fc3b7326f691742b9a438f1174858"
FROZEN_MODEL_TAU = 0.236312
FROZEN_BASELINE_TAU = float("inf")


# ── D4.0 core functions (shared) ──

def causal_first_trigger(candidates_sorted, scores, threshold):
    for idx in range(len(candidates_sorted)):
        if scores[idx] >= threshold:
            return int(candidates_sorted[idx]["candidate_step"]), idx
    return -1, -1


def classify_emission(emit_step, teacher_p_step):
    if emit_step < 0: return "ABSTAIN"
    if emit_step < teacher_p_step: return "EARLY_TRIGGER"
    if emit_step == teacher_p_step: return "EXACT_HIT"
    return "LATE_TRIGGER"


def audit_feature_prefix_equivalence(trace_ids, candidate_table_path, manifest_map, *args):
    """Full 16-feature causality audit. Returns (rows, n_ok, n_mismatch, total)."""
    sys.path.insert(0, os.path.join(PIPELINE_ROOT, "scripts", "stageb"))
    from remap_v4_trace_for_l12 import remap_v4_to_l12
    from gripper_attack.critical_close_selector import rule_based_close_predictor, PREDICTION_HORIZON
    from gripper_attack.phase_detector import _safe_float

    all_cands = list(csv.DictReader(open(candidate_table_path)))
    val_traces = defaultdict(list)
    for c in all_cands:
        tid = c["trace_id"]
        if tid in trace_ids:
            val_traces[tid].append(c)

    rows = []; n_ok = 0; n_mm = 0
    for tid in sorted(val_traces):
        cands = sorted(val_traces[tid], key=lambda x: int(x["candidate_step"]))
        if not cands or tid not in manifest_map: continue
        fp = manifest_map[tid].get("source_path", "")
        if not fp or not os.path.isfile(fp): continue

        trace_rows, _, _ = remap_v4_to_l12(fp, "/dev/null", raise_on_invariant=False)
        preds = rule_based_close_predictor(trace_rows, horizon=PREDICTION_HORIZON, teacher_anchor=-1)
        open_steps = [int(r.get("step", i)) for i, r in enumerate(trace_rows)
                      if int(_safe_float(r.get("decoded_open_bool", 0))) == 1]
        all_close = sorted([p["step"] for p in preds if p.get("is_close_event_candidate")])

        for idx, c in enumerate(cands):
            step = int(c["candidate_step"]); pred = preds[step] if step < len(preds) else {}
            causal_idx = sum(1 for s in all_close if s < step)
            prevs = [s for s in all_close if s < step]
            causal_prev = step - max(prevs) if prevs else ""
            priors = [s for s in open_steps if s < step]
            causal_open = step - max(priors) if priors else ""

            checks = {
                "total_score": str(round(pred.get("score",0),4)) if pred else "",
                "raw_crossing_bonus": str(pred.get("raw_crossing_bonus","")) if pred else "",
                "close_streak_bonus": str(pred.get("close_streak_bonus","")) if pred else "",
                "close_onset_qpos_bonus": str(pred.get("close_onset_qpos_bonus","")) if pred else "",
                "eef_deceleration_bonus": str(pred.get("eef_deceleration_bonus","")) if pred else "",
                "qpos_ready_bonus": str(pred.get("qpos_ready_bonus","")) if pred else "",
                "eef_speed_now": str(pred.get("eef_speed_now","")) if pred else "",
                "eef_speed_prev": str(pred.get("eef_speed_prev","")) if pred else "",
                "eef_deceleration_delta": _audit_decel_delta(pred),
                "close_streak": str(pred.get("close_streak_value","")) if pred else "",
                "raw_crossing": str(int(pred.get("raw_open_to_close_crossing",0))) if pred else "",
                "close_onset": str(int(pred.get("close_onset",0))) if pred else "",
                "qpos": str(pred.get("qpos","")) if pred else "",
                "time_since_prev_close": str(causal_prev) if causal_prev != "" else "",
                "time_since_last_open": str(causal_open) if causal_open != "" else "",
                "candidate_index": str(causal_idx),
            }
            for fn, cv in checks.items():
                fv = c.get(fn, "")
                ok, diff = _cmp(fv, cv)
                rows.append({"trace_id": tid, "candidate_step": step, "feature_name": fn,
                             "committed_value": fv, "causal_value": cv, "match": ok, "difference": diff})
                if ok: n_ok += 1
                else: n_mm += 1
    return rows, n_ok, n_mm, len(rows)


def _audit_decel_delta(pred):
    if not pred: return ""
    sn, sp = pred.get("eef_speed_now",""), pred.get("eef_speed_prev","")
    if sn != "" and sp != "":
        try: return str(round(float(sn)-float(sp),6))
        except: return ""
    return ""


def _cmp(a, b):
    if a == "" and b == "": return True, ""
    if a == "" and b != "":
        try:
            if float(b) == 0.0: return False, "empty_vs_zero"
        except: pass
        return False, "empty_vs_nonempty"
    if b == "" and a != "":
        try:
            if float(a) == 0.0: return False, "zero_vs_empty"
        except: pass
        return False, "nonempty_vs_empty"
    try:
        if abs(float(a)-float(b)) < 0.001: return True, ""
        return False, f"diff={abs(float(a)-float(b))}"
    except: return a == b, "string_mismatch" if a != b else ""


# ── D4.1 functions ──

def run_causal_replay(candidates_by_trace, model, means, stdevs, impute, device, threshold):
    """Run causal first-trigger for all traces. Returns per-trace results."""
    model.eval()
    results = []
    with torch.no_grad():
        for tid in sorted(candidates_by_trace):
            cands = sorted(candidates_by_trace[tid], key=lambda x: int(x["candidate_step"]))
            X = normalize_features(cands, means, stdevs, impute).to(device)
            scores = model(X).cpu().numpy()
            emit_step, emit_idx = causal_first_trigger(cands, scores, threshold)
            tp_step = next((int(c["candidate_step"]) for c in cands if int(c.get("is_teacher_p", 0)) == 1), -1)
            classification = classify_emission(emit_step, tp_step)
            delay = emit_step - tp_step if emit_step >= 0 and tp_step >= 0 else ""
            abs_delay = abs(delay) if delay != "" else ""
            results.append({
                "trace_id": tid, "task_key": cands[0]["task_key"], "state_id": cands[0]["state_id"],
                "n_candidates": len(cands), "teacher_p_step": tp_step,
                "emit_step": emit_step, "classification": classification,
                "signed_delay": delay, "absolute_delay": abs_delay,
            })
    return results


def run_baseline_replay(candidates_by_trace, threshold):
    """Baseline replay using total_score."""
    results = []
    for tid in sorted(candidates_by_trace):
        cands = sorted(candidates_by_trace[tid], key=lambda x: int(x["candidate_step"]))
        scores = np.array([float(c.get("total_score", 0)) for c in cands])
        emit_step, emit_idx = causal_first_trigger(cands, scores, threshold)
        tp_step = next((int(c["candidate_step"]) for c in cands if int(c.get("is_teacher_p", 0)) == 1), -1)
        classification = classify_emission(emit_step, tp_step)
        delay = emit_step - tp_step if emit_step >= 0 and tp_step >= 0 else ""
        abs_delay = abs(delay) if delay != "" else ""
        results.append({
            "trace_id": tid, "task_key": cands[0]["task_key"], "state_id": cands[0]["state_id"],
            "n_candidates": len(cands), "teacher_p_step": tp_step,
            "emit_step": emit_step, "classification": classification,
            "signed_delay": delay, "absolute_delay": abs_delay,
        })
    return results


def summarize(results, label):
    n = len(results)
    exact = sum(1 for r in results if r["classification"] == "EXACT_HIT")
    early = sum(1 for r in results if r["classification"] == "EARLY_TRIGGER")
    late = sum(1 for r in results if r["classification"] == "LATE_TRIGGER")
    abstain = sum(1 for r in results if r["classification"] == "ABSTAIN")
    emitted = n - abstain
    cond_exact = exact / emitted if emitted > 0 else 0
    abs_delays = [r["absolute_delay"] for r in results if r["absolute_delay"] != ""]
    cond_abs_delay = np.mean(abs_delays) if abs_delays else "NA"
    print(f"\n{label} (n={n}):")
    print(f"  EXACT: {exact}  EARLY: {early}  LATE: {late}  ABSTAIN: {abstain}")
    print(f"  Coverage: {emitted}/{n}  Exact/emitted: {exact}/{emitted}={cond_exact:.3f}" if emitted else f"  Coverage: 0/{n}")
    print(f"  Cond abs delay: {cond_abs_delay}")
    return {"n": n, "exact": exact, "early": early, "late": late, "abstain": abstain,
            "coverage": emitted, "cond_exact_rate": round(cond_exact, 4),
            "cond_abs_delay": cond_abs_delay}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidate-table-test", required=True)
    ap.add_argument("--candidate-table-fresh", required=True)
    ap.add_argument("--trace-status", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-fresh", default="")
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--norm-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # ── Preflight: verify checkpoint ──
    actual_ckpt = sha256_file(args.checkpoint)
    assert actual_ckpt == FROZEN_CHECKPOINT_SHA, f"Checkpoint SHA mismatch"
    print(f"Checkpoint: {actual_ckpt[:16]}... VERIFIED")
    print(f"Model τ: {FROZEN_MODEL_TAU}  Baseline τ: {FROZEN_BASELINE_TAU}")

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    means = ckpt["normalization"]["means"]
    stdevs = ckpt["normalization"]["stdevs"]
    impute = ckpt["normalization"]["impute"]

    # ── Load test21 ──
    test_cands_all = list(csv.DictReader(open(args.candidate_table_test)))
    split = {r["trace_id"]: r["split"] for r in csv.DictReader(open(args.split_manifest))}
    manifest_map = {r["trace_id"]: r for r in csv.DictReader(open(args.manifest))}

    test21 = {}
    for c in test_cands_all:
        tid = c["trace_id"]
        if tid in split and split[tid] == "test":
            test21[tid] = test21.get(tid, []) + [c]

    for tid in test21:
        test21[tid] = sorted(test21[tid], key=lambda x: int(x["candidate_step"]))
    assert len(test21) == 21, f"Expected 21 test traces, got {len(test21)}"
    for tid, cands in test21.items():
        n_pos = sum(1 for c in cands if int(c.get("is_teacher_p", 0)) == 1)
        assert n_pos == 1, f"{tid}: expected 1 positive"
    print(f"Test21: {len(test21)} traces VERIFIED")

    # ── Load fresh25 ──
    fresh_cands_all = list(csv.DictReader(open(args.candidate_table_fresh)))
    status_rows = list(csv.DictReader(open(args.trace_status)))
    fresh25 = select_eligible_multi_traces(fresh_cands_all, status_rows)
    assert len(fresh25) == 25, f"Expected 25 fresh traces, got {len(fresh25)}"
    n_tasks = len(set(fresh25[tid][0]["task_key"] for tid in fresh25))
    assert n_tasks == 9, f"Expected 9 tasks, got {n_tasks}"
    print(f"Fresh25: {len(fresh25)} traces, {n_tasks} tasks VERIFIED")

    # ── Feature causality audit ──
    print("\n=== CAUSALITY AUDIT ===")
    all_test_ids = set(test21.keys())
    all_fresh_ids = set(fresh25.keys())

    # Inject source paths for audit
    for tid in test21:
        if tid in manifest_map:
            for c in test21[tid]:
                c["_source_path"] = manifest_map[tid]["source_path"]

    audit_test, ok_t, bad_t, tot_t = audit_feature_prefix_equivalence(
        all_test_ids, args.candidate_table_test, manifest_map)
    print(f"  test21: {ok_t}/{tot_t} ok, {bad_t} mismatch")
    assert bad_t == 0, f"test21 causality audit FAIL: {bad_t} mismatches"

    fresh_manifest = args.manifest_fresh or args.manifest
    if fresh_manifest and os.path.exists(fresh_manifest):
        fresh_rows = list(csv.DictReader(open(fresh_manifest)))
        if "filename" in fresh_rows[0] and "trace_id" not in fresh_rows[0]:
            fresh_manifest_map = {r["filename"].replace(".csv", ""): r for r in fresh_rows}
        else:
            fresh_manifest_map = {r["trace_id"]: r for r in fresh_rows}
    else:
        fresh_manifest_map = manifest_map
    audit_fresh, ok_f, bad_f, tot_f = audit_feature_prefix_equivalence(
        all_fresh_ids, args.candidate_table_fresh, fresh_manifest_map)
    print(f"  fresh25: {ok_f}/{tot_f} ok, {bad_f} mismatch")
    assert bad_f == 0, f"fresh25 causality audit FAIL: {bad_f} mismatches"

    # ── All gates passed — create sentinel ──
    sentinel_path = out / "d4_replay_started.json"
    assert not sentinel_path.exists(), "Sentinel already exists"
    with open(sentinel_path, "w") as f:
        json.dump({
            "checkpoint_sha": actual_ckpt, "model_tau": FROZEN_MODEL_TAU,
            "baseline_tau": "inf", "test_traces": 21, "fresh_traces": 25,
            "fresh_tasks": n_tasks, "test_audit_ok": ok_t, "fresh_audit_ok": ok_f,
            "timestamp": str(datetime.now(timezone.utc)),
            "runner_sha": sha256_file(__file__),
        }, f, indent=2)
    print("\nSentinel created.\n")

    # ── Build model ──
    model = CandidateRanker(n_features=16).to(device)
    model.load_state_dict(ckpt["model_state"])

    # ── Run causal replay ──
    print("=== TEST21 ===")
    test_model = run_causal_replay(test21, model, means, stdevs, impute, device, FROZEN_MODEL_TAU)
    test_baseline = run_baseline_replay(test21, FROZEN_BASELINE_TAU)
    test_m_sum = summarize(test_model, "Model test21")
    test_b_sum = summarize(test_baseline, "Baseline test21")

    print("\n=== FRESH25 ===")
    fresh_model = run_causal_replay(fresh25, model, means, stdevs, impute, device, FROZEN_MODEL_TAU)
    fresh_baseline = run_baseline_replay(fresh25, FROZEN_BASELINE_TAU)
    fresh_m_sum = summarize(fresh_model, "Model fresh25")
    fresh_b_sum = summarize(fresh_baseline, "Baseline fresh25")

    # ── Write outputs ──
    pfields = list(test_model[0].keys())
    for name, rows in [("test21_model", test_model), ("test21_baseline", test_baseline),
                        ("fresh25_model", fresh_model), ("fresh25_baseline", fresh_baseline)]:
        with open(out / f"d4_{name}_predictions.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=pfields); w.writeheader(); w.writerows(rows)

    with open(out / "d4_replay_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set","policy","n","exact","early","late","abstain","coverage","cond_exact_rate","cond_abs_delay"])
        w.writeheader()
        for name, s in [("test21_model", test_m_sum), ("test21_baseline", test_b_sum),
                         ("fresh25_model", fresh_m_sum), ("fresh25_baseline", fresh_b_sum)]:
            s["set"], s["policy"] = name.split("_")
            w.writerow(s)

    with open(out / "d4_replay_run_log.txt", "w") as f:
        f.write(f"D4.1 RUN LOG\n")
        f.write(f"checkpoint_sha: {actual_ckpt}\n")
        f.write(f"model_tau: {FROZEN_MODEL_TAU}\n")
        f.write(f"baseline_tau: inf\n")
        f.write(f"test_traces: 21\n")
        f.write(f"fresh_traces: 25\n")
        f.write(f"fresh_tasks: {n_tasks}\n")
        f.write(f"test_audit: {ok_t}/{tot_t}\n")
        f.write(f"fresh_audit: {ok_f}/{tot_f}\n")
        f.write(f"runner_sha: {sha256_file(__file__)}\n")

    print(f"\nD4.1 COMPLETE. Output: {out}")


if __name__ == "__main__":
    main()
