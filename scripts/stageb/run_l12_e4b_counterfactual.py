#!/usr/bin/env python3
"""E4B.1: Corrected causal policy counterfactual + separability audit.

Each policy returns four distinct fields:
  first_threshold_step  — step where score first crossed threshold
  selected_event_step   — the close event the policy picks
  decision_step         — step when the policy makes its decision
  actuation_step        — step when attack could actually begin (= decision_step)

Online causal error is measured from actuation_step, not selected_event_step.

Separability: per-candidate feature rankings, Teacher-P rank, paired diffs.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "stageb"))

from remap_v4_trace_for_l12 import remap_v4_to_l12, REMAPPER_VERSION
from gripper_attack.phase_detector import (
    teacher_rule_critical_close_anchor,
    teacher_privileged_critical_close_anchor,
)
from gripper_attack.critical_close_selector import (
    rule_based_close_predictor,
    WINDOW_LEN, PRE_OFFSET, PREDICTION_HORIZON,
)

SCORE_THRESHOLD = 1.5
NEAR_THRESHOLD = 4


def _sha256_file(path):
    import hashlib
    if not os.path.exists(path): return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(65536), b""): h.update(c)
    return h.hexdigest()


def _git_rev_parse():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    except: return "unknown"


def _git_is_clean():
    try:
        return subprocess.check_output(["git", "status", "--porcelain", "-uno"], cwd=str(REPO_ROOT), text=True).strip() == ""
    except: return False


# ── Corrected causal policies ──

def policy_first_threshold(preds, threshold=SCORE_THRESHOLD):
    for p in preds:
        if not p["abstain"] and p["score"] >= threshold:
            return {"first_threshold_step": p["step"], "selected_event_step": p["step"],
                    "decision_step": p["step"], "actuation_step": p["step"],
                    "causal": True, "policy": "first_threshold"}
    return {"first_threshold_step": -1, "selected_event_step": -1,
            "decision_step": -1, "actuation_step": -1,
            "causal": True, "policy": "first_threshold"}


def policy_bounded_peak_hold(preds, threshold=SCORE_THRESHOLD, hold_steps=4):
    first_t = None
    for p in preds:
        if not p["abstain"] and p["score"] >= threshold:
            first_t = p["step"]; break
    if first_t is None:
        return {"first_threshold_step": -1, "selected_event_step": -1,
                "decision_step": -1, "actuation_step": -1,
                "causal": True, "policy": f"peak_hold_{hold_steps}"}

    decision_t = first_t + hold_steps
    window = [p for p in preds
              if first_t <= p["step"] <= decision_t and not p["abstain"]]
    if not window:
        # Only first crossing step visible; decide at end of hold window
        return {"first_threshold_step": first_t, "selected_event_step": first_t,
                "decision_step": decision_t, "actuation_step": decision_t,
                "causal": True, "policy": f"peak_hold_{hold_steps}"}
    best = max(window, key=lambda p: p["score"])
    return {"first_threshold_step": first_t, "selected_event_step": best["step"],
            "decision_step": decision_t, "actuation_step": decision_t,
            "causal": True, "policy": f"peak_hold_{hold_steps}"}


def policy_local_maximum(preds, threshold=SCORE_THRESHOLD):
    above = False
    for i, p in enumerate(preds):
        if not p["abstain"] and p["score"] >= threshold:
            above = True
        if above and not p["abstain"] and i + 1 < len(preds):
            if not preds[i + 1]["abstain"] and preds[i + 1]["score"] < p["score"]:
                # Decision confirmed at i+1; event at i
                first_t = _first_above(preds, threshold)
                return {"first_threshold_step": first_t, "selected_event_step": p["step"],
                        "decision_step": i + 1, "actuation_step": i + 1,
                        "causal": True, "policy": "local_maximum"}
    # No decline found → cannot decide causally
    first_t = _first_above(preds, threshold)
    return {"first_threshold_step": first_t, "selected_event_step": -1,
            "decision_step": -1, "actuation_step": -1,
            "causal": False, "decision_reason": "no_decline_detected",
            "policy": "local_maximum"}


def policy_non_causal_record_high(preds, threshold=SCORE_THRESHOLD, margin=0.3):
    """NONCAUSAL: scans full trajectory for final record-high. Diagnostic only."""
    running_peak = 0.0; running_step = -1; first_t = None
    for p in preds:
        if p["abstain"]: continue
        if first_t is None and p["score"] >= threshold: first_t = p["step"]
        if p["score"] > running_peak + margin:
            running_peak = p["score"]; running_step = p["step"]
    return {"first_threshold_step": first_t if first_t else -1,
            "selected_event_step": running_step,
            "decision_step": -1, "actuation_step": -1,
            "causal": False, "policy": "noncausal_record_high"}


def policy_non_causal_global_argmax(preds):
    """NONCAUSAL: best close-event candidate across full trajectory. Diagnostic only."""
    close = [p for p in preds if p.get("is_close_event_candidate") and not p["abstain"]]
    if not close:
        return {"first_threshold_step": -1, "selected_event_step": -1,
                "decision_step": -1, "actuation_step": -1,
                "causal": False, "policy": "noncausal_global_score_argmax"}
    best = max(close, key=lambda p: p["score"])
    return {"first_threshold_step": -1, "selected_event_step": best["step"],
            "decision_step": -1, "actuation_step": -1,
            "causal": False, "policy": "noncausal_global_score_argmax"}


def _first_above(preds, threshold):
    for p in preds:
        if not p["abstain"] and p["score"] >= threshold: return p["step"]
    return -1


POLICIES = [
    ("first_threshold", policy_first_threshold),
    ("peak_hold_2", lambda p: policy_bounded_peak_hold(p, hold_steps=2)),
    ("peak_hold_4", lambda p: policy_bounded_peak_hold(p, hold_steps=4)),
    ("peak_hold_8", lambda p: policy_bounded_peak_hold(p, hold_steps=8)),
    ("local_maximum", policy_local_maximum),
    ("noncausal_record_high", policy_non_causal_record_high),
    ("noncausal_global_score_argmax", policy_non_causal_global_argmax),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tracked-tables-dir", default=None)
    args = ap.parse_args()

    RUNNER_COMMIT = _git_rev_parse(); START_TIME = datetime.now(timezone.utc)
    gate_errors = []

    if not _git_is_clean():
        gate_errors.append("G1: tracked worktree dirty")

    out = Path(args.output_dir)
    if out.exists(): gate_errors.append(f"FATAL: output dir exists: {out}")
    else: out.mkdir(parents=True)

    with open(args.manifest, "r", newline="") as f:
        manifest_rows = list(csv.DictReader(f))
    if len(manifest_rows) != 12:
        gate_errors.append(f"G2: expected 12 manifest rows, got {len(manifest_rows)}")

    for mr in manifest_rows:
        actual = _sha256_file(mr["trace_path"])
        if actual != mr["expected_sha256"]:
            gate_errors.append(f"G3: SHA mismatch {mr['task_key']}_s{mr['state_id']}")

    if gate_errors:
        for e in gate_errors: print(e)
        sys.exit(1)

    policy_rows = []; separability_rows = []; rank_rows = []
    invariant_total = 0; field_issue_total = 0

    for mr in manifest_rows:
        task = mr["task_key"]; state = int(mr["state_id"])
        expected_rows = int(mr["expected_row_count"])
        print(f"  {task}_s{state}")

        remap_out = str(out / f"remap_{task}_s{state}.csv")
        l12_rows, inv, fi = remap_v4_to_l12(mr["trace_path"], remap_out, raise_on_invariant=False)
        invariant_total += len(inv); field_issue_total += len(fi)
        if len(l12_rows) != expected_rows:
            gate_errors.append(f"G5: row count mismatch {task}_s{state}")

        p_anchor = teacher_privileged_critical_close_anchor(l12_rows)
        r_anchor = teacher_rule_critical_close_anchor(l12_rows)
        p_avail = p_anchor >= 0
        preds = rule_based_close_predictor(l12_rows, horizon=PREDICTION_HORIZON,
                                            teacher_anchor=p_anchor if p_avail else -1)

        # ── E4B-A: Policy counterfactuals ──
        for pol_name, pol_fn in POLICIES:
            result = pol_fn(preds)
            actuation = result.get("actuation_step", -1)
            selected = result.get("selected_event_step", -1)
            # Online causal error: measured from actuation_step (when attack can start)
            online_err = abs(actuation - p_anchor) if p_avail and actuation >= 0 else ""
            online_near = int(abs(actuation - p_anchor) <= NEAR_THRESHOLD) if p_avail and actuation >= 0 else ""
            # Selected-event error (diagnostic only — when was the identified event?)
            selected_err = abs(selected - p_anchor) if p_avail and selected >= 0 else ""

            policy_rows.append({
                "task_key": task, "state_id": state,
                "teacher_p_available": p_avail, "teacher_p_anchor": p_anchor,
                "policy": result["policy"], "causal": result.get("causal", ""),
                "first_threshold_step": result.get("first_threshold_step", ""),
                "selected_event_step": selected,
                "decision_step": result.get("decision_step", ""),
                "actuation_step": actuation,
                "online_abs_error_vs_P": online_err,
                "online_is_near_P": online_near,
                "selected_event_error_vs_P": selected_err,
                "actual_delay": result.get("decision_step", 0) - result.get("first_threshold_step", 0) if result.get("decision_step", -1) >= 0 and result.get("first_threshold_step", -1) >= 0 else "",
                "decision_reason": result.get("decision_reason", ""),
            })

        # ── E4B-B: Candidate separability ──
        close_cands = [p for p in preds if p.get("is_close_event_candidate") and not p.get("abstain")]
        # Find last OPEN step before each candidate
        open_steps = [p["step"] for p in preds if p.get("decoded_open_bool")]
        for idx, c in enumerate(close_cands):
            step = c["step"]
            prev_close = close_cands[idx - 1]["step"] if idx > 0 else None
            # Last OPEN before this candidate
            last_open = max([s for s in open_steps if s < step]) if [s for s in open_steps if s < step] else ""
            eef_delta = (float(c.get("eef_speed_now", 0) or 0) - float(c.get("eef_speed_prev", 0) or 0)) if c.get("eef_speed_now", "") != "" else ""

            separability_rows.append({
                "task_key": task, "state_id": state,
                "candidate_step": step, "candidate_index": idx,
                "is_teacher_p": int(step == p_anchor) if p_avail else 0,
                "is_teacher_r": int(step == r_anchor),
                "total_score": c["score"],
                "raw_crossing_bonus": c.get("raw_crossing_bonus", ""),
                "close_streak_bonus": c.get("close_streak_bonus", ""),
                "close_onset_qpos_bonus": c.get("close_onset_qpos_bonus", ""),
                "eef_deceleration_bonus": c.get("eef_deceleration_bonus", ""),
                "qpos_ready_bonus": c.get("qpos_ready_bonus", ""),
                "eef_speed_now": c.get("eef_speed_now", ""),
                "eef_speed_prev": c.get("eef_speed_prev", ""),
                "eef_deceleration_delta": eef_delta,
                "close_onset": c.get("close_onset", ""),
                "close_streak": c.get("close_streak_value", ""),
                "qpos": c.get("qpos", ""),
                "raw_crossing": int(c.get("raw_open_to_close_crossing", 0)),
                "time_since_prev_close": step - prev_close if prev_close is not None else "",
                "time_since_last_open": step - last_open if last_open != "" else "",
                "distance_to_teacher_p": step - p_anchor if p_avail else "",
                "disabled_features": ",".join(c.get("disabled_features", [])),
            })

        # Per-trace Teacher-P rank (by score among close candidates)
        if p_avail:
            sorted_cands = sorted(close_cands, key=lambda x: x["score"], reverse=True)
            p_rank = next((i + 1 for i, cc in enumerate(sorted_cands) if cc["step"] == p_anchor), -1)
            p_score_val = next((cc["score"] for cc in close_cands if cc["step"] == p_anchor), 0)
            max_score = sorted_cands[0]["score"] if sorted_cands else 0
            rank_rows.append({
                "task_key": task, "state_id": state,
                "teacher_p_anchor": p_anchor,
                "teacher_p_score": p_score_val,
                "teacher_p_rank_by_score": p_rank,
                "num_close_candidates": len(close_cands),
                "max_candidate_score": max_score,
                "n_tied_with_max": sum(1 for cc in close_cands if abs(cc["score"] - max_score) < 0.01),
            })

    # Remaining gates
    if invariant_total > 0: gate_errors.append(f"G7: {invariant_total} RC1a invariants")
    if field_issue_total > 0: gate_errors.append(f"G8: {field_issue_total} field issues")

    def _wcsv(rows, name):
        if not rows: return
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    _wcsv(policy_rows, "l12_e4b_policy_counterfactuals.csv")
    _wcsv(separability_rows, "l12_e4b_candidate_separability.csv")
    _wcsv(rank_rows, "l12_e4b_teacher_p_rank.csv")

    # Summary: online causal error
    n_p_avail = len(rank_rows)
    print(f"\n=== E4B.1 POLICY COUNTERFACTUAL ({n_p_avail} P-available) ===")
    for pol_name, _ in POLICIES:
        subset = [r for r in policy_rows if r["policy"].startswith(pol_name.split("_")[0]) and r["teacher_p_available"]]
        # Use actual policy name match
        subset = [r for r in policy_rows if r["policy"] == pol_name and r["teacher_p_available"]]
        n_correct = sum(1 for r in subset if r["online_is_near_P"] == 1)
        delays = [r["actual_delay"] for r in subset if isinstance(r["actual_delay"], (int, float)) and r["actual_delay"] >= 0]
        avg_d = sum(delays) / len(delays) if delays else 0
        causal_flag = subset[0]["causal"] if subset else ""
        print(f"  {pol_name:30s} causal={causal_flag} online_correct={n_correct}/{len(subset)} avg_delay={avg_d:.1f}")

    # Teacher-P rank summary
    n_top1 = sum(1 for r in rank_rows if r["teacher_p_rank_by_score"] == 1)
    n_top2 = sum(1 for r in rank_rows if r["teacher_p_rank_by_score"] <= 2)
    print(f"\n=== E4B.1 TEACHER-P RANK ===")
    print(f"  Top-1 by score: {n_top1}/{n_p_avail}")
    print(f"  Top-2 by score: {n_top2}/{n_p_avail}")

    # Run log
    log_path = out / "l12_e4b_run_log.txt"
    with open(log_path, "w") as f:
        f.write(f"E4B.1 RUN LOG\nrunner_commit: {RUNNER_COMMIT}\nremapper_version: {REMAPPER_VERSION}\n")
        f.write(f"start: {START_TIME.isoformat()}\ninput_traces: {len(manifest_rows)}\n")
        f.write(f"invariant_violations: {invariant_total}\nfield_issues: {field_issue_total}\n")
        f.write(f"gate_errors: {len(gate_errors)}\npolicies: {len(POLICIES)}\n")
        f.write(f"worktree_clean: {_git_is_clean()}\nexit_code: {1 if gate_errors else 0}\n")

    if args.tracked_tables_dir:
        tracked = Path(args.tracked_tables_dir)
        tracked.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.csv"): shutil.copy2(f, tracked / f.name)
        shutil.copy2(log_path, tracked / "l12_e4b_run_log.txt")

    if gate_errors:
        print(f"\nGATE FAILURES ({len(gate_errors)}):")
        for e in gate_errors: print(f"  {e}")
        sys.exit(1)

    print(f"\nOutput: {out}\nE4B.1 COMPLETE")


if __name__ == "__main__":
    try: main()
    except Exception: traceback.print_exc(); sys.exit(1)
