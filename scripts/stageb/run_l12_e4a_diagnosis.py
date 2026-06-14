#!/usr/bin/env python3
"""E4A.1: Corrected failure taxonomy and score decomposition diagnosis.

Outputs per-anchor score decomposition, then classifies online trigger
outcome using the frozen ±4 near-anchor standard. Distinguishes:
  correct/near-correct, policy-only failure, collision with/without
  higher spurious, spurious-ranked-higher, etc.

CPU only. Same 12 frozen traces. No tuning, no new features.
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
    select_online_trigger,
    WINDOW_LEN, PRE_OFFSET, PREDICTION_HORIZON,
)

NEAR_THRESHOLD = 4


def _sha256_file(path: str) -> str:
    import hashlib
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev_parse() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    except Exception:
        return "unknown"


def _git_is_clean() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "-uno"], cwd=str(REPO_ROOT), text=True).strip()
        return out == ""
    except Exception:
        return False


def _classify_failure(first_trigger, p_anchor):
    """Binary check: is the online trigger correct per frozen ±4 standard?"""
    if p_anchor < 0:
        return "teacher_reference_unavailable", True
    if first_trigger < 0:
        return "no_trigger", True
    if abs(first_trigger - p_anchor) <= NEAR_THRESHOLD:
        return "correct_or_near_teacher_anchor", True
    return None, False  # needs sub-classification


def _subclassify_early_failure(preds, first_trigger, p_anchor):
    """For traces where trigger is already known to be wrong (early),
    determine WHY the early trigger fired instead of the Teacher-P anchor."""
    # Filter close-event candidates only
    close_candidates = [p for p in preds
                        if p.get("is_close_event_candidate") and not p.get("abstain")]
    all_valid = [p for p in preds if not p.get("abstain")]

    # Global best close candidate
    best_close = max(close_candidates, key=lambda p: p["score"]) if close_candidates else None
    best_all = max(all_valid, key=lambda p: p["score"]) if all_valid else None

    p_pred = preds[p_anchor] if 0 <= p_anchor < len(preds) else None
    trigger_pred = preds[first_trigger] if 0 <= first_trigger < len(preds) else None

    p_score = p_pred["score"] if p_pred else None
    trigger_score = trigger_pred["score"] if trigger_pred else None

    # Is P the unique global best close candidate?
    p_is_global_best = (best_close is not None and best_close["step"] == p_anchor)
    num_tied = sum(1 for p in close_candidates
                   if abs(p["score"] - (p_score or 0)) < 0.01)
    num_higher = sum(1 for p in close_candidates
                     if p["score"] > (p_score or 0) + 0.01)

    extra = {
        "trigger_error": first_trigger - p_anchor,
        "p_is_global_best_close": p_is_global_best,
        "num_candidates_tied_with_p": num_tied,
        "num_candidates_higher_than_p": num_higher,
        "best_close_step": best_close["step"] if best_close else -1,
        "best_close_score": best_close["score"] if best_close else 0.0,
        "best_nonabstain_step": best_all["step"] if best_all else -1,
        "best_nonabstain_score": best_all["score"] if best_all else 0.0,
    }

    # Policy-only: P is unique global best, higher than trigger, but trigger fired first
    if p_is_global_best and num_higher == 0 and p_score is not None and trigger_score is not None:
        if p_score > trigger_score + 0.01:
            return ("policy_only__P_is_unique_global_best_and_higher", extra)
        # P is best AND tied with trigger
        return ("trigger_P_exact_score_collision__no_higher_spurious", extra)

    # Collision: trigger and P have same score, but higher spurious exists
    if (trigger_score is not None and p_score is not None and
        abs(trigger_score - p_score) < 0.01 and not p_is_global_best):
        return ("trigger_P_score_collision__higher_spurious_exists", extra)

    # Spurious ranked higher
    if trigger_score is not None and p_score is not None and trigger_score > p_score + 0.01:
        return ("spurious_trigger_ranked_higher_than_P", extra)

    return ("early_unknown", extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tracked-tables-dir", default=None)
    args = ap.parse_args()

    RUNNER_COMMIT = _git_rev_parse()
    START_TIME = datetime.now(timezone.utc)

    # Provenance gates
    gate_errors = []
    if not _git_is_clean():
        gate_errors.append("G1: tracked worktree dirty")

    out = Path(args.output_dir)
    if out.exists():
        gate_errors.append(f"FATAL: output dir exists: {out}")
    else:
        out.mkdir(parents=True)

    with open(args.manifest, "r", newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    if len(manifest_rows) != 12:
        gate_errors.append(f"G2: expected 12 manifest rows, got {len(manifest_rows)}")

    for mr in manifest_rows:
        actual = _sha256_file(mr["trace_path"])
        if actual != mr["expected_sha256"]:
            gate_errors.append(f"G3: SHA mismatch {mr['task_key']}_s{mr['state_id']}")

    if gate_errors:
        for e in gate_errors:
            print(e)
        sys.exit(1)

    diagnosis_rows = []
    candidate_decomp_rows = []
    invariant_total = 0
    field_issue_total = 0

    for mr in manifest_rows:
        task = mr["task_key"]
        state = int(mr["state_id"])

        remap_out = str(out / f"remap_{task}_s{state}.csv")
        l12_rows, inv, fi = remap_v4_to_l12(mr["trace_path"], remap_out, raise_on_invariant=False)
        invariant_total += len(inv)
        field_issue_total += len(fi)
        if len(l12_rows) != int(mr["expected_row_count"]):
            gate_errors.append(f"G5: row count mismatch {task}_s{state}")

        p_anchor = teacher_privileged_critical_close_anchor(l12_rows)
        r_anchor = teacher_rule_critical_close_anchor(l12_rows)
        p_avail = p_anchor >= 0

        preds = rule_based_close_predictor(l12_rows, horizon=PREDICTION_HORIZON,
                                            teacher_anchor=p_anchor if p_avail else -1)
        win_on = select_online_trigger(preds, mode="close_interception")
        first_trigger = win_on.get("trigger_step", -1)

        # Step 1: binary correct/not
        cat, is_terminal = _classify_failure(first_trigger, p_anchor)
        extra = {}
        if not is_terminal:
            cat, extra = _subclassify_early_failure(preds, first_trigger, p_anchor)

        # Score decomposition at key anchors
        close_cands = [p for p in preds if p.get("is_close_event_candidate") and not p.get("abstain")]
        best_close = max(close_cands, key=lambda p: p["score"]) if close_cands else None

        for label, pred_obj in [
            ("Teacher-P", preds[p_anchor] if p_avail and 0 <= p_anchor < len(preds) else None),
            ("Teacher-R", preds[r_anchor] if r_anchor >= 0 and r_anchor < len(preds) else None),
            ("first_trigger", preds[first_trigger] if first_trigger >= 0 and first_trigger < len(preds) else None),
            ("best_close_candidate", best_close),
        ]:
            if pred_obj is None:
                continue
            candidate_decomp_rows.append({
                "task_key": task, "state_id": state,
                "anchor_label": label,
                "step": pred_obj["step"],
                "total_score": pred_obj["score"],
                "raw_crossing_bonus": pred_obj.get("raw_crossing_bonus", ""),
                "close_streak_bonus": pred_obj.get("close_streak_bonus", ""),
                "close_onset_qpos_bonus": pred_obj.get("close_onset_qpos_bonus", ""),
                "eef_deceleration_bonus": pred_obj.get("eef_deceleration_bonus", ""),
                "qpos_ready_bonus": pred_obj.get("qpos_ready_bonus", ""),
                "decoded_open_penalty": pred_obj.get("decoded_open_penalty", ""),
                "eef_speed_now": pred_obj.get("eef_speed_now", ""),
                "eef_speed_prev": pred_obj.get("eef_speed_prev", ""),
                "disabled_features": ",".join(pred_obj.get("disabled_features", [])),
            })

        p_pred = preds[p_anchor] if p_avail and 0 <= p_anchor < len(preds) else None
        trigger_pred = preds[first_trigger] if first_trigger >= 0 and first_trigger < len(preds) else None

        diagnosis_rows.append({
            "task_key": task, "state_id": state,
            "teacher_p_available": p_avail,
            "teacher_p_anchor": p_anchor,
            "teacher_p_score": p_pred["score"] if p_pred else "",
            "teacher_r_anchor": r_anchor,
            "online_first_trigger": first_trigger,
            "trigger_score": trigger_pred["score"] if trigger_pred else "",
            "trigger_error": extra.get("trigger_error", first_trigger - p_anchor if p_avail else ""),
            "is_near_correct": int(abs(first_trigger - p_anchor) <= NEAR_THRESHOLD) if p_avail and first_trigger >= 0 else "",
            "p_is_global_best_close": extra.get("p_is_global_best_close", ""),
            "num_candidates_tied_with_p": extra.get("num_candidates_tied_with_p", ""),
            "num_candidates_higher_than_p": extra.get("num_candidates_higher_than_p", ""),
            "best_close_step": extra.get("best_close_step", ""),
            "best_close_score": extra.get("best_close_score", ""),
            "failure_classification": cat,
        })
        print(f"  {task}_s{state}: {cat}")

    # Remaining gates
    if invariant_total > 0:
        gate_errors.append(f"G7: {invariant_total} RC1a invariant violations")
    if field_issue_total > 0:
        gate_errors.append(f"G8: {field_issue_total} field issues")

    # Write tables
    def _wcsv(rows, name):
        if not rows:
            return
        p = out / name
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _wcsv(diagnosis_rows, "l12_e4a_failure_diagnosis.csv")
    _wcsv(candidate_decomp_rows, "l12_e4a_score_decomposition.csv")

    # Run log
    log_path = out / "l12_e4a_run_log.txt"
    with open(log_path, "w") as f:
        f.write(f"E4A.1 RUN LOG\n")
        f.write(f"runner_commit: {RUNNER_COMMIT}\n")
        f.write(f"remapper_version: {REMAPPER_VERSION}\n")
        f.write(f"start: {START_TIME.isoformat()}\n")
        f.write(f"input_traces: {len(manifest_rows)}\n")
        f.write(f"invariant_violations: {invariant_total}\n")
        f.write(f"field_issues: {field_issue_total}\n")
        f.write(f"gate_errors: {len(gate_errors)}\n")
        f.write(f"worktree_clean: {_git_is_clean()}\n")
        f.write(f"exit_code: {1 if gate_errors else 0}\n")

    # Summary
    counts = Counter(r["failure_classification"] for r in diagnosis_rows)
    n_p_avail = sum(1 for r in diagnosis_rows if r["teacher_p_available"])
    n_correct = sum(1 for r in diagnosis_rows if r["failure_classification"] == "correct_or_near_teacher_anchor")
    print(f"\n=== E4A.1 DIAGNOSIS SUMMARY ===")
    print(f"Traces: {len(diagnosis_rows)} (P-available: {n_p_avail})")
    print(f"Correct/near-correct: {n_correct}/{n_p_avail}")
    for cat, n in counts.most_common():
        print(f"  {cat}: {n}")

    if gate_errors:
        print(f"\nGATE FAILURES ({len(gate_errors)}):")
        for e in gate_errors:
            print(f"  {e}")
        sys.exit(1)

    if args.tracked_tables_dir:
        tracked = Path(args.tracked_tables_dir)
        tracked.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.csv"):
            shutil.copy2(f, tracked / f.name)
        shutil.copy2(log_path, tracked / "l12_e4a_run_log.txt")
        print(f"\nTracked: {tracked}")

    print(f"\nOutput: {out}")
    print("E4A.1 COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
