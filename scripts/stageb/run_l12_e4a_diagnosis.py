#!/usr/bin/env python3
"""E4A: Rule-family scoring/policy diagnosis.

For each Teacher-P-available trace, pairs online first trigger,
Teacher-R anchor, Teacher-P anchor, and highest-scoring candidate.
Classifies failure mode using score decomposition.

CPU only. Same 12 frozen traces. No tuning, no model, no attack.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import traceback
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


def _classify_failure(preds, p_anchor, r_anchor, first_trigger, best_candidate_step,
                       best_score, p_score, trigger_score):
    """Classify the failure mode for this trace."""
    if p_anchor < 0:
        return "teacher_reference_unavailable"

    # Find scores at key points
    if best_candidate_step == p_anchor and first_trigger == p_anchor:
        return "correct_near_anchor"

    # Check if Teacher-P candidate has highest score
    if p_score is not None and best_score is not None:
        if p_score >= best_score and first_trigger != p_anchor and first_trigger < p_anchor:
            return "first_hit_policy_failure__critical_ranked_higher_but_later"

    # Check score equality
    if trigger_score is not None and p_score is not None:
        if abs(trigger_score - p_score) < 0.01:
            return "exact_score_collision__first_hit_picks_earlier"

    # Check ranking
    if trigger_score is not None and p_score is not None:
        if trigger_score > p_score and first_trigger < p_anchor:
            return "spurious_candidate_ranked_higher"
        if p_score > trigger_score and first_trigger < p_anchor:
            return "first_hit_policy_failure__critical_ranked_higher_but_later"

    if first_trigger < p_anchor:
        return "early_trigger_unknown_cause"
    if first_trigger > p_anchor:
        return "late_trigger"

    return "unclassified"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tracked-tables-dir", default=None)
    args = ap.parse_args()

    RUNNER_COMMIT = _git_rev_parse()
    out = Path(args.output_dir)
    if out.exists():
        print(f"FATAL: output dir exists: {out}")
        sys.exit(1)
    out.mkdir(parents=True)

    with open(args.manifest, "r", newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    for mr in manifest_rows:
        actual = _sha256_file(mr["trace_path"])
        if actual != mr["expected_sha256"]:
            print(f"FATAL: SHA mismatch {mr['task_key']}_s{mr['state_id']}")
            sys.exit(1)

    diagnosis_rows = []
    candidate_decomp_rows = []

    for mr in manifest_rows:
        task = mr["task_key"]
        state = int(mr["state_id"])
        print(f"  {task}_s{state} ...", end=" ", flush=True)

        remap_out = str(out / f"remap_{task}_s{state}.csv")
        l12_rows, inv, fi = remap_v4_to_l12(mr["trace_path"], remap_out, raise_on_invariant=False)

        p_anchor = teacher_privileged_critical_close_anchor(l12_rows)
        r_anchor = teacher_rule_critical_close_anchor(l12_rows)
        p_avail = p_anchor >= 0

        preds = rule_based_close_predictor(l12_rows, horizon=PREDICTION_HORIZON,
                                            teacher_anchor=p_anchor if p_avail else -1)
        win_on = select_online_trigger(preds, mode="close_interception")
        first_trigger = win_on.get("trigger_step", -1)

        # Find highest-scoring non-abstaining step
        valid_preds = [p for p in preds if not p["abstain"]]
        best_pred = max(valid_preds, key=lambda p: p["score"]) if valid_preds else None
        best_step = best_pred["step"] if best_pred else -1
        best_score = best_pred["score"] if best_pred else 0.0

        # Scores at key points
        p_pred = preds[p_anchor] if p_avail and 0 <= p_anchor < len(preds) else None
        r_pred = preds[r_anchor] if r_anchor >= 0 and r_anchor < len(preds) else None
        t_pred = preds[first_trigger] if first_trigger >= 0 and first_trigger < len(preds) else None

        p_score = p_pred["score"] if p_pred else None
        r_score = r_pred["score"] if r_pred else None
        trigger_score = t_pred["score"] if t_pred else None

        failure = _classify_failure(preds, p_anchor, r_anchor, first_trigger,
                                     best_step, best_score, p_score, trigger_score)

        # Score decomposition at key anchors
        for label, pred_obj in [("Teacher-P", p_pred), ("Teacher-R", r_pred),
                                 ("first_trigger", t_pred), ("best_candidate", best_pred)]:
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
                "is_close_event_candidate": pred_obj.get("is_close_event_candidate", ""),
                "qpos": str(pred_obj.get("qpos", ""))[:10],
                "close_onset": pred_obj.get("close_onset", ""),
            })

        # Diagnosis row
        diagnosis_rows.append({
            "task_key": task, "state_id": state,
            "teacher_p_available": p_avail,
            "teacher_p_anchor": p_anchor, "teacher_p_score": p_score if p_score is not None else "",
            "teacher_r_anchor": r_anchor, "teacher_r_score": r_score if r_score is not None else "",
            "online_first_trigger": first_trigger, "trigger_score": trigger_score if trigger_score is not None else "",
            "best_candidate_step": best_step, "best_candidate_score": best_score,
            "failure_classification": failure,
        })
        print(failure)

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

    # Summary
    from collections import Counter
    counts = Counter(r["failure_classification"] for r in diagnosis_rows)
    print(f"\n=== E4A DIAGNOSIS SUMMARY ===")
    for cat, n in counts.most_common():
        pct = n / len(diagnosis_rows) * 100
        print(f"  {cat}: {n} ({pct:.0f}%)")

    if args.tracked_tables_dir:
        tracked = Path(args.tracked_tables_dir)
        tracked.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.csv"):
            shutil.copy2(f, tracked / f.name)
        print(f"\nTracked: {tracked}")

    print(f"\nOutput: {out}")
    print("E4A COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
