#!/usr/bin/env python3
"""L12 clean-only window selector runner.

Reads clean V6 trace CSVs, runs Layer1 (phase estimation) and
Layer2 (critical-close scoring), outputs frozen WindowProposals.

Usage:
  python run_l12_clean_window_selector.py \
    --trace-dir /path/to/clean/traces \
    --task butter --state-id 2 \
    --commit $(git rev-parse HEAD) \
    --output proposals.csv

No attack data is read or used.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.window_contract import WindowProposal, validate_proposals
from gripper_attack.phase_detector import (
    teacher_rule_phase_labels,
    teacher_rule_critical_close_anchor,
    teacher_privileged_critical_close_anchor,
    teacher_window_proposal,
)
from gripper_attack.critical_close_selector import (
    rule_based_close_predictor,
    select_best_window,
    select_online_trigger,
    build_clean_proposal,
    WINDOW_LEN, PRE_OFFSET, PREDICTION_HORIZON,
)


def _sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_clean_traces(trace_dir: str, task: str,
                      state_id: int) -> list[tuple[str, list[dict]]]:
    """Load clean observer traces for a specific task/state."""
    results = []
    pattern = os.path.join(
        trace_dir,
        f"trace_{task}_s{state_id}_*clean_observer*.csv"
    )
    for fp in sorted(glob.glob(pattern)):
        with open(fp) as f:
            reader = csv.DictReader(f)
            records = list(reader)
        results.append((fp, records))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--state-id", type=int, required=True)
    parser.add_argument("--commit", default="unknown")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    traces = load_clean_traces(args.trace_dir, args.task, args.state_id)
    if not traces:
        print(f"No clean traces found for {args.task}_s{args.state_id}")
        return

    proposals = []
    for trace_path, records in traces:
        trace_sha = _sha256_file(trace_path)

        # Layer1: Teacher phase estimation
        phases = teacher_rule_phase_labels(records)

        # Teacher-R: rule-based anchor (deployment-safe, fast baseline)
        anchor_r = teacher_rule_critical_close_anchor(records)
        ws_r, we_r = teacher_window_proposal(anchor_r, WINDOW_LEN, PRE_OFFSET)

        # Teacher-P: privileged anchor (object/target pose, may abstain)
        anchor_p = teacher_privileged_critical_close_anchor(records)
        ws_p, we_p = teacher_window_proposal(anchor_p, WINDOW_LEN, PRE_OFFSET) if anchor_p >= 0 else (-1, -1)

        # Layer2: Causal student selector
        # Teacher-P is the SOLE evaluation target. When Teacher-P abstains,
        # we do NOT fall back to Teacher-R — we mark teacher_reference_unavailable.
        teacher_p_available = anchor_p >= 0
        horizon_anchor = anchor_p if teacher_p_available else -1

        # --- Mode A: offline clean-repeat ---
        preds_offline = rule_based_close_predictor(
            records, horizon=PREDICTION_HORIZON, teacher_anchor=horizon_anchor)
        win_offline = select_best_window(preds_offline, WINDOW_LEN, PRE_OFFSET)

        # --- Mode B: online streaming ---
        preds_online = rule_based_close_predictor(
            records, horizon=PREDICTION_HORIZON, teacher_anchor=horizon_anchor)
        win_online = select_online_trigger(preds_online)

        task_key = f"{args.task}"

        # Build offline proposal — student anchor for phase_label (no Teacher-P pollution)
        student_anchor_off = win_offline.get("anchor_step", -1)
        p_offline = build_clean_proposal(
            task_key=task_key,
            state_id=args.state_id,
            trace_path=trace_path,
            trace_sha256=trace_sha,
            commit=args.commit,
            window_info=win_offline,
            phase_label=phases[student_anchor_off] if student_anchor_off >= 0 and student_anchor_off < len(phases) else "",
            selection_mode="offline_clean_repeat",
            is_online=False,
            first_close_horizon=PREDICTION_HORIZON,
        )

        # Build online proposal — phase_label left empty (teacher_rule_phase_labels
        # is not strictly causal — it scans forward and uses absolute T).
        # A causal phase estimator is not yet implemented; using empty string
        # prevents future leakage in online proposals.
        p_online = build_clean_proposal(
            task_key=task_key,
            state_id=args.state_id,
            trace_path=trace_path,
            trace_sha256=trace_sha,
            commit=args.commit,
            window_info=win_online,
            phase_label="",  # online: no future-capable phase label available
            selection_mode="online_streaming",
            is_online=True,
            first_close_horizon=0,
            prediction_mode=win_online.get("prediction_mode", "observed_close_interception"),
        )

        # Annotate offline proposal — Teacher-P is evaluation target, Teacher-R is baseline
        # anchor_error: None when teacher unavailable OR student abstains (NOT numeric fallback)
        student_available_off = student_anchor_off >= 0
        anchor_error_p_off = (
            abs(anchor_p - student_anchor_off)
            if teacher_p_available and student_available_off
            else None
        )
        anchor_error_r_off = (
            abs(anchor_r - student_anchor_off)
            if anchor_r >= 0 and student_available_off
            else None
        )

        proposals.append({
            "proposal": p_offline,
            "trace_path": trace_path,
            "mode": "offline",
            "teacher_p_anchor": anchor_p,
            "teacher_p_window": f"[{ws_p},{we_p}]" if anchor_p >= 0 else "ABSTAIN",
            "teacher_reference_unavailable": not teacher_p_available,
            "teacher_r_anchor": anchor_r,
            "teacher_r_window": f"[{ws_r},{we_r}]" if anchor_r >= 0 else "N/A",
            "student_anchor": win_offline["anchor_step"],
            "student_window": f"[{win_offline['window_start']},{win_offline['window_end']}]",
            "anchor_error_vs_p": anchor_error_p_off,    # int or None
            "anchor_error_vs_r": anchor_error_r_off,
            "teacher_p_abstain": not teacher_p_available,
            "n_steps": len(records),
            "abstain": win_offline.get("abstain_reason", ""),
        })

        # Annotate online proposal
        online_trigger = win_online.get("trigger_step", -1)
        student_available_on = online_trigger >= 0
        anchor_error_p_on = (
            abs(anchor_p - online_trigger)
            if teacher_p_available and student_available_on
            else None
        )
        anchor_error_r_on = (
            abs(anchor_r - online_trigger)
            if anchor_r >= 0 and student_available_on
            else None
        )

        proposals.append({
            "proposal": p_online,
            "trace_path": trace_path,
            "mode": "online",
            "teacher_p_anchor": anchor_p,
            "teacher_p_window": f"[{ws_p},{we_p}]" if anchor_p >= 0 else "ABSTAIN",
            "teacher_reference_unavailable": not teacher_p_available,
            "teacher_r_anchor": anchor_r,
            "teacher_r_window": f"[{ws_r},{we_r}]" if anchor_r >= 0 else "N/A",
            "student_anchor": online_trigger,
            "student_window": f"[{win_online['window_start']},{win_online['window_end']}]" if online_trigger >= 0 else "NO_TRIGGER",
            "anchor_error_vs_p": anchor_error_p_on,    # int or None
            "anchor_error_vs_r": anchor_error_r_on,
            "teacher_p_abstain": not teacher_p_available,
            "n_steps": len(records),
            "abstain": win_online.get("abstain_reason", ""),
        })

    # Validate
    prop_objects = [p["proposal"] for p in proposals]
    issues, valid = validate_proposals(prop_objects)
    if issues:
        print(f"VALIDATION ISSUES ({len(issues)}):")
        for i in issues:
            print(f"  {i}")

    # Write CSV
    with open(args.output, "w", newline="") as f:
        fieldnames = list(prop_objects[0].to_dict().keys()) + [
            "mode",
            "teacher_p_anchor", "teacher_p_window",
            "teacher_r_anchor", "teacher_r_window",
            "teacher_reference_unavailable",
            "student_anchor", "student_window",
            "anchor_error_vs_p", "anchor_error_vs_r",
            "teacher_p_abstain", "n_steps",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for p in proposals:
            d = p["proposal"].to_dict()
            d.update({
                "mode": p["mode"],
                "teacher_p_anchor": p["teacher_p_anchor"],
                "teacher_p_window": p["teacher_p_window"],
                "teacher_r_anchor": p["teacher_r_anchor"],
                "teacher_r_window": p["teacher_r_window"],
                "teacher_reference_unavailable": p["teacher_reference_unavailable"],
                "student_anchor": p["student_anchor"],
                "student_window": p["student_window"],
                "anchor_error_vs_p": p["anchor_error_vs_p"] if p["anchor_error_vs_p"] is not None else "",
                "anchor_error_vs_r": p["anchor_error_vs_r"],
                "teacher_p_abstain": p["teacher_p_abstain"],
                "n_steps": p["n_steps"],
            })
            w.writerow(d)

    # Summary
    offline_props = [p for p in proposals if p["mode"] == "offline"]
    online_props = [p for p in proposals if p["mode"] == "online"]
    n_eligible_off = sum(1 for p in offline_props if p["proposal"].eligible)
    n_abstain_off = sum(1 for p in offline_props if p["proposal"].abstain_reason)
    n_eligible_on = sum(1 for p in online_props if p["proposal"].eligible)
    n_abstain_on = sum(1 for p in online_props if p["proposal"].abstain_reason)
    n_p_abstain = sum(1 for p in offline_props if p["teacher_p_abstain"])
    print(f"Task: {args.task}_s{args.state_id}")
    print(f"Traces: {len(traces)}")
    print(f"Proposals: {len(proposals)} total "
          f"({len(offline_props)} offline, {len(online_props)} online)")
    print(f"Offline: {n_eligible_off} eligible, {n_abstain_off} abstain")
    print(f"Online:  {n_eligible_on} eligible, {n_abstain_on} abstain")
    print(f"Teacher-P abstains: {n_p_abstain}/{len(offline_props)}")
    print()
    for p in proposals:
        err_p = p["anchor_error_vs_p"]
        err_r = p["anchor_error_vs_r"]
        if err_p is None or err_p == "":
            err_p_str = "P_ABSTAIN"
        elif err_p >= 0:
            err_p_str = f"err_p={err_p}"
        else:
            err_p_str = "N/A"
        err_r_str = f"err_r={err_r}" if isinstance(err_r, (int, float)) and err_r >= 0 else "N/A"
        print(f"  [{p['mode']}] {p['trace_path']}: student={p['student_anchor']} "
              f"teacher_p={p['teacher_p_anchor']} {err_p_str} "
              f"teacher_r={p['teacher_r_anchor']} {err_r_str} "
              f"window={p['student_window']} "
              f"abstain={p['abstain']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
