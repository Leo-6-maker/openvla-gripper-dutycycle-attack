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
    teacher_phase_labels,
    teacher_rule_critical_close_anchor,
    teacher_privileged_critical_close_anchor,
    teacher_window_proposal,
)
from gripper_attack.critical_close_selector import (
    rule_based_close_predictor,
    select_best_window,
    build_clean_proposal,
    WINDOW_LEN, PRE_OFFSET,
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
        phases = teacher_phase_labels(records)

        # Teacher-R: rule-based anchor (deployment-safe, fast baseline)
        anchor_r = teacher_rule_critical_close_anchor(records)
        ws_r, we_r = teacher_window_proposal(anchor_r, WINDOW_LEN, PRE_OFFSET)

        # Teacher-P: privileged anchor (object/target pose, may abstain)
        anchor_p = teacher_privileged_critical_close_anchor(records)
        ws_p, we_p = teacher_window_proposal(anchor_p, WINDOW_LEN, PRE_OFFSET) if anchor_p >= 0 else (-1, -1)

        # Layer2: Causal student selector
        preds = rule_based_close_predictor(records)
        win = select_best_window(preds, WINDOW_LEN, PRE_OFFSET)

        task_key = f"{args.task}"
        pid = f"{task_key}_s{args.state_id}_l12v1"

        # Build student proposal (teacher anchor = Teacher-P if available, else Teacher-R)
        primary_anchor = anchor_p if anchor_p >= 0 else anchor_r
        p = build_clean_proposal(
            task_key=task_key,
            state_id=args.state_id,
            trace_path=trace_path,
            trace_sha256=trace_sha,
            commit=args.commit,
            window_info=win,
            phase_label=phases[primary_anchor] if 0 <= primary_anchor < len(phases) else "",
        )

        # Annotate with teacher comparison (both Teacher-P and Teacher-R)
        anchor_error_p = abs(anchor_p - win["anchor_step"]) if anchor_p >= 0 else -1
        anchor_error_r = abs(anchor_r - win["anchor_step"]) if anchor_r >= 0 else -1

        proposals.append({
            "proposal": p,
            "trace_path": trace_path,
            "teacher_p_anchor": anchor_p,
            "teacher_p_window": f"[{ws_p},{we_p}]" if anchor_p >= 0 else "ABSTAIN",
            "teacher_r_anchor": anchor_r,
            "teacher_r_window": f"[{ws_r},{we_r}]" if anchor_r >= 0 else "N/A",
            "student_anchor": win["anchor_step"],
            "student_window": f"[{win['window_start']},{win['window_end']}]",
            "anchor_error_vs_p": anchor_error_p,
            "anchor_error_vs_r": anchor_error_r,
            "teacher_p_abstain": anchor_p < 0,
            "n_steps": len(records),
            "abstain": win.get("abstain_reason", ""),
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
            "teacher_p_anchor", "teacher_p_window",
            "teacher_r_anchor", "teacher_r_window",
            "student_anchor", "student_window",
            "anchor_error_vs_p", "anchor_error_vs_r",
            "teacher_p_abstain", "n_steps",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for p in proposals:
            d = p["proposal"].to_dict()
            d.update({
                "teacher_p_anchor": p["teacher_p_anchor"],
                "teacher_p_window": p["teacher_p_window"],
                "teacher_r_anchor": p["teacher_r_anchor"],
                "teacher_r_window": p["teacher_r_window"],
                "student_anchor": p["student_anchor"],
                "student_window": p["student_window"],
                "anchor_error_vs_p": p["anchor_error_vs_p"],
                "anchor_error_vs_r": p["anchor_error_vs_r"],
                "teacher_p_abstain": p["teacher_p_abstain"],
                "n_steps": p["n_steps"],
            })
            w.writerow(d)

    # Summary
    n_eligible = sum(1 for p in prop_objects if p.eligible)
    n_abstain = sum(1 for p in prop_objects if p.abstain_reason)
    n_p_abstain = sum(1 for p in proposals if p["teacher_p_abstain"])
    print(f"Task: {args.task}_s{args.state_id}")
    print(f"Traces: {len(traces)}")
    print(f"Proposals: {len(proposals)} ({n_eligible} eligible, {n_abstain} abstain)")
    print(f"Teacher-P abstains: {n_p_abstain}/{len(proposals)}")
    for p in proposals:
        err_p = p["anchor_error_vs_p"]
        err_r = p["anchor_error_vs_r"]
        err_p_str = f"err_p={err_p}" if err_p >= 0 else "P_ABSTAIN"
        err_r_str = f"err_r={err_r}" if err_r >= 0 else "N/A"
        print(f"  {p['trace_path']}: student={p['student_anchor']} "
              f"teacher_p={p['teacher_p_anchor']} {err_p_str} "
              f"teacher_r={p['teacher_r_anchor']} {err_r_str} "
              f"window={p['student_window']} "
              f"abstain={p['abstain']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
