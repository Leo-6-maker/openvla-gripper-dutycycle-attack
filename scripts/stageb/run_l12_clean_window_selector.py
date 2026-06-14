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
    teacher_critical_close_anchor,
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
        anchor = teacher_critical_close_anchor(records)
        ws, we = teacher_window_proposal(anchor, WINDOW_LEN, PRE_OFFSET)

        # Layer2: Causal student selector
        preds = rule_based_close_predictor(records)
        win = select_best_window(preds, WINDOW_LEN, PRE_OFFSET)

        task_key = f"{args.task}"
        pid = f"{task_key}_s{args.state_id}_l12v1"

        # Build student proposal
        p = build_clean_proposal(
            task_key=task_key,
            state_id=args.state_id,
            trace_path=trace_path,
            trace_sha256=trace_sha,
            commit=args.commit,
            window_info=win,
            phase_label=phases[anchor] if 0 <= anchor < len(phases) else "",
        )

        # Annotate with teacher comparison
        teacher_ws, teacher_we = ws, we
        anchor_error = abs(anchor - win["anchor_step"]) if anchor >= 0 else -1

        proposals.append({
            "proposal": p,
            "trace_path": trace_path,
            "teacher_anchor": anchor,
            "teacher_window": f"[{teacher_ws},{teacher_we}]",
            "student_anchor": win["anchor_step"],
            "student_window": f"[{win['window_start']},{win['window_end']}]",
            "anchor_error": anchor_error,
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
            "teacher_anchor", "teacher_window", "student_anchor",
            "student_window", "anchor_error", "n_steps",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for p in proposals:
            d = p["proposal"].to_dict()
            d.update({
                "teacher_anchor": p["teacher_anchor"],
                "teacher_window": p["teacher_window"],
                "student_anchor": p["student_anchor"],
                "student_window": p["student_window"],
                "anchor_error": p["anchor_error"],
                "n_steps": p["n_steps"],
            })
            w.writerow(d)

    # Summary
    n_eligible = sum(1 for p in prop_objects if p.eligible)
    n_abstain = sum(1 for p in prop_objects if p.abstain_reason)
    print(f"Task: {args.task}_s{args.state_id}")
    print(f"Traces: {len(traces)}")
    print(f"Proposals: {len(proposals)} ({n_eligible} eligible, {n_abstain} abstain)")
    for p in proposals:
        err = p["anchor_error"]
        err_str = f"err={err}" if err >= 0 else "N/A"
        print(f"  {p['trace_path']}: student={p['student_anchor']} "
              f"teacher={p['teacher_anchor']} {err_str} "
              f"window={p['student_window']} "
              f"abstain={p['abstain']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
