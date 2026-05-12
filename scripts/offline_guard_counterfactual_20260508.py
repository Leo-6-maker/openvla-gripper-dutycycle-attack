#!/usr/bin/env python3
"""Offline command-layer gripper guard counterfactual for state7 seedrep runs.

DEPRECATED (2026-05-10): This script was a diagnostic prototype that performed
log-level counterfactual analysis only.  The paper-facing causal defense
experiment is now implemented as an online guard in v4_run_eval_openvla.py
(flag: --guard_enabled).  See the plan at .claude/plans/ for the full
experimental design.

This script is retained for provenance and should NOT be cited as a defense
evaluation in the paper.

Purpose (historical):
    Read logged step records, detect consecutive unsafe open commands during
    contact-like phases, clamp those command-layer opens to close, and report
    original vs guarded open streaks and unsafe-open segments.
"""

from __future__ import annotations

import csv
import os
import json
from pathlib import Path

ROOT = Path(os.environ.get("OPENVLA_STATE7_SEEDREP_ROOT", "outputs/state7_seedrep_20260508/runs"))
OUT = Path(os.environ.get("OPENVLA_OFFLINE_GUARD_OUT", "outputs/state7_seedrep_20260508/tables/offline_guard_counterfactual.csv"))

RUNS = [
    "S7_ORACLE_continuous_seed1_state7_z004",
    "S7_ORACLE_continuous_seed2_state7_z004",
    "S7_VIS_margin_prevdelta_seed1_state7_z004",
    "S7_VIS_margin_prevdelta_seed2_state7_z004",
    "S7_CTRL_same_gate_zero_margin_seed1_state7_z004",
    "S7_CTRL_same_gate_zero_margin_seed2_state7_z004",
    "S7_CTRL_random_direction_seed1_state7_z004",
    "S7_CTRL_random_direction_seed2_state7_z004_retry_gpu67b",
    "S7_clean_seed1_state7_z004",
    "S7_clean_seed2_state7_z004_retry_gpu45b",
]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fnum(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def is_open(row: dict) -> bool:
    return fnum(row.get("executed_gripper_env"), 1.0) < -0.5


def in_contact_phase(row: dict) -> bool:
    if row.get("attack_active"):
        return True
    if fnum(row.get("proxy_lift_carry_eef_z_delta_from_min")) >= 0.04:
        return True
    if abs(fnum(row.get("grasp_bowl_z_delta"))) >= 0.02 and fnum(row.get("grasp_eef_bowl_dist"), 999) <= 0.14:
        return True
    return False


def segments(open_flags: list[bool]) -> list[tuple[int, int]]:
    out = []
    start = None
    for i, flag in enumerate(open_flags):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(open_flags) - 1):
            end = i if flag and i == len(open_flags) - 1 else i - 1
            out.append((start, end))
            start = None
    return out


def max_streak(flags: list[bool]) -> int:
    best = cur = 0
    for flag in flags:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return best


def summarize(run_id: str) -> dict:
    rd = ROOT / run_id / run_id
    steps = read_jsonl(rd / "step_records.jsonl")
    eps = read_jsonl(rd / "episode_records.jsonl")
    ep = eps[0] if eps else {}

    attack_or_contact = [row for row in steps if in_contact_phase(row)]
    original_flags = [is_open(row) for row in attack_or_contact]
    original_segments = [(a, b) for a, b in segments(original_flags) if (b - a + 1) >= 2]
    guarded_flags = original_flags[:]
    for a, b in original_segments:
        for i in range(a, b + 1):
            guarded_flags[i] = False
    guarded_segments = [(a, b) for a, b in segments(guarded_flags) if (b - a + 1) >= 2]

    original_streak = max_streak(original_flags)
    guarded_streak = max_streak(guarded_flags)
    original_lift_fail = str(ep.get("failure_phase", "")) == "lift_fail"
    guarded_proxy = "possibly_prevented" if original_lift_fail and original_segments and not guarded_segments else "not_prevented"
    if not original_lift_fail:
        guarded_proxy = "not_applicable_success"

    return {
        "run_id": run_id,
        "episode_id": 0,
        "guard_enabled": True,
        "original_open_streak": original_streak,
        "guarded_open_streak": guarded_streak,
        "original_unsafe_open_segments": len(original_segments),
        "guarded_unsafe_open_segments": len(guarded_segments),
        "original_lift_fail": original_lift_fail,
        "guarded_lift_fail_proxy": guarded_proxy,
        "clean_success_impact": "check_false_positive" if "clean" in run_id else "",
        "failure_phase": ep.get("failure_phase", ""),
        "success": ep.get("success", ""),
        "notes": "command-layer counterfactual only; no physical qpos replay",
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = [summarize(run) for run in RUNS]
    with OUT.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT)


if __name__ == "__main__":
    main()
