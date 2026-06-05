#!/usr/bin/env python3
"""Prepare full-window VIS-1R screening commands without running them."""

from __future__ import annotations

import argparse
import csv
import os


FIELDS = [
    "target_id", "task_key", "state_id", "window_start", "window_end",
    "condition", "eps_raw_pixels", "pgd_steps", "pgd_restarts", "objective",
    "output_dir", "command", "runtime_status",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="tables/object_phase_response_batch4_candidates.csv")
    ap.add_argument("--output-csv", default="tables/vis_1r_full_window_screening_commands.csv")
    ap.add_argument("--output-report", default="reports/VIS_1R_FULL_WINDOW_SCREENING_PLAN.md")
    ap.add_argument("--output-dir", default="/data/liuyu/outputs/vis_1r_full_window_screening_20260605")
    ap.add_argument("--eps-raw-pixels", type=int, default=6)
    ap.add_argument("--pgd-steps", type=int, default=40)
    ap.add_argument("--pgd-restarts", type=int, default=1)
    ap.add_argument("--objective", default="prefix_locked_gripper_open_margin")
    ap.add_argument("--gpu-pair-placeholder", default="<GPU_PAIR>")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def norm(v):
    return str(v if v is not None else "").strip()


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{norm(k).lstrip("\ufeff"): v for k, v in row.items()} for row in csv.DictReader(f)]


def blocked_proxy(row):
    text = " ".join(norm(v).lower() for v in row.values())
    return any(tok in text for tok in ["phase_d", "phase_e", "command_proxy", "low_budget"])


def command_for(row, args):
    return (
        "python scripts/vis_phase_conditioned_attack.py "
        f"--task {norm(row.get('task_key'))} "
        f"--state-id {norm(row.get('state_id'))} "
        "--seed 0 --condition vis_pgd --window-source fixed "
        f"--fixed-window-start {norm(row.get('window_start'))} "
        f"--fixed-window-end {norm(row.get('window_end'))} "
        f"--eps_raw_pixels {args.eps_raw_pixels} "
        f"--pgd_steps {args.pgd_steps} --pgd_restarts {args.pgd_restarts} "
        f"--objective {args.objective} "
        f"--gpu_pair {args.gpu_pair_placeholder} "
        f"--output-dir {args.output_dir}"
    )


def main():
    args = parse_args()
    candidates = read_csv(args.candidates)
    rows = []
    for i, row in enumerate(candidates, start=1):
        if blocked_proxy(row):
            continue
        ws = norm(row.get("window_start"))
        we = norm(row.get("window_end"))
        if not ws or not we:
            continue
        rows.append({
            "target_id": norm(row.get("target_id")) or f"vis1r_{i:03d}",
            "task_key": norm(row.get("task_key")),
            "state_id": norm(row.get("state_id")),
            "window_start": ws,
            "window_end": we,
            "condition": "vis_pgd",
            "eps_raw_pixels": str(args.eps_raw_pixels),
            "pgd_steps": str(args.pgd_steps),
            "pgd_restarts": str(args.pgd_restarts),
            "objective": args.objective,
            "output_dir": args.output_dir,
            "command": command_for(row, args),
            "runtime_status": "not_run_cpu_only_plan",
        })
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join([
            "# VIS-1R Full-Window Screening Plan",
            "",
            f"**Candidates input**: `{args.candidates}`",
            f"**Commands prepared**: {len(rows)}",
            f"**pgd_restarts**: {args.pgd_restarts}",
            f"**pgd_steps**: {args.pgd_steps}",
            "",
            "This is a CPU-only command plan. It did not run GPU, VIS, rollout, watcher, or detector training.",
            "",
            "## Label Boundary",
            "",
            "- 1R positives may be treated as `silver_positive_1r` only after audit.",
            "- 1R negatives are `pending_negative_1r`, never gold negatives.",
            "- Gold labels require full VIS 3R confirmation.",
        ]))
    if args.dry_run:
        print(f"DRY RUN: prepared {len(rows)} VIS-1R commands")
        for row in rows[:5]:
            print(row["command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
