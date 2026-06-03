#!/usr/bin/env python3
"""vis_phase_conditioned_attack.py — run VIS prefix_margin on phase-selected windows.

Wraps scripts/vis_rollout_adaptive_v3.py with phase-specific window selection.
"""

from __future__ import annotations
import argparse, csv, os, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get("ATTACK_REPO", "/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524"))
VIS_ROLLOUT = str(REPO / "scripts/vis_rollout_adaptive_v3.py")
PYTHON = os.environ.get("PYTHON_BIN", "python")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", choices=["clean","random_linf","vis_pgd"], required=True)
    ap.add_argument("--window-source", choices=["fixed","heuristic_phase","proprionostep_offset","phase_selector"], default="fixed")
    ap.add_argument("--phase", choices=["pre_grasp","grasp_formation","post_grasp","release"], default="grasp_formation")
    ap.add_argument("--fixed-window-start", type=int, default=10)
    ap.add_argument("--fixed-window-end", type=int, default=27)
    ap.add_argument("--phase-csv", default="tables/phase_alignment_clean_rollouts.csv")
    ap.add_argument("--window-proposals-csv", default="tables/phase_selector_window_proposals.csv")
    ap.add_argument("--proprionostep-trigger-step", type=int, default=93)
    ap.add_argument("--offset", type=int, default=-10)
    ap.add_argument("--eps_raw_pixels", type=int, default=6)
    ap.add_argument("--objective", default="prefix_locked_gripper_open_margin")
    ap.add_argument("--pgd_steps", type=int, default=40)
    ap.add_argument("--pgd_restarts", type=int, default=3)
    ap.add_argument("--gpu_pair", default="6,7")
    ap.add_argument("--output-dir", default=".")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def get_window_from_source(args):
    """Determine attack window from selected source."""
    ws, we = None, None
    window_source = args.window_source
    selector_type = "unknown"

    if window_source == "fixed":
        ws, we = args.fixed_window_start, args.fixed_window_end
        selector_type = "fixed"
    elif window_source == "proprionostep_offset":
        T = args.proprionostep_trigger_step
        ws = max(0, T + args.offset)
        we = min(299, ws + 17)
        selector_type = f"proprionostep_offset_{args.offset}"
    elif window_source == "heuristic_phase" and os.path.exists(args.phase_csv):
        with open(args.phase_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        tr = [r for r in rows if r.get("task")==args.task and int(r.get("seed",-1))==args.seed]
        if args.phase == "grasp_formation":
            gs = [int(r["policy_step"]) for r in tr if r.get("phase_label_3class")=="grasp_formation"]
            ws = min(gs) if gs else 10
            we = min(ws + 17, 299)
        selector_type = "heuristic_phase"
    elif window_source == "phase_selector" and os.path.exists(args.window_proposals_csv):
        with open(args.window_proposals_csv, newline="") as f:
            props = list(csv.DictReader(f))
        for p in props:
            if p.get("task")==args.task and int(p.get("seed",-1))==args.seed:
                ws = int(p["window_start"]); we = int(p["window_end"])
                selector_type = p.get("selector_type","phase_selector")
                break

    if ws is None:
        ws, we = args.fixed_window_start, args.fixed_window_end
        selector_type = "fallback_fixed"
    return ws, we, selector_type


def main():
    args = parse_args()
    ws, we, selector_type = get_window_from_source(args)
    print(f"Window: [{ws},{we}] source={args.window_source} selector={selector_type}")

    if args.dry_run:
        print(f"DRY RUN: would run {args.condition} on {args.task} seed={args.seed} [{ws},{we}]")
        print(f"Command: {PYTHON} -u {VIS_ROLLOUT} --task {args.task} --condition {args.condition} ...")
        return

    cmd = [
        PYTHON, "-u", VIS_ROLLOUT,
        "--task", args.task,
        "--condition", args.condition,
        "--eps_raw_pixels", str(args.eps_raw_pixels),
        "--perturb_start", str(ws),
        "--perturb_end", str(we),
        "--objective", args.objective,
        "--seed", str(args.seed),
        "--gpu_pair", args.gpu_pair,
    ]
    if args.condition == "vis_pgd":
        cmd += ["--pgd_steps", str(args.pgd_steps), "--pgd_restarts", str(args.pgd_restarts)]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
