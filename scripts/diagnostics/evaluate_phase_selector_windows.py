#!/usr/bin/env python3
"""evaluate_phase_selector_windows.py — predicted phase → attack window proposals."""

from __future__ import annotations
import argparse, csv, os, sys, json
from pathlib import Path
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--feature-schema")
    ap.add_argument("--phase-csv", default="tables/phase_alignment_clean_rollouts.csv")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--consecutive-k", type=int, default=2)
    ap.add_argument("--window-policy", choices=["T_to_Tplus17","Tminus3_to_Tplus14"], default="T_to_Tplus17")
    ap.add_argument("--output-csv", default="tables/phase_selector_window_proposals.csv")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.dry_run:
        print("DRY RUN: evaluate_phase_selector_windows")
        print(f"  Window policy: {args.window_policy}")
        print(f"  Threshold: {args.threshold}, K: {args.consecutive_k}")
        return

    if not os.path.exists(args.phase_csv):
        print(f"Phase CSV not found: {args.phase_csv}")
        return

    with open(args.phase_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    # Group by rollout
    from collections import defaultdict
    rollouts = defaultdict(list)
    for r in rows:
        rid = r.get("rollout_id", r.get("task","?") + "_" + r.get("seed","?"))
        rollouts[rid].append(r)

    proposals = []
    for rid, rlist in rollouts.items():
        r0 = rlist[0]
        task = r0.get("task","?"); seed = r0.get("seed","?")
        T_gform_str = r0.get("T_grasp_formation_start","")

        # Use heuristic T_gform as ground truth for now
        if T_gform_str:
            T_eg = int(T_gform_str)
            label_source = "heuristic"
        else:
            T_eg = 10  # fallback
            label_source = "fallback"

        if args.window_policy == "T_to_Tplus17":
            ws, we = T_eg, min(T_eg + 17, 299)
        else:
            ws, we = max(0, T_eg - 3), min(T_eg + 14, 299)

        # Compute clean natural OPEN in window
        wrows = [r for r in rlist if ws <= int(r.get("policy_step",-1)) <= we]
        n_steps = len(wrows)
        clean_open = sum(1 for r in wrows if r.get("phase_label_3class","") == "post_grasp"
                         or r.get("phase_label_6class","") == "release_or_done")
        clean_ratio = clean_open / max(n_steps, 1)

        proposals.append({
            "task": task, "seed": seed, "T_eg": T_eg,
            "window_start": ws, "window_end": we,
            "window_policy": args.window_policy,
            "clean_natural_open_ratio": round(clean_ratio, 4),
            "natural_release_confounded": clean_ratio > 0.5,
            "online_feasible": True,  # heuristic labels are available before action
            "selector_confidence": 1.0,
            "label_source": label_source,
            "selector_type": "heuristic_phase",
        })

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(proposals[0].keys())); w.writeheader(); w.writerows(proposals)
    print(f"Wrote {len(proposals)} proposals to {args.output_csv}")


if __name__ == "__main__":
    main()
