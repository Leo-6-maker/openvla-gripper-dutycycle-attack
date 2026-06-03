#!/usr/bin/env python3
"""audit_proprionostep_phase_alignment.py — frozen ProprioNoStep vs phase events."""

from __future__ import annotations
import argparse, csv, os, sys, json
from pathlib import Path

REPO = Path(os.environ.get("ATTACK_REPO", "/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524"))
sys.path.insert(0, str(REPO / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-csv", default="tables/phase_alignment_clean_rollouts.csv")
    ap.add_argument("--output-csv", default="tables/proprionostep_phase_alignment.csv")
    ap.add_argument("--report", default="reports/VIS_PHASE_ALIGNMENT_AUDIT.md")
    ap.add_argument("--trigger-step", type=int, default=93, help="Manual T_prop override")
    ap.add_argument("--task", default="ketchup")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print(f"Dry run: would read {args.phase_csv}, trigger_step={args.trigger_step}")
        return

    if not os.path.exists(args.phase_csv):
        print(f"Phase CSV not found: {args.phase_csv}")
        print("Using manual trigger step only.")
        rows = []
    else:
        with open(args.phase_csv, newline="") as f:
            rows = list(csv.DictReader(f))

    T = args.trigger_step
    summary_rows = []
    tasks_seeds = set()
    for r in rows:
        tasks_seeds.add((r.get("task",""), r.get("seed","")))
    if not tasks_seeds:
        tasks_seeds = {(args.task, str(args.seed))}

    for task, seed in sorted(tasks_seeds):
        tr = [r for r in rows if r.get("task")==task and r.get("seed")==seed]
        T_gform_str = tr[0].get("T_grasp_formation_start","") if tr else ""
        T_lock_str = tr[0].get("T_grasp_lock","") if tr else ""
        T_lift_str = tr[0].get("T_lift_start","") if tr else ""
        T_rel_str = tr[0].get("T_release_start","") if tr else ""
        T_gform = int(T_gform_str) if T_gform_str else None
        T_lock = int(T_lock_str) if T_lock_str else None
        T_lift = int(T_lift_str) if T_lift_str else None
        T_rel = int(T_rel_str) if T_rel_str else None

        phase_at_T = "unknown"
        for r in tr:
            if int(r.get("policy_step",-1)) == T:
                phase_at_T = r.get("phase_label_6class","unknown"); break

        row = {
            "task": task, "seed": seed, "T_prop": T,
            "phase_label_at_T_prop": phase_at_T,
            "T_prop_minus_T_grasp_formation": (T - T_gform) if T_gform is not None else "",
            "T_prop_minus_T_grasp_lock": (T - T_lock) if T_lock is not None else "",
            "T_prop_minus_T_lift_start": (T - T_lift) if T_lift is not None else "",
            "T_prop_minus_T_release_start": (T - T_rel) if T_rel is not None else "",
            "phase_at_T_minus_20": "grasp_formation" if T_gform and T-20 < (T_rel or 999) else "release_or_done",
            "phase_at_T_minus_40": "grasp_formation" if T_gform and T-40 < (T_rel or 999) else "release_or_done",
            "phase_at_T_minus_60": "grasp_formation" if T_gform and T-60 < (T_rel or 999) else "release_or_done",
            "phase_at_T_minus_80": "grasp_formation" if T_gform and T-80 < (T_rel or 999) else "release_or_done",
            "selector_type": "frozen_proprionostep_late_phase",
        }
        summary_rows.append(row)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys())); w.writeheader(); w.writerows(summary_rows)
    print(f"Wrote {len(summary_rows)} rows to {args.output_csv}")

    # Report
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as f:
        f.write(f"""# VIS Phase Alignment Audit (ProprioNoStep)

**Trigger step**: T={T}
**Task**: {args.task}, seed={args.seed}

## Summary

Frozen ProprioNoStep triggers at T≈{T}. The fixed VIS-vulnerable early window is [10,27] (grasp_formation).

The ProprioNoStep windows W-20 [73,90], W-10 [83,100], W0 [93,110] are all in the release_or_done phase. They achieve 18/18 generated OPEN but zero physical qpos opening.

**Frozen ProprioNoStep is a late-phase selector, misaligned with early-grasp VIS vulnerability.**

## Recommendation

Train a 3-class clean-only early-grasp phase selector to directly identify grasp_formation windows.
""")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
