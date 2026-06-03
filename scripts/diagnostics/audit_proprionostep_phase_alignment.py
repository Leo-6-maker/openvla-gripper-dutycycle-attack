#!/usr/bin/env python3
"""audit_proprionostep_phase_alignment.py — frozen ProprioNoStep trigger vs phase events.

Reads phase-labeled CSV and computes offsets from ProprioNoStep trigger to phase events.
Supports manual trigger step or detector-score CSV.
"""

from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path

REPO = Path(os.environ.get("ATTACK_REPO", "/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524"))
_src = str(REPO / "src")
if os.path.isdir(_src): sys.path.insert(0, _src)
try: from gripper_attack.gripper_semantics import raw_gripper_is_open
except ImportError: raw_gripper_is_open = lambda v: float(v) < 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-csv", default="tables/phase_alignment_clean_rollouts.csv")
    ap.add_argument("--detector-score-csv", default=None,
        help="NOT YET IMPLEMENTED — will raise SystemExit if provided")
    ap.add_argument("--output-csv", default="tables/proprionostep_phase_alignment.csv")
    ap.add_argument("--report", default="reports/VIS_PHASE_ALIGNMENT_AUDIT.md")
    ap.add_argument("--trigger-step", type=int, default=93)
    ap.add_argument("--task", default="ketchup")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print(f"DRY RUN: audit_proprionostep_phase_alignment trigger_step={args.trigger_step}")
        return

    if args.detector_score_csv:
        print("FATAL: --detector-score-csv is not yet implemented. Use --trigger-step for manual T_prop.")
        sys.exit(1)

    if not os.path.exists(args.phase_csv):
        print(f"Phase CSV not found: {args.phase_csv}")
        print("Using manual trigger only.")
        rows = []
    else:
        with open(args.phase_csv, newline="") as f:
            rows = list(csv.DictReader(f))

    T = args.trigger_step
    trigger_source = "manual"

    # Group by rollout
    from collections import defaultdict
    rollouts = defaultdict(list)
    for r in rows:
        rid = r.get("rollout_id", f"{r.get('task','')}_{r.get('seed','')}")
        rollouts[rid].append(r)

    alignment_rows = []
    for rid, rlist in rollouts.items():
        r0 = rlist[0]; task = r0.get("task",""); seed = r0.get("seed","")

        # Lookup actual phase at T and offsets
        step_map = {int(r["policy_step"]): r for r in rlist}
        phase_at = {}
        for offset in [0, -20, -40, -60, -80]:
            s = T + offset
            if s in step_map:
                r = step_map[s]
                phase_at[f"T_prop{offset:+d}"] = {
                    "step": s,
                    "phase_3class": r.get("phase_label_3class","unknown"),
                    "phase_6class": r.get("phase_label_6class","unknown"),
                }

        # Phase event offsets
        T_gform = r0.get("T_grasp_formation_start","")
        T_lock = r0.get("T_grasp_lock","")
        T_lift = r0.get("T_lift_start","")
        T_rel = r0.get("T_release_start","")

        # Clean natural OPEN in W0/W-10/W-20 using raw_gripper_is_open
        def clean_open_ratio(ws, we):
            wr = [r for r in rlist if ws <= int(r.get("policy_step",-1)) <= we]
            if not wr: return 0.0
            return sum(1 for r in wr if raw_gripper_is_open(float(r.get("raw_gripper",0.996)))) / len(wr)

        row = {
            "task":task,"seed":seed,"T_prop":T,"trigger_source":trigger_source,
            "T_grasp_formation_start":T_gform,"T_grasp_lock":T_lock,
            "T_lift_start":T_lift,"T_release_start":T_rel,
            "T_prop_minus_T_gform": (T - int(T_gform)) if T_gform else "",
            "T_prop_minus_T_lock": (T - int(T_lock)) if T_lock else "",
            "T_prop_minus_T_lift": (T - int(T_lift)) if T_lift else "",
            "T_prop_minus_T_rel": (T - int(T_rel)) if T_rel else "",
            "phase_at_T_prop": phase_at.get("T_prop+0",{}).get("phase_6class","unknown"),
            "phase_at_T_prop_minus_20": phase_at.get("T_prop-20",{}).get("phase_6class","unknown"),
            "phase_at_T_prop_minus_40": phase_at.get("T_prop-40",{}).get("phase_6class","unknown"),
            "phase_at_T_prop_minus_60": phase_at.get("T_prop-60",{}).get("phase_6class","unknown"),
            "phase_at_T_prop_minus_80": phase_at.get("T_prop-80",{}).get("phase_6class","unknown"),
            "clean_open_W0": round(clean_open_ratio(93,110),4),
            "clean_open_Wm10": round(clean_open_ratio(83,100),4),
            "clean_open_Wm20": round(clean_open_ratio(73,90),4),
        }
        alignment_rows.append(row)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(alignment_rows[0].keys())); w.writeheader(); w.writerows(alignment_rows)
    print(f"Wrote {len(alignment_rows)} rows to {args.output_csv}")

    # Report
    r0 = alignment_rows[0] if alignment_rows else {}
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report,"w") as f:
        f.write(f"""# Phase Alignment Audit (ProprioNoStep)

**T_prop**: {T} (source: {trigger_source})
**Task**: {args.task}

## Phase at T_prop

T_prop={T}: phase = {r0.get("phase_at_T_prop","?")}
T_prop-20={T-20}: phase = {r0.get("phase_at_T_prop_minus_20","?")}
T_prop-40={T-40}: phase = {r0.get("phase_at_T_prop_minus_40","?")}
T_prop-60={T-60}: phase = {r0.get("phase_at_T_prop_minus_60","?")}
T_prop-80={T-80}: phase = {r0.get("phase_at_T_prop_minus_80","?")}

## Offsets from Phase Events

T_prop - T_gform = {r0.get("T_prop_minus_T_gform","?")}
T_prop - T_lock = {r0.get("T_prop_minus_T_lock","?")}
T_prop - T_rel = {r0.get("T_prop_minus_T_rel","?")}

## Clean Natural OPEN in ProprioNoStep Windows

W0  [93,110]: {r0.get("clean_open_W0","?")}
W-10 [83,100]: {r0.get("clean_open_Wm10","?")}
W-20 [73,90]: {r0.get("clean_open_Wm20","?")}

## Interpretation

Frozen ProprioNoStep triggers at T≈{T}. The fixed VIS-vulnerable early window is [10,27] (grasp_formation).
The ProprioNoStep is **{r0.get("phase_at_T_prop","?")}** phase — a late-phase selector misaligned with early-grasp VIS vulnerability.
""")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
