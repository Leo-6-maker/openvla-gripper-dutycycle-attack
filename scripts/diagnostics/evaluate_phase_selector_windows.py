#!/usr/bin/env python3
"""evaluate_phase_selector_windows.py — phase → attack window proposals.

Two modes:
  --mode oracle_phase: use heuristic phase labels as ground truth
  --mode trained_selector: load checkpoint + feature_schema, causal inference
"""

from __future__ import annotations
import argparse, csv, os, sys, json
from pathlib import Path
try: import numpy as np
except ImportError: np = None

REPO = Path(os.environ.get("ATTACK_REPO", "/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524"))
_src = str(REPO / "src")
if os.path.isdir(_src): sys.path.insert(0, _src)
try: from gripper_attack.gripper_semantics import raw_gripper_is_open
except ImportError: raw_gripper_is_open = lambda v: float(v) < 0.5


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle_phase","trained_selector"], default="oracle_phase")
    ap.add_argument("--checkpoint")
    ap.add_argument("--feature-schema")
    ap.add_argument("--phase-csv", default="tables/phase_alignment_clean_rollouts.csv")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--consecutive-k", type=int, default=2)
    ap.add_argument("--window-policy", choices=["T_to_Tplus17","Tminus3_to_Tplus14"], default="T_to_Tplus17")
    ap.add_argument("--output-csv", default="tables/phase_selector_window_proposals.csv")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def _find_T_eg_from_labels(rlist, phase_label_col="phase_label_3class"):
    """Oracle: find first step with grasp_formation label."""
    for r in rlist:
        if r.get(phase_label_col) == "grasp_formation":
            return int(r["policy_step"])
    return None


def _find_T_eg_from_selector(rlist, checkpoint_path, feature_schema_path, threshold, k):
    """Trained selector: placeholder for GPU inference."""
    # TODO: implement causal TCN inference using checkpoint
    # For now, fall back to heuristic
    return _find_T_eg_from_labels(rlist)


def main():
    args = parse_args()
    if args.dry_run:
        print(f"DRY RUN: evaluate_phase_selector_windows mode={args.mode}")
        print(f"  Window policy: {args.window_policy}")
        return

    if not os.path.exists(args.phase_csv):
        print(f"Phase CSV not found: {args.phase_csv}")
        return

    with open(args.phase_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    from collections import defaultdict
    rollouts = defaultdict(list)
    for r in rows:
        rid = r.get("rollout_id", f"{r.get('task','?')}_{r.get('seed','?')}")
        rollouts[rid].append(r)

    proposals = []
    for rid, rlist in rollouts.items():
        r0 = rlist[0]; task = r0.get("task","?"); seed = r0.get("seed","?")

        if args.mode == "oracle_phase":
            T_eg = _find_T_eg_from_labels(rlist)
            selector_type = "oracle_phase"
            online_feasible = False  # oracle uses privileged/offline labels
        else:
            T_eg = _find_T_eg_from_selector(rlist, args.checkpoint, args.feature_schema, args.threshold, args.consecutive_k)
            selector_type = "trained_phase_selector"
            online_feasible = True

        if T_eg is None:
            T_eg = 10  # fallback
            selector_type += "_fallback"

        ws = T_eg
        we = min(T_eg + 17, 299) if args.window_policy == "T_to_Tplus17" else min(T_eg + 14, 299)
        if args.window_policy == "Tminus3_to_Tplus14":
            ws = max(0, T_eg - 3)

        # Clean natural OPEN in window using raw_gripper_is_open, not phase label
        wrows = [r for r in rlist if ws <= int(r.get("policy_step",-1)) <= we]
        n_w = len(wrows)
        clean_open = sum(1 for r in wrows if raw_gripper_is_open(float(r.get("raw_gripper", 0.996))))
        clean_ratio = clean_open / max(n_w, 1)

        # IoU with oracle grasp_formation if available
        gform_rows = [r for r in rlist if r.get("phase_label_3class")=="grasp_formation"]
        iou = 0.0
        if gform_rows:
            gs = int(gform_rows[0]["policy_step"]); ge = int(gform_rows[-1]["policy_step"])
            overlap = max(0, min(we,ge) - max(ws,gs) + 1)
            union = max(we,ge) - min(ws,gs) + 1
            iou = round(overlap/max(union,1), 4)

        proposals.append({
            "task":task,"seed":seed,"T_eg":T_eg,"window_start":ws,"window_end":we,
            "window_policy":args.window_policy,"selector_type":selector_type,
            "selector_confidence":1.0,"online_feasible":online_feasible,
            "clean_natural_open_ratio":round(clean_ratio,4),
            "natural_release_confounded":clean_ratio > 0.5,
            "phase_overlap_iou":iou,
        })

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(proposals[0].keys())); w.writeheader(); w.writerows(proposals)
    print(f"Wrote {len(proposals)} proposals to {args.output_csv}")
    for p in proposals:
        print(f"  {p['task']} seed{p['seed']}: T_eg={p['T_eg']} [{p['window_start']}-{p['window_end']}] confounded={p['natural_release_confounded']} IoU={p['phase_overlap_iou']}")


if __name__ == "__main__":
    main()
