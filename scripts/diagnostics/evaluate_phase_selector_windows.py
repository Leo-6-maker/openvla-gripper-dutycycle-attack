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

# Shared gripper field fallback (must match audit_phase_conditioned_vis.py)
GRIP_FIELDS = ("adv_grip", "raw_gripper", "clean_grip", "clean_gripper_action", "adv_gripper_action", "action_gripper")
def get_raw_gripper(row):
    for k in GRIP_FIELDS:
        if k in row and row[k] not in ("", None):
            try: return float(row[k])
            except (ValueError, TypeError): continue
    return None


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
    """Trained selector: NOT IMPLEMENTED. Hard fail."""
    raise SystemExit(
        "trained_selector mode requires GPU inference from checkpoint. "
        "Not implemented. Use --mode oracle_phase instead."
    )


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

    PROPOSAL_FIELDS = [
        "task","seed","T_eg","window_start","window_end","window_policy",
        "selector_type","selector_confidence","online_feasible",
        "clean_natural_open_ratio","natural_release_confounded","phase_overlap_iou",
        "proposal_valid","invalid_reason",
    ]
    proposals = []
    for rid, rlist in rollouts.items():
        r0 = rlist[0]; task = r0.get("task","?"); seed = r0.get("seed","?")

        if args.mode == "oracle_phase":
            T_eg = _find_T_eg_from_labels(rlist)
            selector_type = "oracle_phase"
            online_feasible = False  # oracle uses privileged/offline labels
        else:
            _find_T_eg_from_selector(rlist, args.checkpoint, args.feature_schema, args.threshold, args.consecutive_k)
            selector_type = "trained_selector_placeholder_NOT_IMPLEMENTED"
            online_feasible = False

        if T_eg is None:
            proposals.append({
                "task":task,"seed":seed,"T_eg":"","window_start":"","window_end":"",
                "window_policy":args.window_policy,"selector_type":selector_type,
                "selector_confidence":0.0,"online_feasible":False,
                "clean_natural_open_ratio":"","natural_release_confounded":"",
                "phase_overlap_iou":"",
                "proposal_valid":False,
                "invalid_reason":"no_grasp_formation_label",
            })
            continue

        ws = T_eg
        we = min(T_eg + 17, 299) if args.window_policy == "T_to_Tplus17" else min(T_eg + 14, 299)
        if args.window_policy == "Tminus3_to_Tplus14":
            ws = max(0, T_eg - 3)

        # Clean natural OPEN — use get_raw_gripper helper (same as audit)
        wrows = [r for r in rlist if ws <= int(r.get("policy_step",-1)) <= we]
        n_w = len(wrows)
        clean_open = 0; clean_missing = 0
        for r in wrows:
            g = get_raw_gripper(r)
            if g is None: clean_missing += 1
            elif raw_gripper_is_open(g): clean_open += 1
        clean_ratio = clean_open / max(n_w, 1)
        if clean_missing > 0:
            proposals.append({
                "task":task,"seed":seed,"T_eg":T_eg,"window_start":ws,"window_end":we,
                "window_policy":args.window_policy,"selector_type":selector_type,
                "selector_confidence":0.0,"online_feasible":False,
                "clean_natural_open_ratio":"","natural_release_confounded":"",
                "phase_overlap_iou":"",
                "proposal_valid":False,
                "invalid_reason":f"missing_clean_gripper_for_confound_audit_{clean_missing}",
            })
            continue

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
            "proposal_valid":True,"invalid_reason":"",
        })

    if not proposals:
        print("ERROR: no proposals generated — check phase CSV or task/seed match.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=PROPOSAL_FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(proposals)
    print(f"Wrote {len(proposals)} proposals to {args.output_csv}")
    for p in proposals:
        valid = str(p.get("proposal_valid",""))
        reason = p.get("invalid_reason","")
        tag = f"[{reason}]" if reason else ""
        print(f"  {p['task']} seed{p['seed']}: T_eg={p.get('T_eg','?')} [{p.get('window_start','?')}-{p.get('window_end','?')}] valid={valid} {tag}")


if __name__ == "__main__":
    main()
