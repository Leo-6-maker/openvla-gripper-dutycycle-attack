#!/usr/bin/env python3
"""vis_phase_conditioned_attack.py — VIS prefix_margin on phase-selected windows.

Wraps vis_rollout_adaptive_v3.py. After subprocess completes, patches trace CSV
with phase/window metadata columns.

CRITICAL: selector modes NEVER fall back to fixed window.
  - heuristic_phase with missing label → SystemExit (unless --allow-fallback-fixed-window)
  - phase_selector with invalid proposal → SystemExit
  - fallback_fixed ONLY when --window-source fixed or --allow-fallback-fixed-window
"""

from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(os.environ.get("ATTACK_REPO", "/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524"))
VIS_ROLLOUT = str(REPO / "scripts/vis_rollout_adaptive_v3.py")
PYTHON = os.environ.get("PYTHON_BIN", "python")
CANONICAL_OPEN_SEMANTICS_VERSION = "v1.0_decoded_action_lt_0.5_is_open_20260603"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", choices=["clean","random_linf","vis_pgd"], required=True)
    ap.add_argument("--window-source", choices=["fixed","heuristic_phase","proprionostep_offset","phase_selector"],
                    default="fixed")
    ap.add_argument("--phase", default="grasp_formation")
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
    ap.add_argument("--allow-fallback-fixed-window", action="store_true",
        help="Only when you explicitly accept fixed-window fallback")
    ap.add_argument("--allow-partial-labels", action="store_true",
        help="Allow label_validity=partial_missing_qpos (default: heuristic only)")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def get_window_from_source(args):
    """Determine attack window. Returns (ws, we, selector_type, selection_meta)."""
    ws, we = None, None; selector_type = "unknown"
    window_source = args.window_source
    selection_meta = {
        "phase_label_validity": "", "phase_window_selection_validity": "",
        "phase_window_selection_reason": "", "clean_natural_open_ratio": "",
        "natural_release_confounded": "",
    }

    if window_source == "fixed":
        ws, we = args.fixed_window_start, args.fixed_window_end
        selector_type = "fixed"
        selection_meta["phase_window_selection_validity"] = "fixed_manual"

    elif window_source == "proprionostep_offset":
        T = args.proprionostep_trigger_step
        ws = max(0, T + args.offset); we = min(299, ws + 17)
        selector_type = f"proprionostep_offset_{args.offset}"
        selection_meta["phase_window_selection_validity"] = "proprionostep_offset"

    elif window_source == "heuristic_phase":
        if not os.path.exists(args.phase_csv):
            raise SystemExit("phase-csv not found for heuristic_phase")
        with open(args.phase_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        tr = [r for r in rows if r.get("task")==args.task and int(r.get("seed",-1))==args.seed]
        if not tr:
            raise SystemExit(f"No rows for {args.task} seed {args.seed} in phase CSV")

        # Validate label_validity across ALL rows in rollout (not just r0)
        validities = sorted(set(r.get("label_validity", "unknown") for r in tr))
        _allowed = ["heuristic"]
        if args.allow_partial_labels:
            _allowed.append("partial_missing_qpos")
        _rejected = [v for v in validities if v not in _allowed]
        if _rejected or len(validities) > 1:
            # Mixed or invalid: reject
            raise SystemExit(
                f"Phase label_validity for {args.task} seed {args.seed}: "
                f"{validities}. Only single-value {_allowed} allowed. "
                f"Rejected: {_rejected}. Mixed: {len(validities)>1}."
            )
        _validity = validities[0]
        selection_meta["phase_label_validity"] = _validity
        selection_meta["phase_window_selection_validity"] = "ok"
        selection_meta["phase_window_selection_reason"] = "heuristic_phase_validated"

        gs = [int(r["policy_step"]) for r in tr if r.get("phase_label_3class")==args.phase]
        if not gs:
            if args.allow_fallback_fixed_window:
                ws, we = args.fixed_window_start, args.fixed_window_end
                selector_type = "heuristic_phase_fallback_fixed"
            else:
                raise SystemExit(
                    f"No '{args.phase}' label found for {args.task} seed {args.seed}. "
                    "Use --allow-fallback-fixed-window to fall back, or fix phase labels."
                )
        else:
            ws = min(gs); we = min(ws + 17, 299)
            selector_type = "heuristic_phase"

    elif window_source == "phase_selector":
        if not os.path.exists(args.window_proposals_csv):
            raise SystemExit("window-proposals-csv not found for phase_selector")
        with open(args.window_proposals_csv, newline="") as f:
            props = list(csv.DictReader(f))
        matched = [p for p in props
                   if p.get("task")==args.task and int(p.get("seed",-1))==args.seed]
        if not matched:
            raise SystemExit(f"No proposal found for {args.task} seed {args.seed}")
        p = matched[0]
        pv = str(p.get("proposal_valid","True")).strip().lower()
        if pv not in ("true","1","yes") and "proposal_valid" in p:
            raise SystemExit(
                f"Invalid proposal for {args.task} seed {args.seed}: "
                f"{p.get('invalid_reason','unknown')}"
            )
        ws_str = p.get("window_start",""); we_str = p.get("window_end","")
        if not ws_str or not we_str:
            if args.allow_fallback_fixed_window:
                ws, we = args.fixed_window_start, args.fixed_window_end
                selector_type = "phase_selector_fallback_fixed"
            else:
                raise SystemExit(
                    f"Empty window in proposal for {args.task} seed {args.seed}. "
                    "Use --allow-fallback-fixed-window to fall back."
                )
        else:
            ws = int(ws_str); we = int(we_str)
            selector_type = p.get("selector_type","phase_selector")

    if ws is None:
        raise SystemExit(f"Could not resolve window for source={window_source}. This is a bug.")
    return ws, we, selector_type, selection_meta


def patch_trace_with_metadata(trace_path, metadata):
    if not trace_path or not os.path.exists(trace_path): return
    with open(trace_path, newline="") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())
    for key in metadata:
        if key not in fieldnames: fieldnames.append(key)
    for r in rows:
        for k, v in metadata.items(): r[k] = v
    tmp = trace_path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, trace_path)


def main():
    args = parse_args()
    ws, we, selector_type, selection_meta = get_window_from_source(args)

    if args.dry_run:
        print(f"DRY RUN: {args.condition} {args.task} seed={args.seed} [{ws},{we}]")
        print(f"  window_source={args.window_source} selector={selector_type}")
        print(f"  selection_meta: {selection_meta}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir,
        f"phase_conditioned_attack_{args.task}_seed{args.seed}_{args.condition}_{ws}_{we}.log")

    cmd = [PYTHON, "-u", VIS_ROLLOUT,
        "--task", args.task, "--condition", args.condition,
        "--eps_raw_pixels", str(args.eps_raw_pixels),
        "--perturb_start", str(ws), "--perturb_end", str(we),
        "--objective", args.objective,
        "--seed", str(args.seed), "--gpu_pair", args.gpu_pair]
    if args.condition == "vis_pgd":
        cmd += ["--pgd_steps", str(args.pgd_steps), "--pgd_restarts", str(args.pgd_restarts)]

    print(f"Running: {' '.join(cmd)}")
    print(f"Log: {log_path}")
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, cwd=str(REPO), stdout=log_f, stderr=subprocess.STDOUT, timeout=7200)

    # Find trace path from log file
    trace_path = None
    with open(log_path) as log_f:
        for line in log_f:
            if "Saved:" in line and "_trace.csv" in line:
                trace_path = line.split("Saved:")[-1].strip(); break

    metadata = {
        "window_source": args.window_source, "phase": args.phase,
        "selector_type": selector_type, "selector_checkpoint": "",
        "detector_trigger_step": args.proprionostep_trigger_step,
        "phase_label_3class": "", "phase_label_6class": "",
        "phase_label_validity": selection_meta.get("phase_label_validity", ""),
        "phase_window_selection_validity": selection_meta.get("phase_window_selection_validity", "invalid"),
        "phase_window_selection_reason": selection_meta.get("phase_window_selection_reason", ""),
        "clean_natural_open_ratio": selection_meta.get("clean_natural_open_ratio", ""),
        "natural_release_confounded": selection_meta.get("natural_release_confounded", ""),
        "phase_conditioned_wrapper_version": CANONICAL_OPEN_SEMANTICS_VERSION,
    }
    if trace_path and os.path.exists(trace_path):
        patch_trace_with_metadata(trace_path, metadata)
        print(f"Patched trace: {trace_path}")
    elif result.returncode == 0:
        print("FATAL: subprocess rc=0 but no trace CSV found. Cannot patch metadata.")
        manifest = {"task":args.task,"seed":args.seed,"condition":args.condition,
            "window_source":args.window_source,"window_start":ws,"window_end":we,
            "selector_type":selector_type,"trace_path":None,"rc":result.returncode,
            "trace_patch_failed":True}
        with open(os.path.join(args.output_dir, "phase_conditioned_attack_manifest.json"), "a") as mf:
            json.dump(manifest, mf); mf.write("\n")
        sys.exit(2)

    # Write success manifest
    manifest = {"task":args.task,"seed":args.seed,"condition":args.condition,
        "window_source":args.window_source,"window_start":ws,"window_end":we,
        "selector_type":selector_type,"trace_path":trace_path,"rc":result.returncode}
    with open(os.path.join(args.output_dir, "phase_conditioned_attack_manifest.json"), "a") as mf:
        json.dump(manifest, mf); mf.write("\n")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
