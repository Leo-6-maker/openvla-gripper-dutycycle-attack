#!/usr/bin/env python3
"""vis_phase_conditioned_attack.py — VIS prefix_margin on phase-selected windows.

Wraps vis_rollout_adaptive_v3.py. After subprocess completes, patches trace CSV
with phase/window metadata columns.
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
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def get_window_from_source(args):
    ws, we = None, None; selector_type = "unknown"
    window_source = args.window_source

    if window_source == "fixed":
        ws, we = args.fixed_window_start, args.fixed_window_end
        selector_type = "fixed"
    elif window_source == "proprionostep_offset":
        T = args.proprionostep_trigger_step
        ws = max(0, T + args.offset); we = min(299, ws + 17)
        selector_type = f"proprionostep_offset_{args.offset}"
    elif window_source == "heuristic_phase" and os.path.exists(args.phase_csv):
        with open(args.phase_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        tr = [r for r in rows if r.get("task")==args.task and int(r.get("seed",-1))==args.seed]
        if args.phase == "grasp_formation":
            gs = [int(r["policy_step"]) for r in tr if r.get("phase_label_3class")=="grasp_formation"]
            ws = min(gs) if gs else 10; we = min(ws+17, 299)
        selector_type = "heuristic_phase"
    elif window_source == "phase_selector" and os.path.exists(args.window_proposals_csv):
        with open(args.window_proposals_csv, newline="") as f:
            props = list(csv.DictReader(f))
        for p in props:
            if p.get("task")==args.task and int(p.get("seed",-1))==args.seed:
                ws = int(p["window_start"]); we = int(p["window_end"])
                selector_type = p.get("selector_type","phase_selector"); break

    if ws is None:
        ws, we = args.fixed_window_start, args.fixed_window_end
        selector_type = "fallback_fixed"
    return ws, we, selector_type


def patch_trace_with_metadata(trace_path, metadata):
    """Add phase/window metadata columns to existing trace CSV."""
    if not trace_path or not os.path.exists(trace_path):
        return
    with open(trace_path, newline="") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())
    for key in metadata:
        if key not in fieldnames:
            fieldnames.append(key)
    for r in rows:
        for k, v in metadata.items():
            r[k] = v
    # Write back
    tmp = trace_path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, trace_path)


def main():
    args = parse_args()
    ws, we, selector_type = get_window_from_source(args)

    if args.dry_run:
        print(f"DRY RUN: {args.condition} {args.task} seed={args.seed} [{ws},{we}]")
        print(f"  window_source={args.window_source} selector={selector_type}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    cmd = [PYTHON, "-u", VIS_ROLLOUT,
        "--task", args.task, "--condition", args.condition,
        "--eps_raw_pixels", str(args.eps_raw_pixels),
        "--perturb_start", str(ws), "--perturb_end", str(we),
        "--objective", args.objective,
        "--seed", str(args.seed), "--gpu_pair", args.gpu_pair]
    if args.condition == "vis_pgd":
        cmd += ["--pgd_steps", str(args.pgd_steps), "--pgd_restarts", str(args.pgd_restarts)]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=7200)

    # Find newest trace from output
    trace_path = None
    for line in (result.stdout + result.stderr).split("\n"):
        if "Saved:" in line and "_trace.csv" in line:
            trace_path = line.split("Saved:")[-1].strip(); break

    # Patch trace with phase metadata
    metadata = {
        "window_source": args.window_source, "phase": args.phase,
        "selector_type": selector_type, "selector_checkpoint": "",
        "detector_trigger_step": args.proprionostep_trigger_step,
        "phase_label_3class": "", "phase_label_6class": "",
        "clean_natural_open_ratio": "", "natural_release_confounded": "",
        "phase_conditioned_wrapper_version": CANONICAL_OPEN_SEMANTICS_VERSION,
    }
    if trace_path:
        patch_trace_with_metadata(trace_path, metadata)
        print(f"Patched trace: {trace_path}")

    # Write manifest
    manifest = {"task":args.task,"seed":args.seed,"condition":args.condition,
        "window_source":args.window_source,"window_start":ws,"window_end":we,
        "selector_type":selector_type,"trace_path":trace_path,"rc":result.returncode}
    manifest_path = os.path.join(args.output_dir, "phase_conditioned_attack_manifest.json")
    with open(manifest_path, "a") as f:
        json.dump(manifest, f); f.write("\n")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
