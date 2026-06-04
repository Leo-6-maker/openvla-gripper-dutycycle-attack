#!/usr/bin/env python3
"""run_object_teacher_delay50_chain.py — CHAIN runner for Object teacher-oracle delay=-50 VIS smoke.

For each batch1 candidate:
  clean → random_linf → audit denominator → VIS only if denominator clean

Usage:
  python scripts/run_object_teacher_delay50_chain.py \
    --candidate-csv tables/object_teacher_delay50_vis_smoke_batch1.csv \
    --gpu-pairs "2,3;4,5;6,7" \
    --output-dir /data/liuyu/outputs/object_teacher_delay50_smoke_20260604

This is OFFLINE teacher-oracle smoke, NOT online detector-driven VIS.
"""

from __future__ import annotations
import argparse, csv, os, subprocess, sys, time
from pathlib import Path

REPO = Path(os.environ.get("ATTACK_REPO",
    "/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524"))
PYTHON = "/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python"
WRAPPER = str(REPO / "scripts/vis_phase_conditioned_attack.py")
AUDIT_SCRIPT = str(REPO / "scripts/diagnostics/audit_phase_conditioned_vis.py")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-csv", default="tables/object_teacher_delay50_vis_smoke_batch1.csv")
    ap.add_argument("--gpu-pairs", default="2,3;4,5;6,7")
    ap.add_argument("--output-dir", default="/data/liuyu/outputs/object_teacher_delay50_smoke_20260604")
    ap.add_argument("--eps-raw-pixels", type=int, default=6)
    ap.add_argument("--pgd-steps", type=int, default=40)
    ap.add_argument("--pgd-restarts", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-clean", action="store_true", help="Skip clean (use existing)")
    ap.add_argument("--skip-random", action="store_true", help="Skip random (use existing)")
    ap.add_argument("--skip-vis", action="store_true", help="Skip VIS (audit only)")
    return ap.parse_args()


def run_one(condition, task_key, state_id, ws, we, gpu_pair, out_dir, args):
    """Run a single condition. Returns (rc, trace_path)."""
    ep_id = f"{task_key}_s{state_id}"
    cmd = [PYTHON, "-u", WRAPPER,
           "--task", task_key, "--state-id", str(state_id),
           "--condition", condition,
           "--window-source", "fixed",
           "--fixed-window-start", str(ws), "--fixed-window-end", str(we),
           "--eps_raw_pixels", str(args.eps_raw_pixels),
           "--objective", "prefix_locked_gripper_open_margin",
           "--seed", "0", "--gpu_pair", gpu_pair,
           "--output-dir", out_dir,
           "--episode-id", ep_id]
    if condition == "vis_pgd":
        cmd += ["--pgd_steps", str(args.pgd_steps), "--pgd_restarts", str(args.pgd_restarts)]

    log_path = os.path.join(out_dir, f"chain_{ep_id}_{condition}.log")
    print(f"  [{condition}] {task_key} state={state_id} [{ws},{we}] GPU={gpu_pair}")
    print(f"    log: {log_path}")

    if args.dry_run:
        return 0, ""

    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, cwd=str(REPO), stdout=log_f, stderr=subprocess.STDOUT, timeout=7200)

    if result.returncode != 0:
        print(f"    FAILED rc={result.returncode}")
        return result.returncode, ""

    # Find trace CSV
    trace_dir = os.path.join(out_dir, "runs")
    trace_files = list(Path(trace_dir).rglob(f"*{ep_id}*trace.csv"))
    if not trace_files:
        print(f"    WARNING: no trace CSV found")
        return -1, ""
    return 0, str(trace_files[0])


def audit_denominator(clean_path, random_path, audit_dir):
    """Check clean/random denominators. Returns True if clean."""
    if not clean_path or not random_path:
        return False
    summary_path = os.path.join(audit_dir, "den_check_summary.csv")
    cmd = [PYTHON, "-u", AUDIT_SCRIPT,
           "--run-dirs", os.path.dirname(clean_path), os.path.dirname(random_path),
           "--output-csv", os.path.join(audit_dir, "den_check_prov.csv"),
           "--summary-csv", summary_path]
    result = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"    Audit FAILED: {result.stderr[:200]}")
        return False
    if not os.path.exists(summary_path):
        return False
    import csv as _csv
    with open(summary_path) as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        return False
    r = rows[0]
    den_clean = str(r.get("denominator_clean","")).lower() in ("true","1","yes")
    rand_open = int(float(r.get("random_OPEN_max", -1) or -1))
    rand_done = str(r.get("random_done_all_true","")).lower() in ("true","1","yes")
    clean_open = float(r.get("clean_OPEN_mean", 1) or 1)
    is_clean = den_clean and rand_open == 0 and rand_done and clean_open <= 0.1
    print(f"    Denominator: den_clean={den_clean} rand_open={rand_open} rand_done={rand_done} clean_open={clean_open} -> {'PASS' if is_clean else 'FAIL'}")
    return is_clean


def main():
    args = parse_args()

    if not os.path.exists(args.candidate_csv):
        print(f"ERROR: candidate CSV not found: {args.candidate_csv}")
        sys.exit(1)

    with open(args.candidate_csv, newline="") as f:
        candidates = list(csv.DictReader(f))
    print(f"Loaded {len(candidates)} candidates from {args.candidate_csv}")

    gpu_pairs = [p.strip() for p in args.gpu_pairs.split(";")]
    print(f"GPU pairs: {gpu_pairs}")
    print(f"Output: {args.output_dir}")

    if args.dry_run:
        print("\nDRY RUN — checking all candidates:")
        for c in candidates:
            task_key = c["task_key"]; state_id = c["state_id"]
            ws = c["window_start"]; we = c["window_end"]
            print(f"  {task_key:20s} state={state_id:3s} [{ws},{we}]")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for i, c in enumerate(candidates):
        task_key = c["task_key"]; state_id = c["state_id"]
        ws = int(c["window_start"]); we = int(c["window_end"])
        ep_id = f"{task_key}_s{state_id}"
        gpu = gpu_pairs[i % len(gpu_pairs)]
        ep_dir = os.path.join(args.output_dir, ep_id)
        os.makedirs(ep_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(candidates)}] {task_key} state={state_id} [{ws},{we}] GPU={gpu}")
        print(f"{'='*60}")

        clean_path = ""; random_path = ""; vis_path = ""

        # 1. Clean
        if not args.skip_clean:
            rc, clean_path = run_one("clean", task_key, state_id, ws, we, gpu, ep_dir, args)
            if rc != 0:
                results.append(dict(episode_id=ep_id, task_key=task_key, state_id=state_id,
                                    window=f"[{ws},{we}]", clean_rc=rc, random_rc="", vis_rc="",
                                    denominator_clean=False, status="clean_failed"))
                continue
            time.sleep(10)

        # 2. Random
        if not args.skip_random:
            rc, random_path = run_one("random_linf", task_key, state_id, ws, we, gpu, ep_dir, args)
            if rc != 0:
                results.append(dict(episode_id=ep_id, task_key=task_key, state_id=state_id,
                                    window=f"[{ws},{we}]", clean_rc=0, random_rc=rc, vis_rc="",
                                    denominator_clean=False, status="random_failed"))
                continue
            time.sleep(10)

        # 3. Audit denominator
        audit_dir = os.path.join(ep_dir, "audit")
        os.makedirs(audit_dir, exist_ok=True)
        den_ok = audit_denominator(clean_path, random_path, audit_dir)

        if not den_ok:
            results.append(dict(episode_id=ep_id, task_key=task_key, state_id=state_id,
                                window=f"[{ws},{we}]", clean_rc=0, random_rc=0, vis_rc="",
                                denominator_clean=False, status="denominator_polluted"))
            continue

        # 4. VIS
        if not args.skip_vis:
            rc, vis_path = run_one("vis_pgd", task_key, state_id, ws, we, gpu, ep_dir, args)
            if rc != 0:
                results.append(dict(episode_id=ep_id, task_key=task_key, state_id=state_id,
                                    window=f"[{ws},{we}]", clean_rc=0, random_rc=0, vis_rc=rc,
                                    denominator_clean=True, status="vis_failed"))
                continue
            time.sleep(10)

        results.append(dict(episode_id=ep_id, task_key=task_key, state_id=state_id,
                            window=f"[{ws},{we}]", clean_rc=0, random_rc=0, vis_rc=0,
                            denominator_clean=True, status="complete"))

    # Write results manifest
    manifest_path = os.path.join(args.output_dir, "chain_results.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["status"])
        w.writeheader(); w.writerows(results)
    print(f"\nResults: {manifest_path}")
    for r in results:
        print(f"  {r['status']:25s} {r['episode_id']:30s} {r['window']}")


if __name__ == "__main__":
    main()
