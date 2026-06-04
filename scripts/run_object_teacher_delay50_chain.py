#!/usr/bin/env python3
"""run_object_teacher_delay50_chain.py v2 — CHAIN runner with trace localization and audit isolation.

v2 fixes:
  - Finds localized trace from <ep_dir>/traces/<task>_s<state>_<cond>_w<ws>_<we>_trace.csv
  - Audit uses only <ep_dir>/traces (never global runs).
  - Verifies trace metadata after each run (task, state_id, condition, window).
  - Hard-fail if localization fails or metadata mismatch.
"""

from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, time
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
    ap.add_argument("--skip-clean", action="store_true")
    ap.add_argument("--skip-random", action="store_true")
    ap.add_argument("--skip-vis", action="store_true")
    return ap.parse_args()


def find_localized_trace(ep_dir, task_key, state_id, condition, ws, we):
    """Find the localized trace copied by the wrapper."""
    traces_dir = os.path.join(ep_dir, "traces")
    if not os.path.isdir(traces_dir):
        return ""
    # Match pattern: <...>_<condition>_w<ws>_<we>_trace.csv
    pattern = f"_{condition}_w{ws}_{we}_trace.csv"
    matches = []
    for f in os.listdir(traces_dir):
        if pattern in f and f.endswith("_trace.csv"):
            matches.append(os.path.join(traces_dir, f))
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        print(f"    ERROR: no localized trace found in {traces_dir} for {condition} [{ws},{we}]")
        return ""
    print(f"    ERROR: {len(matches)} ambiguous traces found: {matches}")
    return ""


def verify_trace_metadata(trace_path, task_key, state_id, condition, ws, we, eps):
    """Verify first row of trace matches expected metadata."""
    if not trace_path or not os.path.exists(trace_path):
        return False, f"trace_missing_{trace_path}"
    with open(trace_path, newline="") as f:
        r0 = next(csv.DictReader(f), None)
    if r0 is None:
        return False, "trace_empty"
    checks = [
        ("task", r0.get("task",""), task_key),
        ("state_id", str(r0.get("state_id","")), str(state_id)),
        ("condition", r0.get("condition",""), condition),
        ("window_start", str(r0.get("window_start","")), str(ws)),
        ("window_end", str(r0.get("window_end","")), str(we)),
    ]
    failures = [f"{k}={v} != {exp}" for k, v, exp in checks if v != exp]
    if failures:
        return False, "trace_metadata_mismatch: " + ", ".join(failures)
    return True, "ok"


def run_one(condition, task_key, state_id, ws, we, gpu_pair, out_dir, args):
    """Run a single condition. Returns (rc, localized_trace_path, error_reason)."""
    ep_id = f"{task_key}_s{state_id}"
    ep_dir = os.path.join(out_dir, ep_id)
    os.makedirs(ep_dir, exist_ok=True)

    cmd = [PYTHON, "-u", WRAPPER,
           "--task", task_key, "--state-id", str(state_id),
           "--condition", condition,
           "--window-source", "fixed",
           "--fixed-window-start", str(ws), "--fixed-window-end", str(we),
           "--eps_raw_pixels", str(args.eps_raw_pixels),
           "--objective", "prefix_locked_gripper_open_margin",
           "--seed", "0", "--gpu_pair", gpu_pair,
           "--output-dir", ep_dir, "--episode-id", ep_id]
    if condition == "vis_pgd":
        cmd += ["--pgd_steps", str(args.pgd_steps), "--pgd_restarts", str(args.pgd_restarts)]

    log_path = os.path.join(ep_dir, f"chain_{condition}.log")
    print(f"  [{condition}] {task_key} state={state_id} [{ws},{we}] GPU={gpu_pair}")

    if args.dry_run:
        return 0, f"{ep_dir}/traces/{ep_id}_{condition}_w{ws}_{we}_trace.csv", "ok"

    env = os.environ.copy(); env["PYTHON_BIN"] = PYTHON
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, cwd=str(REPO), stdout=log_f, stderr=subprocess.STDOUT,
                                timeout=7200, env=env)

    localized = find_localized_trace(ep_dir, task_key, state_id, condition, ws, we)
    if not localized:
        return -1, "", f"localization_failed_no_trace_in_{ep_dir}/traces"

    ok, reason = verify_trace_metadata(localized, task_key, state_id, condition, ws, we, args.eps_raw_pixels)
    if not ok:
        return -2, localized, reason

    print(f"    trace={os.path.basename(localized)} verified={ok}")
    return result.returncode, localized, "ok"


def audit_denominator(clean_path, random_path, ep_dir):
    """Audit denominator using only localized traces. Returns (passed, reason)."""
    traces_dir = os.path.join(ep_dir, "traces")
    if not os.path.isdir(traces_dir):
        return False, f"traces_dir_missing_{traces_dir}"
    summary_path = os.path.join(ep_dir, "audit", "den_check_summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    cmd = [PYTHON, "-u", AUDIT_SCRIPT,
           "--run-dirs", traces_dir,
           "--output-csv", os.path.join(ep_dir, "audit", "den_check_prov.csv"),
           "--summary-csv", summary_path]
    env = os.environ.copy(); env["PYTHON_BIN"] = PYTHON
    result = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=300, env=env)
    if result.returncode != 0:
        return False, f"audit_script_failed_rc={result.returncode}"
    if not os.path.exists(summary_path):
        return False, "no_summary_csv"

    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        return False, f"ambiguous_audit_groups_{len(rows)}"
    r = rows[0]
    den_clean = str(r.get("denominator_clean","")).lower() in ("true","1","yes")
    rand_open = int(float(r.get("random_OPEN_max", "-1") or -1))
    rand_done = str(r.get("random_done_all_true","")).lower() in ("true","1","yes")
    clean_open = float(r.get("clean_OPEN_mean", "1") or 1)
    ok = den_clean and rand_open == 0 and rand_done and clean_open <= 0.1
    reason = f"den_clean={den_clean} rand_open={rand_open} rand_done={rand_done} clean_open={clean_open}"
    if len(rows) != 1:
        reason += f" groups={len(rows)}"
    return ok, reason


def main():
    args = parse_args()
    if not os.path.exists(args.candidate_csv):
        print(f"ERROR: candidate CSV not found: {args.candidate_csv}"); sys.exit(1)
    with open(args.candidate_csv, newline="") as f:
        candidates = list(csv.DictReader(f))
    print(f"Loaded {len(candidates)} candidates")
    gpu_pairs = [p.strip() for p in args.gpu_pairs.split(";")]
    print(f"GPU pairs: {gpu_pairs}")

    if args.dry_run:
        for c in candidates:
            print(f"  {c['task_key']:20s} state={c['state_id']:3s} [{c['window_start']},{c['window_end']}]")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for i, c in enumerate(candidates):
        task_key = c["task_key"]; state_id = int(c["state_id"])
        ws = int(c["window_start"]); we = int(c["window_end"])
        ep_id = f"{task_key}_s{state_id}"
        gpu = gpu_pairs[i % len(gpu_pairs)]
        ep_dir = os.path.join(args.output_dir, ep_id)

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(candidates)}] {task_key} state={state_id} [{ws},{we}] GPU={gpu}")
        print(f"{'='*60}")

        clean_path = ""; random_path = ""; vis_path = ""

        # 1. Clean
        if not args.skip_clean:
            rc, clean_path, reason = run_one("clean", task_key, state_id, ws, we, gpu, args.output_dir, args)
            if rc != 0:
                results.append(dict(episode_id=ep_id, task_key=task_key, state_id=state_id,
                    window=f"[{ws},{we}]", clean_rc=rc, random_rc="", vis_rc="",
                    denominator_clean=False, status=f"clean_failed_{reason}"))
                continue
            time.sleep(10)

        # 2. Random
        if not args.skip_random:
            rc, random_path, reason = run_one("random_linf", task_key, state_id, ws, we, gpu, args.output_dir, args)
            if rc != 0:
                results.append(dict(episode_id=ep_id, task_key=task_key, state_id=state_id,
                    window=f"[{ws},{we}]", clean_rc=0, random_rc=rc, vis_rc="",
                    denominator_clean=False, status=f"random_failed_{reason}"))
                continue
            time.sleep(10)

        # 3. Audit denominator (isolated to <ep_dir>/traces)
        passed, den_reason = audit_denominator(clean_path, random_path, ep_dir)
        print(f"  Denominator: {'PASS' if passed else 'FAIL'} ({den_reason})")

        if not passed:
            results.append(dict(episode_id=ep_id, task_key=task_key, state_id=state_id,
                window=f"[{ws},{we}]", clean_rc=0, random_rc=0, vis_rc="",
                denominator_clean=False, status=f"denominator_polluted_{den_reason}"))
            continue

        # 4. VIS
        if args.skip_vis:
            results.append(dict(episode_id=ep_id, task_key=task_key, state_id=state_id,
                window=f"[{ws},{we}]", clean_rc=0, random_rc=0, vis_rc="",
                denominator_clean=True, status="denominator_clean_vis_skipped"))
            continue

        rc, vis_path, reason = run_one("vis_pgd", task_key, state_id, ws, we, gpu, args.output_dir, args)
        status = "complete" if rc == 0 else f"vis_failed_{reason}"
        results.append(dict(episode_id=ep_id, task_key=task_key, state_id=state_id,
            window=f"[{ws},{we}]", clean_rc=0, random_rc=0, vis_rc=rc,
            denominator_clean=True, status=status))
        time.sleep(10)

    # Write manifest
    manifest_path = os.path.join(args.output_dir, "chain_results.csv")
    fields = ["episode_id","task_key","state_id","window","clean_rc","random_rc","vis_rc",
              "denominator_clean","status"]
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(results)
    print(f"\nResults: {manifest_path}")
    for r in results:
        print(f"  {r['status']:40s} {r['episode_id']:30s} {r['window']}")


if __name__ == "__main__":
    main()
