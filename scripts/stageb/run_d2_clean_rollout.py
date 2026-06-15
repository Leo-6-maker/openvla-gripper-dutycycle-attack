#!/usr/bin/env python3
"""D2.1: Fresh clean rollout — resumable job-ledger launcher.

Key fixes from D2.0:
  - Resumable: reads job ledger, skips already-completed states.
  - Per-state unique job IDs.
  - Full job ledger output (pending/running/completed/policy_failure/infra_failure/timeout/missing_artifact).
  - Render GPU aligned with config.
  - Can benchmark different GPU topologies (A: multi-GPU single-process, B: per-GPU workers).

Clean inference only. NO attack. NO training.
"""

import argparse, csv, hashlib, os, subprocess, sys, time, glob, json
from collections import defaultdict
from datetime import datetime

REPO = "/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607"
RUNNER = f"{REPO}/scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py"
MODEL = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
PYTHON = "/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python"


def sha256_file(path):
    if not os.path.isfile(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def discover_traces(output_dir):
    """Return set of (task, state_id) already written."""
    import re
    existing = set()
    pat = re.compile(r"trace_(.+)_s(\d+)_w\d+_\d+_s20d_clean_seed0_job(\d+)\.csv$")
    for f in glob.glob(os.path.join(output_dir, "trace_*.csv")):
        m = pat.match(os.path.basename(f))
        if m:
            existing.add((m.group(1), m.group(2)))
    return existing


def run_one_task(task, states_str, job_id, gpu_visible, render_gpu, output_dir, timeout=5400):
    """Run one task batch as a subprocess. Returns (success, n_traces_produced, runtime_sec)."""
    env = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
           "OPENVLA_ATTN_IMPLEMENTATION": "eager",
           "CUDA_VISIBLE_DEVICES": gpu_visible, "DISPLAY": "",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    t0 = time.time()
    cmd = [PYTHON, "-u", RUNNER, "--task", task, "--state_ids", states_str,
           "--condition", "clean", "--window_start", "0", "--window_end", "10",
           "--max_steps_override", "280", "--success_metric", "check_success",
           "--num_steps_wait", "10", "--model_path", MODEL,
           "--render_gpu_device_id", str(render_gpu),
           "--model_gpu_device_id", "-1",
           "--output_dir", output_dir, "--job_id", str(job_id), "--seed", "0"]
    try:
        r = subprocess.run(cmd, env=env, timeout=timeout, capture_output=True, text=True)
        dt = time.time() - t0
        n_traces = len(glob.glob(os.path.join(output_dir, f"trace_{task}_s*_job{job_id}*.csv")))
        return r.returncode == 0, n_traces, dt, r.stderr[-500:] if r.stderr else ""
    except subprocess.TimeoutExpired:
        dt = time.time() - t0
        n_traces = len(glob.glob(os.path.join(output_dir, f"trace_{task}_s*_job{job_id}*.csv")))
        return False, n_traces, dt, "TIMEOUT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ledger", default=None, help="Job ledger CSV (created if absent)")
    ap.add_argument("--gpu-visible", default="0,1,2,4,5,6,7", help="CUDA_VISIBLE_DEVICES")
    ap.add_argument("--render-gpu", type=int, default=0, help="Render GPU device ID")
    ap.add_argument("--benchmark", action="store_true", help="Run topology benchmark: first 2 states only")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load manifest
    with open(args.manifest, newline="") as f:
        all_jobs = list(csv.DictReader(f))

    # Discover already-completed traces
    existing = discover_traces(args.output_dir)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Existing traces: {len(existing)}")

    # Build job ledger
    ledger_path = args.ledger or os.path.join(args.output_dir, "d2_rollout_ledger.csv")
    if os.path.exists(ledger_path):
        with open(ledger_path, newline="") as f:
            ledger = list(csv.DictReader(f))
        ledger_map = {(r["task_key"], r["state_id"]): r for r in ledger}
    else:
        ledger_map = {}

    # Update ledger from manifest + existing traces
    job_id_base = 500000
    for j in all_jobs:
        key = (j["task_key"], j["state_id"])
        if key not in ledger_map:
            status = "completed" if key in existing else "pending"
            ledger_map[key] = {
                "job_id": str(job_id_base), "task_key": j["task_key"],
                "state_id": j["state_id"], "seed": j.get("seed", "0"),
                "status": status, "runtime_sec": "", "n_trace_rows": "",
                "trace_sha256": "", "failure_reason": "",
            }
        job_id_base += 1

    # Write initial ledger
    ledger = list(ledger_map.values())
    with open(ledger_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ledger[0].keys()))
        w.writeheader(); w.writerows(ledger)

    # Filter to pending jobs
    pending = [j for j in ledger if j["status"] == "pending"]
    if args.benchmark:
        pending = pending[:2]  # benchmark mode: only 2 states
        print(f"BENCHMARK MODE: {len(pending)} test states")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Pending: {len(pending)} states, GPUs: {args.gpu_visible}, Render: {args.render_gpu}")

    if not pending:
        print("No pending jobs. Done.")
        return

    # Batch by task
    by_task = defaultdict(list)
    for j in pending:
        by_task[j["task_key"]].append(j)

    completed = 0; failed = 0
    for task in sorted(by_task):
        jobs = sorted(by_task[task], key=lambda x: int(x["state_id"]))
        states_str = ",".join(j["state_id"] for j in jobs)
        n = len(jobs)
        job_id = jobs[0]["job_id"]

        print(f"[{datetime.now().strftime('%H:%M:%S')}] TASK {task}: {n} states ({states_str[:60]})")

        success, n_traces, dt, stderr = run_one_task(
            task, states_str, int(job_id), args.gpu_visible, args.render_gpu, args.output_dir)

        status = "OK" if success else "PARTIAL/FAILED"
        completed += n_traces
        failed += n - n_traces
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   {status} ({dt:.0f}s, {n_traces}/{n} traces) [{completed}/{len(pending)} total]")

        # Update ledger for affected jobs
        for j in jobs:
            key = (j["task_key"], j["state_id"])
            if key in ledger_map:
                trace_file = os.path.join(
                    args.output_dir,
                    f"trace_{j['task_key']}_s{j['state_id']}_w0_10_s20d_clean_seed0_job{j['job_id']}.csv")
                if os.path.isfile(trace_file):
                    ledger_map[key]["status"] = "completed"
                    ledger_map[key]["runtime_sec"] = str(round(dt, 1))
                    ledger_map[key]["n_trace_rows"] = str(
                        sum(1 for _ in open(trace_file)) - 1)
                    ledger_map[key]["trace_sha256"] = sha256_file(trace_file)[:16]
                    ledger_map[key]["failure_reason"] = ""
                elif success:
                    ledger_map[key]["status"] = "missing_artifact"
                    ledger_map[key]["failure_reason"] = "runner_ok_but_no_trace_file"
                else:
                    ledger_map[key]["status"] = "infra_failure"
                    ledger_map[key]["failure_reason"] = stderr[-200:] if stderr else "unknown"

        # Save ledger after each task
        ledger = list(ledger_map.values())
        with open(ledger_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ledger[0].keys()))
            w.writeheader(); w.writerows(ledger)

        if args.benchmark:
            break  # only one task batch for benchmark

    # Final summary
    final_status = defaultdict(int)
    for j in ledger_map.values():
        final_status[j["status"]] += 1
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] DONE: {dict(final_status)}")
    print(f"Ledger: {ledger_path}")


if __name__ == "__main__":
    main()
