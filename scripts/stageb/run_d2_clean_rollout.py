#!/usr/bin/env python3
"""D2.1a: Fresh clean rollout — 4-group parallel scheduler with full job ledger.

Each worker group gets dedicated GPUs, runs its task batches sequentially.
Four groups run concurrently via ProcessPoolExecutor.

Ledger keyed by (task_key, state_id). Per-group batch_job_id tracked
separately. Full SHA256 recorded. Proper failure taxonomy.

Clean inference only. NO attack. NO training.
"""

import argparse, csv, hashlib, os, subprocess, sys, time, glob, json, traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPO = "/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607"
RUNNER = f"{REPO}/scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py"
MODEL = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
PYTHON = "/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python"

# GPU group assignments: (group_id, gpu_visible, render_gpu_device_id)
# render_gpu_device_id=0 uses the FIRST GPU in the visible set (local index)
GPU_GROUPS = [
    ("A", "0,1", 0),
    ("B", "2,3", 0),
    ("C", "4,5", 0),
    ("D", "6,7", 0),
]

# Task allocation (balanced by state count, 92 remaining after 6 done)
TASK_ALLOCATION = {
    "A": [("tomato_sauce", 15), ("butter", 8)],
    "B": [("salad_dressing", 13), ("chocolate_pudding", 9)],
    "C": [("milk", 9), ("orange_juice", 9), ("bbq_sauce", 7)],
    "D": [("cream_cheese", 8), ("ketchup", 8), ("alphabet_soup", 6)],
}

LEDGER_FIELDS = [
    "task_key", "state_id", "seed", "worker_id", "gpu_group",
    "batch_job_id", "status", "start_time", "end_time", "runtime_sec",
    "trace_path", "full_trace_sha256", "row_count",
    "failure_class", "failure_detail", "attempt",
]


def sha256_file_full(path):
    if not os.path.isfile(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def discover_existing(output_dir):
    """Return {(task, state_id)} of traces already on disk, with their SHAs."""
    import re
    existing = {}
    pat = re.compile(r"trace_(.+)_s(\d+)_w\d+_\d+_s20d_clean_seed(\d+)_job(\d+)\.csv$")
    for f in glob.glob(os.path.join(output_dir, "trace_*.csv")):
        m = pat.match(os.path.basename(f))
        if m:
            key = (m.group(1), m.group(2))
            existing[key] = {
                "trace_path": f, "seed": m.group(3),
                "job_id": m.group(4),
                "sha256": sha256_file_full(f),
                "row_count": sum(1 for _ in open(f)) - 1,
            }
    return existing


def run_worker_group(worker_id, gpu_visible, render_gpu, tasks, manifest_rows, output_dir, ledger_path):
    """Single worker process: runs multiple task batches sequentially."""
    worker_log = []
    worker_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Worker {worker_id} start, GPUs={gpu_visible}")

    # Discover existing traces
    existing = discover_existing(output_dir)

    # Filter manifest to only states not yet done and in this worker's tasks
    worker_task_set = set(t for t, _ in tasks)
    pending = []
    for row in manifest_rows:
        key = (row["task_key"], row["state_id"])
        if key not in existing and row["task_key"] in worker_task_set:
            pending.append(row)

    worker_log.append(f"Pending states: {len(pending)}")

    # Batch by task
    by_task = defaultdict(list)
    for j in pending:
        by_task[j["task_key"]].append(j)

    completed = 0; total = len(pending)
    job_id_base = {"A": 510000, "B": 520000, "C": 530000, "D": 540000}[worker_id]
    updated_ledger = {}

    env = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
           "OPENVLA_ATTN_IMPLEMENTATION": "eager",
           "CUDA_VISIBLE_DEVICES": gpu_visible, "DISPLAY": "",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    # Sort tasks by allocation order
    task_order = [t for t, _ in tasks]
    for task in task_order:
        if task not in by_task:
            continue
        jobs = sorted(by_task[task], key=lambda x: int(x["state_id"]))
        n = len(jobs)
        states_str = ",".join(j["state_id"] for j in jobs)
        batch_job_id = job_id_base
        job_id_base += 100

        t0 = time.time()
        worker_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {worker_id} TASK {task}: {n} states [{states_str[:60]}]")

        # Mark running
        for j in jobs:
            key = (j["task_key"], j["state_id"])
            updated_ledger[key] = {
                "task_key": j["task_key"], "state_id": j["state_id"],
                "seed": j.get("seed", "0"), "worker_id": worker_id,
                "gpu_group": worker_id, "batch_job_id": str(batch_job_id),
                "status": "running", "start_time": datetime.now().isoformat(),
                "end_time": "", "runtime_sec": "", "trace_path": "",
                "full_trace_sha256": "", "row_count": "",
                "failure_class": "", "failure_detail": "", "attempt": "1",
            }

        cmd = [PYTHON, "-u", RUNNER, "--task", task, "--state_ids", states_str,
               "--condition", "clean", "--window_start", "0", "--window_end", "10",
               "--max_steps_override", "280", "--success_metric", "check_success",
               "--num_steps_wait", "10", "--model_path", MODEL,
               "--render_gpu_device_id", str(render_gpu),
               "--model_gpu_device_id", "-1",
               "--output_dir", output_dir, "--job_id", str(batch_job_id), "--seed", "0"]

        try:
            r = subprocess.run(cmd, env=env, timeout=5400, capture_output=True, text=True)
            dt = time.time() - t0
            rc = r.returncode
            stderr_tail = r.stderr[-300:] if r.stderr else ""
        except subprocess.TimeoutExpired:
            dt = time.time() - t0
            rc = -1
            stderr_tail = "TIMEOUT_5400s"

        # Update ledger for each state in batch
        for j in jobs:
            key = (j["task_key"], j["state_id"])
            # Search for trace file by task+state (don't rely on batch_job_id)
            trace_file = os.path.join(
                output_dir,
                f"trace_{j['task_key']}_s{j['state_id']}_w0_10_s20d_clean_seed0_job{batch_job_id}.csv")
            if os.path.isfile(trace_file):
                sha = sha256_file_full(trace_file)
                rows = sum(1 for _ in open(trace_file)) - 1
                updated_ledger[key].update({
                    "status": "completed", "end_time": datetime.now().isoformat(),
                    "runtime_sec": str(round(dt, 1)), "trace_path": trace_file,
                    "full_trace_sha256": sha, "row_count": str(rows),
                    "failure_class": "", "failure_detail": "",
                })
                completed += 1
            elif rc == -1:
                updated_ledger[key].update({
                    "status": "timeout", "end_time": datetime.now().isoformat(),
                    "runtime_sec": str(round(dt, 1)),
                    "failure_class": "timeout", "failure_detail": "5400s",
                })
            elif rc != 0:
                # Check for OOM
                if "OutOfMemory" in stderr_tail or "out of memory" in stderr_tail.lower():
                    updated_ledger[key].update({
                        "status": "oom", "end_time": datetime.now().isoformat(),
                        "runtime_sec": str(round(dt, 1)),
                        "failure_class": "oom", "failure_detail": stderr_tail[:200],
                    })
                else:
                    updated_ledger[key].update({
                        "status": "infra_failure", "end_time": datetime.now().isoformat(),
                        "runtime_sec": str(round(dt, 1)),
                        "failure_class": "process_nonzero", "failure_detail": stderr_tail[:200],
                    })
            else:
                updated_ledger[key].update({
                    "status": "missing_artifact", "end_time": datetime.now().isoformat(),
                    "runtime_sec": str(round(dt, 1)),
                    "failure_class": "runner_success_missing_artifact",
                    "failure_detail": "runner exited 0 but no trace file found",
                })

        worker_log.append(f"[{datetime.now().strftime('%H:%M:%S')}]   {completed}/{total} ok ({dt:.0f}s)")

    worker_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Worker {worker_id} DONE: {completed}/{total}")
    return worker_id, updated_ledger, worker_log


def load_ledger(ledger_path):
    if not os.path.exists(ledger_path):
        return {}
    with open(ledger_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {(r["task_key"], r["state_id"]): r for r in rows}


def save_ledger(ledger_map, ledger_path):
    with open(ledger_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in ledger_map.values():
            # Ensure all fields present
            row = {k: r.get(k, "") for k in LEDGER_FIELDS}
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--benchmark", action="store_true",
                    help="Run only 1 state per worker group (4 total)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ledger_path = args.ledger or os.path.join(args.output_dir, "d2_rollout_ledger.csv")

    with open(args.manifest, newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    # Load existing ledger
    ledger_map = load_ledger(ledger_path)
    # Discover existing files not yet in ledger
    existing = discover_existing(args.output_dir)
    for key, info in existing.items():
        if key not in ledger_map or ledger_map[key].get("status") != "completed":
            ledger_map[key] = {
                "task_key": key[0], "state_id": key[1], "seed": info.get("seed", "0"),
                "worker_id": "", "gpu_group": "", "batch_job_id": info.get("job_id", ""),
                "status": "completed", "start_time": "", "end_time": "",
                "runtime_sec": "", "trace_path": info["trace_path"],
                "full_trace_sha256": info["sha256"], "row_count": str(info["row_count"]),
                "failure_class": "", "failure_detail": "", "attempt": "0",
            }
    save_ledger(ledger_map, ledger_path)

    # Count completed
    n_completed = sum(1 for r in ledger_map.values() if r.get("status") == "completed")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed: {n_completed}, Pending remaining")

    # Select active groups
    groups_to_use = GPU_GROUPS
    if args.benchmark:
        print("BENCHMARK MODE: 1 state per group (4 total)")
        # Use only first state from each group's first task
        # Filter manifest to benchmark: one state per group's first task
        bench_tasks = {grp_id: TASK_ALLOCATION[grp_id][0][0] for grp_id, _, _ in groups_to_use}
        bench_rows = []
        seen_tasks = set()
        for row in manifest_rows:
            task = row["task_key"]
            for grp_id in bench_tasks:
                if bench_tasks[grp_id] == task:
                    key = (task, row["state_id"])
                    if key not in existing and task not in seen_tasks:
                        bench_rows.append(row)
                        seen_tasks.add(task)
                        break
        manifest_rows = bench_rows[:len(groups_to_use)]

    # Launch parallel workers
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Launching {len(groups_to_use)} worker groups...")
    futures = {}
    with ProcessPoolExecutor(max_workers=len(groups_to_use)) as executor:
        for grp_id, gpu_visible, render_gpu in groups_to_use:
            tasks = TASK_ALLOCATION[grp_id]
            f = executor.submit(
                run_worker_group, grp_id, gpu_visible, render_gpu,
                tasks, manifest_rows, args.output_dir, ledger_path)
            futures[f] = grp_id
            print(f"  Group {grp_id}: GPUs {gpu_visible}, {sum(n for _, n in tasks)} states, tasks: {[t for t, _ in tasks]}")

        # Collect results as they complete
        for f in as_completed(futures):
            grp_id = futures[f]
            try:
                worker_id, new_ledger, worker_log = f.result()
                # Merge ledger
                for key, row in new_ledger.items():
                    ledger_map[key] = row
                save_ledger(ledger_map, ledger_path)
                # Print worker log
                for line in worker_log:
                    print(f"  {line}")
            except Exception as e:
                print(f"  Group {grp_id} FAILED: {e}")
                traceback.print_exc()

    # Final summary
    final_status = defaultdict(int)
    for r in ledger_map.values():
        final_status[r.get("status", "unknown")] += 1
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] FINAL: {dict(final_status)}")
    save_ledger(ledger_map, ledger_path)
    print(f"Ledger: {ledger_path}")


if __name__ == "__main__":
    main()
