#!/usr/bin/env python3
"""D2.1a: Fresh clean rollout — 4-group parallel master + standalone workers.

Master launches 4 worker subprocesses concurrently via subprocess.Popen.
Each worker is an independent Python process with dedicated GPU pair.
Workers write progress to group-specific logs. Master monitors completion
and merges final ledger.

Clean inference only. NO attack. NO training.
"""

import argparse, csv, glob, os, subprocess, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

GPU_GROUPS = [
    ("A", "0,1", 0),   # render_gpu = first physical GPU in group
    ("B", "2,3", 2),
    ("C", "4,5", 4),
    ("D", "6,7", 6),
]

TASK_ALLOCATION = {
    "A": "tomato_sauce,butter",
    "B": "salad_dressing,chocolate_pudding",
    "C": "milk,orange_juice,bbq_sauce",
    "D": "cream_cheese,ketchup,alphabet_soup",
}

PYTHON = "/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python"
WORKER_SCRIPT = "/data/liuyu/l12_e4c2_pipeline/scripts/stageb/run_d2_clean_rollout.py"


def launch_worker(worker_id, gpu_visible, render_gpu, tasks, manifest, output_dir, ledger_path, benchmark):
    """Launch one worker subprocess. Returns Popen object."""
    cmd = [
        PYTHON, WORKER_SCRIPT, "--worker",
        "--worker-id", worker_id,
        "--gpu-visible", gpu_visible,
        "--render-gpu", str(render_gpu),
        "--tasks", tasks,
        "--manifest", manifest,
        "--output-dir", output_dir,
        "--ledger", ledger_path,
    ]
    if benchmark:
        cmd.append("--benchmark-single")
    log_file = os.path.join(output_dir, f"worker_{worker_id}.log")
    with open(log_file, "w") as f:
        f.write(f"Worker {worker_id} starting at {datetime.now().isoformat()}\n")
        f.write(f"GPUs: {gpu_visible}  Tasks: {tasks}\n")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_visible}
    return subprocess.Popen(cmd, env=env, stdout=open(log_file, "a"), stderr=subprocess.STDOUT)


def worker_main():
    """Entry point when launched as --worker subprocess."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-id", default="")
    ap.add_argument("--gpu-visible", default="0,1")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--ledger", default="")
    ap.add_argument("--benchmark-single", action="store_true")
    args = ap.parse_args()

    import hashlib, glob, re, json
    from datetime import datetime

    REPO = "/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607"
    RUNNER = f"{REPO}/scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py"
    MODEL = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"

    def sha256_full(path):
        if not os.path.isfile(path): return ""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
        return h.hexdigest()

    # Discover existing
    existing = {}
    pat = re.compile(r"trace_(.+)_s(\d+)_w\d+_\d+_s20d_clean_seed(\d+)_job(\d+)\.csv$")
    for f in glob.glob(os.path.join(args.output_dir, "trace_*.csv")):
        m = pat.match(os.path.basename(f))
        if m:
            existing[(m.group(1), m.group(2))] = True

    # Load manifest
    with open(args.manifest, newline="") as f:
        all_rows = list(csv.DictReader(f))

    # Filter to this worker's tasks
    my_tasks = set(args.tasks.split(","))
    pending = [r for r in all_rows
               if r["task_key"] in my_tasks
               and (r["task_key"], r["state_id"]) not in existing]

    if args.benchmark_single:
        # One state from first task
        first_task = args.tasks.split(",")[0]
        pending = [r for r in pending if r["task_key"] == first_task][:1]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Worker {args.worker_id}: {len(pending)} pending states")
    if not pending:
        print(f"Worker {args.worker_id}: nothing to do.")
        return

    # Batch by task in allocation order
    task_order = args.tasks.split(",")
    job_id_map = {"A": 510000, "B": 520000, "C": 530000, "D": 540000,
                  "RETRY_A": 560000, "RETRY_D": 570000,
                  "GPU26": 580000, "GPU45": 590000}
    job_id_base = job_id_map.get(args.worker_id, 590000 + hash(args.worker_id) % 100000)

    env = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
           "OPENVLA_ATTN_IMPLEMENTATION": "eager",
           "CUDA_VISIBLE_DEVICES": args.gpu_visible, "DISPLAY": "",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    completed = 0; total = len(pending)

    for task in task_order:
        task_jobs = sorted([r for r in pending if r["task_key"] == task],
                           key=lambda x: int(x["state_id"]))
        if not task_jobs:
            continue

        states_str = ",".join(j["state_id"] for j in task_jobs)
        batch_job_id = job_id_base
        job_id_base += 100
        n = len(task_jobs)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args.worker_id} TASK {task}: {n} states [{states_str[:60]}]")

        t0 = time.time()
        cmd = [PYTHON, "-u", RUNNER, "--task", task, "--state_ids", states_str,
               "--condition", "clean", "--window_start", "0", "--window_end", "10",
               "--max_steps_override", "280", "--success_metric", "check_success",
               "--num_steps_wait", "10", "--model_path", MODEL,
               "--render_gpu_device_id", str(args.render_gpu),
               "--model_gpu_device_id", "-1",
               "--output_dir", args.output_dir, "--job_id", str(batch_job_id), "--seed", "0"]

        try:
            r = subprocess.run(cmd, env=env, timeout=5400, capture_output=True, text=True)
            dt = time.time() - t0
            rc = r.returncode
            stderr_tail = r.stderr[-300:] if r.stderr else ""
        except subprocess.TimeoutExpired:
            dt = time.time() - t0
            rc = -1
            stderr_tail = "TIMEOUT"

        # Update ledger
        try:
            ledger_map = {}
            if os.path.exists(args.ledger):
                with open(args.ledger, newline="") as f:
                    for row in csv.DictReader(f):
                        ledger_map[(row["task_key"], row["state_id"])] = row

            for j in task_jobs:
                key = (j["task_key"], j["state_id"])
                trace_file = os.path.join(
                    args.output_dir,
                    f"trace_{j['task_key']}_s{j['state_id']}_w0_10_s20d_clean_seed0_job{batch_job_id}.csv")
                if os.path.isfile(trace_file):
                    sha = sha256_full(trace_file)
                    rows_n = sum(1 for _ in open(trace_file)) - 1
                    status = "completed"; fc = ""; fd = ""
                    completed += 1
                elif rc == -1:
                    status = "timeout"; fc = "timeout"; fd = "5400s"; sha = ""; rows_n = ""
                elif "OutOfMemory" in stderr_tail:
                    status = "oom"; fc = "oom"; fd = stderr_tail[:200]; sha = ""; rows_n = ""
                elif rc != 0:
                    status = "infra_failure"; fc = "process_nonzero"; fd = stderr_tail[:200]; sha = ""; rows_n = ""
                else:
                    status = "missing_artifact"; fc = "runner_success_missing_artifact"
                    fd = "no trace file"; sha = ""; rows_n = ""

                ledger_map[key] = {
                    "task_key": key[0], "state_id": key[1],
                    "seed": j.get("seed", "0"), "worker_id": args.worker_id,
                    "gpu_group": args.worker_id, "batch_job_id": str(batch_job_id),
                    "status": status, "start_time": "", "end_time": "",
                    "runtime_sec": str(round(dt, 1)) if dt else "",
                    "trace_path": trace_file, "full_trace_sha256": sha,
                    "row_count": str(rows_n) if rows_n else "",
                    "failure_class": fc, "failure_detail": fd, "attempt": "1",
                }

            # Save ledger
            with open(args.ledger, "w", newline="") as f:
                fields = list(next(iter(ledger_map.values())).keys())
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
                w.writerows(ledger_map.values())
        except Exception as e:
            print(f"  LEDGER UPDATE ERROR: {e}")

        print(f"[{datetime.now().strftime('%H:%M:%S')}]   {completed}/{total} ok ({dt:.0f}s)")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Worker {args.worker_id} DONE: {completed}/{total}")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--ledger", default="")
    ap.add_argument("--benchmark", action="store_true")

    # Worker-only args (hidden from --help)
    ap.add_argument("--worker-id", default="", help=argparse.SUPPRESS)
    ap.add_argument("--gpu-visible", default="", help=argparse.SUPPRESS)
    ap.add_argument("--render-gpu", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--tasks", default="", help=argparse.SUPPRESS)
    ap.add_argument("--benchmark-single", action="store_true", help=argparse.SUPPRESS)

    args = ap.parse_args()

    if args.worker:
        worker_main()
        return

    # ── Master ──
    os.makedirs(args.output_dir, exist_ok=True)
    ledger_path = args.ledger or os.path.join(args.output_dir, "d2_rollout_ledger.csv")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] D2.1a MASTER: launching 4 workers")
    if args.benchmark:
        print("BENCHMARK MODE: 1 state per group")

    workers = []
    for grp_id, gpu_visible, render_gpu in GPU_GROUPS:
        tasks = TASK_ALLOCATION[grp_id]
        n_tasks = len(tasks.split(","))
        print(f"  Group {grp_id}: GPUs {gpu_visible}, tasks: {tasks}")

        p = launch_worker(grp_id, gpu_visible, render_gpu, tasks,
                          args.manifest, args.output_dir, ledger_path, args.benchmark)
        workers.append((grp_id, p))

    print(f"Waiting for {len(workers)} workers...")
    for grp_id, p in workers:
        rc = p.wait()
        print(f"  Group {grp_id}: exit {rc}")

    # Merge final ledger summary
    if os.path.exists(ledger_path):
        with open(ledger_path, newline="") as f:
            rows = list(csv.DictReader(f))
        status_counts = defaultdict(int)
        for r in rows:
            status_counts[r.get("status", "unknown")] += 1
        print(f"\nFINAL LEDGER: {dict(status_counts)}")
        n_traces = len(glob.glob(os.path.join(args.output_dir, "trace_*.csv")))
        print(f"Total trace files: {n_traces}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] MASTER DONE")


if __name__ == "__main__":
    main()
