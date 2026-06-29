#!/usr/bin/env python3
"""Launch a Table 1 condition: distribute manifest across idle GPUs, start workers.

Checks GPU free memory, splits jobs evenly across workers on GPUs with >= min_free
memory, and launches one worker per GPU (or multiple if manifest > jobs_per_worker).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

PYTHON = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python"
WORKER = "/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_vis_formal_worker.py"
MIN_FREE_MB = 20480  # 20GB minimum free for one bridge
GPU_DENYLIST = {2}  # GPU2 remains quarantined


def get_gpu_free_memory() -> dict[int, int]:
    """Query nvidia-smi for free memory per GPU. Returns {gpu_id: free_mb}."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10)
    except Exception:
        return {}
    free = {}
    for line in out.strip().split("\n"):
        parts = line.strip().split(",")
        if len(parts) >= 2:
            gpu = int(parts[0].strip())
            mb = int(parts[1].strip())
            free[gpu] = mb
    return free


def split_jobs_evenly(jobs: list[dict], n_workers: int) -> list[list[dict]]:
    """Distribute jobs across workers round-robin."""
    splits = [[] for _ in range(n_workers)]
    for i, job in enumerate(jobs):
        splits[i % n_workers].append(job)
    return splits


def main():
    ap = argparse.ArgumentParser(description="Launch a Table 1 condition")
    ap.add_argument("--manifest", required=True, help="Condition formal_manifest.jsonl")
    ap.add_argument("--condition_id", required=True, help="Condition ID for logging")
    ap.add_argument("--launch_dir", required=True, help="Directory for launch manifests and logs")
    ap.add_argument("--max_workers_per_gpu", type=int, default=1,
                    help="Max workers per GPU (default 1 for stability)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    # Load manifest
    jobs = []
    with open(args.manifest) as f:
        for line in f:
            jobs.append(json.loads(line))
    print(f"Loaded {len(jobs)} jobs from manifest")

    # Check GPUs
    gpu_free = get_gpu_free_memory()
    print(f"GPU free memory (MB): {gpu_free}")

    available_gpus = []
    for gpu, free_mb in sorted(gpu_free.items()):
        if gpu in GPU_DENYLIST:
            continue
        if free_mb >= MIN_FREE_MB:
            available_gpus.append(gpu)
            print(f"  GPU {gpu}: {free_mb} MB — OK")
        else:
            print(f"  GPU {gpu}: {free_mb} MB — SKIP (< {MIN_FREE_MB})")

    if not available_gpus:
        sys.exit("No GPUs with sufficient free memory")

    n_workers = len(available_gpus) * args.max_workers_per_gpu
    splits = split_jobs_evenly(jobs, n_workers)

    os.makedirs(args.launch_dir, exist_ok=True)

    print(f"\nDistribution: {n_workers} workers on {len(available_gpus)} GPUs")
    for wi, (gpu_idx, split) in enumerate(zip(
        available_gpus * args.max_workers_per_gpu, splits)):
        gpu = available_gpus[wi // args.max_workers_per_gpu]
        manifest_path = f"{args.launch_dir}/manifest_gpu{gpu}_w{wi}.jsonl"
        log_path = f"{args.launch_dir}/worker_gpu{gpu}_w{wi}.log"

        if not args.dry_run:
            with open(manifest_path, "w") as f:
                for j in split:
                    f.write(json.dumps(j) + "\n")
            cmd = [PYTHON, "-u", WORKER, str(gpu), manifest_path]
            with open(log_path, "w") as log_f:
                subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            print(f"  GPU {gpu} w{wi}: {len(split)} jobs — launched (PID from nohup)")
        else:
            print(f"  GPU {gpu} w{wi}: {len(split)} jobs — DRY_RUN")

    if not args.dry_run:
        time.sleep(2)
        print(f"\nLaunched {n_workers} workers. Logs: {args.launch_dir}/")
        print("Monitor: tail -f {}/worker_*.log".format(args.launch_dir))


if __name__ == "__main__":
    main()
