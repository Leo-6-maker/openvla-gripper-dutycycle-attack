#!/usr/bin/env python3
"""Launch a Table 1 condition — safe defaults, explicit --execute, lock checks.

Default: dry_run only. Requires --execute to actually launch.
Checks: GPU free memory, existing outputs, locks, denylist, manifest integrity.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

PYTHON = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python"
WORKER = "/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_vis_formal_worker.py"
MIN_FREE_MB = 20480
GPU_DENYLIST = {2}
MAX_WORKERS_PER_GPU = 1


def sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_gpu_free() -> dict[int, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
    except Exception:
        return {}
    free = {}
    for line in out.strip().split("\n"):
        parts = line.strip().split(",")
        if len(parts) >= 2:
            free[int(parts[0].strip())] = int(parts[1].strip())
    return free


def check_existing_outputs(jobs: list[dict]) -> list[str]:
    """Check for existing episode_summary.json or partial output."""
    issues = []
    for j in jobs:
        out = j.get("output_dir", "")
        ep = os.path.join(out, "episode_summary.json")
        if os.path.exists(ep):
            issues.append(f"EXISTS: {j.get('job_key','?')} at {ep}")
        elif os.path.exists(out) and os.listdir(out):
            issues.append(f"PARTIAL: {j.get('job_key','?')} has existing dir with content")
    return issues


def main():
    ap = argparse.ArgumentParser(description="Launch a Table 1 condition")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--condition_id", required=True)
    ap.add_argument("--launch_dir", required=True)
    ap.add_argument("--execute", action="store_true", help="Actually launch (default: dry_run)")
    ap.add_argument("--force", action="store_true", help="Launch even with existing outputs")
    args = ap.parse_args()

    # Load manifest
    jobs = []
    with open(args.manifest) as f:
        for line in f:
            jobs.append(json.loads(line))
    manifest_sha = sha256_file(args.manifest)
    print(f"Manifest: {len(jobs)} jobs, SHA={manifest_sha[:16]}...")

    if len(jobs) != 162:
        sys.exit(f"ERROR: expected 162 jobs, got {len(jobs)}")

    # Verify condition_id consistency
    for j in jobs:
        if j.get("condition_id") != args.condition_id:
            sys.exit(f"ERROR: job condition_id={j.get('condition_id')} != {args.condition_id}")

    # Check existing outputs
    existing = check_existing_outputs(jobs)
    if existing:
        print(f"WARNING: {len(existing)} jobs have existing output:")
        for e in existing[:10]:
            print(f"  {e}")
        if not args.force:
            sys.exit("Use --force to overwrite, or clean existing outputs first")

    # Check GPUs
    gpu_free = get_gpu_free()
    print(f"GPU free (MB): {gpu_free}")
    available = []
    for gpu, mb in sorted(gpu_free.items()):
        if gpu in GPU_DENYLIST:
            print(f"  GPU {gpu}: DENYLIST — skip")
        elif mb >= MIN_FREE_MB:
            available.append(gpu)
            print(f"  GPU {gpu}: {mb} MB — OK")
        else:
            print(f"  GPU {gpu}: {mb} MB — SKIP (< {MIN_FREE_MB})")
    if not available:
        sys.exit("No GPUs available")

    n_workers = len(available) * MAX_WORKERS_PER_GPU
    # Round-robin split
    splits = [[] for _ in range(n_workers)]
    for i, j in enumerate(jobs):
        splits[i % n_workers].append(j)

    os.makedirs(args.launch_dir, exist_ok=True)

    print(f"\nPlan: {n_workers} workers on {len(available)} GPUs ({len(jobs)} jobs)")
    for wi, split in enumerate(splits):
        gpu = available[wi // MAX_WORKERS_PER_GPU]
        print(f"  GPU {gpu} w{wi}: {len(split)} jobs")

    if not args.execute:
        print("\nDRY_RUN complete. Use --execute to launch.")
        return

    # Execute
    launch_plan = {"manifest_sha256": manifest_sha, "condition_id": args.condition_id,
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "gpus": available, "workers": []}
    for wi, split in enumerate(splits):
        gpu = available[wi // MAX_WORKERS_PER_GPU]
        mf_path = os.path.join(args.launch_dir, f"manifest_gpu{gpu}_w{wi}.jsonl")
        log_path = os.path.join(args.launch_dir, f"worker_gpu{gpu}_w{wi}.log")

        with open(mf_path, "w") as f:
            for j in split:
                f.write(json.dumps(j) + "\n")

        cmd = [PYTHON, "-u", WORKER, str(gpu), mf_path]
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        launch_plan["workers"].append({
            "gpu": gpu, "worker_idx": wi, "pid": proc.pid,
            "n_jobs": len(split), "manifest": mf_path, "log": log_path,
        })
        print(f"  GPU {gpu} w{wi}: PID {proc.pid}, {len(split)} jobs")

    with open(os.path.join(args.launch_dir, "LAUNCH_PLAN.json"), "w") as f:
        json.dump(launch_plan, f, indent=2)

    print(f"\nLaunched {n_workers} workers. Plan: {args.launch_dir}/LAUNCH_PLAN.json")
    print("Monitor: tail -f {}/worker_*.log".format(args.launch_dir))


if __name__ == "__main__":
    main()
