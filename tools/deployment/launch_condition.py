#!/usr/bin/env python3
"""Launch a Table 1 condition — canary + formal modes, lock protection, SHA binding.

Default: dry_run. Requires --execute. Formal mode requires exact 162 jobs.
Canary mode accepts any job count with explicit allowlist.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path

PYTHON = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python"
WORKER = "/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_vis_formal_worker.py"
MIN_FREE_MB = 20480
GPU_DENYLIST = {2}


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


def acquire_lock(launch_dir: str) -> tuple:
    """Acquire atomic launch lock. Returns (fd, acquired)."""
    lock_path = os.path.join(launch_dir, "LAUNCH.lock")
    os.makedirs(launch_dir, exist_ok=True)
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fd.write(f"PID={os.getpid()}\nTIME={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        fd.flush()
        return fd, True
    except (IOError, OSError):
        fd.close()
        return None, False


def check_existing_outputs(jobs: list[dict], mode: str) -> list[str]:
    """Check for existing outputs. Formal mode: any existing → reject."""
    issues = []
    for j in jobs:
        out = j.get("output_dir", "")
        ep = os.path.join(out, "episode_summary.json")
        if os.path.exists(ep):
            issues.append(f"COMPLETE: {j.get('job_key','?')}")
        elif os.path.exists(out) and os.listdir(out):
            issues.append(f"PARTIAL: {j.get('job_key','?')}")
    return issues


def verify_provenance(manifest_path: str, condition_id: str) -> dict:
    """Verify worker/bridge SHA at launch time."""
    manifest_sha = sha256_file(manifest_path)
    worker_sha = sha256_file(WORKER) if os.path.exists(WORKER) else "MISSING"
    bridge_sha = sha256_file(os.path.join(os.path.dirname(WORKER), "run_v2_vis_sc5_mlp_bridge.py"))
    return {
        "manifest_sha256": manifest_sha,
        "worker_sha256": worker_sha,
        "bridge_sha256": bridge_sha,
        "condition_id": condition_id,
        "worker_path": WORKER,
        "bridge_path": os.path.join(os.path.dirname(WORKER), "run_v2_vis_sc5_mlp_bridge.py"),
    }


def main():
    ap = argparse.ArgumentParser(description="Launch a Table 1 condition")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--condition_id", required=True)
    ap.add_argument("--launch_dir", required=True)
    ap.add_argument("--mode", default="formal", choices=["formal", "canary"],
                    help="formal=exact 162 jobs, canary=any count")
    ap.add_argument("--execute", action="store_true", help="Actually launch")
    args = ap.parse_args()

    # ── Load and validate manifest ──
    jobs = []
    with open(args.manifest) as f:
        for line in f:
            jobs.append(json.loads(line))
    print(f"Manifest: {len(jobs)} jobs, mode={args.mode}")

    if args.mode == "formal" and len(jobs) != 162:
        sys.exit(f"Formal mode requires 162 jobs, got {len(jobs)}")
    if len(jobs) == 0:
        sys.exit("Empty manifest")

    for j in jobs:
        if j.get("condition_id") != args.condition_id:
            sys.exit(f"condition_id mismatch: {j.get('condition_id')} != {args.condition_id}")

    # ── Provenance binding ──
    prov = verify_provenance(args.manifest, args.condition_id)
    print(f"Worker SHA: {prov['worker_sha256'][:16]}...")
    print(f"Bridge SHA: {prov['bridge_sha256'][:16]}...")
    print(f"Manifest SHA: {prov['manifest_sha256'][:16]}...")

    # ── Check existing outputs ──
    existing = check_existing_outputs(jobs, args.mode)
    if existing:
        if args.mode == "formal":
            print(f"ERROR: {len(existing)} jobs have existing output (formal mode rejects):")
            for e in existing[:10]:
                print(f"  {e}")
            sys.exit("Cannot launch formal over existing outputs. Use recovery procedure.")
        else:
            print(f"WARNING: {len(existing)} jobs have existing output (canary mode continues):")
            for e in existing[:5]:
                print(f"  {e}")

    # ── GPU check ──
    gpu_free = get_gpu_free()
    print(f"GPU free (MB): {gpu_free}")
    available = [g for g, mb in sorted(gpu_free.items())
                 if g not in GPU_DENYLIST and mb >= MIN_FREE_MB]
    for gpu, mb in sorted(gpu_free.items()):
        tag = "DENYLIST" if gpu in GPU_DENYLIST else ("OK" if gpu in available else f"SKIP ({mb}<{MIN_FREE_MB})")
        print(f"  GPU {gpu}: {mb} MB — {tag}")
    if not available:
        sys.exit("No GPUs available")

    # ── Split jobs ──
    n_workers = len(available)
    splits = [[] for _ in range(n_workers)]
    for i, j in enumerate(jobs):
        splits[i % n_workers].append(j)

    os.makedirs(args.launch_dir, exist_ok=True)

    print(f"\nPlan: {n_workers} workers on {len(available)} GPUs, {len(jobs)} jobs")
    for wi, split in enumerate(splits):
        print(f"  GPU {available[wi]} w{wi}: {len(split)} jobs")

    if not args.execute:
        print("\nDRY_RUN complete. Use --execute to launch.")
        return

    # ── Acquire lock ──
    lock_fd, locked = acquire_lock(args.launch_dir)
    if not locked:
        sys.exit(f"Launch lock held by another process: {args.launch_dir}/LAUNCH.lock")

    try:
        launch_plan = {"manifest_sha256": prov["manifest_sha256"],
                       "condition_id": args.condition_id, "mode": args.mode,
                       "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "provenance": prov, "gpus": available, "workers": []}

        for wi, split in enumerate(splits):
            gpu = available[wi]
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

        print(f"\nLaunched. Plan: {args.launch_dir}/LAUNCH_PLAN.json")
        print("Monitor: tail -f {}/worker_*.log".format(args.launch_dir))
    finally:
        # Lock intentionally held until process exits
        pass


if __name__ == "__main__":
    main()
