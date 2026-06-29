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


def verify_provenance(manifest_path: str, condition_id: str,
                      expected_worker: str = None, expected_bridge: str = None) -> dict:
    """Verify worker/bridge SHA at launch time. Fails on expected mismatch."""
    manifest_sha = sha256_file(manifest_path)
    worker_sha = sha256_file(WORKER) if os.path.exists(WORKER) else "MISSING"
    bridge_path = os.path.join(os.path.dirname(WORKER), "run_v2_vis_sc5_mlp_bridge.py")
    bridge_sha = sha256_file(bridge_path) if os.path.exists(bridge_path) else "MISSING"

    errors = []
    if expected_worker and worker_sha != expected_worker:
        errors.append(f"Worker SHA mismatch: expected {expected_worker[:16]}... got {worker_sha[:16]}...")
    if expected_bridge and bridge_sha != expected_bridge:
        errors.append(f"Bridge SHA mismatch: expected {expected_bridge[:16]}... got {bridge_sha[:16]}...")
    if errors:
        for e in errors:
            print(f"PROVENANCE FAIL: {e}")
        raise RuntimeError("Provenance verification failed — worker/bridge may have drifted")

    return {
        "manifest_sha256": manifest_sha,
        "worker_sha256": worker_sha,
        "bridge_sha256": bridge_sha,
        "condition_id": condition_id,
        "worker_path": WORKER,
        "bridge_path": bridge_path,
        "expected_worker_sha256": expected_worker,
        "expected_bridge_sha256": expected_bridge,
    }


def main():
    ap = argparse.ArgumentParser(description="Launch a Table 1 condition")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--condition_id", required=True)
    ap.add_argument("--launch_dir", required=True)
    ap.add_argument("--mode", default="formal", choices=["formal", "canary"],
                    help="formal=exact 162 jobs, canary=any count")
    ap.add_argument("--expected_worker_sha", help="Required worker SHA-256")
    ap.add_argument("--expected_bridge_sha", help="Required bridge SHA-256")
    ap.add_argument("--expected_manifest_sha", help="Pre-approved manifest SHA (reject on mismatch)")
    ap.add_argument("--gpus", type=int, nargs="*", help="Approved GPU list (REQUIRED for --execute)")
    ap.add_argument("--condition_spec", help="Condition spec JSON (REQUIRED for --execute)")
    ap.add_argument("--expected_condition_spec_sha", help="Pre-approved condition spec SHA-256")
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

    # Execution status gate: check RUNNING marker
    running_marker = os.path.join(args.launch_dir, "RUNNING")
    if os.path.exists(running_marker):
        try:
            running_info = json.load(f)
            sys.exit(f"Condition already RUNNING since {running_info.get('started','?')}. "
                     f"Wait for completion or use recovery procedure.")
        except (json.JSONDecodeError, KeyError) as e:
            sys.exit(f"Corrupt RUNNING marker at {running_marker}: {e}. "
                     f"Manual recovery required.")

    # ── Provenance binding ──
    if args.execute and not args.expected_worker_sha:
        sys.exit("--execute requires --expected_worker_sha and --expected_bridge_sha")
    if args.execute and not args.expected_bridge_sha:
        sys.exit("--execute requires --expected_worker_sha and --expected_bridge_sha")

    prov = verify_provenance(args.manifest, args.condition_id,
                              args.expected_worker_sha, args.expected_bridge_sha)
    print(f"Worker SHA: {prov['worker_sha256'][:16]}...")
    print(f"Bridge SHA: {prov['bridge_sha256'][:16]}...")
    print(f"Manifest SHA: {prov['manifest_sha256'][:16]}...")

    # ── Manifest SHA approval ──
    manifest_sha = prov["manifest_sha256"]
    if args.expected_manifest_sha:
        if manifest_sha != args.expected_manifest_sha:
            sys.exit(f"Manifest SHA mismatch: expected {args.expected_manifest_sha[:16]}... "
                     f"got {manifest_sha[:16]}... — manifest must be pre-approved")
    elif args.execute:
        sys.exit("--execute requires --expected_manifest_sha (pre-approved manifest SHA)")

    # ── Check existing outputs ──
    existing = check_existing_outputs(jobs, args.mode)
    if existing:
        print(f"ERROR: {len(existing)} jobs have existing output (rejected in all modes):")
        for e in existing[:10]:
            print(f"  {e}")
        sys.exit("Cannot launch over existing outputs. Use recovery procedure.")

    # ── Condition spec gate ──
    if args.execute:
        if not args.condition_spec:
            sys.exit("--execute requires --condition_spec")
        if not os.path.exists(args.condition_spec):
            sys.exit(f"Condition spec not found: {args.condition_spec}")
        spec = json.loads(open(args.condition_spec).read())
        if spec.get("execution_status") != "FROZEN":
            sys.exit(f"Condition execution_status={spec.get('execution_status')} (must be FROZEN)")
        if spec.get("condition_id") != args.condition_id:
            sys.exit(f"Spec condition_id={spec.get('condition_id')} != {args.condition_id}")
        if args.expected_condition_spec_sha:
            actual_spec_sha = sha256_file(args.condition_spec)
            if actual_spec_sha != args.expected_condition_spec_sha:
                sys.exit(f"Condition spec SHA mismatch: expected {args.expected_condition_spec_sha[:16]}... "
                         f"got {actual_spec_sha[:16]}...")

    # ── GPU check ──
    gpu_free = get_gpu_free()
    print(f"GPU free (MB): {gpu_free}")
    if args.execute and args.gpus is None:
        sys.exit("--execute requires --gpus (explicit GPU allowlist)")
    if args.gpus is not None and len(args.gpus) == 0:
        sys.exit("--gpus cannot be empty")
    if args.gpus:
        approved_gpus = set(args.gpus)
        for g in approved_gpus:
            if g in GPU_DENYLIST:
                sys.exit(f"GPU {g} is denylisted")
            if gpu_free.get(g, 0) < MIN_FREE_MB:
                sys.exit(f"GPU {g} has {gpu_free.get(g, 0)} MB < {MIN_FREE_MB}")
        available = sorted(approved_gpus)
    else:
        available = [g for g, mb in sorted(gpu_free.items())
                     if g not in GPU_DENYLIST and mb >= MIN_FREE_MB]
    for gpu, mb in sorted(gpu_free.items()):
        tag = "DENYLIST" if gpu in GPU_DENYLIST else ("APPROVED" if gpu in available else f"SKIP")
        print(f"  GPU {gpu}: {mb} MB — {tag}")
    if not available:
        sys.exit("No GPUs available")

    # ── Split jobs ──
    n_workers = min(len(jobs), len(available))  # never more workers than jobs
    splits = [[] for _ in range(n_workers)]
    for i, j in enumerate(jobs):
        splits[i % n_workers].append(j)
    empty_splits = [i for i, s in enumerate(splits) if len(s) == 0]
    if empty_splits:
        sys.exit(f"Empty worker splits: indices {empty_splits}. n_jobs={len(jobs)} n_workers={n_workers}")

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
        # Write durable RUNNING marker before spawning workers
        running_info = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "condition_id": args.condition_id, "mode": args.mode,
                        "pid": os.getpid()}
        with open(running_marker, "w") as f:
            json.dump(running_info, f)

        launch_plan = {"manifest_sha256": manifest_sha,
                       "condition_id": args.condition_id, "mode": args.mode,
                       "timestamp": running_info["started"],
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
