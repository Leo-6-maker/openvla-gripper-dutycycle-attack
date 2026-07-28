"""[DeepSeek] FIT670 Supervisor — Gate F670-G.

Spawns one independent worker process per GPU from the shard plan.
Monitors completion, collects per-worker manifests, reports progress.

Usage:
  python n5/phase2_labels/run_fit670_supervisor.py \
    --shard-plan /path/to/FIT670_GPU_SHARD_PLAN.json \
    --identity-allowlist /path/to/FIT670_IDENTITY_ALLOWLIST.json \
    --transition-receipt /path/to/transition_root \
    --model-path /path/to/model \
    --official-worker /path/to/official_clean_worker.py \
    --registry-root /path/to/registry/per_task \
    --alias-ledger /path/to/ALIAS_LEDGER.json \
    --upstream-root /path/to/upstream \
    --output-root /path/to/d670_output \
    --seed 20260717 \
    --gpus 0,1,2,3,4,5
"""
import argparse, json, os, subprocess, sys, time, uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-plan", type=Path, required=True)
    parser.add_argument("--identity-allowlist", type=Path, required=True)
    parser.add_argument("--transition-receipt", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--official-worker", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--alias-ledger", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--gpus", default="0,1,2,3,4,5",
                        help="Comma-separated physical GPU indices (default: 0-5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    gpu_list = [int(x.strip()) for x in args.gpus.split(",")]
    n_workers = len(gpu_list)

    # Load shard plan
    shard_plan = json.loads(Path(args.shard_plan).read_text(encoding="utf-8"))
    n_shards = shard_plan.get("n_shards", 0)
    if n_shards != n_workers:
        print(f"WARNING: shard plan has {n_shards} shards but {n_workers} GPUs specified")

    shards = shard_plan.get("shards", [])
    if len(shards) < n_workers:
        raise SystemExit(f"only {len(shards)} shards in plan, need {n_workers}")

    print(f"FIT670 Supervisor: {n_workers} workers")
    print(f"  GPUs: {gpu_list}")
    print(f"  Total identities: {shard_plan['n_identities']}")
    print(f"  Output: {args.output_root}")
    print()

    # Build worker commands
    worker_script = Path(__file__).resolve().parent / "run_fit670_atomic_worker.py"
    commands = []
    for i, gpu in enumerate(gpu_list):
        if i >= len(shards):
            break
        shard = shards[i]
        cmd = [
            sys.executable, "-u", str(worker_script),
            "--shard-id", str(shard["shard_id"]),
            "--gpu", str(gpu),
            "--model-path", str(args.model_path),
            "--official-worker", str(args.official_worker),
            "--transition-receipt", str(args.transition_receipt),
            "--identity-allowlist", str(args.identity_allowlist),
            "--shard-plan", str(args.shard_plan),
            "--registry-root", str(args.registry_root),
            "--alias-ledger", str(args.alias_ledger),
            "--upstream-root", str(args.upstream_root),
            "--output-root", str(args.output_root),
            "--seed", str(args.seed),
        ]
        commands.append((gpu, shard["shard_id"], shard["n_identities"], shard["total_cost"], cmd))

    if args.dry_run:
        print("DRY RUN — commands that would be executed:")
        for gpu, sid, nid, cost, cmd in commands:
            print(f"\n  GPU {gpu} (shard {sid}, {nid} identities, cost={cost}):")
            print(f"    {' '.join(cmd)}")
        return

    # Launch workers
    processes = []
    for gpu, sid, nid, cost, cmd in commands:
        print(f"Launching GPU {gpu} (shard {sid}, {nid} ids, cost={cost})...")
        log_file = Path(args.output_root) / f"worker_gpu_{gpu}.log"
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd, stdout=lf, stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            )
        processes.append((gpu, sid, proc, log_file))

    print(f"\nAll {len(processes)} workers launched. Monitoring...\n")

    # Wait for completion
    t_start = time.time()
    while processes:
        still_running = []
        for gpu, sid, proc, log_file in processes:
            ret = proc.poll()
            if ret is None:
                still_running.append((gpu, sid, proc, log_file))
            else:
                elapsed = time.time() - t_start
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(f"  GPU {gpu} (shard {sid}): {status}  elapsed={elapsed:.0f}s  log={log_file}")
        processes = still_running
        if processes:
            elapsed = time.time() - t_start
            print(f"  [{elapsed:.0f}s] {len(processes)} workers still running...")
            time.sleep(30)

    total_elapsed = time.time() - t_start
    print(f"\nAll workers finished in {total_elapsed:.0f}s")
    print(f"Output: {args.output_root}")


if __name__ == "__main__":
    main()
