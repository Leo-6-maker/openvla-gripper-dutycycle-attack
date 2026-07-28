"""Fail-closed multi-GPU supervisor for FIT670 V2."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import fit670_strict_contract as strict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-plan", type=Path, required=True)
    parser.add_argument("--identity-allowlist", type=Path, required=True)
    parser.add_argument("--transition-receipt", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--official-worker", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--alias-ledger", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--mode", choices=("canary", "formal"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]
    gpu_list = [int(value) for value in args.gpus.split(",")]
    if not gpu_list or len(gpu_list) != len(set(gpu_list)):
        raise SystemExit("GPU list must be non-empty and unique")
    plan, _ = strict.validate_shard_plan(
        args.shard_plan, args.identity_allowlist, len(gpu_list)
    )
    expected_mapping = {
        str(shard_id): gpu_list[shard_id] for shard_id in range(len(gpu_list))
    }
    transition = strict.load_json(args.transition_receipt / "TRANSITION_MANIFEST.json")
    if transition.get("shard_to_physical_gpu") != expected_mapping:
        raise SystemExit("CLI GPU order differs from frozen transition mapping")

    source_files = {
        "fit670_strict_contract.py": here / "fit670_strict_contract.py",
        "run_fit670_atomic_worker_v2.py": here / "run_fit670_atomic_worker_v2.py",
        "run_fit670_atomic_worker.py": here / "run_fit670_atomic_worker.py",
        "fit_collection_core.py": here / "fit_collection_core.py",
        "run_fit670_supervisor_v2.py": here / "run_fit670_supervisor_v2.py",
        "finalize_fit670_collection_v2.py": here / "finalize_fit670_collection_v2.py",
        "run_fit670_v2.sh": here / "run_fit670_v2.sh",
    }
    registry_summary = args.registry_root.parent / "ENTITY_REGISTRY_V2_SUMMARY.json"

    # Validate every shard/GPU binding before any process or CUDA import.
    for shard_id, gpu in enumerate(gpu_list):
        strict.validate_transition_v2(
            args.transition_receipt,
            allowlist_path=args.identity_allowlist,
            shard_plan_path=args.shard_plan,
            output_root=args.output_root,
            physical_gpu=gpu,
            shard_id=shard_id,
            model_path=args.model_path,
            official_worker=args.official_worker,
            registry_summary=registry_summary,
            alias_ledger=args.alias_ledger,
            repo_root=repo_root,
            upstream_root=args.upstream_root,
            libero_root=args.libero_root,
            source_files=source_files,
        )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    worker = here / "run_fit670_atomic_worker_v2.py"
    max_identities = 1 if args.mode == "canary" else 0
    commands = []
    for shard_id, gpu in enumerate(gpu_list):
        shard = plan["shards"][shard_id]
        command = [
            sys.executable, "-u", str(worker),
            "--shard-id", str(shard_id),
            "--gpu", str(gpu),
            "--model-path", str(args.model_path),
            "--official-worker", str(args.official_worker),
            "--transition-receipt", str(args.transition_receipt),
            "--identity-allowlist", str(args.identity_allowlist),
            "--shard-plan", str(args.shard_plan),
            "--registry-root", str(args.registry_root),
            "--alias-ledger", str(args.alias_ledger),
            "--upstream-root", str(args.upstream_root),
            "--libero-root", str(args.libero_root),
            "--output-root", str(output_root),
            "--seed", str(args.seed),
            "--max-identities", str(max_identities),
        ]
        commands.append((shard_id, gpu, shard["n_identities"], command))

    if args.dry_run:
        for shard_id, gpu, count, command in commands:
            print(f"shard={shard_id} gpu={gpu} assigned={count}")
            print(" ".join(command))
        return

    processes = []
    for shard_id, gpu, count, command in commands:
        log_path = output_root / f"worker_shard_{shard_id}_gpu_{gpu}.log"
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONPATH": os.pathsep.join(
                    [
                        str(args.libero_root.resolve()),
                        str(args.upstream_root.resolve()),
                        os.environ.get("PYTHONPATH", ""),
                    ]
                ),
            },
        )
        processes.append((shard_id, gpu, count, process, handle, log_path))
        print(f"launched shard={shard_id} gpu={gpu} pid={process.pid}")

    failures = []
    started = time.time()
    try:
        while processes:
            live = []
            for shard_id, gpu, count, process, handle, log_path in processes:
                code = process.poll()
                if code is None:
                    live.append((shard_id, gpu, count, process, handle, log_path))
                    continue
                handle.close()
                if code != 0:
                    failures.append((shard_id, gpu, code, str(log_path)))
                print(f"finished shard={shard_id} gpu={gpu} exit={code}")
            processes = live
            if processes:
                print(f"elapsed={time.time()-started:.0f}s live={len(processes)}")
                time.sleep(30)
    except BaseException:
        for _, _, _, process, handle, _ in processes:
            process.terminate()
            handle.close()
        raise
    if failures:
        raise SystemExit(f"FIT670 workers failed: {failures}")
    print(f"all workers passed in {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
