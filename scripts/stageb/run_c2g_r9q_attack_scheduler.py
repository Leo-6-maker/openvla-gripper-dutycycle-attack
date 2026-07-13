#!/usr/bin/env python3
"""Run R9Q persistent workers on physical GPUs 6 and 7.

There are exactly four suite workers per GPU. Model loading is serialized by a
single advisory lock; worker and GPU owner locks are held for the run. The
scheduler never kills or migrates unrelated processes.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SUITES = ("object", "spatial", "goal", "l10")
GPUS = (6, 7)
EXPECTED_WORKERS = tuple(f"g{gpu}_{suite}" for gpu in GPUS for suite in SUITES)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def nvidia_snapshot(gpu: int) -> dict[str, Any]:
    query = ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"]
    try:
        output = subprocess.check_output(query, text=True, stderr=subprocess.STDOUT)
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if fields and int(fields[0]) == gpu:
                return {
                    "gpu": gpu,
                    "memory_total_mib": int(fields[1]),
                    "memory_used_mib": int(fields[2]),
                    "memory_free_mib": int(fields[3]),
                    "utilization_percent": float(fields[4]),
                    "temperature_c": float(fields[5]),
                }
    except Exception as exc:
        return {"gpu": gpu, "error": f"{type(exc).__name__}: {exc}"}
    return {"gpu": gpu, "error": "GPU not returned by nvidia-smi"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preview", "run"), required=True)
    parser.add_argument("--plan-root", required=True)
    parser.add_argument("--detector-bundle", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--worker-budget-mib", type=int, default=18000)
    parser.add_argument("--gpu-reserve-mib", type=int, default=8000)
    parser.add_argument("--max-resident-workers-per-gpu", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--model-load-lock-file", default="/tmp/c2g_r9q_global_model_load.lock")
    return parser.parse_args()


def load_worker_rows(plan_root: Path) -> dict[str, Path]:
    shards = plan_root / "shards"
    result = {path.stem: path for path in shards.glob("g*.jsonl")}
    if set(result) != set(EXPECTED_WORKERS):
        raise SystemExit(f"expected exactly {EXPECTED_WORKERS}, got {sorted(result)}")
    for worker_id, path in result.items():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise SystemExit(f"empty worker manifest: {worker_id}")
        if {row["assigned_worker_id"] for row in rows} != {worker_id}:
            raise SystemExit(f"worker assignment mismatch: {worker_id}")
        parent_conditions = {}
        for row in rows:
            parent_conditions.setdefault(row["parent_key"], set()).add(row["condition"])
        if any(len(conditions) != 4 for conditions in parent_conditions.values()):
            raise SystemExit(f"worker {worker_id} does not have four conditions per parent")
    return result


def build_report(args: argparse.Namespace, *, status: str, before: dict[str, Any], after: dict[str, Any], workers: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": status,
        "expected_git_commit": args.expected_git_commit,
        "plan_root": str(Path(args.plan_root).resolve()),
        "detector_bundle": str(Path(args.detector_bundle).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "requested_resident_workers_per_gpu": args.max_resident_workers_per_gpu,
        "worker_budget_mib": args.worker_budget_mib,
        "gpu_reserve_mib": args.gpu_reserve_mib,
        "gpus": list(GPUS),
        "workers": list(EXPECTED_WORKERS),
        "gpu_mapping": {worker: int(worker[1]) for worker in EXPECTED_WORKERS},
        "memory_before": before,
        "memory_after": after,
        "workers_report": workers,
        "concurrency_degraded": False,
        "concurrency_degradation_reason": "",
    }


def main() -> int:
    args = parse_args()
    plan_root = Path(args.plan_root).resolve()
    bundle = Path(args.detector_bundle).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite output root: {output_root}")
    if args.max_resident_workers_per_gpu != 4:
        raise SystemExit("R9Q scheduler requires exactly four logical workers per GPU")
    worker_manifests = load_worker_rows(plan_root)
    if not bundle.is_dir() or not (bundle / "checkpoint.pt").is_file():
        raise SystemExit("detector bundle is incomplete")
    before = {str(gpu): nvidia_snapshot(gpu) for gpu in GPUS}
    for gpu in GPUS:
        snap = before[str(gpu)]
        if "memory_free_mib" not in snap:
            raise SystemExit(f"cannot verify GPU {gpu}: {snap}")
        required = 4 * args.worker_budget_mib + args.gpu_reserve_mib
        if int(snap["memory_free_mib"]) < required:
            raise SystemExit(f"GPU {gpu} has insufficient free memory: {snap} required={required} MiB")

    preview = {
        "status": "PASS_C2G_R9Q_ATTACK_SCHEDULER_PREVIEW",
        "expected_git_commit": args.expected_git_commit,
        "worker_manifests": {worker: str(path) for worker, path in sorted(worker_manifests.items())},
        "worker_rows": {
            worker: len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
            for worker, path in sorted(worker_manifests.items())
        },
        "gpu_mapping": {worker: int(worker[1]) for worker in EXPECTED_WORKERS},
        "memory_before": before,
        "requested_resident_workers_per_gpu": 4,
        "achieved_resident_workers_per_gpu": 4,
        "concurrency_degraded": False,
        "global_model_load_lock": args.model_load_lock_file,
    }
    if args.mode == "preview":
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0

    output_root.mkdir(parents=True)
    for gpu in GPUS:
        (output_root / "gpu_locks").mkdir(parents=True, exist_ok=True)
    owner_handles = []
    try:
        for gpu in GPUS:
            lock_path = Path(f"/tmp/c2g_r9q_gpu_{gpu}.owner.lock")
            handle = lock_path.open("w")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                raise SystemExit(f"GPU owner lock is busy: {lock_path}")
            owner_handles.append(handle)

        statuses = output_root / "statuses"
        logs = output_root / "logs"
        statuses.mkdir()
        logs.mkdir()
        processes: dict[str, subprocess.Popen[str]] = {}
        for worker_id in EXPECTED_WORKERS:
            gpu = int(worker_id[1])
            status_file = statuses / f"{worker_id}.json"
            stdout_file = logs / f"{worker_id}.stdout.log"
            stderr_file = logs / f"{worker_id}.stderr.log"
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "C2G_PHYSICAL_GPU": str(gpu),
                "C2G_WORKER_ID": worker_id,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "PYTHONPATH": f"{Path(__file__).resolve().parents[2] / 'src'}:{Path(__file__).resolve().parents[2]}" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""),
            })
            command = [
                args.python, str(Path(__file__).resolve().with_name("run_c2g_r9q_attack_worker.py")),
                "--manifest", str(worker_manifests[worker_id]),
                "--detector-bundle", str(bundle),
                "--output-root", str(output_root),
                "--worker-id", worker_id,
                "--physical-gpu", str(gpu),
                "--expected-git-commit", args.expected_git_commit,
                "--model-load-lock-file", args.model_load_lock_file,
                "--status-file", str(status_file),
            ]
            out_handle = stdout_file.open("w", encoding="utf-8")
            err_handle = stderr_file.open("w", encoding="utf-8")
            processes[worker_id] = subprocess.Popen(command, env=env, stdout=out_handle, stderr=err_handle, text=True)

        heartbeat = output_root / "scheduler_heartbeat.jsonl"
        while processes:
            snapshot = {worker: {
                "pid": process.pid,
                "returncode": process.poll(),
                "status": json.loads((statuses / f"{worker}.json").read_text(encoding="utf-8"))
                if (statuses / f"{worker}.json").is_file() else {},
            } for worker, process in processes.items()}
            with heartbeat.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"timestamp": time.time(), "workers": snapshot}, sort_keys=True) + "\n")
            done = [worker for worker, process in processes.items() if process.poll() is not None]
            if done:
                for worker in done:
                    processes.pop(worker)
            if processes:
                time.sleep(max(1, args.poll_seconds))
        after = {str(gpu): nvidia_snapshot(gpu) for gpu in GPUS}
        workers_report = {}
        failed = False
        for worker_id in EXPECTED_WORKERS:
            status_path = statuses / f"{worker_id}.json"
            status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
            # The process return codes are recovered from the final status and log; a missing PASS is fail-closed.
            worker_failed = status.get("phase") != "PASS" or int(status.get("failed_cell_count", 1)) != 0
            failed = failed or worker_failed
            workers_report[worker_id] = status
        report = build_report(args, status="HOLD_C2G_R9Q_ATTACK_WORKER_FAILURE" if failed else "PASS_C2G_R9Q_ATTACK_SCHEDULER_RUN", before=before, after=after, workers=workers_report)
        write_json(output_root / "scheduler_report.json", report)
        return 1 if failed else 0
    finally:
        for handle in owner_handles:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
