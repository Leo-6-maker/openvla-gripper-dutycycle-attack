#!/usr/bin/env python3
"""Dynamically schedule four suite-local R8T canary shards on GPUs 4-7.

The scheduler keeps at most one OpenVLA process on each physical GPU, polls
``nvidia-smi`` before assignment, and never migrates a running shard.  Each shard
uses a separate immutable output root and retains its logs on failure.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.stageb.run_c2g_r8t_teacher_v2_canary_shard import (
    RUN_STATUS,
    load_plan,
)
from tools.multisuite_detector.build_c2g_r8t_teacher_v2_canary import (
    AUTHORIZATION_TOKEN,
    sha256_file,
)

RUNNER = REPO / "scripts" / "stageb" / "run_c2g_r8t_teacher_v2_canary_shard.py"
SCHEMA = "c2g.r8t.dynamic_gpu_scheduler.2026-07-11.v1"
PASS_STATUS = "PASS_C2G_R8T_DYNAMIC_GPU_CANARY"
HOLD_STATUS = "HOLD_C2G_R8T_DYNAMIC_GPU_CANARY"


def parse_gpu_list(value: str) -> list[int]:
    output = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not output or len(output) != len(set(output)) or any(item < 0 for item in output):
        raise ValueError("--gpus must be a nonempty unique comma-separated list")
    return output


def gpu_snapshot(allowed: Sequence[int]) -> dict[int, dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    text = subprocess.check_output(command, text=True)
    output: dict[int, dict[str, int]] = {}
    for line in text.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 4:
            continue
        index, free, total, util = map(int, parts)
        if index in allowed:
            output[index] = {
                "index": index,
                "memory_free_mib": free,
                "memory_total_mib": total,
                "utilization_percent": util,
            }
    missing = sorted(set(allowed) - set(output))
    if missing:
        raise RuntimeError(f"nvidia-smi did not report allowed GPUs: {missing}")
    return output


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def shard_command(
    *,
    plan_report: Path,
    expected_plan_sha: str,
    shard_id: str,
    output_root: Path,
    suite_model_map: Path,
    suite_model_report: Path,
    goal_model_manifest: Path,
    max_steps: int,
    dummy_wait: int,
    base_seed: int,
    authorization: str,
) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "run",
        "--plan-report", str(plan_report),
        "--expected-plan-report-sha256", expected_plan_sha,
        "--shard-id", shard_id,
        "--output-root", str(output_root),
        "--suite-model-map", str(suite_model_map),
        "--suite-model-report", str(suite_model_report),
        "--goal-model-manifest", str(goal_model_manifest),
        "--device", "cuda:0",
        "--max-steps", str(max_steps),
        "--dummy-wait", str(dummy_wait),
        "--base-seed", str(base_seed),
        "--authorization", authorization,
    ]


def run_scheduler(
    *,
    mode: str,
    plan_report: Path,
    expected_plan_sha: str,
    output_root: Path,
    suite_model_map: Path,
    suite_model_report: Path,
    goal_model_manifest: Path,
    gpus: Sequence[int],
    min_free_mib: int,
    max_utilization: int,
    max_concurrent: int,
    poll_seconds: int,
    max_wait_seconds: int,
    max_steps: int,
    dummy_wait: int,
    base_seed: int,
    authorization: str,
) -> dict[str, Any]:
    if mode not in {"preview", "run"}:
        raise ValueError("mode must be preview or run")
    if output_root.exists():
        raise FileExistsError(output_root)
    if min_free_mib <= 0 or not (0 <= max_utilization <= 100):
        raise ValueError("invalid GPU thresholds")
    if max_concurrent <= 0 or max_concurrent > len(gpus):
        raise ValueError("max_concurrent must be between 1 and number of GPUs")
    plan = load_plan(plan_report, expected_plan_sha)
    shards = [dict(row) for row in plan.get("shards", [])]
    if len(shards) != 4 or {row.get("suite") for row in shards} != {
        "libero_object", "libero_spatial", "libero_goal", "libero_10"
    }:
        raise ValueError("R8T dynamic scheduler requires exactly four suite shards")
    if authorization != AUTHORIZATION_TOKEN and mode == "run":
        raise PermissionError(f"exact authorization token required: {AUTHORIZATION_TOKEN}")
    initial_snapshot = gpu_snapshot(gpus)
    preview_commands = {
        str(row["shard_id"]): shard_command(
            plan_report=plan_report.resolve(),
            expected_plan_sha=expected_plan_sha,
            shard_id=str(row["shard_id"]),
            output_root=output_root.resolve() / "shards" / str(row["shard_id"]),
            suite_model_map=suite_model_map.resolve(),
            suite_model_report=suite_model_report.resolve(),
            goal_model_manifest=goal_model_manifest.resolve(),
            max_steps=max_steps,
            dummy_wait=dummy_wait,
            base_seed=base_seed,
            authorization=authorization,
        )
        for row in shards
    }
    preview = {
        "schema": SCHEMA,
        "status": "PASS_C2G_R8T_DYNAMIC_GPU_PREVIEW",
        "mode": mode,
        "plan_report": str(plan_report.resolve()),
        "plan_report_sha256": sha256_file(plan_report.resolve()),
        "output_root": str(output_root.resolve()),
        "allowed_gpus": list(gpus),
        "initial_gpu_snapshot": initial_snapshot,
        "min_free_mib": min_free_mib,
        "max_utilization": max_utilization,
        "max_concurrent": max_concurrent,
        "shard_count": len(shards),
        "commands": preview_commands,
    }
    if mode == "preview":
        return preview

    output_root.mkdir(parents=True)
    log_root = output_root / "logs"
    log_root.mkdir()
    pending = list(shards)
    running: dict[int, dict[str, Any]] = {}
    completed_rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    started = time.time()
    stop_launching = False

    while pending or running:
        now = time.time()
        if now - started > max_wait_seconds:
            stop_launching = True
            if not running:
                break
        snapshot = gpu_snapshot(gpus)
        snapshots.append({"timestamp": now, "gpus": snapshot})

        for gpu, state in list(running.items()):
            returncode = state["process"].poll()
            if returncode is None:
                continue
            state["stdout"].close()
            state["stderr"].close()
            try:
                fcntl.flock(state["lock"].fileno(), fcntl.LOCK_UN)
            finally:
                state["lock"].close()
            row = {
                "shard_id": state["shard_id"],
                "suite": state["suite"],
                "physical_gpu": gpu,
                "pid": state["pid"],
                "returncode": returncode,
                "started_at": state["started_at"],
                "finished_at": now,
                "stdout_log": state["stdout_path"],
                "stderr_log": state["stderr_path"],
                "output_root": state["output_root"],
            }
            receipt = Path(state["output_root"]) / "c2g_r8t_canary_shard_receipt.json"
            if returncode == 0 and receipt.is_file():
                value = json.loads(receipt.read_text(encoding="utf-8"))
                row["receipt"] = str(receipt)
                row["receipt_sha256"] = sha256_file(receipt)
                row["receipt_status"] = value.get("status")
                if value.get("status") != RUN_STATUS:
                    row["returncode"] = 99
            else:
                row["receipt"] = None
                row["receipt_sha256"] = None
                row["receipt_status"] = None
            completed_rows.append(row)
            if row["returncode"] != 0:
                stop_launching = True
            del running[gpu]

        if pending and not stop_launching:
            available = [
                gpu
                for gpu in gpus
                if gpu not in running
                and snapshot[gpu]["memory_free_mib"] >= min_free_mib
                and snapshot[gpu]["utilization_percent"] <= max_utilization
            ]
            available.sort(
                key=lambda gpu: (
                    -snapshot[gpu]["memory_free_mib"],
                    snapshot[gpu]["utilization_percent"],
                    gpu,
                )
            )
            while pending and available and len(running) < max_concurrent:
                gpu = available.pop(0)
                lock_path = Path(f"/tmp/c2g_r8t_gpu_{gpu}.lock")
                lock_handle = lock_path.open("a+")
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock_handle.close()
                    continue
                shard = pending.pop(0)
                shard_id = str(shard["shard_id"])
                shard_output = output_root / "shards" / shard_id
                stdout_path = log_root / f"{shard_id}.stdout.log"
                stderr_path = log_root / f"{shard_id}.stderr.log"
                stdout_handle = stdout_path.open("w", encoding="utf-8")
                stderr_handle = stderr_path.open("w", encoding="utf-8")
                command = preview_commands[shard_id]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(
                    command,
                    cwd=REPO,
                    env=env,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                )
                running[gpu] = {
                    "process": process,
                    "pid": process.pid,
                    "shard_id": shard_id,
                    "suite": shard["suite"],
                    "started_at": time.time(),
                    "stdout": stdout_handle,
                    "stderr": stderr_handle,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "output_root": str(shard_output),
                    "lock": lock_handle,
                }

        if not running and pending and stop_launching:
            break
        if pending or running:
            time.sleep(poll_seconds)

    failed = [row for row in completed_rows if int(row["returncode"]) != 0]
    status = PASS_STATUS if not pending and not failed and len(completed_rows) == len(shards) else HOLD_STATUS
    report = {
        "schema": SCHEMA,
        "status": status,
        "git_commit": plan["expected_git_commit"],
        "plan_report": str(plan_report.resolve()),
        "plan_report_sha256": sha256_file(plan_report.resolve()),
        "output_root": str(output_root.resolve()),
        "allowed_gpus": list(gpus),
        "min_free_mib": min_free_mib,
        "max_utilization": max_utilization,
        "max_concurrent": max_concurrent,
        "poll_seconds": poll_seconds,
        "elapsed_seconds": time.time() - started,
        "completed_shard_count": len(completed_rows),
        "failed_shard_count": len(failed),
        "pending_shard_ids": [row["shard_id"] for row in pending],
        "shards": sorted(completed_rows, key=lambda row: row["shard_id"]),
        "gpu_snapshots": snapshots,
        "boundaries": {
            "one_process_per_gpu": True,
            "clean_only": True,
            "train_only": True,
            "attacks_launched": 0,
            "training_epochs": 0,
            "storage_deletions": 0,
        },
    }
    report_path = output_root / "c2g_r8t_dynamic_gpu_scheduler_report.json"
    write_json(report_path, report)
    return {**report, "report": str(report_path), "report_sha256": sha256_file(report_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preview", "run"))
    parser.add_argument("--plan-report", type=Path, required=True)
    parser.add_argument("--expected-plan-report-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suite-model-map", type=Path, required=True)
    parser.add_argument("--suite-model-report", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path, required=True)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--min-free-mib", type=int, default=24000)
    parser.add_argument("--max-utilization", type=int, default=40)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--max-wait-seconds", type=int, default=7200)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dummy-wait", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260711)
    parser.add_argument("--authorization", default=os.environ.get("R8T_COLLECTION_AUTHORIZATION", ""))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_scheduler(
        mode=args.mode,
        plan_report=args.plan_report,
        expected_plan_sha=args.expected_plan_report_sha256,
        output_root=args.output_root,
        suite_model_map=args.suite_model_map,
        suite_model_report=args.suite_model_report,
        goal_model_manifest=args.goal_model_manifest,
        gpus=parse_gpu_list(args.gpus),
        min_free_mib=args.min_free_mib,
        max_utilization=args.max_utilization,
        max_concurrent=args.max_concurrent,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
        max_steps=args.max_steps,
        dummy_wait=args.dummy_wait,
        base_seed=args.base_seed,
        authorization=args.authorization,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
