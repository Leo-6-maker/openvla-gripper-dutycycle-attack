#!/usr/bin/env python3
"""Preview or run the fixed R8W 16-worker GPU4-7 schedule."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.stageb.run_c2g_r8w_full_clean_shard import (
    RECEIPT_SCHEMA,
    RUN_STATUS,
    load_plan,
    select_shard,
)
from scripts.stageb.verify_c2g_suite_model_map_strict import verify as verify_model_map
from tools.multisuite_detector.build_c2g_r8w_full_clean_2000_plan import (
    AUTHORIZATION_TOKEN,
    CANARY_PURPOSE,
    GPUS,
    SUITES,
    sha256_file,
    worker_id,
)

RUNNER = REPO / "scripts" / "stageb" / "run_c2g_r8w_full_clean_shard.py"
PREVIEW_STATUS = "PASS_C2G_R8W_16WORKER_PREVIEW"
RUN_STATUS_SCHEDULER = "PASS_C2G_R8W_16WORKER_COLLECTION"
HOLD_STATUS = "HOLD_C2G_R8W_16WORKER_COLLECTION"
CANARY_PREVIEW_STATUS = "PASS_C2G_R8W_FRESH_CANARY_SCHEDULER_PREVIEW"
CANARY_RUN_STATUS = "PASS_C2G_R8W_FRESH_CANARY_SCHEDULER"
CANARY_HOLD_STATUS = "HOLD_C2G_R8W_FRESH_CANARY_SCHEDULER"
SCHEMA = "c2g.r8w.gpu4567_16worker_scheduler.2026-07-12.v1"
TARGET_LOGICAL_WORKERS_PER_GPU = 4
TARGET_RESIDENT_WORKERS_PER_GPU = 4
HARD_MAX_WORKERS_PER_GPU = 4
GPU_RESERVE_MIB = 8000
MODEL_LOAD_LOCK = Path("/tmp/c2g_r8w_global_model_load.lock")


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int
    utilization_percent: int
    temperature_c: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def nvidia_snapshot(gpus: Sequence[int] = GPUS) -> dict[int, GpuSnapshot]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    )
    requested = set(gpus)
    snapshots: dict[int, GpuSnapshot] = {}
    for line in output.splitlines():
        values = [int(value.strip()) for value in line.split(",")]
        if len(values) != 6 or values[0] not in requested:
            continue
        snapshot = GpuSnapshot(*values)
        snapshots[snapshot.index] = snapshot
    missing = requested - snapshots.keys()
    if missing:
        raise RuntimeError(f"nvidia-smi did not report GPUs: {sorted(missing)}")
    return snapshots


def ceil_to_1024(value: float) -> int:
    return int(math.ceil(value / 1024.0) * 1024)


def measured_worker_budget_mib(observed_worker_mib: float) -> int:
    if observed_worker_mib <= 0:
        raise ValueError("observed worker memory must be positive")
    return ceil_to_1024(max(18000.0, observed_worker_mib * 1.15))


def safe_resident_workers(total_mib: int, worker_budget_mib: int, reserve_mib: int = GPU_RESERVE_MIB) -> int:
    if total_mib <= 0 or worker_budget_mib <= 0 or reserve_mib < 0:
        raise ValueError("invalid memory admission values")
    return max(0, min(HARD_MAX_WORKERS_PER_GPU, (total_mib - reserve_mib) // worker_budget_mib))


def memory_admission(snapshot: GpuSnapshot, worker_budget_mib: int) -> tuple[bool, str]:
    if snapshot.memory_free_mib < worker_budget_mib + GPU_RESERVE_MIB:
        return False, "INSUFFICIENT_FREE_MEMORY"
    if snapshot.utilization_percent > 40:
        return False, "GPU_UTILIZATION_ABOVE_40_PERCENT"
    return True, "PASS"


def stable_admission(samples: Sequence[GpuSnapshot], worker_budget_mib: int) -> tuple[bool, str]:
    if len(samples) != 3:
        return False, "REQUIRES_EXACTLY_THREE_SAMPLES"
    if len({sample.index for sample in samples}) != 1:
        return False, "MIXED_GPU_SAMPLES"
    for sample in samples:
        ok, reason = memory_admission(sample, worker_budget_mib)
        if not ok:
            return False, reason
    if max(sample.memory_free_mib for sample in samples) - min(sample.memory_free_mib for sample in samples) > 1024:
        return False, "FREE_MEMORY_NOT_STABLE"
    return True, "PASS"


def validate_worker_layout(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    shards = [dict(row) for row in plan.get("shards", [])]
    if len(shards) != 16:
        raise ValueError(f"R8W scheduler requires exactly 16 workers, got {len(shards)}")
    seen: set[str] = set()
    for gpu in GPUS:
        gpu_rows = [row for row in shards if row.get("physical_gpu") == gpu]
        if len(gpu_rows) != 4 or {row.get("suite") for row in gpu_rows} != set(SUITES):
            raise ValueError(f"GPU {gpu} worker layout mismatch")
        if sum(int(row.get("episode_count", -1)) for row in gpu_rows) != 500:
            raise ValueError(f"GPU {gpu} episode count mismatch")
        for suite in SUITES:
            expected = worker_id(gpu, suite)
            matches = [row for row in gpu_rows if row.get("worker_id") == expected]
            if len(matches) != 1 or matches[0].get("episode_count") != 125:
                raise ValueError(f"worker mapping mismatch: {expected}")
            if expected in seen:
                raise ValueError(f"duplicate worker: {expected}")
            seen.add(expected)
    for suite in SUITES:
        if sum(int(row["episode_count"]) for row in shards if row["suite"] == suite) != 500:
            raise ValueError(f"suite episode count mismatch: {suite}")
    order = []
    for suite in SUITES:
        for gpu in GPUS:
            order.append(next(row for row in shards if row["worker_id"] == worker_id(gpu, suite)))
    return order


def validate_canary_layout(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    if plan.get("plan_kind") != CANARY_PURPOSE:
        raise ValueError("not an R8W fresh shadow canary plan")
    shards = [dict(row) for row in plan.get("shards", [])]
    if len(shards) != 4:
        raise ValueError("R8W shadow canary requires exactly four workers")
    # Validate each shard against plan's own physical_gpu assignment
    output = []
    for shard in shards:
        gpu = shard.get("physical_gpu")
        suite = shard.get("suite")
        if gpu is None or suite is None:
            raise ValueError(f"R8W shadow canary shard missing gpu/suite: {shard}")
        if shard.get("episode_count") != 2:
            raise ValueError(f"R8W shadow canary mapping mismatch: GPU {gpu}/{suite}")
        output.append(shard)
    if len(output) != 4:
        raise ValueError("R8W shadow canary requires exactly four valid shards")
    return output


def worker_command(
    *,
    plan_report: Path,
    plan_sha256: str,
    shard: Mapping[str, Any],
    output_root: Path,
    suite_model_map: Path,
    suite_model_report: Path,
    goal_model_manifest: Path,
    model_verification_report: Path,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "run",
        "--plan-report", str(plan_report.resolve()),
        "--expected-plan-report-sha256", plan_sha256,
        "--worker-id", str(shard["worker_id"]),
        "--output-root", str((output_root / "workers" / str(shard["worker_id"])).resolve()),
        "--suite-model-map", str(suite_model_map.resolve()),
        "--suite-model-report", str(suite_model_report.resolve()),
        "--goal-model-manifest", str(goal_model_manifest.resolve()),
        "--model-verification-report", str(model_verification_report.resolve()),
        "--model-load-lock-file", str(MODEL_LOAD_LOCK),
        "--authorization", AUTHORIZATION_TOKEN,
    ]
    if resume:
        command.append("--resume")
    return command


def verify_models_once(
    output_root: Path,
    suite_model_map: Path,
    suite_model_report: Path,
    goal_model_manifest: Path,
    frozen_verification_report: Path,
) -> tuple[Path, str]:
    frozen = json.loads(frozen_verification_report.read_text(encoding="utf-8"))
    if not isinstance(frozen, dict) or not str(frozen.get("status", "")).startswith("PASS"):
        raise ValueError("frozen model verification report is not PASS")
    result = verify_model_map(
        suite_model_map.resolve(),
        suite_model_report.resolve(),
        goal_model_manifest.resolve(),
    )
    if not str(result.get("status", "")).startswith("PASS"):
        raise RuntimeError("strict suite model verification did not PASS")
    path = output_root / "c2g_r8w_model_verification_once.json"
    write_json(path, {
        **result,
        "frozen_verification_report": str(frozen_verification_report.resolve()),
        "frozen_verification_report_sha256": sha256_file(frozen_verification_report.resolve()),
    })
    return path, sha256_file(path)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def storage_estimate(r8t_collection_root: Path) -> dict[str, Any]:
    episode_dirs = [path.parent for path in r8t_collection_root.rglob("episode_metadata.json")]
    sizes = sorted(directory_bytes(path) for path in episode_dirs)
    if len(sizes) < 8:
        raise ValueError(f"R8T storage estimate requires at least 8 episodes, got {len(sizes)}")
    p95_index = max(0, math.ceil(0.95 * len(sizes)) - 1)
    p95 = sizes[p95_index]
    file_counts = sorted(sum(1 for item in path.rglob("*") if item.is_file()) for path in episode_dirs)
    p95_files = file_counts[p95_index]
    return {
        "source_episode_count": len(sizes),
        "mean_episode_bytes": statistics.fmean(sizes),
        "p95_episode_bytes": p95,
        "p95_episode_file_count": p95_files,
        "estimated_required_bytes": math.ceil(p95 * 2000 * 1.20),
        "estimated_required_files": math.ceil(p95_files * 2000 * 1.20),
    }


def storage_preflight(target_root: Path, estimate: Mapping[str, Any]) -> dict[str, Any]:
    anchor = target_root.resolve()
    while not anchor.exists():
        if anchor.parent == anchor:
            raise FileNotFoundError(target_root)
        anchor = anchor.parent
    usage = shutil.disk_usage(anchor)
    stats = os.statvfs(anchor)
    available_inodes = int(stats.f_favail)
    required_bytes = int(estimate["estimated_required_bytes"])
    required_files = int(estimate["estimated_required_files"])
    return {
        "filesystem_anchor": str(anchor),
        "available_bytes": usage.free,
        "available_inodes": available_inodes,
        "required_bytes": required_bytes,
        "required_inodes_with_headroom": math.ceil(required_files * 1.20),
        "bytes_pass": usage.free > required_bytes,
        "inodes_pass": available_inodes > math.ceil(required_files * 1.20),
    }


def host_snapshot() -> dict[str, Any]:
    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            memory[key] = value.strip()
    except Exception as exc:
        memory = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        nofile = subprocess.check_output(["bash", "-lc", "ulimit -n"], text=True).strip()
    except Exception as exc:
        nofile = f"UNRESOLVED: {exc}"
    return {"memory": memory, "ulimit_nofile": nofile}


def read_status(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def phase_is_resident(phase: str) -> bool:
    return phase in {"MODEL_READY", "CREATING_ENVIRONMENT", "RUNNING_EPISODES", "FINALIZING", "PASS"}


def worker_failure_policy(returncode: int, valid_receipt: bool) -> dict[str, bool]:
    failed = returncode != 0 or not valid_receipt
    return {
        "worker_failed": failed,
        "stop_new_launches": failed,
        "preserve_worker_output": True,
        "allow_other_running_workers_to_finish": True,
    }


def acquire_owner_locks() -> list[Any]:
    import fcntl

    handles = []
    try:
        for gpu in GPUS:
            path = Path(f"/tmp/c2g_r8w_gpu_{gpu}.owner.lock")
            handle = path.open("a+", encoding="utf-8")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "gpu": gpu, "timestamp": utc_now()}) + "\n")
            handle.flush()
            handles.append(handle)
        return handles
    except Exception:
        for handle in handles:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        raise


def release_owner_locks(handles: Sequence[Any]) -> None:
    import fcntl

    for handle in handles:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def write_heartbeat(
    path: Path,
    states: Mapping[str, Mapping[str, Any]],
    pending: Sequence[Mapping[str, Any]],
) -> None:
    phases = CounterLike(state.get("phase", "CREATED") for state in states.values())
    snapshots = nvidia_snapshot()
    value = {
        "timestamp": utc_now(),
        "workers_created": len(states),
        "workers_waiting": len(pending) + phases.get("WAITING_MODEL_LOAD_LOCK", 0),
        "workers_loading": phases.get("LOADING_PROCESSOR", 0) + phases.get("LOADING_MODEL", 0),
        "workers_ready": phases.get("MODEL_READY", 0),
        "workers_running": phases.get("RUNNING_EPISODES", 0) + phases.get("CREATING_ENVIRONMENT", 0),
        "workers_completed": phases.get("PASS", 0),
        "workers_failed": phases.get("FAILED", 0),
        "per_gpu": {
            str(gpu): {
                **asdict(snapshot),
                "resident_worker_count": sum(
                    phase_is_resident(str(state.get("phase", "")))
                    and int(state.get("physical_gpu", -1)) == gpu
                    for state in states.values()
                ),
            }
            for gpu, snapshot in snapshots.items()
        },
    }
    write_json(path, value)


def CounterLike(values: Sequence[str] | Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def collect_stable_samples(
    gpu: int,
    *,
    sample_count: int,
    interval_seconds: float,
    snapshot_fn: Callable[[Sequence[int]], dict[int, GpuSnapshot]] = nvidia_snapshot,
) -> list[GpuSnapshot]:
    samples = []
    for index in range(sample_count):
        samples.append(snapshot_fn([gpu])[gpu])
        if index + 1 < sample_count:
            time.sleep(interval_seconds)
    return samples


def run_canary_scheduler(
    *,
    mode: str,
    plan_report: Path,
    expected_plan_report_sha256: str,
    output_root: Path,
    suite_model_map: Path,
    suite_model_report: Path,
    goal_model_manifest: Path,
    model_verification_report: Path,
    authorization: str,
    poll_seconds: float = 5.0,
    stable_sample_interval: float = 20.0,
) -> dict[str, Any]:
    if mode not in {"canary-preview", "canary-run"}:
        raise ValueError("invalid canary scheduler mode")
    plan = load_plan(plan_report, expected_plan_report_sha256)
    if git_output("rev-parse", "HEAD") != plan["expected_git_commit"]:
        raise RuntimeError("canary scheduler HEAD differs from plan HEAD")
    if git_output("status", "--porcelain"):
        raise RuntimeError("canary scheduler requires a clean worktree")
    order = validate_canary_layout(plan)
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    before = nvidia_snapshot()
    planned_verification_report = output_root / "c2g_r8w_model_verification_once.json"
    commands = {
        str(shard["worker_id"]): worker_command(
            plan_report=plan_report,
            plan_sha256=expected_plan_report_sha256,
            shard=shard,
            output_root=output_root,
            suite_model_map=suite_model_map,
            suite_model_report=suite_model_report,
            goal_model_manifest=goal_model_manifest,
            model_verification_report=planned_verification_report,
            resume=False,
        )
        for shard in order
    }
    preview = {
        "schema": SCHEMA,
        "status": CANARY_PREVIEW_STATUS,
        "mode": mode,
        "git_head": plan["expected_git_commit"],
        "plan_report": str(plan_report.resolve()),
        "plan_report_sha256": expected_plan_report_sha256,
        "output_root": str(output_root),
        "worker_count": 4,
        "episode_count": 8,
        "worker_mapping": [
            {"worker_id": row["worker_id"], "physical_gpu": row["physical_gpu"], "suite": row["suite"], "episode_count": 2}
            for row in order
        ],
        "gpu_snapshot_before": {str(gpu): asdict(value) for gpu, value in before.items()},
        "model_loading_order": [row["worker_id"] for row in order],
        "commands": commands,
        "attacks": 0,
        "training_epochs": 0,
        "materialization_runs": 0,
    }
    if mode == "canary-preview":
        return preview
    if authorization != AUTHORIZATION_TOKEN:
        raise PermissionError("R8W canary scheduler authorization mismatch")
    output_root.mkdir(parents=True)
    (output_root / "logs").mkdir()
    actual_verification_report, actual_verification_sha = verify_models_once(
        output_root,
        suite_model_map,
        suite_model_report,
        goal_model_manifest,
        model_verification_report,
    )
    locks = acquire_owner_locks()
    running: dict[str, dict[str, Any]] = {}
    receipts = []
    failures = []
    prelaunch_free = before[4].memory_free_mib
    min_free_gpu4 = prelaunch_free
    model_ready_free_gpu4: int | None = None
    after_first_env_free_gpu4: int | None = None
    try:
        for shard in order:
            gpu = int(shard["physical_gpu"])
            samples = collect_stable_samples(
                gpu,
                sample_count=3,
                interval_seconds=stable_sample_interval,
            )
            admitted, reason = stable_admission(samples, 18000)
            if not admitted:
                failures.append({"worker_id": shard["worker_id"], "reason": reason})
                break
            wid = str(shard["worker_id"])
            stdout = (output_root / "logs" / f"{wid}.stdout.log").open("w", encoding="utf-8")
            stderr = (output_root / "logs" / f"{wid}.stderr.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                commands[wid],
                cwd=REPO,
                stdout=stdout,
                stderr=stderr,
                env={**os.environ, "R8W_COLLECTION_AUTHORIZATION": AUTHORIZATION_TOKEN},
            )
            state = {
                "process": process,
                "stdout": stdout,
                "stderr": stderr,
                "physical_gpu": gpu,
                "status_file": output_root / "workers" / wid / "worker_status.json",
                "output_root": output_root / "workers" / wid,
                "phase": "CREATED",
            }
            running[wid] = state
            deadline = time.time() + 3600
            while time.time() < deadline and process.poll() is None:
                status = read_status(state["status_file"])
                phase = str(status.get("phase", ""))
                if phase:
                    state["phase"] = phase
                snapshots = nvidia_snapshot([gpu])
                if gpu == 4:
                    free = snapshots[4].memory_free_mib
                    min_free_gpu4 = min(min_free_gpu4, free)
                    if phase_is_resident(phase) and model_ready_free_gpu4 is None:
                        model_ready_free_gpu4 = free
                    if phase in {"CREATING_ENVIRONMENT", "RUNNING_EPISODES"} and after_first_env_free_gpu4 is None:
                        after_first_env_free_gpu4 = free
                if phase_is_resident(phase):
                    break
                if phase == "FAILED":
                    break
                time.sleep(poll_seconds)
            if not phase_is_resident(str(state.get("phase", ""))):
                failures.append({"worker_id": wid, "reason": "MODEL_READY_NOT_REACHED", "phase": state.get("phase")})
                break

        while running:
            for wid, state in list(running.items()):
                status = read_status(state["status_file"])
                if status:
                    state["phase"] = status.get("phase", state["phase"])
                if state["physical_gpu"] == 4:
                    min_free_gpu4 = min(min_free_gpu4, nvidia_snapshot([4])[4].memory_free_mib)
                returncode = state["process"].poll()
                if returncode is None:
                    continue
                state["stdout"].close()
                state["stderr"].close()
                receipt_path = state["output_root"] / "worker_receipt.json"
                if returncode == 0 and receipt_path.is_file():
                    value = read_status(receipt_path)
                    if value.get("schema") == RECEIPT_SCHEMA and value.get("status") == RUN_STATUS:
                        receipts.append({"worker_id": wid, "receipt": str(receipt_path), "receipt_sha256": sha256_file(receipt_path)})
                    else:
                        failures.append({"worker_id": wid, "reason": "INVALID_WORKER_RECEIPT"})
                else:
                    failures.append({"worker_id": wid, "reason": "WORKER_FAILED", "returncode": returncode})
                del running[wid]
            if running:
                time.sleep(poll_seconds)
    finally:
        for state in running.values():
            state["stdout"].close()
            state["stderr"].close()
        release_owner_locks(locks)
    model_ready_free_gpu4 = model_ready_free_gpu4 or min_free_gpu4
    after_first_env_free_gpu4 = after_first_env_free_gpu4 or min_free_gpu4
    observed = max(
        prelaunch_free - min_free_gpu4,
        prelaunch_free - model_ready_free_gpu4,
    )
    budget = measured_worker_budget_mib(float(max(observed, 1)))
    status = CANARY_RUN_STATUS if len(receipts) == 4 and not failures else CANARY_HOLD_STATUS
    report = {
        **preview,
        "status": status,
        "mode": "canary-run",
        "worker_receipts": receipts,
        "worker_receipts_pass": len(receipts),
        "worker_failures": failures,
        "calibration": {
            "gpu": 4,
            "prelaunch_free_mib": prelaunch_free,
            "model_ready_free_mib": model_ready_free_gpu4,
            "after_first_env_free_mib": after_first_env_free_gpu4,
            "minimum_free_mib_observed": min_free_gpu4,
            "observed_worker_mib": observed,
            "worker_budget_mib": budget,
            "gpu_reserve_mib": GPU_RESERVE_MIB,
        },
        "model_verification_report": str(actual_verification_report),
        "model_verification_report_sha256": actual_verification_sha,
        "gpu_snapshot_after": {str(gpu): asdict(value) for gpu, value in nvidia_snapshot().items()},
    }
    report_path = output_root / "c2g_r8w_fresh_canary_scheduler_report.json"
    write_json(report_path, report)
    return {**report, "report": str(report_path), "report_sha256": sha256_file(report_path)}


def run_scheduler(
    *,
    mode: str,
    plan_report: Path,
    expected_plan_report_sha256: str,
    output_root: Path,
    suite_model_map: Path,
    suite_model_report: Path,
    goal_model_manifest: Path,
    model_verification_report: Path,
    r8t_collection_root: Path,
    worker_budget_mib: int,
    authorization: str,
    resume: bool,
    poll_seconds: float = 20.0,
    stable_sample_interval: float = 20.0,
) -> dict[str, Any]:
    if mode not in {"preview", "run"}:
        raise ValueError("mode must be preview or run")
    plan = load_plan(plan_report, expected_plan_report_sha256)
    if git_output("rev-parse", "HEAD") != plan["expected_git_commit"]:
        raise RuntimeError("scheduler HEAD differs from plan HEAD")
    if git_output("status", "--porcelain"):
        raise RuntimeError("scheduler requires a clean worktree")
    load_order = validate_worker_layout(plan)
    output_root = output_root.resolve()
    if output_root.exists() and not resume:
        raise FileExistsError(output_root)
    gpu_before = nvidia_snapshot()
    estimate = storage_estimate(r8t_collection_root.resolve())
    storage = storage_preflight(output_root, estimate)
    if not storage["bytes_pass"] or not storage["inodes_pass"]:
        raise RuntimeError(f"R8W storage preflight failed: {storage}")
    caps = {
        gpu: safe_resident_workers(gpu_before[gpu].memory_total_mib, worker_budget_mib)
        for gpu in GPUS
    }
    if any(cap < 1 for cap in caps.values()):
        raise RuntimeError(f"no safe resident worker slot on one or more GPUs: {caps}")
    planned_verification_report = output_root / "c2g_r8w_model_verification_once.json"
    commands = {
        str(shard["worker_id"]): worker_command(
            plan_report=plan_report,
            plan_sha256=expected_plan_report_sha256,
            shard=shard,
            output_root=output_root,
            suite_model_map=suite_model_map,
            suite_model_report=suite_model_report,
            goal_model_manifest=goal_model_manifest,
            model_verification_report=planned_verification_report,
            resume=resume,
        )
        for shard in load_order
    }
    preview = {
        "schema": SCHEMA,
        "status": PREVIEW_STATUS,
        "mode": mode,
        "git_head": plan["expected_git_commit"],
        "plan_report": str(plan_report.resolve()),
        "plan_report_sha256": expected_plan_report_sha256,
        "output_root": str(output_root),
        "worker_count": 16,
        "requested_resident_workers_per_gpu": TARGET_RESIDENT_WORKERS_PER_GPU,
        "safe_resident_workers_per_gpu": {str(gpu): caps[gpu] for gpu in GPUS},
        "achieved_resident_workers_per_gpu": min(caps.values()),
        "concurrency_degraded": min(caps.values()) < TARGET_RESIDENT_WORKERS_PER_GPU,
        "concurrency_degradation_reason": (
            "MEASURED_MEMORY_ADMISSION" if min(caps.values()) < TARGET_RESIDENT_WORKERS_PER_GPU else "NONE"
        ),
        "measured_worker_budget_mib": worker_budget_mib,
        "gpu_reserve_mib": GPU_RESERVE_MIB,
        "gpu_snapshot": {str(gpu): asdict(snapshot) for gpu, snapshot in gpu_before.items()},
        "host_snapshot": host_snapshot(),
        "storage_estimate": estimate,
        "storage_preflight": storage,
        "model_loading_order": [str(shard["worker_id"]) for shard in load_order],
        "commands": commands,
        "attacks": 0,
        "training_epochs": 0,
        "materialization_runs": 0,
    }
    if mode == "preview":
        return preview
    if authorization != AUTHORIZATION_TOKEN:
        raise PermissionError("R8W scheduler authorization mismatch")
    output_root.mkdir(parents=True, exist_ok=resume)
    (output_root / "logs").mkdir(exist_ok=True)
    actual_verification_report, actual_verification_sha = verify_models_once(
        output_root,
        suite_model_map,
        suite_model_report,
        goal_model_manifest,
        model_verification_report,
    )
    owner_locks = acquire_owner_locks()
    running_by_worker: dict[str, dict[str, Any]] = {}
    running_by_gpu: dict[int, list[str]] = {gpu: [] for gpu in GPUS}
    completed_receipts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    pending = list(load_order)
    stop_launching = False
    heartbeat = output_root / "scheduler_heartbeat.json"
    last_heartbeat = 0.0
    try:
        while pending or running_by_worker:
            for wid, state in list(running_by_worker.items()):
                status = read_status(state["status_file"])
                if status:
                    state["phase"] = status.get("phase", state["phase"])
                returncode = state["process"].poll()
                if returncode is None:
                    continue
                state["stdout"].close()
                state["stderr"].close()
                gpu = state["physical_gpu"]
                running_by_gpu[gpu].remove(wid)
                receipt_path = state["output_root"] / "worker_receipt.json"
                if returncode == 0 and receipt_path.is_file():
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("status") != RUN_STATUS:
                        failures.append({"worker_id": wid, "reason": "INVALID_WORKER_RECEIPT"})
                        stop_launching = True
                    else:
                        completed_receipts.append({
                            "worker_id": wid,
                            "receipt": str(receipt_path),
                            "receipt_sha256": sha256_file(receipt_path),
                        })
                else:
                    failures.append({"worker_id": wid, "returncode": returncode, "reason": "WORKER_FAILED"})
                    stop_launching = True
                del running_by_worker[wid]

            if pending and not stop_launching:
                shard = pending[0]
                wid = str(shard["worker_id"])
                gpu = int(shard["physical_gpu"])
                resident = sum(
                    phase_is_resident(str(running_by_worker[other].get("phase", "")))
                    for other in running_by_gpu[gpu]
                )
                if resident < caps[gpu]:
                    samples = collect_stable_samples(
                        gpu,
                        sample_count=3,
                        interval_seconds=stable_sample_interval,
                    )
                    admitted, reason = stable_admission(samples, worker_budget_mib)
                    if admitted:
                        pending.pop(0)
                        worker_root = output_root / "workers" / wid
                        stdout_path = output_root / "logs" / f"{wid}.stdout.log"
                        stderr_path = output_root / "logs" / f"{wid}.stderr.log"
                        stdout = stdout_path.open("a" if resume else "w", encoding="utf-8")
                        stderr = stderr_path.open("a" if resume else "w", encoding="utf-8")
                        process = subprocess.Popen(
                            commands[wid],
                            cwd=REPO,
                            stdout=stdout,
                            stderr=stderr,
                            env={**os.environ, "R8W_COLLECTION_AUTHORIZATION": AUTHORIZATION_TOKEN},
                        )
                        state = {
                            "worker_id": wid,
                            "physical_gpu": gpu,
                            "process": process,
                            "stdout": stdout,
                            "stderr": stderr,
                            "output_root": worker_root,
                            "status_file": worker_root / "worker_status.json",
                            "phase": "CREATED",
                        }
                        running_by_worker[wid] = state
                        running_by_gpu[gpu].append(wid)
                        deadline = time.time() + 3600
                        while time.time() < deadline and process.poll() is None:
                            status = read_status(state["status_file"])
                            phase = str(status.get("phase", ""))
                            if phase:
                                state["phase"] = phase
                            if phase_is_resident(phase):
                                break
                            if phase == "FAILED":
                                break
                            time.sleep(min(poll_seconds, 5.0))
                        if not phase_is_resident(str(state.get("phase", ""))):
                            failures.append({
                                "worker_id": wid,
                                "reason": "MODEL_READY_NOT_REACHED",
                                "last_phase": state.get("phase"),
                            })
                            stop_launching = True
                    elif reason not in {"INSUFFICIENT_FREE_MEMORY", "GPU_UTILIZATION_ABOVE_40_PERCENT", "FREE_MEMORY_NOT_STABLE"}:
                        failures.append({"worker_id": wid, "reason": reason})
                        stop_launching = True

            now = time.time()
            if now - last_heartbeat >= 60:
                write_heartbeat(heartbeat, running_by_worker, pending)
                last_heartbeat = now
            if running_by_worker or (pending and not stop_launching):
                time.sleep(poll_seconds)
            elif pending and stop_launching:
                break
    finally:
        for state in running_by_worker.values():
            state["stdout"].close()
            state["stderr"].close()
        release_owner_locks(owner_locks)

    status = RUN_STATUS_SCHEDULER if len(completed_receipts) == 16 and not failures and not pending else HOLD_STATUS
    report = {
        **preview,
        "status": status,
        "mode": "run",
        "worker_receipts": completed_receipts,
        "worker_receipts_pass": len(completed_receipts),
        "worker_failures": failures,
        "unlaunched_workers": [row["worker_id"] for row in pending],
        "model_verification_report": str(actual_verification_report),
        "model_verification_report_sha256": actual_verification_sha,
        "gpu_snapshot_after": {str(gpu): asdict(value) for gpu, value in nvidia_snapshot().items()},
    }
    report_path = output_root / "c2g_r8w_16worker_scheduler_report.json"
    write_json(report_path, report)
    return {**report, "report": str(report_path), "report_sha256": sha256_file(report_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preview", "run", "canary-preview", "canary-run"))
    parser.add_argument("--plan-report", type=Path, required=True)
    parser.add_argument("--expected-plan-report-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suite-model-map", type=Path, required=True)
    parser.add_argument("--suite-model-report", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path, required=True)
    parser.add_argument("--model-verification-report", type=Path, required=True)
    parser.add_argument("--r8t-collection-root", type=Path)
    parser.add_argument("--worker-budget-mib", type=int)
    parser.add_argument("--authorization", default=os.environ.get("R8W_COLLECTION_AUTHORIZATION", ""))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--stable-sample-interval", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    common = dict(
        plan_report=args.plan_report,
        expected_plan_report_sha256=args.expected_plan_report_sha256,
        output_root=args.output_root,
        suite_model_map=args.suite_model_map,
        suite_model_report=args.suite_model_report,
        goal_model_manifest=args.goal_model_manifest,
        model_verification_report=args.model_verification_report,
        authorization=args.authorization,
        poll_seconds=args.poll_seconds,
        stable_sample_interval=args.stable_sample_interval,
    )
    if args.mode.startswith("canary-"):
        if args.resume:
            raise ValueError("fresh shadow canary cannot resume")
        result = run_canary_scheduler(mode=args.mode, **common)
    else:
        if args.r8t_collection_root is None or args.worker_budget_mib is None:
            raise ValueError("full scheduler mode requires R8T storage root and worker budget")
        result = run_scheduler(
            mode=args.mode,
            r8t_collection_root=args.r8t_collection_root,
            worker_budget_mib=args.worker_budget_mib,
            resume=args.resume,
            **common,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {
        PREVIEW_STATUS, RUN_STATUS_SCHEDULER, CANARY_PREVIEW_STATUS, CANARY_RUN_STATUS,
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
