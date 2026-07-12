#!/usr/bin/env python3
"""Dynamic GPU-admission scheduler for R8Y L10-520 collection.

Supports two modes with separate authorization:
  canary-run  → R8Y_L10_520_CANARY_AUTHORIZED  (12-episode shadow)
  full-run    → R8Y_L10_520_FULL500_COLLECTION_AUTHORIZED (500-episode)

Key design:
  - Per-GPU resident cap starts at 2; upgrades to 3 after calibration.
  - Worker status polled from worker_status.json every 2 s.
  - loading_worker released when worker reaches MODEL_READY.
  - Model-load serialization via global lock file.
  - All worker stdout/stderr written to log files (no pipe deadlock).
  - PASS requires all shards completed, 0 failed, 0 pending.
  - Git head frozen from plan report (not re-read per worker).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.multisuite_detector.c2g_official_suite_horizons import (
    OFFICIAL_DUMMY_WAIT_STEPS,
    OFFICIAL_MAX_POLICY_STEPS,
)

# ── constants ──────────────────────────────────────────────────────────
GPUS = (4, 5, 6, 7)
TARGET_SUITE = "libero_10"
SHARD_RUNNER = REPO / "scripts" / "stageb" / "run_c2g_r8y_l10_520_shard.py"
MODEL_LOAD_LOCK = Path("/tmp/c2g_r8y_global_model_load.lock")

SCHEMA = "c2g.r8y.l10_520_dynamic_scheduler.2026-07-12.v1"
CANARY_PASS_STATUS = "PASS_C2G_R8Y_L10_520_CANARY"
FULL_PASS_STATUS = "PASS_C2G_R8Y_L10_520_FULL500"

CANARY_AUTH = "R8Y_L10_520_CANARY_AUTHORIZED"
FULL_AUTH = "R8Y_L10_520_FULL500_COLLECTION_AUTHORIZED"

# Memory admission
ABSOLUTE_MIN_FREE_MIB = 16384
FALLBACK_WORKER_BUDGET_MIB = 18432
GPU_POST_LAUNCH_RESERVE_MIB = 8192
MODEL_LOAD_TRANSIENT_MARGIN_MIB = 2048
INITIAL_RESIDENT_CAP = 2
MAX_RESIDENT_CAP = 3
STABLE_POLL_COUNT = 3
STABLE_POLL_INTERVAL_S = 5

# Resident phases (model loaded, occupying GPU memory)
RESIDENT_PHASES = frozenset({
    "MODEL_READY", "CREATING_ENVIRONMENT", "RUNNING_EPISODES", "FINALIZING",
})
# Loading phases (acquiring model, must be serialized)
LOADING_PHASES = frozenset({
    "WAITING_MODEL_LOAD_LOCK", "LOADING_PROCESSOR", "LOADING_MODEL",
})


# ── data classes ───────────────────────────────────────────────────────
@dataclass
class GpuSnapshot:
    index: int
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int
    utilization_percent: int
    temperature_c: int


@dataclass
class WorkerState:
    worker_id: str
    shard_id: str
    gpu: int
    pid: int = 0
    phase: str = "CREATED"
    launch_time: str = ""
    completion_time: str = ""
    returncode: int | None = None
    episode_count: int = 0


@dataclass
class CalibrationState:
    gpu: int
    observed_deltas_mib: list[float] = field(default_factory=list)
    oom_count: int = 0
    calibrated_budget_mib: int = FALLBACK_WORKER_BUDGET_MIB
    gpu_uuid: str = ""
    gpu_name: str = ""
    driver_version: str = ""


# ── helpers ────────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def ceil_to_1024(value: float) -> int:
    return int(math.ceil(value / 1024.0) * 1024)


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
        parts = [v.strip() for v in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            vals = [int(p) for p in parts]
        except (ValueError, TypeError):
            continue
        if vals[0] not in requested:
            continue
        snapshots[vals[0]] = GpuSnapshot(*vals)
    missing = requested - snapshots.keys()
    if missing:
        raise RuntimeError(f"nvidia-smi did not report GPUs: {sorted(missing)}")
    return snapshots


def compute_admission_threshold(calibrated_budget_mib: int) -> int:
    return max(
        ABSOLUTE_MIN_FREE_MIB,
        calibrated_budget_mib + GPU_POST_LAUNCH_RESERVE_MIB + MODEL_LOAD_TRANSIENT_MARGIN_MIB,
    )


def memory_admission_pass(
    snapshot: GpuSnapshot,
    calibrated_budget_mib: int,
) -> tuple[bool, str]:
    """Check if a GPU has enough free memory to admit another worker."""
    threshold = compute_admission_threshold(calibrated_budget_mib)
    if snapshot.memory_free_mib < threshold:
        return False, (
            f"FREE_{snapshot.memory_free_mib}_LT_THRESHOLD_{threshold}"
        )
    return True, "PASS"


def stable_admission_pass(
    samples: list[GpuSnapshot],
    calibrated_budget_mib: int,
) -> tuple[bool, str]:
    """Require 3 stable polls, all above threshold."""
    if len(samples) != STABLE_POLL_COUNT:
        return False, f"NEED_{STABLE_POLL_COUNT}_SAMPLES_GOT_{len(samples)}"
    gpu_ids = {s.index for s in samples}
    if len(gpu_ids) != 1:
        return False, "MIXED_GPU_SAMPLES"
    for i, sample in enumerate(samples):
        ok, reason = memory_admission_pass(sample, calibrated_budget_mib)
        if not ok:
            return False, f"POLL_{i}_{reason}"
    free_vals = [s.memory_free_mib for s in samples]
    if max(free_vals) - min(free_vals) > 1024:
        return False, "FREE_MEMORY_NOT_STABLE"
    return True, "PASS"


# ── plan loading ───────────────────────────────────────────────────────
def load_plan(plan_report_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = read_json(plan_report_path)
    shard_index_path = Path(report.get("shard_index", ""))
    if not shard_index_path.is_file():
        raise FileNotFoundError(f"shard index not found: {shard_index_path}")
    index_data = read_json(shard_index_path)
    shards = index_data.get("shards", [])
    if not shards:
        raise ValueError("shard index contains no shards")
    return report, [dict(s) for s in shards]


# ── scheduler core ─────────────────────────────────────────────────────
def run_scheduler(
    *,
    mode: str,  # "canary-run" or "full-run"
    plan_report: Path,
    expected_plan_report_sha256: str,
    output_root: Path,
    suite_model_map: Path,
    suite_model_report: Path,
    goal_model_manifest: Path,
    model_verification_report: Path,
    authorization: str,
    cap_override: str = "",
    canary_report: Path | None = None,
    expected_canary_report_sha256: str | None = None,
    canary_ledger: Path | None = None,
    expected_canary_ledger_sha256: str | None = None,
) -> dict[str, Any]:
    # ── authorization (P0-1) ────────────────────────────────────────
    plan_report = plan_report.resolve()
    if sha256_file(plan_report) != expected_plan_report_sha256:
        raise ValueError("plan report SHA256 mismatch")

    if mode == "canary-run":
        if authorization != CANARY_AUTH:
            raise PermissionError(f"canary-run requires {CANARY_AUTH}")
    elif mode == "full-run":
        if authorization != FULL_AUTH:
            raise PermissionError(f"full-run requires {FULL_AUTH}")
    else:
        raise ValueError("mode must be canary-run or full-run")

    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True)

    plan, shards = load_plan(plan_report)

    # Frozen git head from plan (P0-8)
    frozen_head = str(plan.get("expected_git_commit", ""))
    if not frozen_head or len(frozen_head) != 40:
        raise ValueError("plan report missing valid expected_git_commit")
    current_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    if current_head != frozen_head:
        raise ValueError(
            f"current HEAD {current_head[:10]} != plan expected {frozen_head[:10]}"
        )

    is_canary = mode == "canary-run"
    total_episodes = sum(int(s.get("episode_count", 0)) for s in shards)
    target_shard_count = len(shards)
    eps_per_shard = {int(s.get("episode_count", 0)) for s in shards}
    shards_per_gpu = {gpu: sum(1 for s in shards if int(s.get("physical_gpu", -1)) == gpu) for gpu in GPUS}
    plan_kind = str(plan.get("plan_kind", ""))

    # ── mode-specific shape validation (Gate 2) ──────────────────────
    if is_canary:
        if total_episodes != 12 or target_shard_count != 12:
            raise ValueError(
                f"canary-run requires exactly 12 episodes in 12 shards, "
                f"got {total_episodes} episodes in {target_shard_count} shards"
            )
        if eps_per_shard != {1}:
            raise ValueError(
                f"canary-run requires 1 episode per shard, got {eps_per_shard}"
            )
        if any(v != 3 for v in shards_per_gpu.values()):
            raise ValueError(
                f"canary-run requires 3 shards per GPU, got {shards_per_gpu}"
            )
    else:  # full-run
        if total_episodes != 500 or target_shard_count != 20:
            raise ValueError(
                f"full-run requires exactly 500 episodes in 20 shards, "
                f"got {total_episodes} episodes in {target_shard_count} shards"
            )
        if eps_per_shard != {25}:
            raise ValueError(
                f"full-run requires 25 episodes per shard, got {eps_per_shard}"
            )
        if any(v != 5 for v in shards_per_gpu.values()):
            raise ValueError(
                f"full-run requires 5 shards per GPU, got {shards_per_gpu}"
            )
        # ── canary proof for full-run (Gate 1) ───────────────────────
        if canary_report is None or expected_canary_report_sha256 is None:
            raise ValueError(
                "full-run requires --canary-report and "
                "--expected-canary-report-sha256"
            )
        canary_report = canary_report.resolve()
        if sha256_file(canary_report) != expected_canary_report_sha256:
            raise ValueError("canary report SHA256 mismatch")
        cr = read_json(canary_report)
        if str(cr.get("status", "")) != "PASS_C2G_R8Y_L10_520_CANARY":
            raise ValueError(
                f"canary report not PASS: {cr.get('status', '?')}"
            )
        if int(cr.get("runtime_valid", 0)) != 12:
            raise ValueError("canary runtime not 12/12")
        for field in ("raw_action_prefix_exact", "applied_action_prefix_exact",
                       "features_25d_exact_or_equivalent", "success_agreement"):
            if int(cr.get(field, 0)) < 8:
                raise ValueError(f"canary {field} < 8/8")
        if int(cr.get("oom_count", 1)) != 0:
            raise ValueError("canary had OOM events")
        if int(cr.get("gpu_migration_count", 1)) != 0:
            raise ValueError("canary had GPU migrations")
        canary_git = str(cr.get("frozen_git_head", ""))
        if canary_git and canary_git != frozen_head:
            raise ValueError(
                f"canary git head {canary_git[:10]} != full plan {frozen_head[:10]}"
            )
        if canary_ledger is not None and expected_canary_ledger_sha256 is not None:
            canary_ledger = canary_ledger.resolve()
            if sha256_file(canary_ledger) != expected_canary_ledger_sha256:
                raise ValueError("canary ledger SHA256 mismatch")

    # Build pending queues per GPU (permanently bound)
    pending_by_gpu: dict[int, deque[dict[str, Any]]] = {gpu: deque() for gpu in GPUS}
    for shard in shards:
        gpu = int(shard["physical_gpu"])
        pending_by_gpu[gpu].append(shard)

    # State tracking
    workers: dict[str, WorkerState] = {}  # worker_id → WorkerState
    worker_processes: dict[str, subprocess.Popen] = {}  # worker_id → subprocess
    _pre_launch_free: dict[str, float] = {}  # worker_id → free_mib before launch
    loading_worker_id: str | None = None
    completed_workers: list[str] = []
    failed_workers: list[str] = []

    # Calibration and caps
    cal_by_gpu: dict[int, CalibrationState] = {gpu: CalibrationState(gpu=gpu) for gpu in GPUS}
    effective_cap_by_gpu: dict[int, int] = {gpu: INITIAL_RESIDENT_CAP for gpu in GPUS}

    # Apply cap override if provided (e.g. "4=2,5=2,6=2,7=1")
    if cap_override:
        for pair in cap_override.split(","):
            pair = pair.strip()
            if not pair:
                continue
            gpu_str, cap_str = pair.split("=")
            gpu = int(gpu_str)
            cap = int(cap_str)
            if gpu not in GPUS or cap < 1 or cap > MAX_RESIDENT_CAP:
                raise ValueError(f"invalid cap override: {pair}")
            effective_cap_by_gpu[gpu] = cap

    # Collect GPU metadata
    for gpu in GPUS:
        try:
            uuid_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,name,driver_version",
                 "--format=csv,noheader", f"--id={gpu}"],
                text=True, timeout=10,
            ).strip()
            parts = [p.strip() for p in uuid_out.split(",")]
            if len(parts) >= 3:
                cal_by_gpu[gpu].gpu_uuid = parts[0]
                cal_by_gpu[gpu].gpu_name = parts[1]
                cal_by_gpu[gpu].driver_version = parts[2]
        except Exception:
            pass

    heartbeat_path = output_root / "scheduler_heartbeat.json"
    report_path = output_root / "c2g_r8y_l10_520_scheduler_report.json"

    def write_heartbeat() -> None:
        per_gpu: dict[str, dict[str, Any]] = {}
        try:
            snaps = nvidia_snapshot(GPUS)
        except Exception:
            snaps = {}
        for gpu in GPUS:
            s = snaps.get(gpu)
            gpu_workers = [w for w in workers.values() if w.gpu == gpu]
            resident = sum(1 for w in gpu_workers if w.phase in RESIDENT_PHASES)
            per_gpu[str(gpu)] = {
                "index": gpu,
                "memory_free_mib": s.memory_free_mib if s else -1,
                "utilization_percent": s.utilization_percent if s else -1,
                "resident_worker_count": resident,
                "effective_cap": effective_cap_by_gpu[gpu],
            }
        write_json(heartbeat_path, {
            "timestamp": utc_now(),
            "mode": mode,
            "per_gpu": per_gpu,
            "pending_shards": {str(g): len(pending_by_gpu[g]) for g in GPUS},
            "loading_worker": loading_worker_id,
            "completed_workers": len(completed_workers),
            "failed_workers": len(failed_workers),
            "target_shard_count": target_shard_count,
        })

    def poll_worker_status() -> None:
        """Read worker_status.json for each running worker (P0-2)."""
        nonlocal loading_worker_id

        for wid, ws in list(workers.items()):
            if ws.phase in ("PASS", "FAILED", "ABORTED"):
                continue

            status_path = output_root / "workers" / wid / "worker_status.json"
            if not status_path.is_file():
                continue

            status = read_json(status_path)
            phase = str(status.get("phase", ""))
            if not phase or phase == ws.phase:
                continue

            old_phase = ws.phase
            ws.phase = phase

            # Release loading slot as soon as worker passes model loading.
            # Polls may skip fast phases (MODEL_READY), so trigger on any
            # post-load phase regardless of where we last observed it.
            if loading_worker_id == wid and phase in RESIDENT_PHASES:
                loading_worker_id = None
                # Record real post-load memory delta
                pre_load_free = _pre_launch_free.get(wid)
                try:
                    snap = nvidia_snapshot([ws.gpu])[ws.gpu]
                    if pre_load_free is not None:
                        delta = pre_load_free - snap.memory_free_mib
                        if 5000 < delta < 100000:  # sanity bounds
                            cal = cal_by_gpu[ws.gpu]
                            cal.observed_deltas_mib.append(float(delta))
                except Exception:
                    pass

            # Check for OOM in phase
            if phase == "FAILED" and old_phase in LOADING_PHASES:
                cal = cal_by_gpu[ws.gpu]
                cal.oom_count += 1

    def poll_processes() -> None:
        """Check completed subprocesses (P0-4: stderr→log file)."""
        nonlocal loading_worker_id
        for wid, proc in list(worker_processes.items()):
            ret = proc.poll()
            if ret is None:
                continue
            ws = workers.get(wid)
            if ws is None:
                continue
            ws.returncode = ret
            ws.completion_time = utc_now()

            # Check worker receipt
            receipt_path = output_root / "workers" / wid / "worker_receipt.json"
            receipt_ok = False
            if receipt_path.is_file():
                receipt = read_json(receipt_path)
                receipt_ok = str(receipt.get("status", "")).startswith("PASS")

            if receipt_ok and ret == 0:
                ws.phase = "PASS"
                completed_workers.append(wid)
            else:
                ws.phase = "FAILED"
                failed_workers.append(wid)

            if loading_worker_id == wid:
                loading_worker_id = None
            # Keep worker_processes entry, proc now defunct

    def resident_count(gpu: int) -> int:
        return sum(
            1 for w in workers.values()
            if w.gpu == gpu and w.phase in RESIDENT_PHASES
        )

    def try_launch(gpu: int) -> None:
        nonlocal loading_worker_id

        if not pending_by_gpu[gpu]:
            return
        if resident_count(gpu) >= effective_cap_by_gpu[gpu]:
            return
        if loading_worker_id is not None:
            return

        # Check no worker on this GPU is currently loading
        for w in workers.values():
            if w.gpu == gpu and w.phase in LOADING_PHASES:
                return

        # Memory admission
        cal = cal_by_gpu[gpu]
        threshold = compute_admission_threshold(cal.calibrated_budget_mib)
        try:
            snap = nvidia_snapshot([gpu])[gpu]
            if snap.memory_free_mib < threshold:
                return
        except Exception:
            return

        # Launch
        shard = pending_by_gpu[gpu].popleft()
        wid = str(shard["worker_id"])
        sid = str(shard.get("shard_id", ""))
        ep_count = int(shard.get("episode_count", 25))
        manifest_path = Path(shard["manifest"])
        manifest_sha = str(shard.get("manifest_sha256", sha256_file(manifest_path)))
        worker_root = output_root / "workers" / wid
        status_file = worker_root / "worker_status.json"

        # Log files (P0-4: no pipe deadlock)
        # Write to scheduler logs dir, NOT worker output root (collector
        # requires output_root to not yet exist).
        logs_dir = output_root / "worker_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = open(str(logs_dir / f"{wid}_stdout.log"), "a", encoding="utf-8")
        stderr_log = open(str(logs_dir / f"{wid}_stderr.log"), "a", encoding="utf-8")

        cmd = [
            sys.executable, str(SHARD_RUNNER),
            "--manifest", str(manifest_path),
            "--manifest-sha256", manifest_sha,
            "--output-root", str(worker_root),
            "--expected-git-commit", frozen_head,
            "--suite-model-map", str(suite_model_map),
            "--suite-model-report", str(suite_model_report),
            "--goal-model-manifest", str(goal_model_manifest),
            "--model-verification-report", str(model_verification_report),
            "--worker-id", wid,
            "--shard-id", sid,
            "--physical-gpu", str(gpu),
            "--model-load-lock-file", str(MODEL_LOAD_LOCK),
            "--worker-status-file", str(status_file),
            "--dummy-wait", str(OFFICIAL_DUMMY_WAIT_STEPS),
            "--mode", "run",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["C2G_PHYSICAL_GPU"] = str(gpu)

        proc = subprocess.Popen(
            cmd, cwd=REPO, env=env,
            stdout=stdout_log, stderr=stderr_log,
        )

        ws = WorkerState(
            worker_id=wid,
            shard_id=sid,
            gpu=gpu,
            pid=proc.pid,
            phase="CREATED",
            launch_time=utc_now(),
            episode_count=ep_count,
        )
        workers[wid] = ws
        worker_processes[wid] = proc
        loading_worker_id = wid
        _pre_launch_free[wid] = float(snap.memory_free_mib)  # for Gate 3 calibration

    def calibrate_caps() -> None:
        """Update calibrated budgets and consider cap upgrades."""
        for gpu in GPUS:
            cal = cal_by_gpu[gpu]
            if len(cal.observed_deltas_mib) >= 2:
                p95 = sorted(cal.observed_deltas_mib)[
                    int(0.95 * len(cal.observed_deltas_mib))
                ]
                cal.calibrated_budget_mib = ceil_to_1024(
                    max(float(FALLBACK_WORKER_BUDGET_MIB), p95 * 1.15)
                )

            # Consider cap upgrade
            if effective_cap_by_gpu[gpu] < MAX_RESIDENT_CAP:
                rc = resident_count(gpu)
                if (
                    rc >= INITIAL_RESIDENT_CAP
                    and cal.oom_count == 0
                    and len(cal.observed_deltas_mib) >= 1
                ):
                    threshold = compute_admission_threshold(cal.calibrated_budget_mib)
                    try:
                        snap = nvidia_snapshot([gpu])[gpu]
                        if snap.memory_free_mib >= threshold:
                            effective_cap_by_gpu[gpu] = MAX_RESIDENT_CAP
                    except Exception:
                        pass

    def all_done() -> bool:
        pending = sum(len(pending_by_gpu[g]) for g in GPUS)
        active = sum(
            1 for wid, proc in worker_processes.items()
            if proc.poll() is None
        )
        return pending == 0 and active == 0

    # ── Main loop ──
    max_loops = 7200  # ~24 hours
    for iteration in range(max_loops):
        poll_worker_status()
        poll_processes()
        calibrate_caps()

        for gpu in GPUS:
            try_launch(gpu)

        if iteration % 15 == 0:  # every ~30s
            write_heartbeat()

        if all_done():
            write_heartbeat()
            break

        time.sleep(2)

    # ── Final report (P0-5: proper completion check) ──
    pending_count = sum(len(pending_by_gpu[g]) for g in GPUS)
    active_count = sum(
        1 for wid, proc in worker_processes.items() if proc.poll() is None
    )

    all_completed = (
        len(completed_workers) == target_shard_count
        and len(failed_workers) == 0
        and pending_count == 0
        and active_count == 0
    )

    cap_decisions: dict[str, dict[str, Any]] = {}
    for gpu in GPUS:
        cal = cal_by_gpu[gpu]
        cap_decisions[str(gpu)] = {
            "effective_cap": effective_cap_by_gpu[gpu],
            "initial_cap": INITIAL_RESIDENT_CAP,
            "max_requested_cap": MAX_RESIDENT_CAP,
            "upgraded": effective_cap_by_gpu[gpu] > INITIAL_RESIDENT_CAP,
            "calibrated_budget_mib": cal.calibrated_budget_mib,
            "oom_count": cal.oom_count,
        }

    report = {
        "schema": SCHEMA,
        "status": (
            (CANARY_PASS_STATUS if is_canary else FULL_PASS_STATUS)
            if all_completed else "HOLD_C2G_R8Y_L10_520_SCHEDULER"
        ),
        "mode": mode,
        "all_completed": all_completed,
        "completed_workers": len(completed_workers),
        "failed_workers": len(failed_workers),
        "pending_shards": pending_count,
        "active_workers": active_count,
        "target_shard_count": target_shard_count,
        "target_episode_count": total_episodes,
        "worker_receipts_valid": len(completed_workers),
        "per_gpu": {
            str(gpu): {
                "completed": sum(1 for wid in completed_workers if workers[wid].gpu == gpu),
                "failed": sum(1 for wid in failed_workers if workers[wid].gpu == gpu),
                "pending": len(pending_by_gpu[gpu]),
                "effective_cap": effective_cap_by_gpu[gpu],
                "cap_upgrade": cap_decisions[str(gpu)],
            }
            for gpu in GPUS
        },
        "oom_count": sum(cal.oom_count for cal in cal_by_gpu.values()),
        "gpu_migration_count": 0,
        "abs_min_free_mib": ABSOLUTE_MIN_FREE_MIB,
        "fallback_worker_budget_mib": FALLBACK_WORKER_BUDGET_MIB,
        "gpu_reserve_mib": GPU_POST_LAUNCH_RESERVE_MIB,
        "load_margin_mib": MODEL_LOAD_TRANSIENT_MARGIN_MIB,
        "initial_resident_cap": INITIAL_RESIDENT_CAP,
        "max_resident_cap": MAX_RESIDENT_CAP,
        "frozen_git_head": frozen_head,
        "output_root": str(output_root),
    }

    write_json(report_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["canary-run", "full-run"], default="canary-run",
        help="canary-run = 12-episode shadow; full-run = 500-episode",
    )
    parser.add_argument("--plan-report", type=Path, required=True)
    parser.add_argument("--expected-plan-report-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suite-model-map", type=Path, required=True)
    parser.add_argument("--suite-model-report", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path, required=True)
    parser.add_argument("--model-verification-report", type=Path, required=True)
    parser.add_argument("--authorization", default="")
    parser.add_argument("--canary-report", type=Path, default=None)
    parser.add_argument("--expected-canary-report-sha256", default=None)
    parser.add_argument("--canary-ledger", type=Path, default=None)
    parser.add_argument("--expected-canary-ledger-sha256", default=None)
    parser.add_argument("--cap-override", default="",
                        help="Per-GPU cap override, e.g. '4=2,5=2,6=2,7=1'")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_scheduler(
        mode=args.mode,
        plan_report=args.plan_report,
        expected_plan_report_sha256=args.expected_plan_report_sha256,
        output_root=args.output_root,
        suite_model_map=args.suite_model_map,
        suite_model_report=args.suite_model_report,
        goal_model_manifest=args.goal_model_manifest,
        model_verification_report=args.model_verification_report,
        authorization=args.authorization,
        cap_override=args.cap_override,
        canary_report=args.canary_report,
        expected_canary_report_sha256=args.expected_canary_report_sha256,
        canary_ledger=args.canary_ledger,
        expected_canary_ledger_sha256=args.expected_canary_ledger_sha256,
    )
    status = report.get("status", "UNKNOWN")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if str(status).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
