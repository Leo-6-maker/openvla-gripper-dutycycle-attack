"""MODE_B/C GPU admission, project leases, and resource receipts.

This module is intentionally independent of the science queue.  The queue owns
parent work; this file owns the physical-GPU admission boundary.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import uuid
from typing import Any, Callable, Iterable, Mapping


MODE_B = "MODE_B_THROUGHPUT_SCIENCE"
MODE_C = "MODE_C_TRAINING"
MIN_FREE_MEMORY_MIB = 20_480


class ResourceContractError(RuntimeError):
    """Fail-closed resource admission or lease error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_uuid(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.removeprefix("gpu-")


def _number(value: Any) -> float | int | None:
    text = str(value or "").strip().split()[0] if str(value or "").strip() else ""
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def _rows_from_csv(text: str, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for values in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(values) < len(fields):
            continue
        row = dict(zip(fields, (item.strip() for item in values)))
        rows.append(row)
    return rows


def _process_identity(pid: int) -> dict[str, Any]:
    """Return best-effort owner/name telemetry without treating it as a gate."""
    command = ""
    owner = "unknown"
    if os.name == "posix":
        cmdline = Path(f"/proc/{pid}/cmdline")
        try:
            command = cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            pass
        try:
            for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("Uid:"):
                    import pwd
                    owner = pwd.getpwuid(int(line.split()[1])).pw_name
                    break
        except (OSError, KeyError, ValueError, IndexError):
            pass
    if not command:
        try:
            result = subprocess.run(
                ["ps", "-o", "user=,args=", "-p", str(pid)],
                capture_output=True, text=True, check=False, timeout=5,
            )
            values = result.stdout.strip().split(None, 1)
            if values:
                owner = values[0]
            if len(values) > 1:
                command = values[1]
        except (OSError, subprocess.SubprocessError):
            pass
    return {"pid": pid, "owner": owner, "command": command}


def combine_inventory(
    gpu_rows: Iterable[Mapping[str, Any]],
    process_rows: Iterable[Mapping[str, Any]] = (),
    *,
    process_identity: Callable[[int], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Join GPU and compute-app telemetry by physical GPU UUID."""
    inventory = []
    by_uuid: dict[str, dict[str, Any]] = {}
    for raw in gpu_rows:
        row = dict(raw)
        key = canonical_uuid(row.get("uuid") or row.get("gpu_uuid"))
        row["gpu_uuid"] = key
        row["gpu_id"] = int(row["gpu_id"])
        row["memory_used_mib"] = _number(row.get("memory_used_mib"))
        row["memory_free_mib"] = _number(row.get("memory_free_mib"))
        row["utilization_gpu_percent"] = _number(row.get("utilization_gpu_percent"))
        row["compute_processes"] = []
        by_uuid[key] = row
        inventory.append(row)
    identify = process_identity or _process_identity
    for raw in process_rows:
        item = dict(raw)
        pid = int(item.get("pid") or 0)
        gpu_uuid = canonical_uuid(item.get("gpu_uuid") or item.get("uuid"))
        identity = dict(identify(pid)) if pid > 0 else {"owner": "unknown", "command": ""}
        process = {
            "pid": pid,
            "process_name": str(item.get("process_name") or ""),
            "used_memory_mib": _number(item.get("used_memory_mib")),
            **identity,
        }
        target = by_uuid.get(gpu_uuid)
        if target is not None:
            target["compute_processes"].append(process)
    for row in inventory:
        row["compute_processes"] = sorted(row["compute_processes"], key=lambda item: int(item.get("pid") or 0))
    return inventory


def query_inventory(
    *,
    gpu_query_command: Iterable[str] = (
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ),
    process_query_command: Iterable[str] = (
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ),
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        gpu_result = subprocess.run(list(gpu_query_command), capture_output=True, text=True, check=False, timeout=20)
        process_result = subprocess.run(list(process_query_command), capture_output=True, text=True, check=False, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"GPU_QUERY_ERROR:{type(exc).__name__}"
    if gpu_result.returncode != 0:
        return [], f"GPU_QUERY_EXIT:{gpu_result.returncode}"
    if process_result.returncode != 0:
        return [], f"GPU_PROCESS_QUERY_EXIT:{process_result.returncode}"
    return combine_inventory(
        _rows_from_csv(gpu_result.stdout, ("gpu_id", "uuid", "memory_used_mib", "memory_free_mib", "utilization_gpu_percent")),
        _rows_from_csv(process_result.stdout, ("gpu_uuid", "pid", "process_name", "used_memory_mib")),
    ), None


def _admission_row(
    row: Mapping[str, Any], *, mode: str, minimum_free_mib: int,
    leased_gpu_ids: set[int], project_pids: set[int], project_process_tokens: tuple[str, ...],
    excluded_gpu_ids: set[int],
) -> dict[str, Any]:
    gpu_id = int(row["gpu_id"])
    processes = [dict(item) for item in row.get("compute_processes", [])]
    reasons: list[str] = []
    if gpu_id in excluded_gpu_ids:
        reasons.append("EXCLUDED_GPU")
    project_processes = [
        item for item in processes
        if int(item.get("pid") or 0) in project_pids
        or any(token and token in str(item.get("command") or "") for token in project_process_tokens)
    ]
    if gpu_id in leased_gpu_ids:
        reasons.append("PROJECT_LEASE_PRESENT")
    if project_processes:
        reasons.append("PROJECT_WORKER_PRESENT")
    free = row.get("memory_free_mib")
    if free is None or float(free) < minimum_free_mib:
        reasons.append("INSUFFICIENT_FREE_MEMORY")
    # Foreign work is explicitly allowed in MODE_B/C; it is telemetry, not a
    # scheduler veto.  Project ownership is vetoed by the lease table above.
    foreign = [item for item in processes if item not in project_processes]
    result = {
        "gpu_id": gpu_id,
        "gpu_uuid": canonical_uuid(row.get("gpu_uuid") or row.get("uuid")),
        "mode": mode,
        "safe": not reasons,
        "reasons": reasons,
        "memory_used_mib": row.get("memory_used_mib"),
        "memory_free_mib": free,
        "utilization_gpu_percent": row.get("utilization_gpu_percent"),
        "foreign_processes": foreign,
        "project_processes": project_processes,
        "foreign_workload_present": bool(foreign),
        "minimum_free_memory_mib": minimum_free_mib,
        "captured_utc": utc_now(),
    }
    return result


def admit_mode_b_or_c(
    inventory: Iterable[Mapping[str, Any]], *, mode: str,
    leased_gpu_ids: Iterable[int] = (), project_pids: Iterable[int] = (),
    project_process_tokens: Iterable[str] = (),
    excluded_gpu_ids: Iterable[int] = (),
    minimum_free_mib: int = MIN_FREE_MEMORY_MIB,
) -> dict[str, Any]:
    if mode not in {MODE_B, MODE_C}:
        raise ResourceContractError(f"UNSUPPORTED_RESOURCE_MODE:{mode}")
    leased = {int(item) for item in leased_gpu_ids}
    project = {int(item) for item in project_pids}
    project_tokens = tuple(str(item) for item in project_process_tokens if str(item))
    excluded = {int(item) for item in excluded_gpu_ids}
    decisions = [_admission_row(row, mode=mode, minimum_free_mib=minimum_free_mib,
                                leased_gpu_ids=leased, project_pids=project,
                                project_process_tokens=project_tokens,
                                excluded_gpu_ids=excluded) for row in inventory]
    safe = sorted(row["gpu_id"] for row in decisions if row["safe"])
    return {
        "schema": "STAGE_V_GPU_RESOURCE_ADMISSION_V1",
        "mode": mode,
        "status": "PASS" if safe else "HOLD_NO_ELIGIBLE_GPU",
        "minimum_free_memory_mib": minimum_free_mib,
        "eligible_gpu_ids": safe,
        "partial_fleet_allowed": True,
        "utilization_is_scheduler_preference_only": True,
        "gpu_decisions": decisions,
        "captured_utc": utc_now(),
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class GpuLeaseStore:
    """Project-local physical-GPU lease table; active rows are never deleted."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        with self._connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS gpu_leases (
                lease_id TEXT PRIMARY KEY,
                lease_token TEXT NOT NULL UNIQUE,
                gpu_id INTEGER NOT NULL,
                gpu_uuid TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                worker_pid INTEGER NOT NULL,
                stage TEXT NOT NULL,
                atomic_job_id TEXT NOT NULL,
                source_commit TEXT NOT NULL,
                source_tree TEXT NOT NULL,
                runtime_root TEXT NOT NULL,
                acquired_utc TEXT NOT NULL,
                released_utc TEXT,
                state TEXT NOT NULL,
                release_reason TEXT,
                launch_snapshot_json TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_gpu_leases_active_gpu
                ON gpu_leases(gpu_id) WHERE state='ACTIVE';
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def active(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM gpu_leases WHERE state='ACTIVE' ORDER BY gpu_id").fetchall()
        return [self._row(row) for row in rows if row is not None]

    def acquire(self, *, gpu_id: int, gpu_uuid: str, worker_id: str, worker_pid: int,
                stage: str, atomic_job_id: str, source_commit: str, source_tree: str,
                runtime_root: str | Path, launch_snapshot: Mapping[str, Any]) -> dict[str, Any]:
        now = utc_now()
        lease = {
            "lease_id": uuid.uuid4().hex,
            "lease_token": uuid.uuid4().hex,
            "gpu_id": int(gpu_id),
            "gpu_uuid": canonical_uuid(gpu_uuid),
            "worker_id": str(worker_id),
            "worker_pid": int(worker_pid),
            "stage": str(stage),
            "atomic_job_id": str(atomic_job_id),
            "source_commit": str(source_commit),
            "source_tree": str(source_tree),
            "runtime_root": str(runtime_root),
            "acquired_utc": now,
            "released_utc": None,
            "state": "ACTIVE",
            "release_reason": None,
            "launch_snapshot_json": json.dumps(dict(launch_snapshot), sort_keys=True, allow_nan=False),
        }
        try:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                if conn.execute("SELECT 1 FROM gpu_leases WHERE gpu_id=? AND state='ACTIVE'", (lease["gpu_id"],)).fetchone():
                    conn.rollback()
                    raise ResourceContractError(f"GPU_LEASE_BUSY:{lease['gpu_id']}")
                conn.execute("""INSERT INTO gpu_leases
                    (lease_id,lease_token,gpu_id,gpu_uuid,worker_id,worker_pid,stage,atomic_job_id,
                     source_commit,source_tree,runtime_root,acquired_utc,released_utc,state,release_reason,launch_snapshot_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", tuple(lease.values()))
                conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ResourceContractError(f"GPU_LEASE_BUSY:{lease['gpu_id']}") from exc
        return lease

    def release(self, lease: Mapping[str, Any], *, reason: str = "WORKER_FINISHED") -> bool:
        now = utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("""SELECT * FROM gpu_leases WHERE lease_id=? AND lease_token=?
                                      AND state='ACTIVE' AND worker_id=? AND worker_pid=?""",
                                   (lease["lease_id"], lease["lease_token"], lease["worker_id"], int(lease["worker_pid"]))).fetchone()
            if current is None:
                conn.rollback()
                return False
            conn.execute("UPDATE gpu_leases SET state='RELEASED',released_utc=?,release_reason=? WHERE lease_id=?",
                         (now, reason, lease["lease_id"]))
            conn.commit()
        return True

    def recover_stale(self, lease_id: str, *, pid_alive: bool, identity_verified: bool,
                      reason: str = "STALE_PID_RECOVERED") -> bool:
        if pid_alive:
            raise ResourceContractError("STALE_RECOVERY_PID_ALIVE")
        if not identity_verified:
            raise ResourceContractError("STALE_RECOVERY_IDENTITY_UNVERIFIED")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT state FROM gpu_leases WHERE lease_id=?", (lease_id,)).fetchone()
            if current is None:
                conn.rollback()
                raise ResourceContractError("UNKNOWN_GPU_LEASE")
            if current["state"] != "ACTIVE":
                conn.rollback()
                return False
            conn.execute("UPDATE gpu_leases SET state='RECOVERED_STALE',released_utc=?,release_reason=? WHERE lease_id=?",
                         (utc_now(), reason, lease_id))
            conn.commit()
        return True


def verify_recheck(snapshot: Mapping[str, Any], *, expected_gpu_id: int, expected_gpu_uuid: str,
                   minimum_free_mib: int = MIN_FREE_MEMORY_MIB) -> None:
    if int(snapshot.get("gpu_id", -1)) != int(expected_gpu_id):
        raise ResourceContractError("GPU_RECHECK_ID_MISMATCH")
    if canonical_uuid(snapshot.get("gpu_uuid")) != canonical_uuid(expected_gpu_uuid):
        raise ResourceContractError("GPU_RECHECK_UUID_MISMATCH")
    free = snapshot.get("memory_free_mib")
    if free is None or float(free) < minimum_free_mib:
        raise ResourceContractError("GPU_RECHECK_MEMORY_INSUFFICIENT")


def write_resource_receipt(path: str | Path, *, phase: str, gpu_snapshot: Mapping[str, Any],
                           lease: Mapping[str, Any] | None = None,
                           atomic_job_id: str | None = None) -> dict[str, Any]:
    receipt = {
        "schema": "STAGE_V_GPU_RESOURCE_RECEIPT_V1",
        "phase": phase,
        "atomic_job_id": atomic_job_id,
        "gpu_id": gpu_snapshot.get("gpu_id"),
        "gpu_uuid": canonical_uuid(gpu_snapshot.get("gpu_uuid")),
        "memory_used_mib": gpu_snapshot.get("memory_used_mib"),
        "memory_free_mib": gpu_snapshot.get("memory_free_mib"),
        "utilization_gpu_percent": gpu_snapshot.get("utilization_gpu_percent"),
        "foreign_processes": list(gpu_snapshot.get("compute_processes", [])),
        "captured_utc": utc_now(),
        "lease": {
            key: lease.get(key)
            for key in ("lease_id", "gpu_id", "gpu_uuid", "worker_id", "worker_pid", "stage",
                        "atomic_job_id", "source_commit", "source_tree", "runtime_root", "acquired_utc")
        } if lease else None,
    }
    _atomic_write_json(Path(path), receipt)
    return receipt
