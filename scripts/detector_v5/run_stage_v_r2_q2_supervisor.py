"""Detached local supervisor for the Q2 clean-control qualification."""
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Any, Mapping

try:
    from .run_stage_v_local_supervisor import ExclusiveLock, check_writable
    from .stage_v_dynamic_common import (
        atomic_write_json, gpu_snapshot, pid_alive, read_json,
        sha256_file, terminate_process_group, utc_now,
    )
except ImportError:  # direct server execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.detector_v5.run_stage_v_local_supervisor import ExclusiveLock, check_writable
    from scripts.detector_v5.stage_v_dynamic_common import (
        atomic_write_json, gpu_snapshot, pid_alive, read_json,
        sha256_file, terminate_process_group, utc_now,
    )


SCHEMA = "STAGE_V_R2_Q2_LOCAL_SUPERVISOR_V1"
SALT = "STAGE_V_R2_Q2_CONTROL_QUALIFICATION_20260807"
BOUNDARIES = {"eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0, "attack_rollouts": 0}


def _proc_values(path: str) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            key, _, raw = line.partition(" ")
            try:
                values[key.rstrip(":")] = int(raw.strip().split()[0])
            except (IndexError, ValueError):
                pass
    except OSError:
        pass
    return values


def _memory_snapshot() -> dict[str, Any]:
    mem = _proc_values("/proc/meminfo")
    vm = _proc_values("/proc/vmstat")
    available = mem.get("MemAvailable")
    swap_used = max(0, mem.get("SwapTotal", 0) - mem.get("SwapFree", 0))
    return {
        "available_ram_bytes": available * 1024 if available is not None else None,
        "available_ram_gib": round(available / (1024 * 1024), 3) if available is not None else None,
        "swap_used_bytes": swap_used * 1024,
        "swap_in": vm.get("pswpin"),
        "swap_out": vm.get("pswpout"),
        "oom_kill": vm.get("oom_kill"),
    }


def _xid_status(start_utc: str) -> tuple[str, str | None]:
    try:
        query_since = _datetime.datetime.fromisoformat(start_utc).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        query_since = start_utc
    try:
        result = subprocess.run(
            ["journalctl", "-k", "--since", query_since, "--no-pager", "-q"],
            check=False, capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "UNKNOWN", f"XID_QUERY_ERROR:{type(exc).__name__}"
    text = result.stdout + result.stderr
    if result.returncode != 0:
        return "UNKNOWN", f"XID_QUERY_EXIT:{result.returncode}"
    if any(token in text.lower() for token in ("xid", "fallen off the bus", "gpu has fallen off")):
        return "XID_DETECTED", "NVIDIA_XID"
    return "CLEAR", None


def _queue_snapshot(db_path: Path) -> dict[str, Any]:
    if not db_path.is_file():
        return {"progress": {}, "active_workers": [], "error": "QUEUE_DB_MISSING"}
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            meta = conn.execute("SELECT state FROM run_meta ORDER BY updated_at DESC LIMIT 1").fetchone()
            tasks = conn.execute(
                """SELECT t.cell_id,t.parent_id,t.suite,t.arm,t.state,t.attempt_count,
                   a.pid,a.gpu_id,a.worker_id,a.heartbeat_at,a.output_dir
                   FROM tasks t LEFT JOIN attempts a ON a.attempt_id=(
                     SELECT a2.attempt_id FROM attempts a2 WHERE a2.cell_id=t.cell_id ORDER BY rowid DESC LIMIT 1)
                   ORDER BY t.cell_id"""
            ).fetchall()
        finally:
            conn.close()
    except (OSError, sqlite3.Error) as exc:
        return {"progress": {}, "active_workers": [], "error": f"QUEUE_READ_ERROR:{type(exc).__name__}"}
    rows = [dict(row) for row in tasks]
    active_states = {"LEASED", "RUNNING", "COMMITTING"}
    active = [row for row in rows if row.get("state") in active_states]
    counts = {
        "total_tasks": len(rows),
        "completed_tasks": sum(row.get("state") == "DONE_VALID" for row in rows),
        "running_tasks": len(active),
        "pending_tasks": sum(row.get("state") in {"PENDING", "RETRY_READY", "LOCKED"} for row in rows),
        "failed_tasks": sum(row.get("state") not in active_states | {"DONE_VALID", "PENDING", "RETRY_READY", "LOCKED"} for row in rows),
    }
    return {
        "run_state": meta["state"] if meta else None,
        "progress": counts,
        "active_workers": active,
        "error": None,
    }


def _mtime_utc(paths: list[Path]) -> str | None:
    existing = [path for path in paths if path.is_file()]
    if not existing:
        return None
    return _datetime.datetime.fromtimestamp(max(path.stat().st_mtime for path in existing), _datetime.timezone.utc).isoformat()


def _write_sums(root: Path) -> None:
    excluded = {"Q2_SHA256SUMS", "Q2_SHA256SUMS.sha256"}
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "Q2_SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (root / "Q2_SHA256SUMS.sha256").write_text(
        f"{sha256_file(root / 'Q2_SHA256SUMS')}  Q2_SHA256SUMS\n", encoding="utf-8",
    )


class Q2Supervisor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.state_root = args.state_root.resolve()
        self.run_root = args.run_root.resolve()
        self.repo_root = args.repo_root.resolve()
        self.protocol = args.protocol.resolve()
        self.candidate = args.candidate_universe.resolve()
        self.started_utc = utc_now()
        self.started_epoch = time.time()
        self.producer: subprocess.Popen[Any] | None = None
        self.lock: ExclusiveLock | None = None
        self.heartbeat_count = 0
        self.baseline_oom: int | None = None
        self.swap_bad_streak = 0
        self.last_resource: dict[str, Any] = {}

    def _source(self) -> dict[str, str]:
        def git(*parts: str) -> str:
            result = subprocess.run(["git", "-C", str(self.repo_root), *parts], check=False, capture_output=True, text=True, timeout=15)
            if result.returncode != 0:
                raise RuntimeError(f"GIT_QUERY_FAIL:{result.stderr.strip()}")
            return result.stdout.strip()
        return {
            "source_commit": git("rev-parse", "HEAD"),
            "source_tree": git("rev-parse", "HEAD^{tree}"),
            "source_status": git("status", "--porcelain", "--untracked-files=all"),
        }

    def _prepare(self) -> None:
        source = self._source()
        if source["source_commit"] != self.args.source_commit or source["source_tree"] != self.args.source_tree:
            raise RuntimeError("SOURCE_OR_TREE_MISMATCH")
        if source["source_status"]:
            raise RuntimeError("SOURCE_WORKTREE_DIRTY")
        protocol = read_json(self.protocol, {})
        if not isinstance(protocol, Mapping) or protocol.get("schema") != "STAGE_Q2_PROTOCOL_V1" or protocol.get("status") != "FROZEN":
            raise RuntimeError("Q2_PROTOCOL_NOT_FROZEN")
        if protocol.get("source_commit") != self.args.source_commit or protocol.get("source_tree") != self.args.source_tree:
            raise RuntimeError("Q2_PROTOCOL_SOURCE_MISMATCH")
        if protocol.get("candidate_universe_sha256") != sha256_file(self.candidate):
            raise RuntimeError("Q2_CANDIDATE_SHA_MISMATCH")
        if self.args.expected_candidate_sha256 and sha256_file(self.candidate) != self.args.expected_candidate_sha256:
            raise RuntimeError("Q2_CANDIDATE_EXPECTED_SHA_MISMATCH")
        if self.run_root.exists() and any(self.run_root.iterdir()):
            raise RuntimeError("Q2_RUN_ROOT_NOT_NEW_OR_EMPTY")
        if self.state_root.exists() and any(self.state_root.iterdir()):
            raise RuntimeError("Q2_STATE_ROOT_NOT_NEW_OR_EMPTY")
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        memory = _memory_snapshot()
        self.baseline_oom = memory.get("oom_kill")
        atomic_write_json(self.state_root / "SUPERVISOR_START.json", {
            "schema": SCHEMA, "control_plane_mode": "LOCAL_AUTONOMOUS", "ssh_is_hard_stop": False,
            "run_root": str(self.run_root), "state_root": str(self.state_root),
            "source_commit": self.args.source_commit, "source_tree": self.args.source_tree,
            "protocol": str(self.protocol), "protocol_sha256": sha256_file(self.protocol),
            "candidate_universe": str(self.candidate), "candidate_universe_sha256": sha256_file(self.candidate),
            "approved_gpus": self.args.gpus, "gpu5_excluded": 5 not in self.args.gpus,
            "supervisor_pid": os.getpid(), "supervisor_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
            "oom_baseline": self.baseline_oom, "started_utc": self.started_utc,
            **BOUNDARIES,
        })

    def _resource_snapshot(self) -> tuple[dict[str, Any], list[str]]:
        memory = _memory_snapshot()
        gpu_rows, gpu_error = gpu_snapshot(self.args.gpu_query_command)
        xid, xid_error = _xid_status(self.started_utc)
        queue = _queue_snapshot(self.run_root / "Q2_CONTROL_QUALIFICATION.sqlite")
        errors: list[str] = []
        available = memory.get("available_ram_bytes")
        if available is None:
            errors.append("MEMORY_QUERY_UNAVAILABLE")
        elif available < self.args.min_available_ram_gib * (1 << 30):
            errors.append("AVAILABLE_RAM_BELOW_HARD_STOP")
        oom = memory.get("oom_kill")
        if self.baseline_oom is not None and oom is not None and oom > self.baseline_oom:
            errors.append("OOM_KILL_COUNTER_INCREASED")
        if memory.get("swap_used_bytes", 0) > 0:
            self.swap_bad_streak += 1
        else:
            self.swap_bad_streak = 0
        if self.swap_bad_streak >= 2:
            errors.append("SWAP_NONZERO_TWO_SAMPLES")
        if gpu_error:
            errors.append(gpu_error)
        by_gpu = {int(row.get("gpu_id")): row for row in gpu_rows if row.get("gpu_id") is not None}
        for gpu in self.args.gpus:
            row = by_gpu.get(gpu)
            if row is None:
                errors.append(f"GPU_MISSING:{gpu}")
            elif row.get("memory_free_mib") is not None and float(row["memory_free_mib"]) < self.args.min_free_memory_mib:
                errors.append(f"GPU_FREE_MEMORY_BELOW_POLICY:{gpu}")
        if 5 in self.args.gpus:
            errors.append("GPU5_FORBIDDEN")
        if xid_error or xid == "XID_DETECTED":
            errors.append(xid_error or "NVIDIA_XID")
        queue_error = queue.get("error")
        # The producer creates SQLite immediately after process start; tolerate
        # only that short startup race, never a missing queue after startup.
        if queue_error and not (
            queue_error == "QUEUE_DB_MISSING"
            and self.producer is not None
            and self.producer.poll() is None
            and time.time() - self.started_epoch < 120
        ):
            errors.append(str(queue_error))
        active = queue.get("active_workers", [])
        active_gpus = [int(row["gpu_id"]) for row in active if row.get("gpu_id") is not None]
        if len(active_gpus) != len(set(active_gpus)):
            errors.append("MULTIPLE_PROJECT_WORKERS_PER_GPU")
        if len(active_gpus) > len(self.args.gpus):
            errors.append("ACTIVE_WORKERS_EXCEED_APPROVED_GPUS")
        if any(gpu == 5 or gpu not in self.args.gpus for gpu in active_gpus):
            errors.append("UNAPPROVED_OR_GPU5_WORKER")
        if self.producer and self.producer.poll() is None:
            if any(row.get("pid") not in (None, self.producer.pid) for row in active):
                errors.append("WORKER_PID_MISMATCH")
        try:
            check_writable(self.state_root)
            check_writable(self.run_root)
        except OSError as exc:
            errors.append(f"FILESYSTEM_WRITE_FAIL:{type(exc).__name__}")
        resource = {
            **memory, "gpu_memory": gpu_rows, "gpu_xid_status": xid,
            "active_worker_pids": sorted({int(row["pid"]) for row in active if row.get("pid")}),
            "gpu_assignments": [
                {"gpu": int(row["gpu_id"]), "pid": row.get("pid"), "parent": row.get("parent_id"), "arm": row.get("arm")}
                for row in active if row.get("gpu_id") is not None
            ],
            "queue": queue, "resource_errors": sorted(set(errors)),
        }
        self.last_resource = resource
        return resource, sorted(set(errors))

    def _report_state(self) -> tuple[dict[str, Any], dict[str, Any] | None]:
        report_path = self.run_root / "Q2_CONTROL_QUALIFICATION_REPORT.json"
        report = read_json(report_path, None)
        queue = self.last_resource.get("queue", {})
        progress = queue.get("progress", {}) if isinstance(queue, Mapping) else {}
        if isinstance(report, Mapping):
            evaluated = int(report.get("evaluated_rows", 0) or 0)
            qualified = sum(int(value) for value in (report.get("qualified_by_suite") or {}).values())
            failed = int(report.get("engineering_invalid_rows", 0) or 0)
        else:
            evaluated = int(progress.get("total_tasks", 0) or 0) // 2
            qualified = 0
            failed = int(progress.get("failed_tasks", 0) or 0) // 2
        return {
            "planned_parents": int(self.protocol_value("candidate_universe_count", 0)),
            "evaluated_parents": evaluated,
            "completed_parents": int(progress.get("completed_tasks", 0) or 0) // 2,
            "accepted_parent_results": qualified,
            "failed_parents": failed,
            "current_parent": (queue.get("active_workers") or [{}])[0].get("parent_id") if queue.get("active_workers") else None,
            "current_branch": "CLEAN_" + str((queue.get("active_workers") or [{}])[0].get("arm")) if queue.get("active_workers") else None,
        }, report if isinstance(report, Mapping) else None

    def protocol_value(self, key: str, default: Any) -> Any:
        value = read_json(self.protocol, {})
        return value.get(key, default) if isinstance(value, Mapping) else default

    def _heartbeat(self, resource: Mapping[str, Any]) -> None:
        self.heartbeat_count += 1
        progress, report = self._report_state()
        heartbeat = {
            "schema": SCHEMA, "control_plane_mode": "LOCAL_AUTONOMOUS", "ssh_is_hard_stop": False,
            "run_root": str(self.run_root), "state_root": str(self.state_root),
            "source_commit": self.args.source_commit, "source_tree": self.args.source_tree,
            "supervisor_pid": os.getpid(), "supervisor_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
            "dispatcher_pid": self.producer.pid if self.producer else None,
            **progress,
            "active_worker_pids": resource.get("active_worker_pids", []), "gpu_assignments": resource.get("gpu_assignments", []),
            "last_artifact_utc": _mtime_utc([
                self.run_root / "Q2_CONTROL_QUALIFICATION_REPORT.json",
                self.run_root / "Q2_CONTROL_QUALIFICATION_ROWS.jsonl",
                self.run_root / "Q2_CONTROL_QUALIFICATION.sqlite",
            ]),
            "available_ram_gib": resource.get("available_ram_gib"), "swap_in": resource.get("swap_in"),
            "swap_out": resource.get("swap_out"), "oom_kill": resource.get("oom_kill"),
            "oom_baseline": self.baseline_oom, "oom_delta": (
                resource.get("oom_kill") - self.baseline_oom
                if resource.get("oom_kill") is not None and self.baseline_oom is not None else None
            ),
            "gpu_memory": resource.get("gpu_memory", []), "gpu_xid_status": resource.get("gpu_xid_status"),
            "filesystem_writable": not any(str(error).startswith("FILESYSTEM_WRITE_FAIL") for error in resource.get("resource_errors", [])),
            "resource_errors": resource.get("resource_errors", []),
            "ssh_probe_success_count": 0, "ssh_probe_failure_count": 0, "longest_ssh_unavailable_interval_seconds": 0,
            "external_root_process_present": bool(self.args.external_pid and pid_alive(self.args.external_pid)),
            "external_root_process_pid": self.args.external_pid or None, "external_root_process_terminated": False,
            "producer_report_status": report.get("status") if report else None,
            "heartbeat_count": self.heartbeat_count, **BOUNDARIES, "updated_utc": utc_now(),
        }
        atomic_write_json(self.state_root / "Q2_LOCAL_HEARTBEAT.json", heartbeat)

    def _start(self) -> None:
        environment = os.environ.copy()
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
        command = [self.args.python_executable, str(self.args.producer_script), *self.args.producer_args]
        stdout = (self.state_root / "Q2_PRODUCER_STDOUT.log").open("a", encoding="utf-8")
        stderr = (self.state_root / "Q2_PRODUCER_STDERR.log").open("a", encoding="utf-8")
        self.producer = subprocess.Popen(
            command, cwd=str(self.repo_root), env=environment, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, start_new_session=(os.name == "posix"),
        )
        stdout.close()
        stderr.close()
        atomic_write_json(self.state_root / "Q2_PIDS.json", {
            "schema": "STAGE_V_R2_Q2_PIDS_V1", "supervisor_pid": os.getpid(),
            "supervisor_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
            "dispatcher_pid": self.producer.pid, "dispatcher_pgid": os.getpgid(self.producer.pid) if hasattr(os, "getpgid") else self.producer.pid,
            "started_utc": utc_now(),
        })

    def _audit(self) -> tuple[int, str]:
        command = [
            self.args.python_executable, str(self.args.auditor_script), "--output-dir", str(self.run_root),
            "--protocol", str(self.protocol), "--candidate-universe", str(self.candidate),
            "--report", str(self.run_root / "Q2_CONTROL_QUALIFICATION_REPORT.json"),
            "--rows", str(self.run_root / "Q2_CONTROL_QUALIFICATION_ROWS.jsonl"),
            "--source-commit", self.args.source_commit, "--source-tree", self.args.source_tree,
        ]
        result = subprocess.run(command, cwd=str(self.repo_root), check=False, capture_output=True, text=True, timeout=self.args.audit_timeout)
        (self.state_root / "Q2_AUDITOR_STDOUT.log").write_text(result.stdout, encoding="utf-8")
        (self.state_root / "Q2_AUDITOR_STDERR.log").write_text(result.stderr, encoding="utf-8")
        return result.returncode, (result.stdout + result.stderr)[-4000:]

    def _abort(self, reason: str) -> int:
        reap: dict[str, Any] = {"owned_pids_before": [], "owned_pids_after": [], "process_reap_complete": True}
        if self.producer is not None:
            before = [self.producer.pid] if pid_alive(self.producer.pid) else []
            terminate_process_group(self.producer, grace_seconds=20)
            reap = {"owned_pids_before": before, "owned_pids_after": [self.producer.pid] if pid_alive(self.producer.pid) else [], "process_reap_complete": not pid_alive(self.producer.pid)}
        payload = {
            "schema": "STAGE_V_R2_Q2_ABORT_V1", "status": "ABORTED_INCOMPLETE", "reason": reason,
            "control_plane_mode": "LOCAL_AUTONOMOUS", "ssh_is_hard_stop": False,
            "supervisor_pid": os.getpid(), "dispatcher_pid": self.producer.pid if self.producer else None,
            "accepted_parent_results": 0, "scientific_validity": 0, **reap, **BOUNDARIES, "aborted_utc": utc_now(),
        }
        atomic_write_json(self.run_root / "ABORTED_INCOMPLETE.json", payload)
        atomic_write_json(self.state_root / "Q2_SUPERVISOR_FAILURE.json", payload)
        if self.last_resource:
            self.last_resource = dict(self.last_resource)
            self.last_resource["resource_errors"] = sorted(set(list(self.last_resource.get("resource_errors", [])) + [reason]))
            self._heartbeat(self.last_resource)
        return 1

    def _finalize(self, producer_code: int) -> int:
        report = self.run_root / "Q2_CONTROL_QUALIFICATION_REPORT.json"
        rows = self.run_root / "Q2_CONTROL_QUALIFICATION_ROWS.jsonl"
        if not report.is_file() or not rows.is_file():
            return self._abort(f"PRODUCER_EXIT_WITHOUT_Q2_ARTIFACTS:{producer_code}")
        audit_code, audit_tail = self._audit()
        audit = read_json(self.run_root / "Q2_CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json", {})
        manifest = self.run_root / "Q2_PARENT_MANIFEST_A.json"
        status = "PASS" if producer_code == 0 and audit_code == 0 and isinstance(audit, Mapping) and audit.get("verdict") == "PASS" and manifest.is_file() else "FAIL"
        complete = {
            "schema": "STAGE_V_R2_Q2_SUPERVISOR_COMPLETE_V1", "status": status,
            "control_plane_mode": "LOCAL_AUTONOMOUS", "ssh_is_hard_stop": False,
            "producer_exit_code": producer_code, "auditor_exit_code": audit_code,
            "audit_verdict": audit.get("verdict") if isinstance(audit, Mapping) else None,
            "audit_tail": audit_tail, "supervisor_pid": os.getpid(),
            "dispatcher_pid": self.producer.pid if self.producer else None,
            "heartbeat_count": self.heartbeat_count, "planned_parent_universe": self.protocol_value("candidate_universe_count", None),
            "q2_parent_manifest": str(manifest) if manifest.is_file() else None,
            "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0, "attack_rollouts": 0,
            "completed_utc": utc_now(),
        }
        atomic_write_json(self.run_root / "Q2_SUPERVISOR_COMPLETE.json", complete)
        atomic_write_json(self.state_root / "Q2_SUPERVISOR_COMPLETE.json", complete)
        _write_sums(self.run_root)
        _write_sums(self.state_root)
        self._heartbeat(self.last_resource)
        return 0 if status == "PASS" else 1

    def run(self) -> int:
        self.lock = ExclusiveLock(self.args.lock_path.resolve(), {
            "schema": "STAGE_V_R2_Q2_LOCK_V1", "supervisor_pid": os.getpid(),
            "supervisor_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
            "run_root": str(self.run_root), "started_utc": self.started_utc,
        })
        try:
            self.lock.acquire(self.state_root)
            self._prepare()
            self._start()
            resource, errors = self._resource_snapshot()
            self._heartbeat(resource)
            if errors:
                return self._abort(";".join(errors))
            while self.producer is not None:
                code = self.producer.poll()
                if code is not None:
                    return self._finalize(code)
                resource, errors = self._resource_snapshot()
                self._heartbeat(resource)
                if errors:
                    return self._abort(";".join(errors))
                time.sleep(max(1.0, self.args.poll_seconds))
        except Exception as exc:
            if self.run_root.exists():
                return self._abort(f"SUPERVISOR_EXCEPTION:{type(exc).__name__}:{exc}")
            return 2
        finally:
            if self.lock:
                self.lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate-universe", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", default="")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--producer-script", type=Path, required=True)
    parser.add_argument("--producer-arg", dest="producer_args", action="append", default=[])
    parser.add_argument("--auditor-script", type=Path, required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--gpus", type=lambda value: [int(item) for item in value.split(",") if item], required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--external-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--audit-timeout", type=float, default=3600.0)
    parser.add_argument("--min-available-ram-gib", type=float, default=128.0)
    parser.add_argument("--min-free-memory-mib", type=float, default=4096.0)
    parser.add_argument("--gpu-query-command", default="nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.gpus or len(args.gpus) != len(set(args.gpus)) or 5 in args.gpus:
        raise SystemExit("Q2 GPU list must be non-empty, unique, and exclude GPU5")
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    return Q2Supervisor(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
