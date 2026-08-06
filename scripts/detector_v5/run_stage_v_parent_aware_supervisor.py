"""Fail-closed, local-only supervisor for Stage V dynamic-8 execution."""
from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Mapping

try:
    from .run_stage_v_local_supervisor import ExclusiveLock
    from .stage_v_dynamic_common import (
        atomic_write_json, gpu_preflight, gpu_snapshot, pid_alive, project_queue, read_json,
        sanitize_key, sha256_file, terminate_process_group, utc_now,
    )
    from .stage_v_science_core_provenance import verify as verify_science_provenance
except ImportError:  # direct server execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.detector_v5.run_stage_v_local_supervisor import ExclusiveLock
    from scripts.detector_v5.stage_v_dynamic_common import (
        atomic_write_json, gpu_preflight, gpu_snapshot, pid_alive, project_queue, read_json,
        sanitize_key, sha256_file, terminate_process_group, utc_now,
    )
    from scripts.detector_v5.stage_v_science_core_provenance import verify as verify_science_provenance

try:
    from scripts.fec.atomic_task_queue import AtomicTaskQueue
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.fec.atomic_task_queue import AtomicTaskQueue


def _proc_values(path: str) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 2:
                try:
                    values[fields[0].rstrip(":")] = int(fields[1])
                except ValueError:
                    pass
    except OSError:
        pass
    return values


def _mem_snapshot() -> dict[str, Any]:
    mem = _proc_values("/proc/meminfo")
    vm = _proc_values("/proc/vmstat")
    available = mem.get("MemAvailable")
    swap_total = mem.get("SwapTotal", 0)
    swap_free = mem.get("SwapFree", 0)
    return {
        "available_ram_bytes": available * 1024 if available is not None else None,
        "available_ram_gib": round(available / (1024 * 1024), 3) if available is not None else None,
        "swap_used_bytes": max(0, swap_total - swap_free) * 1024,
        "swap_in": vm.get("pswpin"), "swap_out": vm.get("pswpout"),
        "oom_kill": vm.get("oom_kill"),
    }


def _xid_status(start_utc: str) -> str | None:
    try:
        completed = subprocess.run(
            ["journalctl", "-k", "--since", start_utc, "--no-pager"],
            capture_output=True, text=True, check=False, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"XID_QUERY_ERROR:{type(exc).__name__}"
    text = completed.stdout + completed.stderr
    if re.search(r"NVRM: Xid|GPU has fallen off|Xid \(", text, re.IGNORECASE):
        return text[-1000:]
    return None


def _write_csv_row(path: Path, row: Mapping[str, Any]) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(row))
        if new:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in sorted(row)})
        handle.flush()
        os.fsync(handle.fileno())


class DynamicSupervisor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.run_root.resolve()
        self.start_epoch = time.time()
        self.start_utc = utc_now()
        self.dispatcher: subprocess.Popen[Any] | None = None
        self.lock: ExclusiveLock | None = None
        self.queue: AtomicTaskQueue | None = None
        self.heartbeat_count = 0
        self.ssh_success_count = 0
        self.ssh_failure_count = 0
        self.ssh_failed_since: float | None = None
        self.longest_ssh_outage = 0.0
        self.baseline_oom: int | None = None
        self.swap_bad_streak = 0
        self.last_heartbeat: dict[str, Any] = {}
        self.timeout_policy: dict[str, Any] = {}

    def _preflight(self) -> dict[str, Any]:
        if not self.args.skip_resource_checks:
            return gpu_preflight(
                required_count=8,
                excluded_gpus=self.args.excluded_gpus,
                canary_peak_mib=self.args.canary_peak_mib,
                protected_pids=self.args.protected_pids,
                gpu_query_command=self.args.gpu_query_command,
            )
        value = read_json(self.args.preflight_file, {})
        if not isinstance(value, Mapping):
            return {"status": "PRELAUNCH_WAITING_FOR_8_GPUS", "reason": "PREFLIGHT_NOT_OBJECT"}
        value = dict(value)
        all_safe = sorted({int(gpu) for gpu in value.get("safe_gpus", value.get("all_safe_gpus", [])) if int(gpu) not in self.args.excluded_gpus})
        approved = all_safe[:8]
        if value.get("status") != "PASS" or len(all_safe) < 8 or 5 in approved:
            value["status"] = "PRELAUNCH_WAITING_FOR_8_GPUS"
            value.setdefault("reason", "LESS_THAN_8_APPROVED_GPUS_OR_GPU5_EXCLUDED")
        else:
            value["all_safe_gpus"] = all_safe
            value["safe_gpus"] = approved
            value["safe_gpu_count"] = len(all_safe)
            value["selected_gpu_count"] = len(approved)
        return value

    def _source(self) -> dict[str, str]:
        def git(*parts: str) -> str:
            result = subprocess.run(["git", "-C", str(self.args.repo_root), *parts], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"GIT_QUERY_FAIL:{result.stderr[-200:]}")
            return result.stdout.strip()
        return {"source_commit": git("rev-parse", "HEAD"), "source_tree": git("rev-parse", "HEAD^{tree}")}

    def _prepare(self) -> None:
        while True:
            preflight = self._preflight()
            atomic_write_json(self.args.preflight_file, preflight)
            if preflight.get("status") == "PASS":
                break
            atomic_write_json(self.args.preflight_file.with_name("PRELAUNCH_WAITING_FOR_8_GPUS.json"), preflight)
            if not getattr(self.args, "wait_for_gpus", False):
                raise RuntimeError("PRELAUNCH_WAITING_FOR_8_GPUS")
            time.sleep(max(1.0, float(getattr(self.args, "preflight_interval_seconds", 300.0))))
        if sorted(int(gpu) for gpu in preflight.get("safe_gpus", [])) != sorted(self.args.approved_gpus):
            raise RuntimeError("PREFLIGHT_APPROVED_GPU_SET_MISMATCH")
        source = self._source()
        if source["source_commit"] != self.args.expected_source_commit or source["source_tree"] != self.args.expected_source_tree:
            raise RuntimeError("SOURCE_OR_TREE_MISMATCH")
        if self.args.science_provenance:
            provenance_ok, provenance_errors = verify_science_provenance(
                self.args.science_provenance,
                expected_commit=self.args.science_source_commit,
                expected_tree=self.args.science_source_tree,
            )
            if not provenance_ok:
                raise RuntimeError("SCIENCE_CORE_PROVENANCE_FAIL:" + ";".join(provenance_errors))
        elif self.args.science_runner:
            raise RuntimeError("SCIENCE_CORE_PROVENANCE_REQUIRED")
        if not self.args.parent_manifest.is_file():
            raise RuntimeError("PARENT_MANIFEST_MISSING")
        manifest_sha = sha256_file(self.args.parent_manifest)
        if self.args.parent_manifest_sha256 and manifest_sha != self.args.parent_manifest_sha256:
            raise RuntimeError("PARENT_MANIFEST_SHA256_MISMATCH")
        if self.args.timeout_policy and self.args.timeout_policy.is_file():
            self.timeout_policy = dict(read_json(self.args.timeout_policy, {}))
        else:
            self.timeout_policy = {"parent_hard_seconds": 10 * 3600, "branch_hard_seconds": 4 * 3600}
        if self.root.exists() and any(self.root.iterdir()):
            raise RuntimeError("RUN_ROOT_NOT_NEW_OR_EMPTY")
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue = AtomicTaskQueue(str(self.args.queue_db), run_id=self.args.run_id)
        self.baseline_oom = _mem_snapshot().get("oom_kill")
        atomic_write_json(self.root / "SUPERVISOR_START.json", {
            "schema": "STAGE_V_PARENT_AWARE_SUPERVISOR_START_V2",
            "control_plane_mode": "LOCAL_AUTONOMOUS",
            "ssh_is_hard_stop": False,
            "run_root": str(self.root), "source_commit": source["source_commit"], "source_tree": source["source_tree"],
            "parent_manifest": str(self.args.parent_manifest), "parent_manifest_sha256": manifest_sha,
            "approved_gpus": sorted(self.args.approved_gpus), "planned_parents": self.args.expected_parent_count,
            "supervisor_pid": os.getpid(), "supervisor_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
            "started_utc": self.start_utc,
        })

    def _worker_statuses(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for worker_root in sorted(path for path in self.root.glob("worker_gpu*") if path.is_dir()):
            heartbeat_path = worker_root / "WORKER_HEARTBEAT.json"
            status_path = worker_root / "WORKER_STATUS.json"
            value = read_json(heartbeat_path, {}) if heartbeat_path.is_file() else read_json(status_path, {})
            if isinstance(value, Mapping):
                row = dict(value)
                row["_worker_root"] = str(worker_root)
                row["_heartbeat_file_present"] = heartbeat_path.is_file()
                rows.append(row)
        return rows

    def _resource_snapshot(self) -> tuple[dict[str, Any], list[str]]:
        memory = _mem_snapshot()
        gpu_rows, gpu_error = gpu_snapshot(self.args.gpu_query_command)
        errors: list[str] = []
        if gpu_error and not self.args.skip_resource_checks:
            errors.append(gpu_error)
        if memory.get("available_ram_bytes") is not None and memory["available_ram_bytes"] < self.args.min_available_ram_gib * (1 << 30):
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
        xid = None if self.args.skip_resource_checks else _xid_status(self.start_utc)
        if xid:
            errors.append("NVIDIA_XID")
        workers = self._worker_statuses()
        active = [row for row in workers if row.get("state") in {"STARTING", "RUNNING"}]
        assigned = [int(row["gpu_id"]) for row in active if row.get("gpu_id") is not None]
        if len(assigned) != len(set(assigned)):
            errors.append("MULTIPLE_PROJECT_WORKERS_PER_GPU")
        if len(active) > len(self.args.approved_gpus):
            errors.append("ACTIVE_WORKERS_EXCEED_APPROVED_GPUS")
        if any(gpu not in self.args.approved_gpus or gpu == 5 for gpu in assigned):
            errors.append("UNAPPROVED_OR_GPU5_WORKER")
        run_manifest = read_json(self.root / "RUN_MANIFEST.json", {})
        registered_dispatcher = run_manifest.get("dispatcher_pid") if isinstance(run_manifest, Mapping) else None
        if registered_dispatcher and self.dispatcher and int(registered_dispatcher) != self.dispatcher.pid:
            errors.append("DISPATCHER_PID_MISMATCH")
        tasks = self.queue.list_tasks() if self.queue else []
        for row in active:
            worker_pid = int(row.get("worker_pid") or 0)
            child_pid = int(row.get("child_pid") or 0)
            if not row.get("_heartbeat_file_present"):
                errors.append("WORKER_HEARTBEAT_MISSING")
            worker_root = Path(str(row.get("_worker_root") or self.root))
            pid_receipt = read_json(self.root / f"{worker_root.name}.pid.json", {})
            expected_worker_pid = int(pid_receipt.get("pid") or 0) if isinstance(pid_receipt, Mapping) else 0
            if expected_worker_pid and worker_pid and expected_worker_pid != worker_pid:
                errors.append("WORKER_PID_MISMATCH")
            parent = row.get("current_parent")
            if parent:
                matching = [task for task in tasks if task.get("parent_id") == parent and task.get("state") in {"LEASED", "RUNNING", "COMMITTING"}]
                accepted_transition = any(task.get("parent_id") == parent and task.get("state") in {"DONE", "DONE_VALID", "DONE_CLASSIFIED_TC"} for task in tasks)
                if (len(matching) != 1 or matching[0].get("lease_owner") != row.get("worker_id")) and not accepted_transition:
                    errors.append("WORKER_PARENT_IDENTITY_MISMATCH")
            updated = row.get("updated_utc")
            age = time.time() - (_datetime.datetime.fromisoformat(str(updated).replace("Z", "+00:00")).timestamp() if updated else self.start_epoch)
            if age > self.args.heartbeat_stale_seconds and not pid_alive(worker_pid) and not pid_alive(child_pid):
                if row.get("current_parent"):
                    self._write_timeout_receipt(row, "WORKER_HEARTBEAT_LOST")
                    errors.append("WORKER_HEARTBEAT_LOST_BOUND")
                else:
                    errors.append("WORKER_HEARTBEAT_LOST")
            self._check_parent_timeout(row, errors)
        metrics = {
            **memory,
            "gpu_memory": gpu_rows,
            "gpu_xid_status": xid,
            "active_worker_pids": [row.get("worker_pid") for row in active],
            "gpu_assignments": [{"gpu_id": row.get("gpu_id"), "worker_pid": row.get("worker_pid"), "parent": row.get("current_parent")} for row in active],
            "resource_errors": sorted(set(errors)),
        }
        return metrics, sorted(set(errors))

    def _check_parent_timeout(self, row: Mapping[str, Any], errors: list[str]) -> None:
        started = float(row.get("parent_started_epoch") or 0)
        if not started or not row.get("current_parent"):
            return
        now = time.time()
        parent_hard = float(self.timeout_policy.get("parent_hard_seconds", 10 * 3600))
        branch_hard = float(self.timeout_policy.get("branch_hard_seconds", 4 * 3600))
        progress_epochs = {
            "simulator_step": float(row.get("last_simulator_progress_epoch") or row.get("last_progress_epoch") or started),
            "branch_progress": float(row.get("last_branch_progress_epoch") or row.get("last_progress_epoch") or started),
            "artifact": float(row.get("last_artifact_epoch") or started),
        }

        def stalled(since: float, threshold: float) -> bool:
            return now - since > threshold and all(now - value > threshold for value in progress_epochs.values())

        branch_started = float(row.get("branch_started_epoch") or started)
        if row.get("current_branch") and stalled(branch_started, branch_hard):
            self._write_timeout_receipt(row, "BRANCH_WATCHDOG_TIMEOUT_BOUND", branch_hard, progress_epochs)
            errors.append("BRANCH_WATCHDOG_TIMEOUT_BOUND")
        if stalled(started, parent_hard):
            self._write_timeout_receipt(row, "PARENT_WATCHDOG_TIMEOUT_BOUND", parent_hard, progress_epochs)
            errors.append("PARENT_WATCHDOG_TIMEOUT_BOUND")

    def _write_timeout_receipt(self, row: Mapping[str, Any], reason: str, threshold: float | None = None,
                               progress_epochs: Mapping[str, float] | None = None) -> None:
        if not row.get("current_parent"):
            return
        directory = self.root / "TIMEOUT_RECEIPTS"
        directory.mkdir(parents=True, exist_ok=True)
        filename = "__".join((sanitize_key(str(row.get("current_parent"))), sanitize_key(str(row.get("current_branch") or "NO_BRANCH")),
                              f"gpu{row.get('gpu_id')}", sanitize_key(reason))) + ".json"
        target = directory / filename
        if target.exists():
            return
        atomic_write_json(target, {
            "schema": "STAGE_V_PARENT_TIMEOUT_RECEIPT_V2",
            "reason": reason,
            "canonical_parent_key": row.get("current_parent"), "branch": row.get("current_branch"),
            "worker_pid": row.get("worker_pid"), "worker_pgid": row.get("worker_pgid"),
            "child_pid": row.get("child_pid"), "gpu_id": row.get("gpu_id"),
            "last_heartbeat_utc": row.get("updated_utc"), "last_simulator_step": row.get("simulator_step"),
            "last_branch_progress": row.get("branch_progress"), "last_artifact_utc": row.get("last_artifact_utc"),
            "elapsed_seconds": time.time() - float(row.get("parent_started_epoch") or time.time()),
            "parent_hard_seconds": self.timeout_policy.get("parent_hard_seconds"),
            "branch_hard_seconds": self.timeout_policy.get("branch_hard_seconds"),
            "threshold_seconds": threshold,
            "timeout_basis": dict(progress_epochs or {}),
            "last_simulator_progress_epoch": row.get("last_simulator_progress_epoch"),
            "last_branch_progress_epoch": row.get("last_branch_progress_epoch"),
            "gpu_state": {"utilization_percent": row.get("gpu_utilization_percent"), "memory_used_mib": row.get("gpu_memory_used_mib")},
            "written_utc": utc_now(),
        })

    def _heartbeat(self, metrics: Mapping[str, Any]) -> None:
        self.heartbeat_count += 1
        tasks = self.queue.list_tasks() if self.queue else []
        active = [row for row in self._worker_statuses() if row.get("state") in {"STARTING", "RUNNING"}]
        current = active[0] if active else {}
        payload = {
            "schema": "STAGE_V_PARENT_AWARE_LOCAL_HEARTBEAT_V2",
            "control_plane_mode": "LOCAL_AUTONOMOUS", "ssh_is_hard_stop": False,
            "run_root": str(self.root), "source_commit": self.args.expected_source_commit, "source_tree": self.args.expected_source_tree,
            "supervisor_pid": os.getpid(), "supervisor_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
            "dispatcher_pid": self.dispatcher.pid if self.dispatcher else None,
            "active_worker_pids": metrics.get("active_worker_pids", []), "gpu_assignments": metrics.get("gpu_assignments", []),
            "planned_parents": len(tasks), "completed_parents": sum(task.get("state") in {"DONE_VALID", "DONE", "DONE_CLASSIFIED_TC"} for task in tasks),
            "accepted_parent_results": sum(task.get("state") == "DONE_VALID" for task in tasks),
            "failed_parents": sum(task.get("state") not in {"PENDING", "RETRY_READY", "LEASED", "RUNNING", "DONE_VALID", "DONE", "DONE_CLASSIFIED_TC"} for task in tasks),
            "current_parent": current.get("current_parent"), "current_branch": current.get("current_branch"),
            "last_artifact_utc": current.get("last_artifact_utc"),
            "available_ram_gib": metrics.get("available_ram_gib"), "swap_in": metrics.get("swap_in"), "swap_out": metrics.get("swap_out"),
            "oom_kill": metrics.get("oom_kill"), "gpu_memory": metrics.get("gpu_memory", []), "gpu_xid_status": metrics.get("gpu_xid_status"),
            "filesystem_writable": os.access(self.root, os.W_OK), "resource_errors": metrics.get("resource_errors", []),
            "ssh_probe_success_count": self.ssh_success_count, "ssh_probe_failure_count": self.ssh_failure_count,
            "longest_ssh_unavailable_interval_seconds": self._ssh_outage_seconds(),
            "external_root_process_present": pid_alive(self.args.external_pid), "external_root_process_pid": self.args.external_pid,
            "external_root_process_terminated": False, "heartbeat_count": self.heartbeat_count, "updated_utc": utc_now(),
        }
        atomic_write_json(self.root / "LOCAL_HEARTBEAT.json", payload)
        atomic_write_json(self.root / "QUEUE_STATE.json", {"schema": "STAGE_V_QUEUE_STATE_V2", "tasks": tasks, "updated_utc": utc_now()})
        _write_csv_row(self.root / "RESOURCE_LEDGER.csv", {
            "utc": payload["updated_utc"], "available_ram_gib": payload["available_ram_gib"], "swap_in": payload["swap_in"],
            "swap_out": payload["swap_out"], "oom_kill": payload["oom_kill"], "active_workers": len(active),
            "gpu_xid_status": payload["gpu_xid_status"], "ssh_failures": self.ssh_failure_count,
        })
        self.last_heartbeat = payload

    def _probe_ssh(self) -> None:
        if not self.args.ssh_probe_command:
            return
        result = subprocess.run(self.args.ssh_probe_command, shell=True, capture_output=True, text=True, timeout=self.args.ssh_probe_timeout)
        now = time.monotonic()
        if result.returncode == 0:
            self.ssh_success_count += 1
            if self.ssh_failed_since is not None:
                self.longest_ssh_outage = max(self.longest_ssh_outage, now - self.ssh_failed_since)
                self.ssh_failed_since = None
        else:
            self.ssh_failure_count += 1
            self.ssh_failed_since = self.ssh_failed_since or now

    def _ssh_outage_seconds(self) -> float:
        current = 0 if self.ssh_failed_since is None else time.monotonic() - self.ssh_failed_since
        return max(self.longest_ssh_outage, current)

    def _terminate_owned(self) -> dict[str, Any]:
        grace_seconds = float(getattr(self.args, "kill_grace_seconds", 20))
        if self.dispatcher is not None:
            terminate_process_group(self.dispatcher, grace_seconds=grace_seconds)
        rows = self._worker_statuses()
        current_pgid = os.getpgid(0) if hasattr(os, "getpgid") else os.getpid()
        terminated_pgids: set[int] = set()
        tracked_pids = [int(row.get(raw) or 0) for row in rows for raw in ("worker_pid", "child_pid") if int(row.get(raw) or 0) > 0]
        for row in rows:
            for raw_pgid in (row.get("worker_pgid"), row.get("child_pgid")):
                pgid = int(raw_pgid or 0)
                if pgid <= 0 or pgid == current_pgid or pgid in terminated_pgids or os.name != "posix":
                    continue
                members: list[int] = []
                for pid in tracked_pids:
                    if not pid_alive(pid):
                        continue
                    try:
                        if os.getpgid(pid) == pgid:
                            members.append(pid)
                    except OSError:
                        pass
                if not members:
                    continue
                terminated_pgids.add(pgid)
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    pass
        deadline = time.time() + grace_seconds
        while time.time() < deadline:
            if not any(pid_alive(pid) for pid in tracked_pids):
                break
            time.sleep(0.2)
        survivors = [pid for pid in tracked_pids if pid_alive(pid)]
        for pgid in terminated_pgids:
            group_survivor = False
            for pid in survivors:
                try:
                    if os.getpgid(pid) == pgid:
                        group_survivor = True
                        break
                except OSError:
                    pass
            if group_survivor:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
        deadline = time.time() + min(5.0, grace_seconds)
        while time.time() < deadline and any(pid_alive(pid) for pid in tracked_pids):
            time.sleep(0.1)
        remaining = [pid for pid in tracked_pids if pid_alive(pid)]
        return {"owned_pids_before": tracked_pids, "owned_pids_after": remaining,
                "process_reap_complete": not remaining}

    def _abort(self, reason: str) -> int:
        reap = self._terminate_owned()
        if not reap["process_reap_complete"]:
            reason = f"{reason};PROCESS_REAP_INCOMPLETE"
        atomic_write_json(self.root / "ABORTED_INCOMPLETE.json", {
            "schema": "STAGE_V_PARENT_AWARE_ABORT_V2", "status": "ABORTED_INCOMPLETE",
            "reason": reason, "control_plane_mode": "LOCAL_AUTONOMOUS", "ssh_is_hard_stop": False,
            "accepted_parent_results": 0, "scientific_validity": 0,
            **reap,
            "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0,
            "heartbeat_count": self.heartbeat_count, "aborted_utc": utc_now(),
        })
        return 1

    def _audit(self) -> int:
        command = [
            sys.executable, str(self.args.auditor_script), "--run-root", str(self.root),
            "--parent-manifest", str(self.args.parent_manifest), "--queue-db", str(self.args.queue_db),
            "--run-id", self.args.run_id, "--expected-parent-count", str(self.args.expected_parent_count),
            "--expected-source-commit", self.args.expected_source_commit, "--expected-source-tree", self.args.expected_source_tree,
        ]
        if self.args.science_source_commit:
            command += ["--science-source-commit", self.args.science_source_commit]
        if self.args.science_source_tree:
            command += ["--science-source-tree", self.args.science_source_tree]
        if self.args.science_provenance:
            command += ["--science-provenance", str(self.args.science_provenance)]
        result = subprocess.run(command, cwd=str(self.args.repo_root), capture_output=True, text=True, check=False, timeout=self.args.audit_timeout)
        (self.root / "AUDITOR_STDOUT.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        return result.returncode

    def _complete(self) -> int:
        if self._audit() != 0:
            return self._abort("INDEPENDENT_AUDITOR_FAIL")
        audit = read_json(self.root / "STAGE_V_COUNTERFACTUAL_AUDIT.json", {})
        if not isinstance(audit, Mapping) or audit.get("verdict") != "PASS":
            return self._abort("AUDIT_VERDICT_NOT_PASS")
        tasks = self.queue.list_tasks() if self.queue else []
        atomic_write_json(self.root / "STAGE_V_CLOSURE_RECEIPT.json", {
            "schema": "STAGE_V_FORMAL_MAP_CLOSURE_RECEIPT_V2",
            "status": "STAGE_V_FORMAL_MAP_CLOSED",
            "source_commit": self.args.expected_source_commit,
            "source_tree": self.args.expected_source_tree,
            "planned_parents": self.args.expected_parent_count,
            "completed_parents": sum(task.get("state") == "DONE_VALID" for task in tasks),
            "accepted_parents": sum(task.get("state") == "DONE_VALID" for task in tasks),
            "manifest_sha256": sha256_file(self.args.parent_manifest),
            "dispatcher_complete": read_json(self.root / "DISPATCHER_COMPLETE.json", {}).get("status") == "PASS",
            "supervisor_complete": True,
            "invalid_branches": 0,
            "duplicate_identities": 0,
            "missing_identities": 0,
            "control_branch_failure": 0,
            "eval160_reads": 0,
            "protected_eval_reads": 0,
            "attack_rollouts": 0,
            "created_utc": utc_now(),
        })
        complete_payload = {
            "schema": "STAGE_V_PARENT_AWARE_SUPERVISOR_COMPLETE_V2", "status": "PASS",
            "control_plane_mode": "LOCAL_AUTONOMOUS", "ssh_is_hard_stop": False,
            "planned_parents": self.args.expected_parent_count,
            "accepted_parent_results": self.args.expected_parent_count,
            "dispatcher_pid": self.dispatcher.pid if self.dispatcher else None,
            "heartbeat_count": self.heartbeat_count, "ssh_probe_success_count": self.ssh_success_count,
            "ssh_probe_failure_count": self.ssh_failure_count, "longest_ssh_unavailable_interval_seconds": self._ssh_outage_seconds(),
            "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0, "completed_utc": utc_now(),
        }
        atomic_write_json(self.root / "SUPERVISOR_COMPLETE.json", complete_payload)
        self._write_seals()
        closure = read_json(self.root / "STAGE_V_CLOSURE_RECEIPT.json", {})
        closure["root_seal"] = sha256_file(self.root / "SHA256SUMS")
        atomic_write_json(self.root / "STAGE_V_CLOSURE_RECEIPT.json", closure)
        return 0

    def _write_seals(self) -> None:
        # ponytail: keep the closure receipt out of the file list so it can
        # bind the completed SHA256SUMS without a circular hash dependency.
        excluded = {"SHA256SUMS", "SHA256SUMS.sha256", "STAGE_V_CLOSURE_RECEIPT.json"}
        rows: list[str] = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name not in excluded:
                rows.append(f"{sha256_file(path)}  {path.relative_to(self.root).as_posix()}")
        (self.root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (self.root / "SHA256SUMS.sha256").write_text(sha256_file(self.root / "SHA256SUMS") + "  SHA256SUMS\n", encoding="utf-8")

    def run(self) -> int:
        lock_parent = self.args.lock_path.resolve().parent
        self.lock = ExclusiveLock(self.args.lock_path.resolve(), {"schema": "STAGE_V_DYNAMIC_LOCK_V2", "supervisor_pid": os.getpid(), "run_root": str(self.root), "started_utc": self.start_utc})
        try:
            self.lock.acquire(lock_parent)
            self._prepare()
            dispatcher_script = self.args.dispatcher_script.resolve()
            command = [sys.executable, str(dispatcher_script), "--run-root", str(self.root), "--repo-root", str(self.args.repo_root),
                       "--parent-manifest", str(self.args.parent_manifest), "--queue-db", str(self.args.queue_db), "--run-id", self.args.run_id,
                       "--source-commit", self.args.expected_source_commit, "--source-tree", self.args.expected_source_tree,
                       "--expected-parent-count", str(self.args.expected_parent_count), "--required-workers", "8",
                       "--excluded-gpus", ",".join(map(str, self.args.excluded_gpus)), "--protected-pids", ",".join(map(str, self.args.protected_pids)),
                       "--canary-peak-mib", str(self.args.canary_peak_mib), "--preflight-file", str(self.args.preflight_file),
                       "--preflight-output", str(self.args.preflight_file), "--worker-heartbeat-seconds", str(self.args.worker_heartbeat_seconds),
                       "--max-attempts", str(self.args.max_attempts), "--probe-limit", str(self.args.probe_limit),
                       "--science-source-commit", self.args.science_source_commit,
                       "--science-source-tree", self.args.science_source_tree]
            if self.args.science_provenance:
                command += ["--science-provenance", str(self.args.science_provenance)]
            for value in (self.args.science_runner, self.args.science_repo_root, self.args.science_parent_manifest):
                if value:
                    command += ["--science-runner" if value == self.args.science_runner else "--science-repo-root" if value == self.args.science_repo_root else "--science-parent-manifest", str(value)]
            if self.args.worker_command:
                command += ["--worker-command", self.args.worker_command]
            log = (self.root / "DISPATCHER_STDOUT.log").open("a", encoding="utf-8")
            self.dispatcher = subprocess.Popen(command, cwd=str(self.args.repo_root), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self._heartbeat(self._resource_snapshot()[0])
            while True:
                if self.dispatcher.poll() is not None:
                    if self.dispatcher.returncode != 0:
                        return self._abort(f"DISPATCHER_EXIT:{self.dispatcher.returncode}")
                    return self._complete()
                metrics, errors = self._resource_snapshot()
                self._probe_ssh()
                self._heartbeat(metrics)
                if errors:
                    return self._abort(";".join(errors))
                time.sleep(self.args.poll_seconds)
        except RuntimeError as exc:
            if "PRELAUNCH_WAITING_FOR_8_GPUS" in str(exc):
                return 75
            if self.root.exists():
                return self._abort(str(exc))
            return 2
        except Exception as exc:
            if self.root.exists():
                return self._abort(f"SUPERVISOR_EXCEPTION:{type(exc).__name__}:{exc}")
            return 2
        finally:
            if self.queue:
                self.queue.close()
            if self.lock:
                self.lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--parent-manifest-sha256", default="")
    parser.add_argument("--queue-db", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-parent-count", type=int, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--approved-gpus", type=lambda value: [int(item) for item in value.split(",") if item], required=True)
    parser.add_argument("--excluded-gpus", type=lambda value: [int(item) for item in value.split(",") if item], default=[5])
    parser.add_argument("--protected-pids", type=lambda value: [int(item) for item in value.split(",") if item], default=[])
    parser.add_argument("--external-pid", type=int, default=0)
    parser.add_argument("--preflight-file", type=Path, required=True)
    parser.add_argument("--wait-for-gpus", action="store_true")
    parser.add_argument("--preflight-interval-seconds", type=float, default=300.0)
    parser.add_argument("--timeout-policy", type=Path)
    parser.add_argument("--dispatcher-script", type=Path, required=True)
    parser.add_argument("--auditor-script", type=Path, required=True)
    parser.add_argument("--science-runner", type=Path)
    parser.add_argument("--science-provenance", type=Path)
    parser.add_argument("--science-source-commit", default="")
    parser.add_argument("--science-source-tree", default="")
    parser.add_argument("--science-repo-root", type=Path)
    parser.add_argument("--science-parent-manifest", type=Path)
    parser.add_argument("--worker-command", default="")
    parser.add_argument("--canary-peak-mib", type=float, default=0.0)
    parser.add_argument("--probe-limit", type=int, default=24)
    parser.add_argument("--worker-heartbeat-seconds", type=float, default=30)
    parser.add_argument("--heartbeat-stale-seconds", type=float, default=600)
    parser.add_argument("--parent-hard-seconds", type=float, default=10 * 3600)
    parser.add_argument("--min-available-ram-gib", type=float, default=128)
    parser.add_argument("--gpu-query-command", default="nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--audit-timeout", type=float, default=3600)
    parser.add_argument("--ssh-probe-command", default="")
    parser.add_argument("--ssh-probe-timeout", type=float, default=10)
    parser.add_argument("--kill-grace-seconds", type=float, default=20)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--skip-resource-checks", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.approved_gpus) != 8 or 5 in args.approved_gpus:
        raise SystemExit("exactly eight approved GPUs are required and GPU5 is excluded")
    return DynamicSupervisor(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
