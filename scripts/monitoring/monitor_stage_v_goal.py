"""Local fail-closed Goal Gatekeeper for the Stage V formal map.

The gatekeeper is deliberately boring: it observes one existing Stage V root,
never starts a second dispatcher, and only launches a pre-registered read-only
Stage V2 command after an independent closure audit and a root seal pass.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Linux is the production target.
    fcntl = None
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "detector_v5"))

from run_stage_v_local_supervisor import (  # noqa: E402
    discover_workers,
    parse_csv_ints,
    pid_alive,
    process_command,
    process_cpu_percent,
    validate_workers,
)

from audit_stage_v_closure import (  # noqa: E402
    atomic_write_json,
    audit_closure,
    parent_progress,
    sha256_file,
    verify_root_seal,
    write_root_seal,
)
from audit_stage_v_live_parents import (  # noqa: E402
    markdown as live_parent_markdown,
    triage as live_parent_triage,
)


SCHEMA = "STAGE_V_GOAL_GATEKEEPER_V1"
V2_COMMAND_SCHEMA = "STAGE_V2_COMMAND_V2"
V2_COMMAND_PLAN_SCHEMA = "STAGE_V2_COMMAND_PLAN_V1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TERMINAL_EVENT_STATES = {
    "STAGE_V_HARD_STOP",
    "STAGE_V_FORMAL_MAP_CLOSED",
    "STAGE_V2_STARTED",
    "STAGE_V2_PASS",
    "STAGE_V2_FAIL",
}
FORBIDDEN_V2 = re.compile(
    r"eval160|protected[_-]?eval|\bvis\b|\bpgd\b|attack[_-]?(?:rollout|matrix)|run[_-]?attack|final[_-]?detector|\bstudent\b|threshold|guard",
    re.IGNORECASE,
)
_WINDOWS_LOCK_PATHS: set[str] = set()


class MonitorError(RuntimeError):
    pass


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def parse_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, raw = line.partition(":")
            fields = raw.strip().split()
            if fields:
                values[key] = int(fields[0]) * (1024 if len(fields) > 1 and fields[1] == "kB" else 1)
    except (OSError, ValueError):
        pass
    return values


def read_vmstat() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/vmstat").read_text(encoding="utf-8").splitlines():
            key, _, raw = line.partition(" ")
            try:
                values[key] = int(raw.strip())
            except ValueError:
                continue
    except (OSError, ValueError):
        pass
    return values


def run_command(argv: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    try:
        result = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", str(exc)


def read_gpu_snapshot() -> tuple[list[dict[str, Any]], str | None]:
    code, stdout, stderr = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if code != 0:
        return [], f"GPU_QUERY_FAILED:{stderr.strip() or code}"
    rows: list[dict[str, Any]] = []
    try:
        for line in stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 3:
                continue
            rows.append(
                {
                    "index": int(fields[0]),
                    "memory_used_mib": int(float(fields[1])),
                    "memory_free_mib": int(float(fields[2])),
                    "utilization_gpu_percent": int(float(fields[3])) if len(fields) > 3 else None,
                }
            )
    except ValueError as exc:
        return [], f"GPU_QUERY_PARSE_FAILED:{exc}"
    if not rows:
        return [], "GPU_QUERY_EMPTY"
    return rows, None


def read_xid_status(since: str) -> tuple[str, str | None]:
    try:
        start = _datetime.datetime.fromisoformat(since).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        start = since
    code, stdout, stderr = run_command(["journalctl", "-k", "--since", start, "--no-pager"], timeout=15)
    if code != 0:
        return "UNKNOWN", f"KERNEL_LOG_QUERY_FAILED:{stderr.strip() or code}"
    text = stdout.lower()
    if "xid" in text or "fallen off the bus" in text or "gpu has fallen off" in text:
        return "XID_DETECTED", "NVIDIA_XID"
    return "CLEAR", None


def _read_oom_before(root: Path) -> int | None:
    for name in ("OOM_KILL_BEFORE.txt", "OOM_KILL_BEFORE"):
        path = root / name
        if not path.is_file():
            continue
        match = re.search(r"(?:oom[_ -]?kill\s*[=:]\s*)?(\d+)", path.read_text(encoding="utf-8", errors="replace"))
        if match:
            return int(match.group(1))
    return None


def _read_int(value: Any) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _boundary_counters(*values: Any) -> dict[str, int]:
    keys = ("eval160_reads", "protected_eval_reads", "attack_rollouts", "vis_rollouts", "pgd_rollouts", "vis_pgd_attack_rollouts")
    result = {key: 0 for key in keys}
    for value in values:
        if not isinstance(value, Mapping):
            continue
        for key in keys:
            number = _read_int(value.get(key))
            if number is not None:
                result[key] = max(result[key], number)
    return result


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return True


class ExclusiveMonitorLock:
    def __init__(self, path: Path, metadata: Mapping[str, Any]):
        self.path = path
        self.metadata = dict(metadata)
        self.handle = None

    def acquire(self, monitor_root: Path) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = self.path.read_text(encoding="utf-8", errors="replace") if self.path.is_file() else ""
        except OSError:
            existing = ""
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            if fcntl is not None:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:  # pragma: no cover - Windows test fallback.
                key = str(self.path.resolve()).lower()
                if key in _WINDOWS_LOCK_PATHS:
                    raise OSError("duplicate lock in this process")
                _WINDOWS_LOCK_PATHS.add(key)
        except (BlockingIOError, OSError) as exc:
            self.handle.close()
            self.handle = None
            if fcntl is None:
                _WINDOWS_LOCK_PATHS.discard(str(self.path.resolve()).lower())
            raise MonitorError("DUPLICATE_MONITOR") from exc
        if existing.strip():
            atomic_write_json(
                monitor_root / "STAGE_V_MONITOR_STALE_LOCK_AUDIT.json",
                {
                    "schema": "STAGE_V_MONITOR_STALE_LOCK_AUDIT_V1",
                    "lock_path": str(self.path),
                    "observed_metadata": existing,
                    "audited_utc": utc_now(),
                },
            )
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps(self.metadata, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        if self.handle is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
                else:  # pragma: no cover - Windows test fallback.
                    _WINDOWS_LOCK_PATHS.discard(str(self.path.resolve()).lower())
            finally:
                self.handle.close()
                self.handle = None


class Gatekeeper:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.stage_v_root = Path(args.stage_v_root).resolve()
        self.goal_root = Path(args.goal_root).resolve()
        self.monitor_root = self.goal_root / "MONITOR"
        self.monitor_root.mkdir(parents=True, exist_ok=True)
        self.expected_gpus = list(args.expected_gpus)
        self.reserved_gpus = set(args.reserved_gpus)
        self.protected_pid = args.protected_pid
        self.monitor_start_utc = utc_now()
        start_metadata = parse_json(self.stage_v_root / "SUPERVISOR_START.json")
        self.start_utc = (
            str(start_metadata.get("started_utc"))
            if isinstance(start_metadata, Mapping) and start_metadata.get("started_utc")
            else self.monitor_start_utc
        )
        self.baseline_oom = _read_oom_before(self.stage_v_root)
        self.baseline_oom_source = "OOM_KILL_BEFORE.txt" if self.baseline_oom is not None else "live_probe"
        self.swap_bad_streak = 0
        self.last_event: str | None = None
        self.last_status: str | None = None
        self.last_heartbeat_mtime: float | None = None
        self.last_heartbeat_count: int | None = None
        self.expected_heartbeat_seconds, self.heartbeat_source = self._heartbeat_interval()
        self.stage_v2_process: subprocess.Popen[Any] | None = None
        self.stage_v2_root: Path | None = None
        self.lock = ExclusiveMonitorLock(
            Path(args.lock_path).resolve(),
            {
                "schema": "STAGE_V_MONITOR_LOCK_V1",
                "monitor_pid": os.getpid(),
                "monitor_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
                "stage_v_root": str(self.stage_v_root),
                "started_utc": self.monitor_start_utc,
            },
        )

    def _heartbeat_interval(self) -> tuple[float, str]:
        for name in ("RUN_MANIFEST.json", "SUPERVISOR_START.json"):
            value = parse_json(self.stage_v_root / name)
            if isinstance(value, Mapping):
                for key in ("heartbeat_interval_seconds", "heartbeat_interval", "expected_heartbeat_seconds"):
                    parsed = value.get(key)
                    try:
                        if parsed is not None and float(parsed) > 0:
                            return float(parsed), f"{name}:{key}"
                    except (TypeError, ValueError):
                        pass
        start = parse_json(self.stage_v_root / "SUPERVISOR_START.json")
        pid = _read_int(start.get("supervisor_pid")) if isinstance(start, Mapping) else None
        command = process_command(pid)
        match = re.search(r"--heartbeat-interval\s+(\d+(?:\.\d+)?)", command)
        if match:
            return float(match.group(1)), "supervisor_cmdline"
        return 30.0, "default_30_seconds"

    def _write_state(self, status: str, **extra: Any) -> None:
        self.last_status = status
        payload = {
            "schema": SCHEMA,
            "status": status,
            "stage_v_root": str(self.stage_v_root),
            "goal_root": str(self.goal_root),
            "source_commit": self.args.expected_source_commit,
            "source_tree": self.args.expected_source_tree,
            "expected_parent_count": self.args.expected_parent_count,
            "expected_gpus": self.expected_gpus,
            "reserved_gpus": sorted(self.reserved_gpus),
            "protected_pid": self.protected_pid,
            "protected_pid_signal_sent": False,
            "ssh_is_hard_stop": False,
            "control_plane_mode": "LOCAL_AUTONOMOUS",
            "updated_utc": utc_now(),
        }
        payload.update(extra)
        atomic_write_json(self.monitor_root / "STAGE_V_MONITOR_STATE.json", payload)

    def _emit(self, event: str, **extra: Any) -> None:
        if event == self.last_event:
            return
        record = {"schema": "STAGE_V_MONITOR_EVENT_V1", "event": event, "utc": utc_now()}
        record.update(extra)
        path = self.monitor_root / "STAGE_V_MONITOR_EVENTS.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.last_event = event

    def _write_monitor_heartbeat(self, resource: Mapping[str, Any], progress: Mapping[str, Any]) -> None:
        heartbeat = {
            "schema": "STAGE_V_MONITOR_HEARTBEAT_V1",
            "monitor_pid": os.getpid(),
            "monitor_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
            "stage_v_root": str(self.stage_v_root),
            "source_commit": self.args.expected_source_commit,
            "source_tree": self.args.expected_source_tree,
            "status": self.last_status,
            "heartbeat_age_seconds": resource.get("heartbeat_age_seconds"),
            "supervisor_pid": resource.get("supervisor_pid"),
            "dispatcher_pid": resource.get("dispatcher_pid"),
            "active_worker_pids": resource.get("active_worker_pids", []),
            "gpu_assignments": resource.get("gpu_assignments", []),
            "planned_parents": progress.get("planned_parent_count"),
            "completed_parents": progress.get("branch_complete_parent_count"),
            "accepted_parents": progress.get("accepted_parent_count"),
            "available_ram_gib": resource.get("available_ram_gib"),
            "swap_used_bytes": resource.get("swap_used_bytes"),
            "oom_kill": resource.get("oom_kill"),
            "oom_delta": resource.get("oom_delta"),
            "gpu_xid_status": resource.get("gpu_xid_status"),
            "gpu_memory": resource.get("gpu_memory", []),
            "ssh_probe_success_count": 0,
            "ssh_probe_failure_count": 0,
            "longest_ssh_unavailable_interval_seconds": None,
            "external_root_process_present": resource.get("protected_pid_present"),
            "external_root_process_pid": self.protected_pid,
            "external_root_process_cpu_percent": resource.get("protected_pid_cpu_percent"),
            "external_root_process_terminated": False,
            "filesystem_writable": resource.get("filesystem_writable"),
            "updated_utc": utc_now(),
        }
        atomic_write_json(self.monitor_root / "STAGE_V_MONITOR_HEARTBEAT.json", heartbeat)

    def _write_resource_sample(self, resource: Mapping[str, Any]) -> None:
        with (self.monitor_root / "STAGE_V_RESOURCE_SAMPLES.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"utc": utc_now(), **resource}, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _read_process_ids(self) -> tuple[int | None, int | None, list[dict[str, int]], list[str]]:
        heartbeat = parse_json(self.stage_v_root / "LOCAL_HEARTBEAT.json")
        start = parse_json(self.stage_v_root / "SUPERVISOR_START.json")
        supervisor = None
        dispatcher = None
        if isinstance(heartbeat, Mapping):
            supervisor = _read_int(heartbeat.get("supervisor_pid"))
            dispatcher = _read_int(heartbeat.get("dispatcher_pid"))
        if supervisor is None and isinstance(start, Mapping):
            supervisor = _read_int(start.get("supervisor_pid"))
        workers, worker_errors = discover_workers(self.stage_v_root, dispatcher)
        return supervisor, dispatcher, workers, worker_errors

    def _provenance_errors(self) -> list[str]:
        errors: list[str] = []
        for name in ("SUPERVISOR_START.json", "RUN_MANIFEST.json", "LOCAL_HEARTBEAT.json"):
            value = parse_json(self.stage_v_root / name)
            if not isinstance(value, Mapping):
                errors.append(f"{name}_missing_or_invalid")
                continue
            if value.get("source_commit") not in (None, self.args.expected_source_commit):
                errors.append(f"{name}_source_commit_mismatch")
            if value.get("source_tree") not in (None, self.args.expected_source_tree):
                errors.append(f"{name}_source_tree_mismatch")
        manifest = parse_json(self.stage_v_root / "RUN_MANIFEST.json")
        if isinstance(manifest, Mapping):
            actual_gpus = {int(item) for item in manifest.get("gpus", []) if str(item).isdigit()}
            if actual_gpus and actual_gpus != set(self.expected_gpus):
                errors.append("manifest_gpu_binding_mismatch")
            for key in ("eval160_reads", "protected_eval_reads", "attack_rollouts"):
                if _read_int(manifest.get(key)) not in (None, 0):
                    errors.append(f"{key}_not_zero")
            if manifest.get("gpu_5_used") is True:
                errors.append("GPU5_FORBIDDEN")
        return sorted(set(errors))

    def _resource_snapshot(self) -> dict[str, Any]:
        supervisor, dispatcher, workers, worker_errors = self._read_process_ids()
        worker_errors.extend(validate_workers(workers, self.expected_gpus, require_live=True))
        mem = read_meminfo()
        vm = read_vmstat()
        available = mem.get("MemAvailable")
        swap_total = mem.get("SwapTotal", 0)
        swap_free = mem.get("SwapFree", 0)
        swap_used = max(0, swap_total - swap_free)
        if swap_used > 0:
            self.swap_bad_streak += 1
        else:
            self.swap_bad_streak = 0
        oom = vm.get("oom_kill")
        if self.baseline_oom is None:
            self.baseline_oom = oom
            self.baseline_oom_source = "live_probe"
        oom_delta = None if oom is None or self.baseline_oom is None else oom - self.baseline_oom
        gpu_memory, gpu_error = read_gpu_snapshot()
        xid_status, xid_error = read_xid_status(self.start_utc)
        heartbeat = parse_json(self.stage_v_root / "LOCAL_HEARTBEAT.json")
        heartbeat_mtime = None
        heartbeat_age = None
        heartbeat_count = None
        if self.stage_v_root.joinpath("LOCAL_HEARTBEAT.json").is_file():
            heartbeat_mtime = self.stage_v_root.joinpath("LOCAL_HEARTBEAT.json").stat().st_mtime
            heartbeat_age = max(0.0, time.time() - heartbeat_mtime)
        if isinstance(heartbeat, Mapping):
            heartbeat_count = _read_int(heartbeat.get("heartbeat_count"))
        heartbeat_warning = self.expected_heartbeat_seconds * 3
        heartbeat_stale = self.expected_heartbeat_seconds * 6
        heartbeat_hard = max(self.expected_heartbeat_seconds * 10, 15 * 60)
        supervisor_alive = pid_alive(supervisor)
        dispatcher_alive = pid_alive(dispatcher)
        heartbeat_progressed = (
            heartbeat_count is not None
            and (self.last_heartbeat_count is None or heartbeat_count > self.last_heartbeat_count)
        )
        if heartbeat_count is not None:
            self.last_heartbeat_count = heartbeat_count
        filesystem_writable = True
        filesystem_error = None
        try:
            fd, name = tempfile.mkstemp(prefix=".stage-v-monitor-write.", dir=str(self.monitor_root))
            os.write(fd, b"monitor\n")
            os.fsync(fd)
            os.close(fd)
            Path(name).unlink(missing_ok=True)
        except OSError as exc:
            filesystem_writable = False
            filesystem_error = f"{type(exc).__name__}:{exc}"
        protected_present = bool(self.protected_pid and pid_alive(self.protected_pid))
        protected_command = process_command(self.protected_pid) if protected_present else ""
        protected_cpu = process_cpu_percent(self.protected_pid) if protected_present else None
        boundary = _boundary_counters(
            parse_json(self.stage_v_root / "RUN_MANIFEST.json"),
            parse_json(self.stage_v_root / "SUPERVISOR_COMPLETE.json"),
            heartbeat,
        )
        errors = list(worker_errors)
        errors.extend(self._provenance_errors())
        if available is not None and available < int(128 * (1 << 30)):
            errors.append("AVAILABLE_RAM_BELOW_HARD_STOP")
        if self.swap_bad_streak >= 2:
            errors.append("SWAP_HARD_STOP")
        if oom_delta is not None and oom_delta > 0:
            errors.append("OOM_KILL_COUNTER_INCREASED")
        if xid_status == "XID_DETECTED":
            errors.append("NVIDIA_XID")
        if xid_error or gpu_error:
            errors.extend(item for item in (xid_error, gpu_error) if item)
        if not filesystem_writable:
            errors.append("FILESYSTEM_WRITE_OR_FSYNC_FAILED")
        if heartbeat_age is not None and heartbeat_age > heartbeat_hard and (not supervisor_alive or not dispatcher_alive):
            errors.append("HEARTBEAT_HARD_STALE_WITH_DEAD_PID")
        if not supervisor_alive and not parse_json(self.stage_v_root / "SUPERVISOR_COMPLETE.json") and not parse_json(self.stage_v_root / "ABORTED_INCOMPLETE.json"):
            errors.append("SUPERVISOR_PID_LOST_BEFORE_TERMINAL_MARKER")
        if not dispatcher_alive and supervisor_alive and not parse_json(self.stage_v_root / "SUPERVISOR_COMPLETE.json") and not parse_json(self.stage_v_root / "ABORTED_INCOMPLETE.json"):
            errors.append("DISPATCHER_PID_LOST_BEFORE_TERMINAL_MARKER")
        if boundary["eval160_reads"] or boundary["protected_eval_reads"]:
            errors.append("PROTECTED_EVAL_BOUNDARY_NONZERO")
        if boundary["vis_rollouts"] or boundary["pgd_rollouts"] or boundary["vis_pgd_attack_rollouts"]:
            errors.append("VIS_PGD_BOUNDARY_NONZERO")
        if self._unexpected_boundary_process():
            errors.append("UNEXPECTED_PROTECTED_OR_ATTACK_PROCESS")
        return {
            "supervisor_pid": supervisor,
            "dispatcher_pid": dispatcher,
            "supervisor_alive": supervisor_alive,
            "dispatcher_alive": dispatcher_alive,
            "active_worker_pids": [item["pid"] for item in workers],
            "gpu_assignments": workers,
            "worker_errors": sorted(set(worker_errors)),
            "available_ram_bytes": available,
            "available_ram_gib": round(available / (1 << 30), 3) if available is not None else None,
            "swap_used_bytes": swap_used,
            "swap_bad_streak": self.swap_bad_streak,
            "swap_in": vm.get("pswpin"),
            "swap_out": vm.get("pswpout"),
            "oom_kill": oom,
            "oom_delta": oom_delta,
            "oom_baseline": self.baseline_oom,
            "oom_baseline_source": self.baseline_oom_source,
            "gpu_memory": gpu_memory,
            "gpu_xid_status": xid_status,
            "heartbeat_age_seconds": round(heartbeat_age, 3) if heartbeat_age is not None else None,
            "heartbeat_count": heartbeat_count,
            "heartbeat_progressed": heartbeat_progressed,
            "heartbeat_warning_seconds": heartbeat_warning,
            "heartbeat_stale_seconds": heartbeat_stale,
            "heartbeat_hard_stop_seconds": heartbeat_hard,
            "heartbeat_threshold_source": self.heartbeat_source,
            "filesystem_writable": filesystem_writable,
            "filesystem_error": filesystem_error,
            "protected_pid_present": protected_present,
            "protected_pid_command": protected_command,
            "protected_pid_cpu_percent": protected_cpu,
            "protected_pid_signal_sent": False,
            "boundary_counters": boundary,
            "hard_stop_errors": sorted(set(errors)),
        }

    def _unexpected_boundary_process(self) -> bool:
        code, stdout, _stderr = run_command(["ps", "-eo", "pid=,args="], timeout=10)
        if code != 0:
            return False
        root = str(self.stage_v_root)
        for line in stdout.splitlines():
            if root not in line:
                continue
            if FORBIDDEN_V2.search(line):
                return True
        return False

    def _write_parent_progress(self) -> dict[str, Any]:
        start = parse_json(self.stage_v_root / "SUPERVISOR_START.json")
        manifest_path = Path(start["parent_manifest"]) if isinstance(start, Mapping) and start.get("parent_manifest") else self.stage_v_root.parent / "STAGE_V_CLEAN_SUCCESS_PARENT_MANIFEST.json"
        full_audit = (self.stage_v_root / "SUPERVISOR_COMPLETE.json").is_file()
        progress = parent_progress(
            self.stage_v_root,
            parent_manifest=manifest_path,
            expected_source_commit=self.args.expected_source_commit,
            expected_source_tree=self.args.expected_source_tree,
            full_audit=full_audit,
        )
        atomic_write_json(self.monitor_root / "STAGE_V_PARENT_PROGRESS.json", progress)
        return progress

    def _write_live_parent_triage(self) -> None:
        start = parse_json(self.stage_v_root / "SUPERVISOR_START.json")
        if not isinstance(start, Mapping) or not start.get("parent_manifest"):
            return
        try:
            report = live_parent_triage(
                self.stage_v_root,
                Path(str(start["parent_manifest"])),
                expected_source_commit=self.args.expected_source_commit,
                expected_source_tree=self.args.expected_source_tree,
                only_nonpass=True,
            )
            atomic_write_json(self.monitor_root / "STAGE_V_LIVE_PARENT_TRIAGE.json", report)
            atomic_write_text(self.monitor_root / "STAGE_V_LIVE_PARENT_TRIAGE.md", live_parent_markdown(report))
        except (OSError, ValueError, TypeError) as exc:
            atomic_write_json(self.monitor_root / "STAGE_V_LIVE_PARENT_TRIAGE_ERROR.json", {"schema": "STAGE_V_LIVE_PARENT_TRIAGE_ERROR_V1", "error": str(exc), "utc": utc_now()})

    def _terminate_project(self, resource: Mapping[str, Any]) -> dict[str, Any]:
        killed: list[int] = []
        supervisor = _read_int(resource.get("supervisor_pid"))
        dispatcher = _read_int(resource.get("dispatcher_pid"))
        start = parse_json(self.stage_v_root / "SUPERVISOR_START.json")
        pgid = _read_int(start.get("supervisor_pgid")) if isinstance(start, Mapping) else None
        protected = self.protected_pid
        if pgid and pgid != os.getpgid(0) and pgid != protected:
            try:
                os.killpg(pgid, signal.SIGTERM)
                killed.append(pgid)
            except (OSError, ProcessLookupError):
                pass
            deadline = time.monotonic() + self.args.kill_grace_seconds
            while time.monotonic() < deadline:
                if supervisor is None or not pid_alive(supervisor):
                    break
                time.sleep(0.2)
            if supervisor and pid_alive(supervisor):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
        for pid in [supervisor, dispatcher, *[item["pid"] for item in resource.get("gpu_assignments", [])]]:
            if not pid or pid == protected or pid in killed or not pid_alive(pid):
                continue
            command = process_command(pid)
            if str(self.stage_v_root) not in command:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                killed.append(pid)
            except (OSError, ProcessLookupError):
                pass
        return {"terminated_project_ids": sorted(set(killed)), "protected_pid_signal_sent": False}

    def _hard_stop(self, reasons: list[str], resource: Mapping[str, Any]) -> str:
        existing = parse_json(self.stage_v_root / "ABORTED_INCOMPLETE.json")
        if not isinstance(existing, Mapping):
            termination = self._terminate_project(resource)
            receipt = {
                "schema": "STAGE_V_HARD_STOP_RECEIPT_V1",
                "status": "ABORTED_INCOMPLETE",
                "stage_v_root": str(self.stage_v_root),
                "source_commit": self.args.expected_source_commit,
                "source_tree": self.args.expected_source_tree,
                "reasons": sorted(set(reasons)),
                "terminated_project_ids": termination["terminated_project_ids"],
                "protected_pid": self.protected_pid,
                "protected_pid_signal_sent": False,
                "external_process_terminated": False,
                "accepted_parent_results": 0,
                "eval160_reads": 0,
                "protected_eval_reads": 0,
                "vis_pgd_attack_rollouts": 0,
                "utc": utc_now(),
            }
            atomic_write_json(self.stage_v_root / "HARD_STOP_RECEIPT.json", receipt)
            atomic_write_text(self.stage_v_root / "STOP", json.dumps(receipt, sort_keys=True) + "\n")
            atomic_write_json(
                self.stage_v_root / "ABORTED_INCOMPLETE.json",
                {
                    "schema": "STAGE_V_GOAL_GATEKEEPER_ABORT_V1",
                    "status": "ABORTED_INCOMPLETE",
                    "control_plane_mode": "LOCAL_AUTONOMOUS",
                    "control_plane_abort_reason": ";".join(sorted(set(reasons))),
                    "accepted_parent_results": 0,
                    "scientific_validity": 0,
                    "eval160_reads": 0,
                    "protected_eval_reads": 0,
                    "vis_pgd_attack_rollouts": 0,
                    "aborted_utc": utc_now(),
                },
            )
            try:
                start = parse_json(self.stage_v_root / "SUPERVISOR_START.json")
                manifest_path = (
                    Path(start["parent_manifest"])
                    if isinstance(start, Mapping) and start.get("parent_manifest")
                    else self.stage_v_root.parent / "STAGE_V_CLEAN_SUCCESS_PARENT_MANIFEST.json"
                )
                audit = audit_closure(
                    self.stage_v_root,
                    parent_manifest=manifest_path,
                    expected_source_commit=self.args.expected_source_commit,
                    expected_source_tree=self.args.expected_source_tree,
                    expected_parent_count=self.args.expected_parent_count,
                    require_root_seal=False,
                )
                atomic_write_json(self.monitor_root / "STAGE_V_HARD_STOP_AUDIT.json", audit)
            except Exception as exc:  # the hard-stop receipt is primary evidence
                atomic_write_json(
                    self.monitor_root / "STAGE_V_HARD_STOP_AUDIT_ERROR.json",
                    {"error": str(exc), "utc": utc_now()},
                )
        self._write_state("ABORTED_INCOMPLETE", hard_stop_reasons=sorted(set(reasons)), accepted_parent_results=0)
        self._emit("STAGE_V_HARD_STOP", reasons=sorted(set(reasons)))
        return "STOP"

    def _run_closure(self, resource: Mapping[str, Any], progress: Mapping[str, Any]) -> str:
        self._write_state("VERIFYING_CLOSURE", parent_progress=progress)
        self._emit("STAGE_V_VERIFYING_CLOSURE")
        parent_manifest = Path(parse_json(self.stage_v_root / "SUPERVISOR_START.json")["parent_manifest"])
        pre_report_path = self.monitor_root / "STAGE_V_CLOSURE_AUDIT_PRESEAL.json"
        pre_report = audit_closure(
            self.stage_v_root,
            parent_manifest=parent_manifest,
            expected_source_commit=self.args.expected_source_commit,
            expected_source_tree=self.args.expected_source_tree,
            expected_parent_count=self.args.expected_parent_count,
            require_root_seal=False,
        )
        atomic_write_json(pre_report_path, pre_report)
        if pre_report["verdict"] != "PASS":
            return self._hard_stop(["CLOSURE_AUDIT_FAIL", *pre_report["errors"]], resource)
        seal = write_root_seal(self.stage_v_root)
        final_report = audit_closure(
            self.stage_v_root,
            parent_manifest=parent_manifest,
            expected_source_commit=self.args.expected_source_commit,
            expected_source_tree=self.args.expected_source_tree,
            expected_parent_count=self.args.expected_parent_count,
            require_root_seal=True,
        )
        final_report_path = self.monitor_root / "STAGE_V_CLOSURE_AUDIT.json"
        atomic_write_json(final_report_path, final_report)
        if final_report["verdict"] != "PASS":
            return self._hard_stop(["FINAL_CLOSURE_AUDIT_FAIL", *final_report["errors"]], resource)
        receipt = {
            "schema": "STAGE_V_CLOSURE_RECEIPT_V1",
            "status": "STAGE_V_FORMAL_MAP_CLOSED",
            "root": str(self.stage_v_root),
            "root_seal": sha256_file(self.stage_v_root / "SHA256SUMS"),
            "source_commit": self.args.expected_source_commit,
            "source_tree": self.args.expected_source_tree,
            "planned_parents": self.args.expected_parent_count,
            "completed_parents": final_report["completed_parent_count"],
            "accepted_parents": final_report["accepted_parent_count"],
            "planned_branches": sum(int(item.get("expected_branch_count") or 0) for item in final_report["parents"]),
            "completed_branches": sum(int(item.get("completed_branch_count") or 0) for item in final_report["parents"]),
            "audit_sha256": sha256_file(final_report_path),
            "manifest_sha256": sha256_file(parent_manifest),
            "root_seal_files": seal["files"],
            "resource_summary": dict(resource),
            "eval160_reads": 0,
            "protected_eval_reads": 0,
            "vis_rollouts": 0,
            "pgd_rollouts": 0,
            "attack_rollouts": 0,
            "finish_utc": utc_now(),
        }
        atomic_write_json(self.stage_v_root / "STAGE_V_CLOSURE_RECEIPT.json", receipt)
        self._write_state("STAGE_V_FORMAL_MAP_CLOSED", closure_receipt=str(self.stage_v_root / "STAGE_V_CLOSURE_RECEIPT.json"), parent_progress=final_report)
        self._emit("STAGE_V_FORMAL_MAP_CLOSED", accepted_parents=final_report["accepted_parent_count"])
        return self._maybe_start_stage_v2(resource)

    @staticmethod
    def _hash_matches(path: Path, expected: Any) -> bool:
        return isinstance(expected, str) and bool(SHA256_RE.fullmatch(expected.lower())) and path.is_file() and sha256_file(path) == expected.lower()

    def _materialize_v2_command(self) -> str | None:
        """Bind the future command to the actual post-closure receipt.

        ponytail: keep the plan immutable until closure; the only dynamic value
        is the receipt hash that cannot exist before the formal map closes.
        """
        command_path = Path(self.args.stage_v2_command_file).resolve()
        if command_path.is_file():
            return None
        plan_path = self.monitor_root / "STAGE_V2_COMMAND_PLAN.json"
        plan = parse_json(plan_path)
        if plan is None:
            return "STAGE_V2_COMMAND_NOT_REGISTERED"
        if not isinstance(plan, Mapping) or plan.get("schema") != V2_COMMAND_PLAN_SCHEMA:
            return "STAGE_V2_COMMAND_PLAN_SCHEMA_INVALID"
        if plan.get("stage") != "V2_TEACHER_ENRICHMENT" or plan.get("read_only") is not True:
            return "STAGE_V2_COMMAND_PLAN_NOT_READ_ONLY_ENRICHMENT"
        if plan.get("stage_v_root") != str(self.stage_v_root):
            return "STAGE_V2_COMMAND_PLAN_ROOT_MISMATCH"
        if plan.get("stage_v_source_commit") != self.args.expected_source_commit or plan.get("stage_v_source_tree") != self.args.expected_source_tree:
            return "STAGE_V2_COMMAND_PLAN_SOURCE_MISMATCH"
        receipt_path = self.stage_v_root / "STAGE_V_CLOSURE_RECEIPT.json"
        receipt = parse_json(receipt_path)
        if not isinstance(receipt, Mapping) or receipt.get("status") != "STAGE_V_FORMAL_MAP_CLOSED":
            return "STAGE_V_CLOSURE_RECEIPT_NOT_READY"
        if not self._hash_matches(Path(plan.get("stage_v2_runner_path", "")), plan.get("stage_v2_runner_sha256")):
            return "STAGE_V2_RUNNER_SHA_MISMATCH"
        if not self._hash_matches(Path(plan.get("stage_v2_auditor_path", "")), plan.get("stage_v2_auditor_sha256")):
            return "STAGE_V2_AUDITOR_SHA_MISMATCH"
        if not self._hash_matches(Path(plan.get("stage_v2_config_path", "")), plan.get("stage_v2_config_sha256")):
            return "STAGE_V2_CONFIG_SHA_MISMATCH"
        run_manifest = self.stage_v_root / "RUN_MANIFEST.json"
        start = parse_json(self.stage_v_root / "SUPERVISOR_START.json")
        if not isinstance(start, Mapping) or not start.get("parent_manifest"):
            return "STAGE_V_PARENT_MANIFEST_NOT_FOUND"
        receipt_sha = sha256_file(receipt_path)
        run_sha = sha256_file(run_manifest) if run_manifest.is_file() else ""
        parent_sha = str(receipt.get("manifest_sha256", ""))
        if plan.get("expected_parent_manifest_sha256") != parent_sha or plan.get("expected_run_manifest_sha256") != run_sha:
            return "STAGE_V2_COMMAND_PLAN_MANIFEST_SHA_MISMATCH"
        command_template = plan.get("command_template")
        if not isinstance(command_template, list) or not command_template or not all(isinstance(item, str) for item in command_template):
            return "STAGE_V2_COMMAND_PLAN_ARGV_INVALID"
        command = [
            self._replace(
                item,
                {
                    "{stage_v_root}": str(self.stage_v_root),
                    "{goal_root}": str(self.goal_root),
                    "{source_commit}": self.args.expected_source_commit,
                    "{source_tree}": self.args.expected_source_tree,
                    "{expected_stage_v_closure_receipt_sha256}": receipt_sha,
                    "{expected_parent_manifest_sha256}": parent_sha,
                    "{expected_run_manifest_sha256}": run_sha,
                },
            )
            for item in command_template
        ]
        final = dict(plan)
        final.update(
            {
                "schema": V2_COMMAND_SCHEMA,
                "stage_v_closure_receipt_sha256": receipt_sha,
                "expected_stage_v_closure_receipt_sha256": receipt_sha,
                "expected_parent_manifest_sha256": parent_sha,
                "parent_manifest_sha256": parent_sha,
                "expected_run_manifest_sha256": run_sha,
                "command": command,
                "registered_utc": utc_now(),
            }
        )
        final.pop("command_template", None)
        atomic_write_json(command_path, final)
        return None

    def _load_v2_spec(self) -> tuple[dict[str, Any] | None, str | None]:
        path = Path(self.args.stage_v2_command_file).resolve()
        value = parse_json(path)
        if value is None:
            return None, "STAGE_V2_COMMAND_NOT_REGISTERED"
        if not isinstance(value, Mapping) or value.get("schema") != V2_COMMAND_SCHEMA:
            return None, "STAGE_V2_COMMAND_SCHEMA_INVALID"
        if value.get("stage") != "V2_TEACHER_ENRICHMENT" or value.get("read_only") is not True:
            return None, "STAGE_V2_COMMAND_NOT_READ_ONLY_ENRICHMENT"
        if value.get("stage_v_root") != str(self.stage_v_root):
            return None, "STAGE_V2_COMMAND_ROOT_MISMATCH"
        if value.get("stage_v_source_commit") != self.args.expected_source_commit or value.get("stage_v_source_tree") != self.args.expected_source_tree:
            return None, "STAGE_V2_COMMAND_SOURCE_MISMATCH"
        if not isinstance(value.get("stage_v2_source_commit"), str) or not isinstance(value.get("stage_v2_source_tree"), str):
            return None, "STAGE_V2_COMMAND_PRODUCER_SOURCE_MISSING"
        receipt_path = self.stage_v_root / "STAGE_V_CLOSURE_RECEIPT.json"
        if not self._hash_matches(receipt_path, value.get("expected_stage_v_closure_receipt_sha256")):
            return None, "STAGE_V2_COMMAND_CLOSURE_RECEIPT_SHA_MISMATCH"
        if value.get("expected_parent_manifest_sha256") != value.get("parent_manifest_sha256"):
            return None, "STAGE_V2_COMMAND_PARENT_MANIFEST_SHA_MISMATCH"
        run_manifest = self.stage_v_root / "RUN_MANIFEST.json"
        if not self._hash_matches(run_manifest, value.get("expected_run_manifest_sha256")):
            return None, "STAGE_V2_COMMAND_RUN_MANIFEST_SHA_MISMATCH"
        for field, path_field in (("stage_v2_runner_sha256", "stage_v2_runner_path"), ("stage_v2_auditor_sha256", "stage_v2_auditor_path"), ("stage_v2_config_sha256", "stage_v2_config_path")):
            if not self._hash_matches(Path(str(value.get(path_field, ""))), value.get(field)):
                return None, f"STAGE_V2_COMMAND_{field.upper()}_MISMATCH"
        if not isinstance(value.get("output_root_template"), str) or "{commit8}" not in value["output_root_template"] or "{utc}" not in value["output_root_template"]:
            return None, "STAGE_V2_COMMAND_OUTPUT_TEMPLATE_INVALID"
        lock_path = value.get("lock_path")
        if not isinstance(lock_path, str) or not lock_path.endswith(".stage_v2_teacher_enrichment.lock"):
            return None, "STAGE_V2_COMMAND_LOCK_PATH_INVALID"
        env = value.get("env")
        if not isinstance(env, Mapping) or env.get("CUDA_VISIBLE_DEVICES") != "" or any(env.get(key) != "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
            return None, "STAGE_V2_COMMAND_CPU_ONLY_ENV_INVALID"
        command = value.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
            return None, "STAGE_V2_COMMAND_MUST_BE_ARGV_LIST"
        rendered = " ".join(command)
        if FORBIDDEN_V2.search(rendered):
            return None, "STAGE_V2_COMMAND_FORBIDDEN_BOUNDARY"
        if not _finite(value):
            return None, "STAGE_V2_COMMAND_NON_FINITE"
        return dict(value), None

    def _maybe_start_stage_v2(self, resource: Mapping[str, Any]) -> str:
        marker = parse_json(self.monitor_root / "STAGE_V2_LAUNCH.json")
        if isinstance(marker, Mapping):
            pid = _read_int(marker.get("pid"))
            if marker.get("status") == "RUNNING" and pid and pid_alive(pid):
                self._write_state("STAGE_V2_RUNNING", stage_v2_pid=pid, stage_v2_root=marker.get("output_root"))
                self._emit("STAGE_V2_STARTED", pid=pid)
                return "CONTINUE"
            complete = parse_json(self.monitor_root / "STAGE_V2_COMPLETE.json")
            if isinstance(complete, Mapping):
                self._write_state(str(complete.get("status", "STAGE_V2_FAIL")), stage_v2_root=complete.get("output_root"))
                return "STOP"
            self._write_state("STAGE_V2_FAIL", stage_v2_launch_error="previous_launch_not_completed")
            self._emit("STAGE_V2_FAIL", reason="previous_launch_not_completed")
            return "STOP"
        materialize_error = self._materialize_v2_command()
        if materialize_error:
            self._write_state("STAGE_V2_WAITING_FOR_REGISTERED_COMMAND", next_transition=materialize_error)
            self._emit("STAGE_V2_NOT_REGISTERED", reason=materialize_error)
            return "CONTINUE"
        spec, error = self._load_v2_spec()
        if error:
            self._write_state("STAGE_V2_WAITING_FOR_REGISTERED_COMMAND", next_transition=error)
            self._emit("STAGE_V2_NOT_REGISTERED", reason=error)
            return "CONTINUE"
        assert spec is not None
        if any(item.get("gpu") in self.reserved_gpus for item in resource.get("gpu_assignments", [])):
            return self._hard_stop(["GPU5_FORBIDDEN"], resource)
        output_template = spec.get("output_root_template")
        if not isinstance(output_template, str) or not output_template:
            self._write_state("STAGE_V2_FAIL", stage_v2_launch_error="output_root_template_missing")
            self._emit("STAGE_V2_FAIL", reason="output_root_template_missing")
            return "STOP"
        output_root = Path(
            self._replace(
                output_template,
                {
                    "{stage_v_root}": str(self.stage_v_root),
                    "{goal_root}": str(self.goal_root),
                    "{source_commit}": self.args.expected_source_commit,
                    "{source_tree}": self.args.expected_source_tree,
                    "{commit8}": self.args.expected_source_commit[:8],
                    "{utc}": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                },
            )
        )
        if not output_root.is_absolute():
            output_root = self.goal_root / output_root
        output_root = output_root.resolve()
        try:
            output_root.relative_to(self.goal_root.resolve())
        except ValueError:
            self._write_state("STAGE_V2_FAIL", stage_v2_launch_error="output_root_outside_goal_root")
            self._emit("STAGE_V2_FAIL", reason="output_root_outside_goal_root")
            return "STOP"
        if output_root.exists():
            self._write_state("STAGE_V2_FAIL", stage_v2_launch_error="fresh_output_root_already_exists")
            self._emit("STAGE_V2_FAIL", reason="fresh_output_root_already_exists")
            return "STOP"
        output_root.mkdir(parents=True)
        replacements = {
            "{stage_v_root}": str(self.stage_v_root),
            "{output_root}": str(output_root),
            "{goal_root}": str(self.goal_root),
            "{source_commit}": self.args.expected_source_commit,
            "{source_tree}": self.args.expected_source_tree,
            "{commit8}": self.args.expected_source_commit[:8],
            "{utc}": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        }
        command = [self._replace(item, replacements) for item in spec["command"]]
        cwd = Path(self._replace(str(spec.get("cwd", self.goal_root)), replacements)).resolve()
        if not cwd.is_dir():
            self._write_state("STAGE_V2_FAIL", stage_v2_launch_error="stage_v2_cwd_missing")
            self._emit("STAGE_V2_FAIL", reason="stage_v2_cwd_missing")
            return "STOP"
        environment = os.environ.copy()
        for key, value in (spec.get("env") or {}).items():
            if not isinstance(key, str) or not isinstance(value, str):
                self._write_state("STAGE_V2_FAIL", stage_v2_launch_error="stage_v2_env_invalid")
                self._emit("STAGE_V2_FAIL", reason="stage_v2_env_invalid")
                return "STOP"
            environment[key] = self._replace(value, replacements)
        wrapped = command
        if os.name == "posix" and shutil.which("nice") and shutil.which("ionice"):
            wrapped = ["nice", "-n", "10", "ionice", "-c", "3", *command]
        stdout_path = self.monitor_root / "STAGE_V2_STDOUT.log"
        stderr_path = self.monitor_root / "STAGE_V2_STDERR.log"
        atomic_write_json(
            self.monitor_root / "STAGE_V2_LAUNCH.json",
            {
                "schema": "STAGE_V2_LAUNCH_V1",
                "status": "LAUNCHING",
                "command": command,
                "output_root": str(output_root),
                "source_commit": self.args.expected_source_commit,
                "source_tree": self.args.expected_source_tree,
                "started_utc": utc_now(),
            },
        )
        try:
            stdout = stdout_path.open("a", encoding="utf-8")
            stderr = stderr_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                wrapped,
                cwd=str(cwd),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=(os.name == "posix"),
            )
        except (OSError, ValueError) as exc:
            atomic_write_json(self.monitor_root / "STAGE_V_LAUNCH_FAIL.json", {"schema": "STAGE_V2_LAUNCH_FAIL_V1", "error": str(exc), "utc": utc_now()})
            self._write_state("STAGE_V2_FAIL", stage_v2_launch_error=str(exc))
            self._emit("STAGE_V2_FAIL", reason=str(exc))
            return "STOP"
        atomic_write_json(
            self.monitor_root / "STAGE_V2_LAUNCH.json",
            {
                "schema": "STAGE_V2_LAUNCH_V1",
                "status": "RUNNING",
                "pid": process.pid,
                "command": command,
                "output_root": str(output_root),
                "source_commit": self.args.expected_source_commit,
                "source_tree": self.args.expected_source_tree,
                "started_utc": utc_now(),
            },
        )
        self.stage_v2_process = process
        self.stage_v2_root = output_root
        self._write_state("STAGE_V2_RUNNING", stage_v2_pid=process.pid, stage_v2_root=str(output_root))
        self._emit("STAGE_V2_STARTED", pid=process.pid, output_root=str(output_root))
        return "CONTINUE"

    @staticmethod
    def _replace(value: str, replacements: Mapping[str, str]) -> str:
        for key, replacement in replacements.items():
            value = value.replace(key, replacement)
        return value

    def _check_stage_v2(self) -> str:
        marker = parse_json(self.monitor_root / "STAGE_V2_LAUNCH.json")
        if not isinstance(marker, Mapping) or marker.get("status") != "RUNNING":
            return "CONTINUE"
        if self.stage_v2_process is not None:
            code = self.stage_v2_process.poll()
        else:
            pid = _read_int(marker.get("pid"))
            if pid and pid_alive(pid):
                return "CONTINUE"
            code = None
        if code is None:
            return "CONTINUE"
        output_root = Path(str(marker.get("output_root")))
        report = parse_json(output_root / "STAGE_V2_TEACHER_ENRICHMENT_REPORT.json") or parse_json(output_root / "STAGE_V2_REPORT.json")
        verdict, reason = self._audit_v2_report(report, code, output_root)
        independent = parse_json(output_root / "STAGE_V2_INDEPENDENT_AUDIT.json")
        complete = {
            "schema": "STAGE_V2_COMPLETE_V1",
            "status": "STAGE_V2_PASS" if verdict == "PASS" else "STAGE_V2_FAIL",
            "exit_code": code,
            "output_root": str(output_root),
            "report": report,
            "independent_audit": independent,
            "reason": reason,
            "eval160_reads": 0,
            "protected_eval_reads": 0,
            "vis_rollouts": 0,
            "pgd_rollouts": 0,
            "attack_rollouts": 0,
            "completed_utc": utc_now(),
        }
        atomic_write_json(self.monitor_root / "STAGE_V2_COMPLETE.json", complete)
        self._write_state(complete["status"], stage_v2_root=str(output_root), stage_v2_reason=reason)
        self._emit(complete["status"], reason=reason)
        return "STOP"

    @staticmethod
    def _audit_v2_report(report: Any, exit_code: int, output_root: Path) -> tuple[str, str]:
        if exit_code != 0:
            return "FAIL", f"stage_v2_exit_code:{exit_code}"
        if not isinstance(report, Mapping):
            return "FAIL", "stage_v2_report_missing"
        independent = parse_json(output_root / "STAGE_V2_INDEPENDENT_AUDIT.json")
        if not isinstance(independent, Mapping) or independent.get("verdict") != "PASS":
            return "FAIL", "stage_v2_independent_audit_not_pass"
        seal_ok, seal_errors, _ = verify_root_seal(output_root)
        if not seal_ok:
            return "FAIL", "stage_v2_root_seal_fail:" + ";".join(seal_errors)
        if not _finite(report):
            return "FAIL", "stage_v2_report_non_finite"
        counters = _boundary_counters(report)
        if any(counters.values()):
            return "FAIL", "stage_v2_protected_boundary_nonzero"
        local_enrichment = report.get("local_vulnerability_enrichment", report.get("local_enrichment_ratio"))
        local_recall = report.get("local_vulnerability_recall", report.get("local_recall"))
        suites = report.get("suite_breakdown", report.get("suites"))
        if report.get("status") != "STAGE_V2_TEACHER_PROPOSAL_PASS":
            return "FAIL", "stage_v2_teacher_proposal_gate_fail"
        try:
            if float(local_enrichment) < 3.0 or float(local_recall) < 0.60:
                return "FAIL", "stage_v2_teacher_proposal_gate_fail"
        except (TypeError, ValueError):
            return "FAIL", "stage_v2_gate_metrics_missing"
        if not isinstance(suites, Mapping):
            return "FAIL", "stage_v2_suite_breakdown_missing"
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
            item = suites.get(suite)
            value = item.get("enrichment", item.get("enrichment_ratio")) if isinstance(item, Mapping) else item
            try:
                if float(value) <= 1.0:
                    return "FAIL", f"stage_v2_suite_gate_fail:{suite}"
            except (TypeError, ValueError):
                return "FAIL", f"stage_v2_suite_metric_missing:{suite}"
        return "PASS", "stage_v2_teacher_proposal_gate_pass"

    def tick(self) -> str:
        if self.last_status in {"STAGE_V2_RUNNING"}:
            return self._check_stage_v2()
        if self.stage_v2_root and self.last_status in {"STAGE_V2_PASS", "STAGE_V2_FAIL"}:
            return "STOP"
        resource = self._resource_snapshot()
        progress = self._write_parent_progress()
        self._write_live_parent_triage()
        self._write_resource_sample(resource)
        self._write_monitor_heartbeat(resource, progress)
        errors = resource["hard_stop_errors"]
        complete = parse_json(self.stage_v_root / "SUPERVISOR_COMPLETE.json")
        aborted = parse_json(self.stage_v_root / "ABORTED_INCOMPLETE.json")
        if errors and not isinstance(aborted, Mapping):
            return self._hard_stop(errors, resource)
        if isinstance(aborted, Mapping):
            self._write_state("ABORTED_INCOMPLETE", accepted_parent_results=0, parent_progress=progress)
            self._emit("STAGE_V_HARD_STOP", reasons=["producer_aborted"])
            return "STOP"
        if not isinstance(complete, Mapping):
            if resource.get("heartbeat_age_seconds") is not None and resource["heartbeat_age_seconds"] > resource["heartbeat_warning_seconds"]:
                self._write_state("DEGRADED", warning="heartbeat_warning", parent_progress=progress)
                self._emit("STAGE_V_DEGRADED", heartbeat_age_seconds=resource["heartbeat_age_seconds"])
            else:
                self._write_state("RUNNING", parent_progress=progress)
                self._emit("STAGE_V_RUNNING", parent_progress=progress)
            return "CONTINUE"
        if complete.get("status") != "PASS":
            return self._hard_stop(["SUPERVISOR_COMPLETE_NOT_PASS"], resource)
        if resource.get("supervisor_alive") or resource.get("dispatcher_alive") or resource.get("active_worker_pids"):
            return self._hard_stop(["RESIDUAL_STAGE_V_PROCESS_AFTER_COMPLETE"], resource)
        return self._run_closure(resource, progress) if not (self.stage_v_root / "STAGE_V_CLOSURE_RECEIPT.json").is_file() else self._maybe_start_stage_v2(resource)

    def run(self) -> int:
        if not self.stage_v_root.is_dir():
            raise MonitorError(f"Stage V root missing: {self.stage_v_root}")
        self.lock.acquire(self.monitor_root)
        atomic_write_text(self.monitor_root / "STAGE_V_MONITOR_PID", f"{os.getpid()}\n")
        self._write_state("STARTING", monitor_pid=os.getpid())
        self._emit("MONITOR_STARTED", monitor_pid=os.getpid())
        try:
            while True:
                action = self.tick()
                if action == "STOP":
                    return 0
                time.sleep(self.args.poll_seconds)
        finally:
            self._write_state(self.last_status or "STOPPED", stopped_utc=utc_now())
            self.lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-v-root", required=True, type=Path)
    parser.add_argument("--goal-root", required=True, type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--expected-parent-count", required=True, type=int)
    parser.add_argument("--expected-gpus", required=True, type=parse_csv_ints)
    parser.add_argument("--reserved-gpus", type=parse_csv_ints, default=[5])
    parser.add_argument("--protected-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--kill-grace-seconds", type=float, default=10.0)
    parser.add_argument("--lock-path", type=Path)
    parser.add_argument("--stage-v2-command-file", type=Path)
    parser.add_argument("--once", action="store_true", help="run one observation cycle")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds <= 0 or args.expected_parent_count <= 0:
        raise SystemExit("poll-seconds and expected-parent-count must be positive")
    args.lock_path = args.lock_path or args.goal_root.resolve() / ".stage_v_goal_gatekeeper.lock"
    args.stage_v2_command_file = args.stage_v2_command_file or args.goal_root.resolve() / "MONITOR" / "STAGE_V2_COMMAND.json"
    monitor = Gatekeeper(args)
    if args.once:
        monitor.lock.acquire(monitor.monitor_root)
        try:
            atomic_write_text(monitor.monitor_root / "STAGE_V_MONITOR_PID", f"{os.getpid()}\n")
            monitor._write_state("STARTING", monitor_pid=os.getpid())
            monitor._emit("MONITOR_STARTED", monitor_pid=os.getpid())
            monitor.tick()
        finally:
            monitor.lock.close()
        return 0
    try:
        return monitor.run()
    except MonitorError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
