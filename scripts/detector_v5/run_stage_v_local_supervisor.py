"""Detached, local-only supervisor for the Stage V counterfactual map.

The supervisor owns one dispatcher process group, one filesystem lock, and one
atomic heartbeat.  SSH is deliberately telemetry only; stop decisions are
made from local process, artifact, memory, swap, GPU, and provenance checks.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

try:  # Linux is the production target; the fallback keeps import/tests usable on Windows.
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - exercised only on Windows
    fcntl = None


SCHEMA = "STAGE_V_LOCAL_SUPERVISOR_V1"
TERMINAL_STATES = {"COMPLETED", "FAILED", "ABORTED", "SKIPPED", "DONE"}
DEFAULT_GPU_QUERY = (
    "nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu "
    "--format=csv,noheader,nounits"
)


class SupervisorError(RuntimeError):
    pass


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write, fsync, and rename in the target directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Windows has no fsync-able directory; file durability is still checked.
            if os.name != "nt":
                raise
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _parse_int(value: Any) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_csv_ints(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    try:
        result = [int(value) for value in values]
    except ValueError as exc:
        raise ValueError(f"invalid GPU/CPU list: {raw!r}") from exc
    if not result or len(result) != len(set(result)) or any(value < 0 for value in result):
        raise ValueError(f"GPU/CPU IDs must be unique non-negative integers: {raw!r}")
    return result


def pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":  # os.kill(pid, 0) is not a harmless probe on Windows.
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            return result.returncode == 0 and str(pid) in result.stdout
        except (OSError, subprocess.SubprocessError):
            return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            state = proc_stat.read_text(encoding="utf-8", errors="replace").split()[2]
            if state == "Z":
                return False
        except (OSError, IndexError):
            pass
    return True


def process_command(pid: int | None) -> str:
    if not pid:
        return ""
    cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        value = cmdline.read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip()
        if value:
            return value
    except OSError:
        pass
    try:
        result = subprocess.run(
            ["ps", "-o", "args=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def process_cpu_percent(pid: int | None) -> float | None:
    if not pid:
        return None
    try:
        result = subprocess.run(
            ["ps", "-o", "%cpu=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return float(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _walk_dicts(value: Any):
    if isinstance(value, Mapping):
        yield value
        for item in value.values():
            yield from _walk_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_dicts(item)


def _manifest_workers(root: Path) -> tuple[list[dict[str, int]], list[str]]:
    workers: list[dict[str, int]] = []
    errors: list[str] = []
    for name in ("RUN_MANIFEST.json", "JOB_MANIFEST.json", "MANIFEST.json"):
        value = _json(root / name)
        if value is None:
            continue
        for item in _walk_dicts(value):
            pid = _parse_int(item.get("worker_pid", item.get("pid")))
            gpu = _parse_int(item.get("gpu", item.get("gpu_id")))
            status = str(item.get("status", "")).upper()
            if pid is None or gpu is None or status in TERMINAL_STATES:
                continue
            workers.append({"pid": pid, "gpu": gpu})
        break
    return workers, errors


def discover_workers(root: Path, dispatcher_pid: int | None) -> tuple[list[dict[str, int]], list[str]]:
    workers: list[dict[str, int]] = []
    errors: list[str] = []
    for path in sorted(root.glob("worker_gpu*.pid")):
        match = re.fullmatch(r"worker_gpu(\d+)\.pid", path.name)
        if not match:
            continue
        pid = _parse_int(path.read_text(encoding="utf-8", errors="replace").strip())
        if pid is None:
            errors.append(f"invalid_worker_pid_file:{path.name}")
            continue
        workers.append({"pid": pid, "gpu": int(match.group(1))})
    manifest_workers, manifest_errors = _manifest_workers(root)
    errors.extend(manifest_errors)
    known = {(item["pid"], item["gpu"]) for item in workers}
    for item in manifest_workers:
        if (item["pid"], item["gpu"]) not in known:
            workers.append(item)
            known.add((item["pid"], item["gpu"]))
    if dispatcher_pid:
        workers = [item for item in workers if item["pid"] != dispatcher_pid]
    return workers, errors


def validate_workers(
    workers: list[dict[str, int]], approved_gpus: list[int], *, require_live: bool = True
) -> list[str]:
    errors: list[str] = []
    by_gpu: dict[int, set[int]] = {}
    for item in workers:
        pid, gpu = item["pid"], item["gpu"]
        by_gpu.setdefault(gpu, set()).add(pid)
        if gpu == 5:
            errors.append("GPU5_FORBIDDEN")
        if gpu not in approved_gpus:
            errors.append(f"GPU_NOT_APPROVED:{gpu}")
        if require_live and not pid_alive(pid):
            errors.append(f"WORKER_PID_NOT_ALIVE:{pid}")
    for gpu, pids in by_gpu.items():
        if len(pids) > 1:
            errors.append(f"MULTIPLE_WORKERS_ON_GPU:{gpu}")
    if len(workers) > len(approved_gpus):
        errors.append("ACTIVE_WORKERS_EXCEED_APPROVED_GPUS")
    return sorted(set(errors))


def hard_stop_errors(
    *,
    available_ram_bytes: int | None,
    min_available_ram_bytes: int,
    baseline_oom: int | None,
    oom_kill: int | None,
    swap_bad_streak: int,
    swap_bad_samples: int,
    xid_status: str,
    query_error: str | None = None,
    worker_errors: list[str] | None = None,
) -> list[str]:
    errors = list(worker_errors or [])
    if available_ram_bytes is not None and available_ram_bytes < min_available_ram_bytes:
        errors.append("AVAILABLE_RAM_BELOW_HARD_STOP")
    if baseline_oom is not None and oom_kill is not None and oom_kill > baseline_oom:
        errors.append("OOM_KILL_COUNTER_INCREASED")
    if swap_bad_streak >= swap_bad_samples:
        errors.append("SWAP_HARD_STOP")
    if xid_status == "XID_DETECTED":
        errors.append("NVIDIA_XID")
    if query_error:
        errors.append(query_error)
    return sorted(set(errors))


def read_meminfo() -> dict[str, int | None]:
    values: dict[str, int | None] = {}
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
    except OSError:
        pass
    return values


def run_text_command(command: str, timeout: float = 10.0) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            shlex.split(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=max(0.1, timeout),
        )
        return result.returncode, result.stdout, result.stderr
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return 127, "", str(exc)


def read_gpu_snapshot(command: str, *, skip_errors: bool) -> tuple[list[dict[str, Any]], str | None]:
    if not command:
        return [], None
    code, stdout, stderr = run_text_command(command)
    if code != 0:
        return [], None if skip_errors else f"GPU_QUERY_FAILED:{stderr.strip() or code}"
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
        return [], None if skip_errors else f"GPU_QUERY_PARSE_FAILED:{exc}"
    return rows, None


def read_xid_status(command: str, since: str, *, skip_errors: bool) -> tuple[str, str | None]:
    if not command:
        return "NOT_CHECKED", None
    rendered = command.format(since=since)
    code, stdout, stderr = run_text_command(rendered)
    if code != 0 and not skip_errors:
        return "UNKNOWN", f"KERNEL_LOG_QUERY_FAILED:{stderr.strip() or code}"
    text = stdout.lower()
    if "xid" in text or "fallen off the bus" in text or "gpu has fallen off" in text:
        return "XID_DETECTED", "NVIDIA_XID"
    return "CLEAR", None


def source_provenance(repo_root: Path) -> dict[str, str]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            raise SupervisorError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    status = git("status", "--porcelain")
    if status:
        raise SupervisorError(f"source worktree is dirty: {status[:200]}")
    return {
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
    }


def _first_int(values: Mapping[str, Any], names: tuple[str, ...]) -> int | None:
    for name in names:
        value = _parse_int(values.get(name))
        if value is not None:
            return value
    return None


def read_counters(root: Path, planned: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "planned_parents": planned,
        "completed_parents": 0,
        "accepted_parent_results": 0,
        "failed_parents": 0,
        "current_parent": None,
        "current_branch": None,
        "artifact_audit_verdict": None,
    }
    candidates = [
        root / "STAGE_V_SUMMARY.json",
        root / "STAGE_V_AUDIT.json",
        root / "RUN_SUMMARY.json",
        root / "CONTROL_CANARY_SUMMARY.json",
        root / "RUN_MANIFEST.json",
        root / "JOB_MANIFEST.json",
    ]
    for path in candidates:
        value = _json(path)
        if not isinstance(value, Mapping):
            continue
        result["completed_parents"] = _first_int(
            value, ("completed_parents", "parents_completed", "completed")
        ) or result["completed_parents"]
        result["accepted_parent_results"] = _first_int(
            value, ("accepted_parent_results", "accepted_parents", "accepted")
        ) or result["accepted_parent_results"]
        result["failed_parents"] = _first_int(
            value, ("failed_parents", "parents_failed", "failed")
        ) or result["failed_parents"]
        result["current_parent"] = value.get("current_parent", result["current_parent"])
        result["current_branch"] = value.get("current_branch", result["current_branch"])
        result["artifact_audit_verdict"] = value.get(
            "artifact_audit_verdict", value.get("verdict", result["artifact_audit_verdict"])
        )
        if "accepted_parent_artifacts" in value:
            result["accepted_parent_artifacts"] = value["accepted_parent_artifacts"]
        break
    return result


def latest_artifact_mtime(root: Path, since: float) -> float:
    newest = since
    try:
        for path in root.rglob("*"):
            if not path.is_file() or path.name in {"LOCAL_HEARTBEAT.json", "SUPERVISOR_STDOUT.log", "SUPERVISOR_STDERR.log"}:
                continue
            newest = max(newest, path.stat().st_mtime)
    except OSError:
        return since
    return newest


def check_writable(root: Path) -> None:
    fd, name = tempfile.mkstemp(prefix=".watchdog.", dir=str(root))
    path = Path(name)
    try:
        os.write(fd, b"watchdog\n")
        os.fsync(fd)
    finally:
        os.close(fd)
        path.unlink(missing_ok=True)


class ExclusiveLock:
    def __init__(self, path: Path, metadata: Mapping[str, Any]):
        self.path = path
        self.metadata = dict(metadata)
        self.handle = None

    def acquire(self, run_root: Path) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        try:
            existing = self.path.read_text(encoding="utf-8")
        except OSError:
            pass
        existing_pid = _parse_int(_json_from_text(existing).get("supervisor_pid"))
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            if fcntl is not None:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:  # pragma: no cover - Windows fallback
                import msvcrt

                self.handle.seek(0)
                self.handle.write("0")
                self.handle.flush()
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except (BlockingIOError, OSError) as exc:
            pid = _parse_int(_json_from_text(existing).get("supervisor_pid"))
            if pid and pid_alive(pid):
                raise SupervisorError(f"DUPLICATE_SUPERVISOR:{pid}") from exc
            run_root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                run_root / "STALE_LOCK_AUDIT.json",
                {
                    "schema": "STAGE_V_STALE_LOCK_AUDIT_V1",
                    "lock_path": str(self.path),
                    "observed_lock_metadata": existing,
                    "observed_pid": pid,
                    "pid_alive": False,
                    "audited_utc": utc_now(),
                },
            )
            self.handle.close()
            self.handle = None
            self.handle = self.path.open("a+", encoding="utf-8")
            try:
                if fcntl is not None:
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:  # pragma: no cover
                    import msvcrt

                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as second_exc:
                self.handle.close()
                self.handle = None
                raise SupervisorError("STALE_LOCK_REACQUIRE_FAILED") from second_exc
        if existing_pid and existing_pid != os.getpid() and not pid_alive(existing_pid):
            run_root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                run_root / "STALE_LOCK_AUDIT.json",
                {
                    "schema": "STAGE_V_STALE_LOCK_AUDIT_V1",
                    "lock_path": str(self.path),
                    "observed_lock_metadata": existing,
                    "observed_pid": existing_pid,
                    "pid_alive": False,
                    "audited_utc": utc_now(),
                },
            )
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps(self.metadata, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        if self.handle is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def _json_from_text(text: str) -> Mapping[str, Any]:
    try:
        value = json.loads(text)
        return value if isinstance(value, Mapping) else {}
    except json.JSONDecodeError:
        return {}


def terminate_process_group(process: subprocess.Popen[Any], grace_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix" and process.pid:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:  # pragma: no cover - Windows fallback
            process.terminate()
    except (OSError, ProcessLookupError):
        pass
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is None:
        try:
            if os.name == "posix" and process.pid:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:  # pragma: no cover
                process.kill()
        except (OSError, ProcessLookupError):
            pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass


class Supervisor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_root = Path(args.run_root).resolve()
        self.repo_root = Path(args.repo_root).resolve()
        self.approved_gpus = list(args.approved_gpus)
        self.started_epoch = time.time()
        self.start_utc = utc_now()
        self.source: dict[str, str] = {}
        self.process: subprocess.Popen[Any] | None = None
        self.lock: ExclusiveLock | None = None
        self.heartbeat_count = 0
        self.ssh_success_count = 0
        self.ssh_failure_count = 0
        self.ssh_failed_since: float | None = None
        self.longest_ssh_outage = 0.0
        self.baseline_oom: int | None = None
        self.previous_swap_nonzero = False
        self.swap_bad_streak = 0
        self.last_artifact_mtime = self.started_epoch
        self.last_heartbeat: dict[str, Any] = {}
        self.dispatcher_miss_streak = 0

    def _prepare(self) -> None:
        self.source = source_provenance(self.repo_root)
        if self.source["source_commit"] != self.args.expected_source_commit:
            raise SupervisorError("source commit mismatch")
        if self.source["source_tree"] != self.args.expected_source_tree:
            raise SupervisorError("source tree mismatch")
        if self.args.parent_manifest:
            manifest = Path(self.args.parent_manifest).resolve()
            if not manifest.is_file():
                raise SupervisorError(f"parent manifest missing: {manifest}")
            actual = sha256_file(manifest)
            if self.args.parent_manifest_sha256 and actual != self.args.parent_manifest_sha256:
                raise SupervisorError("parent manifest SHA256 mismatch")
        if self.run_root.exists() and any(self.run_root.iterdir()):
            raise SupervisorError(f"run root is not new/empty: {self.run_root}")
        self.run_root.mkdir(parents=True, exist_ok=True)
        parent_sha = None
        if self.args.parent_manifest:
            parent_sha = sha256_file(Path(self.args.parent_manifest).resolve())
        atomic_write_json(
            self.run_root / "SUPERVISOR_START.json",
            {
                "schema": SCHEMA,
                "control_plane_mode": "LOCAL_AUTONOMOUS",
                "ssh_is_hard_stop": False,
                "run_root": str(self.run_root),
                "source_commit": self.source["source_commit"],
                "source_tree": self.source["source_tree"],
                "parent_manifest": str(Path(self.args.parent_manifest).resolve()) if self.args.parent_manifest else None,
                "parent_manifest_sha256": parent_sha,
                "approved_gpus": self.approved_gpus,
                "planned_parents": self.args.planned_parents,
                "supervisor_pid": os.getpid(),
                "supervisor_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
                "started_utc": self.start_utc,
            },
        )
        metrics = self._resource_snapshot()
        self.baseline_oom = metrics.get("oom_kill")
        self.last_artifact_mtime = latest_artifact_mtime(self.run_root, self.started_epoch)
        self._write_heartbeat(metrics)

    def _resource_snapshot(self) -> dict[str, Any]:
        mem = read_meminfo()
        vm = read_vmstat()
        available = mem.get("MemAvailable")
        swap_total = mem.get("SwapTotal") or 0
        swap_free = mem.get("SwapFree") or 0
        swap_used = max(0, swap_total - swap_free)
        gpu_rows, gpu_error = read_gpu_snapshot(
            self.args.gpu_query_command, skip_errors=self.args.skip_gpu_check
        )
        xid, xid_error = read_xid_status(
            self.args.kernel_log_command, self.start_utc, skip_errors=self.args.skip_gpu_check
        )
        workers, worker_errors = discover_workers(
            self.run_root, self.process.pid if self.process else None
        )
        worker_errors.extend(validate_workers(workers, self.approved_gpus))
        if self.process and self.process.poll() is None:
            if pid_alive(self.process.pid):
                self.dispatcher_miss_streak = 0
            else:
                self.dispatcher_miss_streak += 1
                if self.dispatcher_miss_streak >= 2:
                    worker_errors.append("DISPATCHER_PID_LOST")
        oom = vm.get("oom_kill")
        if swap_used > 0:
            self.swap_bad_streak += 1
        else:
            self.swap_bad_streak = 0
        worker_errors = hard_stop_errors(
            available_ram_bytes=available,
            min_available_ram_bytes=self.args.min_available_ram_bytes,
            baseline_oom=self.baseline_oom,
            oom_kill=oom,
            swap_bad_streak=self.swap_bad_streak,
            swap_bad_samples=self.args.swap_bad_samples,
            xid_status=xid,
            query_error=xid_error or gpu_error,
            worker_errors=worker_errors,
        )
        check_writable(self.run_root)
        return {
            "available_ram_bytes": available,
            "available_ram_gib": round(available / (1 << 30), 3) if available is not None else None,
            "swap_used_bytes": swap_used,
            "swap_in": vm.get("pswpin"),
            "swap_out": vm.get("pswpout"),
            "oom_kill": oom,
            "gpu_memory": gpu_rows,
            "gpu_xid_status": xid,
            "active_worker_pids": [item["pid"] for item in workers],
            "gpu_assignments": workers,
            "resource_errors": sorted(set(worker_errors)),
        }

    def _probe_ssh(self) -> None:
        if not self.args.ssh_probe_command:
            return
        code, _stdout, _stderr = run_text_command(self.args.ssh_probe_command, timeout=self.args.ssh_probe_timeout)
        now = time.monotonic()
        if code == 0:
            self.ssh_success_count += 1
            if self.ssh_failed_since is not None:
                self.longest_ssh_outage = max(self.longest_ssh_outage, now - self.ssh_failed_since)
                self.ssh_failed_since = None
        else:
            self.ssh_failure_count += 1
            if self.ssh_failed_since is None:
                self.ssh_failed_since = now

    def _ssh_outage_seconds(self) -> float:
        if self.ssh_failed_since is None:
            return self.longest_ssh_outage
        return max(self.longest_ssh_outage, time.monotonic() - self.ssh_failed_since)

    def _write_heartbeat(self, metrics: Mapping[str, Any]) -> None:
        self.heartbeat_count += 1
        counters = read_counters(self.run_root, self.args.planned_parents)
        external_pid = self.args.external_pid
        heartbeat = {
            "schema": SCHEMA,
            "control_plane_mode": "LOCAL_AUTONOMOUS",
            "ssh_is_hard_stop": False,
            "run_root": str(self.run_root),
            "source_commit": self.source.get("source_commit"),
            "source_tree": self.source.get("source_tree"),
            "supervisor_pid": os.getpid(),
            "supervisor_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
            "dispatcher_pid": self.process.pid if self.process else None,
            "active_worker_pids": metrics.get("active_worker_pids", []),
            "gpu_assignments": metrics.get("gpu_assignments", []),
            "planned_parents": counters["planned_parents"],
            "completed_parents": counters["completed_parents"],
            "accepted_parent_results": counters["accepted_parent_results"],
            "failed_parents": counters["failed_parents"],
            "current_parent": counters["current_parent"],
            "current_branch": counters["current_branch"],
            "accepted_parent_artifacts": counters.get("accepted_parent_artifacts", []),
            "last_artifact_utc": utc_now() if latest_artifact_mtime(self.run_root, self.last_artifact_mtime) > self.last_artifact_mtime else None,
            "available_ram_bytes": metrics.get("available_ram_bytes"),
            "available_ram_gib": metrics.get("available_ram_gib"),
            "swap_in": metrics.get("swap_in"),
            "swap_out": metrics.get("swap_out"),
            "oom_kill": metrics.get("oom_kill"),
            "gpu_memory": metrics.get("gpu_memory", []),
            "gpu_xid_status": metrics.get("gpu_xid_status"),
            "filesystem_writable": not bool(metrics.get("filesystem_error")),
            "filesystem_error": metrics.get("filesystem_error"),
            "resource_errors": metrics.get("resource_errors", []),
            "ssh_probe_success_count": self.ssh_success_count,
            "ssh_probe_failure_count": self.ssh_failure_count,
            "longest_ssh_unavailable_interval_seconds": round(self._ssh_outage_seconds(), 3),
            "external_root_process_present": bool(external_pid and pid_alive(external_pid)),
            "external_root_process_pid": external_pid,
            "external_root_process_cpu_percent": process_cpu_percent(external_pid),
            "external_root_process_terminated": False,
            "heartbeat_count": self.heartbeat_count,
            "updated_utc": utc_now(),
        }
        atomic_write_json(self.run_root / "LOCAL_HEARTBEAT.json", heartbeat)
        self.last_heartbeat = heartbeat

    def _run_auditor(self) -> tuple[int, str]:
        if not self.args.audit_command:
            if self.args.allow_no_audit:
                return 0, "NOT_RUN_ALLOWED"
            return 127, "independent auditor command is required"
        command = self.args.audit_command.format(run_root=str(self.run_root))
        code, stdout, stderr = run_text_command(command, timeout=self.args.audit_timeout)
        return code, (stdout + stderr)[-4000:]

    def _finalize_pass(self, audit_code: int, audit_tail: str) -> int:
        counters = read_counters(self.run_root, self.args.planned_parents)
        accepted = counters["accepted_parent_results"]
        if audit_code != 0:
            return self._abort(f"INDEPENDENT_AUDITOR_FAIL:{audit_tail[-500:]}")
        if accepted != self.args.planned_parents:
            return self._abort(f"ACCEPTED_PARENT_COUNT_MISMATCH:{accepted}/{self.args.planned_parents}")
        accepted_artifacts = counters.get("accepted_parent_artifacts", [])
        if not isinstance(accepted_artifacts, list) or len(accepted_artifacts) != accepted:
            return self._abort("ACCEPTED_PARENT_ARTIFACT_AUDIT_INCOMPLETE")
        if any(
            not isinstance(item, Mapping) or item.get("artifact_audit_verdict") != "PASS"
            for item in accepted_artifacts
        ):
            return self._abort("ACCEPTED_PARENT_ARTIFACT_AUDIT_FAIL")
        atomic_write_json(
            self.run_root / "SUPERVISOR_COMPLETE.json",
            {
                "schema": SCHEMA,
                "status": "PASS",
                "control_plane_mode": "LOCAL_AUTONOMOUS",
                "ssh_is_hard_stop": False,
                "audit_verdict": "PASS",
                "audit_exit_code": audit_code,
                "audit_tail": audit_tail,
                "planned_parents": self.args.planned_parents,
                "completed_parents": counters["completed_parents"],
                "accepted_parent_results": accepted,
                "accepted_parent_artifacts": counters.get("accepted_parent_artifacts", []),
                "failed_parents": counters["failed_parents"],
                "heartbeat_count": self.heartbeat_count,
                "ssh_probe_success_count": self.ssh_success_count,
                "ssh_probe_failure_count": self.ssh_failure_count,
                "longest_ssh_unavailable_interval_seconds": round(self._ssh_outage_seconds(), 3),
                "eval160_reads": 0,
                "protected_eval_reads": 0,
                "vis_pgd_attack_rollouts": 0,
                "completed_utc": utc_now(),
            },
        )
        return 0

    def _abort(self, reason: str) -> int:
        if self.process is not None:
            terminate_process_group(self.process)
        payload = {
            "schema": SCHEMA,
            "status": "ABORTED_INCOMPLETE",
            "control_plane_mode": "LOCAL_AUTONOMOUS",
            "control_plane_abort_reason": reason,
            "ssh_is_hard_stop": False,
            "accepted_parent_results": 0,
            "scientific_validity": 0,
            "eval160_reads": 0,
            "protected_eval_reads": 0,
            "vis_pgd_attack_rollouts": 0,
            "heartbeat_count": self.heartbeat_count,
            "ssh_probe_success_count": self.ssh_success_count,
            "ssh_probe_failure_count": self.ssh_failure_count,
            "longest_ssh_unavailable_interval_seconds": round(self._ssh_outage_seconds(), 3),
            "aborted_utc": utc_now(),
        }
        try:
            atomic_write_json(self.run_root / "ABORTED_INCOMPLETE.json", payload)
            if self.last_heartbeat:
                self.last_heartbeat = dict(self.last_heartbeat)
                self.last_heartbeat.update(
                    {
                        "status": "ABORTED_INCOMPLETE",
                        "abort_reason": reason,
                        "accepted_parent_results": 0,
                        "updated_utc": utc_now(),
                    }
                )
                atomic_write_json(self.run_root / "LOCAL_HEARTBEAT.json", self.last_heartbeat)
        except OSError:
            pass
        return 1

    def run(self) -> int:
        self.lock = ExclusiveLock(
            Path(self.args.lock_path).resolve(),
            {
                "schema": "STAGE_V_LOCAL_LOCK_V1",
                "supervisor_pid": os.getpid(),
                "supervisor_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
                "run_root": str(self.run_root),
                "started_utc": self.start_utc,
            },
        )
        try:
            self.lock.acquire(self.run_root)
            self._prepare()
            if hasattr(signal, "SIGHUP"):
                signal.signal(signal.SIGHUP, signal.SIG_IGN)
            command = shlex.split(self.args.dispatcher_command) + list(self.args.dispatcher_args)
            if not command:
                raise SupervisorError("dispatcher command is empty")
            environment = os.environ.copy()
            for item in self.args.dispatcher_env:
                key, separator, value = item.partition("=")
                if not separator or not key:
                    raise SupervisorError(f"invalid dispatcher environment assignment: {item!r}")
                environment[key] = value
            self.process = subprocess.Popen(
                command,
                cwd=str(Path(self.args.dispatcher_cwd or self.repo_root).resolve()),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=self.args.stdout_handle,
                stderr=self.args.stderr_handle,
                start_new_session=(os.name == "posix"),
            )
            self._write_heartbeat(self._resource_snapshot())
            while True:
                return_code = self.process.poll()
                if return_code is not None:
                    if return_code != 0:
                        return self._abort(f"DISPATCHER_EXIT:{return_code}")
                    audit_code, audit_tail = self._run_auditor()
                    return self._finalize_pass(audit_code, audit_tail)
                metrics = self._resource_snapshot()
                self._probe_ssh()
                self._write_heartbeat(metrics)
                errors = metrics.get("resource_errors", [])
                if errors:
                    return self._abort(";".join(errors))
                newest = latest_artifact_mtime(self.run_root, self.last_artifact_mtime)
                if newest > self.last_artifact_mtime:
                    self.last_artifact_mtime = newest
                elif (
                    self.args.parent_timeout_seconds > 0
                    and time.time() - self.last_artifact_mtime > self.args.parent_timeout_seconds
                ):
                    return self._abort("PARENT_WATCHDOG_TIMEOUT")
                time.sleep(self.args.poll_interval)
        except SupervisorError as exc:
            if self.run_root.exists():
                return self._abort(str(exc))
            return 2
        except Exception as exc:
            if self.run_root.exists():
                return self._abort(f"SUPERVISOR_EXCEPTION:{type(exc).__name__}:{exc}")
            return 2
        finally:
            if self.lock is not None:
                self.lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--parent-manifest")
    parser.add_argument("--parent-manifest-sha256")
    parser.add_argument("--lock-path", required=True)
    parser.add_argument("--approved-gpus", type=parse_csv_ints, required=True)
    parser.add_argument("--planned-parents", type=int, required=True)
    parser.add_argument("--dispatcher-command", required=True)
    parser.add_argument("--dispatcher-arg", dest="dispatcher_args", action="append", default=[])
    parser.add_argument("--dispatcher-env", action="append", default=[])
    parser.add_argument("--dispatcher-cwd")
    parser.add_argument("--audit-command")
    parser.add_argument("--allow-no-audit", action="store_true")
    parser.add_argument("--audit-timeout", type=float, default=3600)
    parser.add_argument("--poll-interval", type=float, default=30)
    parser.add_argument("--heartbeat-interval", type=float, default=30)
    parser.add_argument("--parent-timeout-seconds", type=float, default=0)
    parser.add_argument("--min-available-ram-gib", type=float, default=128)
    parser.add_argument("--swap-bad-samples", type=int, default=2)
    parser.add_argument("--gpu-query-command", default=DEFAULT_GPU_QUERY)
    parser.add_argument(
        "--kernel-log-command",
        default="journalctl -k --since '{since}' --no-pager",
    )
    parser.add_argument("--ssh-probe-command", default="")
    parser.add_argument("--ssh-probe-timeout", type=float, default=10)
    parser.add_argument("--external-pid", type=int)
    parser.add_argument("--skip-gpu-check", action="store_true")
    parser.add_argument("--stdout-log")
    parser.add_argument("--stderr-log")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.planned_parents <= 0:
        raise SystemExit("--planned-parents must be positive")
    args.min_available_ram_bytes = int(args.min_available_ram_gib * (1 << 30))
    if args.poll_interval <= 0 or args.heartbeat_interval <= 0:
        raise SystemExit("poll/heartbeat intervals must be positive")
    args.stdout_handle = open(args.stdout_log, "a", encoding="utf-8") if args.stdout_log else subprocess.DEVNULL
    args.stderr_handle = open(args.stderr_log, "a", encoding="utf-8") if args.stderr_log else subprocess.DEVNULL
    try:
        return Supervisor(args).run()
    finally:
        for handle in (args.stdout_handle, args.stderr_handle):
            if hasattr(handle, "close"):
                handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
