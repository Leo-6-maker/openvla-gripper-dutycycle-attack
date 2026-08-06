"""Small, stdlib-only primitives shared by the Stage V dynamic-8 runners."""
from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
from typing import Any, Iterable, Mapping


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))


def canonical_parent_key(row: Mapping[str, Any]) -> str:
    value = row.get("canonical_parent_key") or row.get("parent_key")
    if value:
        return str(value)
    return f"{row['suite']}/task_{int(row['task_index']):02d}/state_{int(row['state_index']):02d}"


def load_rows(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if isinstance(value, Mapping):
        for key in ("parents", "qualified_parents", "rows", "candidates", "all_candidate_audits", "manifest"):
            if isinstance(value.get(key), list):
                return [dict(item) for item in value[key] if isinstance(item, Mapping)]
        return [dict(value)]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    rows: list[dict[str, Any]] = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if isinstance(item, Mapping):
                    rows.append(dict(item))
    except (OSError, json.JSONDecodeError):
        return []
    return rows


def normalize_parent(row: Mapping[str, Any]) -> dict[str, Any]:
    key = canonical_parent_key(row)
    try:
        suite, task, state = key.split("/")
        task_index = int(task.removeprefix("task_"))
        state_index = int(state.removeprefix("state_"))
    except (ValueError, AttributeError):
        task_index = int(row["task_index"])
        state_index = int(row["state_index"])
        suite = str(row["suite"])
    normalized = dict(row)
    normalized.update(
        {
            "canonical_parent_key": key,
            "suite": suite,
            "task_index": task_index,
            "state_index": state_index,
        }
    )
    return normalized


def sanitize_key(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value)


def attempt_dir(run_root: Path, parent_key: str, attempt: int) -> Path:
    return Path(run_root) / "parents" / sanitize_key(parent_key) / f"attempt_{attempt:02d}"


def format_command(template: str, **values: Any) -> list[str]:
    rendered = template.format(**{key: str(value) for key, value in values.items()})
    return shlex.split(rendered)


def run_command(command: list[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None,
                stdout=None, stderr=None, timeout: float | None = None) -> int:
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        start_new_session=(os.name == "posix"),
    )
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_process_group(process)
        return 124


def terminate_process_group(process: subprocess.Popen[Any], grace_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, 15)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, 9)
            except ProcessLookupError:
                return
    else:  # pragma: no cover - production is Linux
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _number(value: str) -> float | int | None:
    value = value.strip().split()[0] if value.strip() else ""
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except ValueError:
        return None


def gpu_snapshot(command: str = "nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits") -> tuple[list[dict[str, Any]], str | None]:
    try:
        completed = subprocess.run(shlex.split(command), check=False, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"GPU_QUERY_ERROR:{type(exc).__name__}"
    if completed.returncode != 0:
        return [], f"GPU_QUERY_EXIT:{completed.returncode}:{completed.stderr[-200:]}"
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        index = _number(fields[0])
        if index is None:
            continue
        rows.append(
            {
                "gpu_id": int(index),
                "memory_used_mib": _number(fields[1]),
                "memory_free_mib": _number(fields[2]),
                "utilization_gpu_percent": _number(fields[3]) if len(fields) > 3 else None,
            }
        )
    return rows, None if rows else "GPU_QUERY_EMPTY"


def gpu_preflight(*, required_count: int, excluded_gpus: Iterable[int], canary_peak_mib: float,
                  protected_pids: Iterable[int] = (), gpu_query_command: str | None = None,
                  active_project_pids: Iterable[int] = ()) -> dict[str, Any]:
    rows, error = gpu_snapshot(gpu_query_command or "nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits")
    excluded = sorted({int(item) for item in excluded_gpus})
    protected = {int(item) for item in protected_pids if int(item) > 0}
    minimum_free = float(canary_peak_mib) * 1.5 + 4096.0
    protected_present: list[int] = []
    try:
        app_query = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=20,
        )
        if app_query.returncode == 0:
            active_pids = {int(item.strip()) for item in app_query.stdout.splitlines() if item.strip().isdigit()}
            protected_present = sorted(active_pids & protected)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        protected_present = []
    safe: list[int] = []
    decisions: list[dict[str, Any]] = []
    for row in rows:
        gpu_id = int(row["gpu_id"])
        reasons: list[str] = []
        if gpu_id in excluded:
            reasons.append("EXCLUDED_GPU")
        free = row.get("memory_free_mib")
        if free is None or float(free) < minimum_free:
            reasons.append("INSUFFICIENT_FREE_MEMORY")
        if protected_present:
            reasons.append("PROTECTED_PROCESS_PRESENT")
        if reasons:
            decisions.append({**row, "safe": False, "reasons": reasons})
        else:
            safe.append(gpu_id)
            decisions.append({**row, "safe": True, "reasons": []})
    safe = sorted(set(safe))
    return {
        "schema": "STAGE_V_GPU_PREFLIGHT_V2",
        "status": "PASS" if len(safe) >= required_count and not error else "PRELAUNCH_WAITING_FOR_8_GPUS",
        "required_gpu_count": required_count,
        "safe_gpu_count": len(safe),
        "safe_gpus": safe,
        "excluded_gpus": excluded,
        "protected_pids": sorted(protected),
        "protected_process_pids_present": protected_present,
        "minimum_free_memory_mib": minimum_free,
        "canary_peak_memory_mib": canary_peak_mib,
        "gpu_rows": decisions,
        "query_error": error,
        "updated_utc": utc_now(),
    }


def project_queue(run_root: Path, tasks: Iterable[Mapping[str, Any]]) -> None:
    root = Path(run_root)
    projection_dirs = {name: root / f"QUEUE_{name}" for name in ("PENDING", "RUNNING", "COMPLETE", "FAILED")}
    for path in projection_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        cell_id = sanitize_key(str(task["cell_id"]))
        payload = dict(task)
        for path in projection_dirs.values():
            target_path = path / f"{cell_id}.json"
            for attempt in range(5):
                try:
                    target_path.unlink(missing_ok=True)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.01 * (attempt + 1))
        state = str(task.get("state", "PENDING"))
        if state in {"PENDING", "RETRY_READY", "LOCKED"}:
            target = projection_dirs["PENDING"]
        elif state in {"LEASED", "RUNNING", "COMMITTING"}:
            target = projection_dirs["RUNNING"]
        elif state in {"DONE", "DONE_VALID", "DONE_CLASSIFIED_TC"}:
            target = projection_dirs["COMPLETE"]
        else:
            target = projection_dirs["FAILED"]
        target_path = target / f"{cell_id}.json"
        for attempt in range(5):
            try:
                atomic_write_json(target_path, payload)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.01 * (attempt + 1))


def parent_output(root: Path, task: Mapping[str, Any]) -> Path:
    return attempt_dir(root, str(task["canonical_parent_key"]), int(task.get("attempt_count") or 1))


def science_artifact_status(output_dir: Path, parent_key: str) -> dict[str, Any]:
    results = list(Path(output_dir).rglob("PARENT_RESULT.json"))
    if len(results) != 1:
        return {"valid": False, "reason": f"PARENT_RESULT_COUNT:{len(results)}", "result": None, "path": None}
    result_path = results[0]
    result = read_json(result_path, {})
    if not isinstance(result, Mapping):
        return {"valid": False, "reason": "PARENT_RESULT_NOT_OBJECT", "result": None, "path": str(result_path)}
    branches = list(Path(result_path.parent).rglob("COUNTERFACTUAL_BRANCHES.jsonl"))
    branch_count = int(result.get("branch_count", -1))
    if str(result.get("canonical_parent_key")) != parent_key:
        return {"valid": False, "reason": "PARENT_IDENTITY_MISMATCH", "result": dict(result), "path": str(result_path)}
    if result.get("status") != "PASS" or result.get("clean_success") is not True or branch_count != 72 or len(branches) != 1:
        return {"valid": False, "reason": "SCIENCE_RESULT_INVALID", "result": dict(result), "path": str(result_path)}
    return {
        "valid": True,
        "reason": "PASS",
        "result": dict(result),
        "path": str(result_path),
        "artifact_sha256": sha256_file(result_path),
        "label_status": "VALID",
    }
