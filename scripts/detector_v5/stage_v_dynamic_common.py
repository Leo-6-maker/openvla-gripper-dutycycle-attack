"""Small, stdlib-only primitives shared by the Stage V dynamic-8 runners."""
from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tempfile
import time
from typing import Any, Iterable, Mapping


# Q2 is frozen to seven GPUs; GPU5 is reserved for the post-Q2 eight-GPU path.
Q2_APPROVED_GPUS = (0, 1, 2, 3, 4, 6, 7)


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
        for attempt in range(8):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if os.name != "nt" or attempt == 7:
                    raise
                # Windows readers can briefly hold the destination open.
                time.sleep(0.01 * (attempt + 1))
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


def bind_source_artifact_rows(rows: Iterable[Mapping[str, Any]], source_manifest: Path) -> list[dict[str, Any]]:
    """Bind provenance-only source metadata; never reads the referenced artifacts."""
    source = read_json(source_manifest, {})
    audits = source.get("all_candidate_audits") if isinstance(source, Mapping) else None
    if not isinstance(audits, list):
        raise ValueError("SOURCE_CLEAN_PARENT_MANIFEST_INVALID")
    by_key: dict[str, Mapping[str, Any]] = {}
    for item in audits:
        if not isinstance(item, Mapping):
            raise ValueError("SOURCE_CLEAN_PARENT_ROW_INVALID")
        key = str(item.get("canonical_parent_key", ""))
        if not key or key in by_key:
            raise ValueError("SOURCE_CLEAN_PARENT_IDENTITY_INVALID")
        if not str(item.get("source_artifact_root", "")):
            raise ValueError(f"SOURCE_ARTIFACT_ROOT_MISSING:{key}")
        by_key[key] = item
    bound_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("PARENT_ROW_INVALID")
        key = canonical_parent_key(row)
        source_row = by_key.get(key)
        if source_row is None:
            raise ValueError(f"SOURCE_CLEAN_PARENT_MISSING:{key}")
        bound = dict(row)
        bound["source_artifact_root"] = str(source_row["source_artifact_root"])
        bound["artifact_recursive_sha256"] = str(source_row.get("artifact_recursive_sha256", ""))
        bound["source_artifact_manifest_sha256"] = str(source_row.get("artifact_manifest_sha256", ""))
        bound_rows.append(bound)
    return bound_rows


def load_rows(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if isinstance(value, Mapping):
        for key in ("parents", "qualified_parents", "selected_parents", "rows", "candidates", "all_candidate_audits", "manifest"):
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


def exposure_binding(parent_keys: Iterable[str], manifest: Path) -> dict[str, Any]:
    """Bind a prospective parent set to the append-only exposure exclusion set."""
    path = Path(manifest).resolve()
    try:
        value = read_json(path)
        manifest_sha = sha256_file(path)
    except OSError as exc:
        return {
            "schema": "STAGE_V_EXPOSURE_BINDING_V1", "status": "FAIL",
            "manifest_path": str(path), "manifest_sha256": None,
            "reason": f"EXPOSURE_MANIFEST_READ_FAIL:{type(exc).__name__}",
            "excluded_parent_count": 0, "overlap_parent_count": 0,
            "overlap_parent_keys": [],
        }
    excluded = value.get("excluded_parent_keys") if isinstance(value, Mapping) else None
    if not isinstance(excluded, list) or any(not isinstance(key, str) or not key for key in excluded):
        return {
            "schema": "STAGE_V_EXPOSURE_BINDING_V1", "status": "FAIL",
            "manifest_path": str(path), "manifest_sha256": manifest_sha,
            "exposure_manifest_schema": value.get("schema") if isinstance(value, Mapping) else None,
            "reason": "EXPOSURE_MANIFEST_EXCLUDED_KEYS_MISSING_OR_INVALID",
            "excluded_parent_count": 0, "overlap_parent_count": 0,
            "overlap_parent_keys": [],
        }
    excluded_keys = [str(key) for key in excluded]
    reason = "EXPOSURE_MANIFEST_DUPLICATE_KEYS" if len(set(excluded_keys)) != len(excluded_keys) else None
    overlap = sorted(set(str(key) for key in parent_keys) & set(excluded_keys))
    if overlap and reason is None:
        reason = "EXPOSURE_PARENT_OVERLAP"
    return {
        "schema": "STAGE_V_EXPOSURE_BINDING_V1",
        "status": "PASS" if reason is None else "FAIL",
        "manifest_path": str(path), "manifest_sha256": manifest_sha,
        "exposure_manifest_schema": value.get("schema") if isinstance(value, Mapping) else None,
        "reason": reason,
        "excluded_parent_count": len(excluded_keys),
        "overlap_parent_count": len(overlap),
        "overlap_parent_keys": overlap,
    }


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
        process.wait()
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
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
    if os.name == "nt":  # os.kill(pid, 0) is not a harmless probe on Windows.
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                check=False, capture_output=True, text=True, timeout=2,
            )
            return result.returncode == 0 and f'"{pid}"' in result.stdout
        except (OSError, subprocess.SubprocessError):
            return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            state = proc_stat.read_text(encoding="utf-8", errors="replace").rsplit(")", 1)[1].split()[0]
            if state == "Z":
                return False
        except (OSError, IndexError):
            pass
    return True


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
    all_safe = sorted(set(safe))
    safe = all_safe[:required_count]
    return {
        "schema": "STAGE_V_GPU_PREFLIGHT_V2",
        "status": "PASS" if len(safe) >= required_count and not error else "PRELAUNCH_WAITING_FOR_8_GPUS",
        "required_gpu_count": required_count,
        "safe_gpu_count": len(all_safe),
        "selected_gpu_count": len(safe),
        "all_safe_gpus": all_safe,
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


def _verify_parent_seal(root: Path) -> tuple[bool, str]:
    sums = root / "SHA256SUMS"
    sums_sha = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sums_sha.is_file():
        return False, "PARENT_SEAL_FILES_MISSING"
    try:
        expected = sums_sha.read_text(encoding="utf-8").strip().split()
        if len(expected) < 2 or expected[1] != "SHA256SUMS" or expected[0] != sha256_file(sums):
            return False, "PARENT_SEAL_SUMS_SHA_MISMATCH"
        for line in sums.read_text(encoding="utf-8").splitlines():
            parts = line.split(None, 1)
            if len(parts) != 2:
                return False, "PARENT_SEAL_ROW_INVALID"
            digest, relative = parts
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                return False, "PARENT_SEAL_PATH_INVALID"
            target = root / relative_path
            if not target.is_file() or sha256_file(target) != digest:
                return False, f"PARENT_SEAL_FILE_MISMATCH:{relative}"
    except (OSError, ValueError):
        return False, "PARENT_SEAL_READ_FAIL"
    return True, "PASS"


def _branch_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                errors.append(f"BRANCH_ROW_NOT_OBJECT:{line_number}")
            else:
                rows.append(dict(value))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"BRANCH_READ_FAIL:{type(exc).__name__}")
    return rows, errors


def _strict_branch_errors(branch_rows: list[dict[str, Any]], parent_key: str) -> list[str]:
    errors: list[str] = []
    identities = set()
    expected_arms = {"OPEN_T3", "OPEN_T5", "OPEN_T10"}
    arm_counts = {arm: 0 for arm in expected_arms}
    for row in branch_rows:
        identity = (row.get("canonical_parent_key"), row.get("probe_step"), row.get("arm"))
        if identity in identities:
            errors.append("DUPLICATE_BRANCH_IDENTITY")
        identities.add(identity)
        if row.get("canonical_parent_key") != parent_key:
            errors.append("BRANCH_PARENT_IDENTITY_MISMATCH")
        if row.get("arm") not in expected_arms:
            errors.append("BRANCH_ARM_INVALID")
        else:
            arm_counts[str(row["arm"])] += 1
        if row.get("control_arm") != "NOOP_T10_REPLAY":
            errors.append("CONTROL_ARM_INVALID")
        if row.get("prefix_replay_exact") is not True:
            errors.append("PREFIX_REPLAY_NOT_EXACT")
        for field in ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts", "attack_rollouts"):
            if row.get(field, 0) != 0:
                errors.append(f"BRANCH_BOUNDARY_VIOLATION:{field}")
        comparison = row.get("comparison") if isinstance(row.get("comparison"), Mapping) else {}
        if comparison.get("control_status") not in ("PASS", "DONE") or comparison.get("open_status") not in ("PASS", "DONE"):
            errors.append("BRANCH_RUNTIME_INVALID")
        if comparison.get("label_status") != "VALID":
            errors.append("BRANCH_LABEL_INVALID")
    if len(branch_rows) != 72:
        errors.append(f"BRANCH_COUNT:{len(branch_rows)}/72")
    if len(identities) != len(branch_rows):
        errors.append("DUPLICATE_BRANCH_IDENTITIES")
    if {row.get("arm") for row in branch_rows} != expected_arms:
        errors.append("BRANCH_ARM_COVERAGE_INVALID")
    if any(count != 24 for count in arm_counts.values()):
        errors.append("BRANCH_ARM_BALANCE_INVALID")
    return sorted(set(errors))


def _m35_coverage_artifact_status(output_dir: Path, parent_key: str, *, expected_source_commit: str | None = None,
                                   expected_source_tree: str | None = None, expected_row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    results = list(Path(output_dir).rglob("PARENT_RESULT.json"))
    if len(results) != 1:
        return {"valid": False, "reason": f"PARENT_RESULT_COUNT:{len(results)}", "result": None, "path": None}
    result_path = results[0]
    result = read_json(result_path, {})
    if not isinstance(result, Mapping) or str(result.get("schema")) != "STAGE_V_M3_5_CLEAN_COVERAGE_RESULT_V1" or str(result.get("canonical_parent_key")) != parent_key:
        return {"valid": False, "reason": "M35_COVERAGE_RESULT_SCHEMA_OR_IDENTITY_INVALID", "result": dict(result) if isinstance(result, Mapping) else None, "path": str(result_path)}
    errors: list[str] = []
    if result.get("status") != "PASS" or result.get("coverage_only") is not True or result.get("parent_atomic") is not True:
        errors.append("M35_COVERAGE_RESULT_INVALID")
    counts = result.get("phase_counts") if isinstance(result.get("phase_counts"), Mapping) else {}
    if set(counts) != {"PRE_CONTACT", "CONTACT_MANIPULATION", "ENGAGED_LIFT", "CARRY"} or any(not isinstance(counts.get(phase), int) or counts.get(phase) < 0 for phase in counts):
        errors.append("M35_COVERAGE_PHASE_COUNTS_INVALID")
    elif result.get("coverage_qualified") is not all(int(counts[phase]) >= 6 for phase in counts):
        errors.append("M35_COVERAGE_QUALIFICATION_MISMATCH")
    if result.get("protected_counters") != {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}:
        errors.append("M35_COVERAGE_PROTECTED_COUNTERS_INVALID")
    if expected_source_commit and result.get("source_commit") != expected_source_commit:
        errors.append("SCIENCE_SOURCE_COMMIT_MISMATCH")
    if expected_source_tree and result.get("source_tree") != expected_source_tree:
        errors.append("SCIENCE_SOURCE_TREE_MISMATCH")
    if expected_row is not None:
        for field in ("suite", "task_index", "state_index"):
            if result.get(field) != expected_row.get(field):
                errors.append(f"PARENT_{field.upper()}_MISMATCH")
    if len(list(result_path.parent.rglob("CLEAN_TRAJECTORY.json"))) != 1 or len(list(result_path.parent.rglob("PHASE_COVERAGE.json"))) != 1:
        errors.append("M35_COVERAGE_INPUT_FILES_INVALID")
    seal_ok, seal_reason = _verify_parent_seal(result_path.parent)
    if not seal_ok:
        errors.append(seal_reason)
    if errors:
        return {"valid": False, "reason": ";".join(sorted(set(errors))), "result": dict(result), "path": str(result_path)}
    return {"valid": True, "reason": "PASS", "result": dict(result), "path": str(result_path), "artifact_sha256": sha256_file(result_path), "label_status": "COVERAGE_ONLY", "parent_seal": "PASS"}


def _m35_artifact_status(output_dir: Path, parent_key: str, *, expected_source_commit: str | None = None,
                         expected_source_tree: str | None = None, expected_row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    results = list(Path(output_dir).rglob("PARENT_RESULT.json"))
    if len(results) != 1:
        return {"valid": False, "reason": f"PARENT_RESULT_COUNT:{len(results)}", "result": None, "path": None}
    result_path = results[0]
    result = read_json(result_path, {})
    if not isinstance(result, Mapping):
        return {"valid": False, "reason": "PARENT_RESULT_NOT_OBJECT", "result": None, "path": str(result_path)}
    if str(result.get("schema")) != "STAGE_V_M3_5_PARENT_RESULT_V1" or str(result.get("canonical_parent_key")) != parent_key:
        return {"valid": False, "reason": "M35_RESULT_SCHEMA_OR_IDENTITY_INVALID", "result": dict(result), "path": str(result_path)}
    errors: list[str] = []
    if result.get("status") != "PASS" or result.get("clean_success") is not True:
        errors.append("M35_RESULT_NOT_PASS")
    if result.get("parent_atomic") is not True or result.get("probe_count") != 24:
        errors.append("M35_PARENT_ACCOUNTING_INVALID")
    if result.get("expected_physical_branches") != 288 or result.get("actual_physical_branches") != 288:
        errors.append("M35_PHYSICAL_BRANCH_COUNT_INVALID")
    if result.get("expected_treatment_label_rows") != 216 or result.get("actual_treatment_label_rows") != 216:
        errors.append("M35_LABEL_ROW_COUNT_INVALID")
    if result.get("protected_counters") != {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}:
        errors.append("M35_PROTECTED_COUNTERS_INVALID")
    if expected_source_commit and result.get("source_commit") != expected_source_commit:
        errors.append("SCIENCE_SOURCE_COMMIT_MISMATCH")
    if expected_source_tree and result.get("source_tree") != expected_source_tree:
        errors.append("SCIENCE_SOURCE_TREE_MISMATCH")
    if expected_row is not None:
        for field in ("suite", "task_index", "state_index"):
            if result.get(field) != expected_row.get(field):
                errors.append(f"PARENT_{field.upper()}_MISMATCH")
    branches = list(Path(result_path.parent).rglob("COUNTERFACTUAL_BRANCHES.jsonl"))
    if len(branches) != 1:
        errors.append("M35_BRANCH_FILE_COUNT_INVALID")
    else:
        rows, parse_errors = _branch_rows(branches[0])
        errors.extend(parse_errors)
        if len(rows) != 288:
            errors.append(f"M35_BRANCH_COUNT:{len(rows)}/288")
        identities = set()
        counts = {arm: 0 for arm in ("CONTROL", "T3", "T5", "T10")}
        for row in rows:
            identity = (row.get("canonical_parent_key"), row.get("probe_step"), row.get("repetition"), row.get("arm"))
            if identity in identities:
                errors.append("M35_DUPLICATE_BRANCH_IDENTITY")
            identities.add(identity)
            if row.get("canonical_parent_key") != parent_key:
                errors.append("M35_BRANCH_PARENT_IDENTITY_MISMATCH")
            arm = row.get("arm")
            if arm not in counts:
                errors.append("M35_BRANCH_ARM_INVALID")
            else:
                counts[arm] += 1
            if row.get("eval160_reads", 0) != 0 or row.get("protected_eval_reads", 0) != 0 or row.get("attack_rollouts", 0) != 0:
                errors.append("M35_BRANCH_BOUNDARY_VIOLATION")
            if arm != "CONTROL" and not isinstance(row.get("pair"), Mapping):
                errors.append("M35_TREATMENT_PAIR_MISSING")
        if any(count != 72 for count in counts.values()):
            errors.append("M35_BRANCH_ARM_BALANCE_INVALID")
    seal_ok, seal_reason = _verify_parent_seal(result_path.parent)
    if not seal_ok:
        errors.append(seal_reason)
    if errors:
        return {"valid": False, "reason": ";".join(sorted(set(errors))), "result": dict(result), "path": str(result_path)}
    return {
        "valid": True, "reason": "PASS", "result": dict(result), "path": str(result_path),
        "artifact_sha256": sha256_file(result_path), "label_status": "VALID", "parent_seal": "PASS",
    }


def _m35_v2_artifact_status(output_dir: Path, parent_key: str, *, expected_source_commit: str | None = None,
                            expected_source_tree: str | None = None, expected_row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    root = Path(output_dir)
    results = list(root.rglob("PARENT_RESULT.json"))
    if len(results) != 1:
        return {"valid": False, "reason": f"PARENT_RESULT_COUNT:{len(results)}", "result": None, "path": None}
    result_path = results[0]
    result = read_json(result_path, {})
    if not isinstance(result, Mapping):
        return {"valid": False, "reason": "PARENT_RESULT_NOT_OBJECT", "result": None, "path": str(result_path)}
    errors: list[str] = []
    counters = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
    if result.get("schema") != "STAGE_V_M3_5_PARENT_RESULT_V2" or result.get("canonical_parent_key") != parent_key:
        errors.append("M35_V2_RESULT_SCHEMA_OR_IDENTITY_INVALID")
    if result.get("status") != "COMPLETE_VALID" or result.get("parent_atomic") is not True or result.get("probe_count") != 24:
        errors.append("M35_V2_PARENT_COMPLETION_INVALID")
    for field, value in (
        ("expected_physical_executions", 288), ("actual_physical_executions", 288),
        ("expected_treatment_repetition_observations", 216), ("actual_treatment_repetition_observations", 216),
        ("expected_collapsed_probe_dose_labels", 72), ("actual_collapsed_probe_dose_labels", 72),
    ):
        if result.get(field) != value:
            errors.append(f"M35_V2_ACCOUNTING_INVALID:{field}")
    if result.get("protected_counters") != counters:
        errors.append("M35_V2_PROTECTED_COUNTERS_INVALID")
    if expected_source_commit and result.get("source_commit") != expected_source_commit:
        errors.append("SCIENCE_SOURCE_COMMIT_MISMATCH")
    if expected_source_tree and result.get("source_tree") != expected_source_tree:
        errors.append("SCIENCE_SOURCE_TREE_MISMATCH")
    if expected_row is not None:
        for field in ("suite", "task_index", "state_index"):
            if result.get(field) != expected_row.get(field):
                errors.append(f"PARENT_{field.upper()}_MISMATCH")

    paths: dict[str, Path] = {}
    for name in ("COUNTERFACTUAL_BRANCHES.jsonl", "TREATMENT_REPETITION_OBSERVATIONS.jsonl", "COLLAPSED_PROBE_DOSE_LABELS.jsonl"):
        matches = list(result_path.parent.rglob(name))
        if len(matches) != 1:
            errors.append(f"M35_V2_FILE_COUNT:{name}:{len(matches)}")
        else:
            paths[name] = matches[0]
    required_json = {
        "M35_RUNTIME_BINDING_RECEIPT.json": "STAGE_V_M3_5_RUNTIME_BINDING_RECEIPT_V1",
        "CLEAN_TRAJECTORY.json": "STAGE_V_M3_5_CLEAN_TRAJECTORY_V1",
        "PROBE_PLAN.json": "STAGE_V_M3_5_PROBE_PLAN_V2",
        "CORRIDOR_COVERAGE.json": "STAGE_V_M3_5_CORRIDOR_COVERAGE_V2",
        "REPEATABILITY_SUMMARY.json": "STAGE_V_M3_5_REPEATABILITY_SUMMARY_V2",
        "BLINDED_TAXONOMY_EVIDENCE_MANIFEST.json": "STAGE_V_M3_5_BLINDED_TAXONOMY_EVIDENCE_MANIFEST_V1",
        "PROGRESS.json": "STAGE_V_M3_5_PROGRESS_V1",
    }
    required_values: dict[str, Mapping[str, Any]] = {}
    for name, schema in required_json.items():
        matches = list(result_path.parent.rglob(name))
        value = read_json(matches[0], {}) if len(matches) == 1 else {}
        if len(matches) != 1 or not isinstance(value, Mapping) or value.get("schema") != schema:
            errors.append(f"M35_V2_REQUIRED_OUTPUT_INVALID:{name}")
        else:
            required_values[name] = value
    runtime_receipt = required_values.get("M35_RUNTIME_BINDING_RECEIPT.json", {})
    if (
        runtime_receipt.get("status") != "PASS" or runtime_receipt.get("parent_key") != parent_key
        or runtime_receipt.get("source_commit") != result.get("source_commit")
        or runtime_receipt.get("source_tree") != result.get("source_tree")
    ):
        errors.append("M35_V2_RUNTIME_BINDING_RECEIPT_INVALID")
    clean = required_values.get("CLEAN_TRAJECTORY.json", {})
    if clean.get("outcomes_read") is not False or clean.get("task_success") is not result.get("clean_success"):
        errors.append("M35_V2_CLEAN_TRAJECTORY_INVALID")
    probe_plan = required_values.get("PROBE_PLAN.json", {})
    if probe_plan.get("outcomes_read") is not False or probe_plan.get("probe_count") != 24 or probe_plan.get("protected_counters") != counters:
        errors.append("M35_V2_PROBE_PLAN_INVALID")
    corridor = required_values.get("CORRIDOR_COVERAGE.json", {})
    if corridor.get("outcomes_read") is not False or corridor.get("corridor_qualified") is not True or corridor.get("protected_counters") != counters:
        errors.append("M35_V2_CORRIDOR_COVERAGE_INVALID")
    blinded_evidence = required_values.get("BLINDED_TAXONOMY_EVIDENCE_MANIFEST.json", {})
    if blinded_evidence.get("canonical_parent_key") != parent_key or blinded_evidence.get("protected_counters") != counters or not isinstance(blinded_evidence.get("complete"), bool):
        errors.append("M35_V2_BLINDED_EVIDENCE_MANIFEST_INVALID")
    progress = required_values.get("PROGRESS.json", {})
    if progress.get("stage") != "COMPLETE" or progress.get("branch_progress") != 288 or progress.get("current_branch") is not None or progress.get("protected_counters") != counters:
        errors.append("M35_V2_FINAL_PROGRESS_INVALID")
    branches, branch_parse = _branch_rows(paths["COUNTERFACTUAL_BRANCHES.jsonl"]) if "COUNTERFACTUAL_BRANCHES.jsonl" in paths else ([], [])
    observations, observation_parse = _branch_rows(paths["TREATMENT_REPETITION_OBSERVATIONS.jsonl"]) if "TREATMENT_REPETITION_OBSERVATIONS.jsonl" in paths else ([], [])
    labels, label_parse = _branch_rows(paths["COLLAPSED_PROBE_DOSE_LABELS.jsonl"]) if "COLLAPSED_PROBE_DOSE_LABELS.jsonl" in paths else ([], [])
    errors.extend(branch_parse + observation_parse + label_parse)
    if len(branches) != 288:
        errors.append(f"M35_V2_PHYSICAL_EXECUTION_COUNT:{len(branches)}/288")
    if len(observations) != 216:
        errors.append(f"M35_V2_TREATMENT_OBSERVATION_COUNT:{len(observations)}/216")
    if len(labels) != 72:
        errors.append(f"M35_V2_COLLAPSED_LABEL_COUNT:{len(labels)}/72")

    branch_by_id: dict[str, dict[str, Any]] = {}
    controls: dict[tuple[Any, Any], dict[str, Any]] = {}
    treatments: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    probe_steps: dict[Any, Any] = {}
    arm_counts = {arm: 0 for arm in ("CONTROL", "T3", "T5", "T10")}
    for row in branches:
        branch_id = str(row.get("branch_id", ""))
        arm = row.get("arm")
        probe_id = row.get("probe_id")
        repetition = row.get("repetition")
        branch = row.get("branch")
        if row.get("schema") != "STAGE_V_M3_5_PHYSICAL_EXECUTION_V2" or row.get("canonical_parent_key") != parent_key:
            errors.append("M35_V2_BRANCH_SCHEMA_OR_PARENT_INVALID")
        if not branch_id or branch_id in branch_by_id:
            errors.append("M35_V2_BRANCH_ID_INVALID_OR_DUPLICATED")
        else:
            branch_by_id[branch_id] = row
        expected_id = f"m35-{sha256_text(f'M35_V1_3::{parent_key}::{probe_id}::R{repetition}::{arm}')}"
        if branch_id != expected_id:
            errors.append("M35_V2_BRANCH_ID_HASH_MISMATCH")
        if not isinstance(branch, Mapping) or row.get("branch_result_sha256") != sha256_json(branch):
            errors.append("M35_V2_BRANCH_RESULT_SHA_MISMATCH")
        if row.get("protected_counters") != counters:
            errors.append("M35_V2_BRANCH_PROTECTED_COUNTERS_INVALID")
        if arm not in arm_counts:
            errors.append("M35_V2_BRANCH_ARM_INVALID")
            continue
        arm_counts[str(arm)] += 1
        if probe_id in probe_steps and probe_steps[probe_id] != row.get("probe_step"):
            errors.append("M35_V2_PROBE_STEP_INCONSISTENT")
        probe_steps[probe_id] = row.get("probe_step")
        if arm == "CONTROL":
            identity = (probe_id, repetition)
            if identity in controls or row.get("pair") is not None or row.get("shared_control_branch_id") is not None or row.get("shared_control_result_sha256") is not None:
                errors.append("M35_V2_CONTROL_IDENTITY_OR_LINEAGE_INVALID")
            controls[identity] = row
        else:
            identity = (probe_id, repetition, arm)
            if identity in treatments or not isinstance(row.get("pair"), Mapping):
                errors.append("M35_V2_TREATMENT_IDENTITY_OR_PAIR_INVALID")
            treatments[identity] = row
    if set(probe_steps) != {f"Q{index:02d}" for index in range(24)} or len(set(probe_steps.values())) != 24:
        errors.append("M35_V2_PROBE_ID_OR_STEP_COVERAGE_INVALID")
    if any(count != 72 for count in arm_counts.values()) or len(controls) != 72 or len(treatments) != 216:
        errors.append("M35_V2_BRANCH_BALANCE_INVALID")
    for (probe_id, repetition, _arm), treatment in treatments.items():
        control = controls.get((probe_id, repetition))
        pair = treatment.get("pair") if isinstance(treatment.get("pair"), Mapping) else {}
        if control is None or any(
            treatment.get(field) != control.get(control_field)
            for field, control_field in (
                ("shared_control_branch_id", "branch_id"),
                ("shared_control_result_sha256", "branch_result_sha256"),
            )
        ):
            errors.append("M35_V2_MATCHED_CONTROL_LINEAGE_INVALID")
        elif pair.get("shared_control_branch_id") != control.get("branch_id") or pair.get("shared_control_result_sha256") != control.get("branch_result_sha256"):
            errors.append("M35_V2_PAIR_CONTROL_LINEAGE_INVALID")

    observation_by_id: dict[str, dict[str, Any]] = {}
    observation_by_treatment: dict[str, dict[str, Any]] = {}
    for row in observations:
        observation_id = str(row.get("observation_id", ""))
        treatment_id = str(row.get("treatment_branch_id", ""))
        identity = {
            "canonical_parent_key": row.get("canonical_parent_key"), "probe_id": row.get("probe_id"),
            "repetition": row.get("repetition"), "dose": row.get("dose"),
        }
        treatment = branch_by_id.get(treatment_id)
        control = branch_by_id.get(str(row.get("shared_control_branch_id", "")))
        if row.get("schema") != "STAGE_V_M3_5_TREATMENT_REPETITION_OBSERVATION_V1" or row.get("canonical_parent_key") != parent_key:
            errors.append("M35_V2_OBSERVATION_SCHEMA_OR_PARENT_INVALID")
        if observation_id != f"m35-observation-{sha256_json(identity)}" or observation_id in observation_by_id:
            errors.append("M35_V2_OBSERVATION_ID_INVALID_OR_DUPLICATED")
        observation_by_id[observation_id] = row
        if treatment_id in observation_by_treatment:
            errors.append("M35_V2_TREATMENT_OBSERVATION_DUPLICATED")
        observation_by_treatment[treatment_id] = row
        if treatment is None or control is None or treatment.get("arm") != row.get("dose"):
            errors.append("M35_V2_OBSERVATION_BRANCH_REFERENCE_INVALID")
        elif (
            row.get("treatment_result_sha256") != treatment.get("branch_result_sha256")
            or row.get("shared_control_result_sha256") != control.get("branch_result_sha256")
            or treatment.get("shared_control_branch_id") != control.get("branch_id")
            or row.get("label_class") != treatment.get("pair", {}).get("label_class")
        ):
            errors.append("M35_V2_OBSERVATION_LINEAGE_OR_LABEL_INVALID")
        if row.get("protected_counters") != counters:
            errors.append("M35_V2_OBSERVATION_PROTECTED_COUNTERS_INVALID")
    if set(observation_by_treatment) != {str(row.get("branch_id")) for row in treatments.values()}:
        errors.append("M35_V2_TREATMENT_OBSERVATION_COVERAGE_INVALID")

    referenced_observations: set[str] = set()
    label_identities: set[tuple[Any, Any]] = set()
    all_binary = True
    for row in labels:
        identity = (row.get("probe_id"), row.get("dose"))
        if row.get("schema") != "STAGE_V_M3_5_COLLAPSED_PROBE_DOSE_LABEL_V1" or row.get("canonical_parent_key") != parent_key or identity in label_identities:
            errors.append("M35_V2_COLLAPSED_LABEL_SCHEMA_OR_IDENTITY_INVALID")
        label_identities.add(identity)
        expected_label_id = f"m35-label-{sha256_json({'parent': parent_key, 'probe': identity[0], 'dose': identity[1]})}"
        if row.get("collapsed_label_id") != expected_label_id:
            errors.append("M35_V2_COLLAPSED_LABEL_ID_HASH_MISMATCH")
        ids = row.get("treatment_observation_ids")
        selected = [observation_by_id.get(str(item)) for item in ids] if isinstance(ids, list) else []
        if len(selected) != 3 or any(item is None for item in selected):
            errors.append("M35_V2_COLLAPSED_OBSERVATION_LINEAGE_INVALID")
            continue
        selected = sorted((item for item in selected if item is not None), key=lambda item: int(item.get("repetition", -1)))
        if [item.get("repetition") for item in selected] != [0, 1, 2] or any((item.get("probe_id"), item.get("dose")) != identity for item in selected):
            errors.append("M35_V2_COLLAPSED_REPETITION_IDENTITY_INVALID")
        referenced_observations.update(str(item) for item in ids)
        classes = [str(item.get("label_class")) for item in selected]
        if len(set(classes)) != 1:
            repeat_status, collapsed = "HOLD_STOCHASTIC_INTERVENTION_OUTCOME", None
        elif classes[0].endswith("_ABSTAIN") or classes[0] in {"UNKNOWN", "HORIZON_CENSORED"}:
            repeat_status, collapsed = "STABLE_ABSTAIN", classes[0]
        elif not all(item.get("treatment_compliant") is True for item in selected):
            repeat_status, collapsed = "TREATMENT_NONCOMPLIANCE_ABSTAIN", None
        else:
            repeat_status, collapsed = "PASS_REPEATABILITY_3_OF_3", classes[0]
        binary = repeat_status == "PASS_REPEATABILITY_3_OF_3" and collapsed in {"V_PHYS", "NO_PHYSICAL_VULNERABILITY"}
        all_binary = all_binary and binary
        treatment_lineage = [{"branch_id": item.get("treatment_branch_id"), "result_sha256": item.get("treatment_result_sha256")} for item in selected]
        control_lineage = [{"branch_id": item.get("shared_control_branch_id"), "result_sha256": item.get("shared_control_result_sha256")} for item in selected]
        if (
            row.get("repeatability_status") != repeat_status or row.get("collapsed_label_class") != collapsed
            or row.get("binary_label_consumable") is not binary
            or row.get("treatment_branch_lineage") != treatment_lineage
            or row.get("matched_control_lineage") != control_lineage
        ):
            errors.append("M35_V2_COLLAPSED_LABEL_RECOMPUTE_MISMATCH")
        if row.get("protected_counters") != counters:
            errors.append("M35_V2_COLLAPSED_LABEL_PROTECTED_COUNTERS_INVALID")
    expected_label_identities = {(f"Q{probe:02d}", dose) for probe in range(24) for dose in ("T3", "T5", "T10")}
    if referenced_observations != set(observation_by_id) or label_identities != expected_label_identities:
        errors.append("M35_V2_COLLAPSED_LABEL_COVERAGE_INVALID")
    expected_label_status = "PASS" if result.get("clean_success") is True and blinded_evidence.get("complete") is True and all_binary and all(
        isinstance(row.get("branch"), Mapping) and row["branch"].get("status") == "PASS" for row in branches
    ) else "FAIL"
    repeatability = required_values.get("REPEATABILITY_SUMMARY.json", {})
    if repeatability.get("collapsed_label_count") != 72 or repeatability.get("collapsed_labels") != labels:
        errors.append("M35_V2_REPEATABILITY_SUMMARY_INVALID")
    if result.get("label_validation_status") != expected_label_status:
        errors.append("M35_V2_LABEL_VALIDATION_STATUS_MISMATCH")
    seal_ok, seal_reason = _verify_parent_seal(result_path.parent)
    if not seal_ok:
        errors.append(seal_reason)
    if errors:
        return {"valid": False, "reason": ";".join(sorted(set(errors))), "result": dict(result), "path": str(result_path)}
    return {
        "valid": True, "reason": "PASS", "result": dict(result), "path": str(result_path),
        "artifact_sha256": sha256_file(result_path), "label_status": expected_label_status, "parent_seal": "PASS",
    }


def science_artifact_status(output_dir: Path, parent_key: str, *, expected_source_commit: str | None = None,
                            expected_source_tree: str | None = None, expected_row: Mapping[str, Any] | None = None,
                            artifact_schema: str = "STAGE_V_PARENT_RESULT_V2") -> dict[str, Any]:
    if artifact_schema == "STAGE_V_M3_5_COVERAGE_RESULT_V1":
        return _m35_coverage_artifact_status(
            output_dir, parent_key, expected_source_commit=expected_source_commit,
            expected_source_tree=expected_source_tree, expected_row=expected_row,
        )
    if artifact_schema == "STAGE_V_M3_5_PARENT_RESULT_V1":
        return _m35_artifact_status(
            output_dir, parent_key, expected_source_commit=expected_source_commit,
            expected_source_tree=expected_source_tree, expected_row=expected_row,
        )
    if artifact_schema == "STAGE_V_M3_5_PARENT_RESULT_V2":
        try:
            return _m35_v2_artifact_status(
                output_dir, parent_key, expected_source_commit=expected_source_commit,
                expected_source_tree=expected_source_tree, expected_row=expected_row,
            )
        except (IndexError, KeyError, OverflowError, TypeError, ValueError) as exc:
            return {
                "valid": False,
                "reason": f"M35_V2_MALFORMED_ARTIFACT:{type(exc).__name__}:{exc}",
                "result": None,
                "path": None,
            }
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
    strict_errors: list[str] = []
    if expected_source_commit or expected_source_tree or expected_row is not None:
        if expected_source_commit and result.get("current_source_commit") != expected_source_commit:
            strict_errors.append("SCIENCE_SOURCE_COMMIT_MISMATCH")
        if expected_source_tree and result.get("current_source_tree") != expected_source_tree:
            strict_errors.append("SCIENCE_SOURCE_TREE_MISMATCH")
        if result.get("current_source_status") not in ("", None):
            strict_errors.append("SCIENCE_SOURCE_WORKTREE_DIRTY")
        if expected_row is not None:
            for result_field, aliases, row_field in (
                ("suite", (), "suite"),
                ("task_idx", ("task_index",), "task_index"),
                ("state_id", ("state_index",), "state_index"),
            ):
                values = [result[name] for name in (result_field, *aliases) if name in result]
                if not values or any(value != values[0] for value in values[1:]) or values[0] != expected_row.get(row_field):
                    strict_errors.append(f"PARENT_{result_field.upper()}_MISMATCH")
        branch_rows, branch_errors = _branch_rows(branches[0])
        strict_errors.extend(branch_errors)
        strict_errors.extend(_strict_branch_errors(branch_rows, parent_key))
        seal_ok, seal_reason = _verify_parent_seal(result_path.parent)
        if not seal_ok:
            strict_errors.append(seal_reason)
    if strict_errors:
        return {"valid": False, "reason": ";".join(sorted(set(strict_errors))), "result": dict(result), "path": str(result_path)}
    return {
        "valid": True,
        "reason": "PASS",
        "result": dict(result),
        "path": str(result_path),
        "artifact_sha256": sha256_file(result_path),
        "label_status": "VALID",
        "parent_seal": "PASS" if expected_source_commit or expected_source_tree or expected_row is not None else "UNVERIFIED",
    }
