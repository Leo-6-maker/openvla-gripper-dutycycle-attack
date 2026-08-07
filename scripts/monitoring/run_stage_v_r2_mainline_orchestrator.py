"""Fail-closed, server-local orchestrator for the Stage V R2 GPU line.

The existing mainline monitor is deliberately only an observer.  This file is
the separate control-plane owner: it waits for explicit receipts, validates
SHA-bound plans, performs process-aware GPU preflight, and launches at most
one registered stage.  SSH is not used here, so losing the parent shell cannot
turn a healthy run into an abort.
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
import time
from typing import Any, Iterable, Mapping

try:
    from .stage_v_dynamic_common import atomic_write_json, read_json, sha256_file, utc_now
except ImportError:  # direct server execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.detector_v5.stage_v_dynamic_common import atomic_write_json, read_json, sha256_file, utc_now


SCHEMA = "STAGE_V_R2_MAINLINE_ORCHESTRATOR_V1"
REGISTRY_SCHEMA = "STAGE_V_R2_ORCHESTRATOR_PLAN_REGISTRY_V2"
LEGACY_REGISTRY_SCHEMA = "STAGE_V_R2_ORCHESTRATOR_PLAN_REGISTRY_V1"
PLAN_SCHEMA = "STAGE_V_R2_ORCHESTRATOR_PLAN_V2"
LEGACY_PLAN_SCHEMA = "STAGE_V_R2_ORCHESTRATOR_PLAN_V1"
WAIT_QUALIFICATION = "WAIT_QUALIFICATION"
WAIT_GPUS = "WAIT_8_SAFE_GPUS"
HARD_STOP = "HARD_STOP"
STAGES = (
    "C0", "R2A", "R2B_DECISION", "R2B", "STAGE_V2", "STAGE_O", "STUDENT_FREEZE",
    "PILOT_QUALIFICATION", "DIRECT_OPEN_PILOT", "VIS_SMALL_MATRIX",
)
FORBIDDEN_BOUNDARY_FIELDS = ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts")
UNIVERSAL_FORBIDDEN_COMMAND = re.compile(r"(?i)(?:eval160|protected[_-]?eval|full[_-]?confirmatory|final[_-]?detector|guard)")
PRE_VIS_FORBIDDEN_COMMAND = re.compile(r"(?i)(?:vis|pgd|(?<!vla_)attack)")


class OrchestratorError(RuntimeError):
    """A fail-closed validation or process-identity error."""


class DuplicateOrchestrator(OrchestratorError):
    pass


def _read(path: Path, default: Any = None) -> Any:
    return read_json(path, default)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    with temporary.open("r+b") as handle:
        try:
            os.fsync(handle.fileno())
        except OSError:
            if os.name != "nt":
                raise
    os.replace(temporary, path)


def _write_sha(path: Path) -> str:
    digest = sha256_file(path)
    _write_text(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n")
    return digest


def pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, check=False, timeout=2,
            )
            return result.returncode == 0 and f'"{pid}"' in result.stdout
        except (OSError, subprocess.SubprocessError):
            return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        try:
            if stat.read_text(encoding="utf-8", errors="replace").rsplit(")", 1)[1].split()[0] == "Z":
                return False
        except (OSError, IndexError):
            pass
    return True


def _proc_identity(pid: int) -> dict[str, Any] | None:
    if not pid_alive(pid):
        return None
    result: dict[str, Any] = {"pid": pid}
    if os.name == "nt":
        result["cmdline"] = []
        result["cwd"] = None
        result["start_ticks"] = None
        return result
    try:
        raw_cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        result["cmdline"] = [item.decode("utf-8", "replace") for item in raw_cmdline if item]
        result["cwd"] = os.readlink(f"/proc/{pid}/cwd")
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace").rsplit(")", 1)[1].split()
        result["start_ticks"] = int(fields[19])  # field 22, after pid/comm/state
    except (OSError, ValueError, IndexError):
        return None
    return result


def _proc_start_ticks(pid: int) -> int | None:
    value = _proc_identity(pid)
    return value.get("start_ticks") if value else None


def _same_path(actual: Any, expected: Path) -> bool:
    try:
        return Path(str(actual)).resolve() == expected.resolve()
    except (OSError, TypeError):
        return False


def source_binding(repo_root: Path) -> dict[str, str]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise OrchestratorError(f"GIT_QUERY_FAIL:{result.stderr[-200:]}")
        return result.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "status_porcelain": git("status", "--porcelain", "--untracked-files=all"),
    }


def active_project_dispatcher_pids(repo_root: Path) -> list[int]:
    """Find existing project dispatchers without relying on SSH or pgrep."""
    if os.name == "nt":
        return []
    pids: list[int] = []
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(entry.name)
            command = " ".join(item.decode("utf-8", "replace") for item in (entry / "cmdline").read_bytes().split(b"\0") if item)
        except (OSError, ValueError):
            continue
        if pid == os.getpid():
            continue
        if "dispatcher" in command.lower() and "run_stage_v" in command and str(repo_root) in command:
            pids.append(pid)
    return sorted(pids)


def _parse_csv_row(line: str) -> list[str]:
    return [field.strip() for field in line.split(",")]


def _query(command: str, timeout: float = 20.0) -> tuple[str, str | None]:
    try:
        result = subprocess.run(shlex.split(command), capture_output=True, text=True, check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", f"QUERY_ERROR:{type(exc).__name__}"
    if result.returncode:
        return result.stdout, f"QUERY_EXIT:{result.returncode}:{result.stderr[-200:]}"
    return result.stdout, None


def _process_info(pid: int) -> dict[str, Any]:
    info = _proc_identity(pid) or {"pid": pid, "cmdline": [], "cwd": None, "start_ticks": None}
    if os.name != "nt":
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="replace")
            uid_line = next((line for line in status.splitlines() if line.startswith("Uid:")), "")
            info["uid"] = int(uid_line.split()[1]) if uid_line.split()[1].isdigit() else None
        except (OSError, IndexError, ValueError):
            info["uid"] = None
    else:
        info["uid"] = None
    return info


def _gpu_rows(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = _parse_csv_row(line)
        if len(fields) < 6:
            continue
        try:
            rows.append({
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": float(fields[3]),
                "memory_used_mib": float(fields[4]),
                "memory_free_mib": float(fields[5]),
                "utilization_gpu_percent": float(fields[6]) if len(fields) > 6 else None,
            })
        except ValueError:
            continue
    return rows


def _app_rows(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = _parse_csv_row(line)
        if len(fields) < 3:
            continue
        try:
            rows.append({"gpu_uuid": fields[0], "pid": int(fields[1]), "used_memory_mib": float(fields[2])})
        except ValueError:
            continue
    return rows


def gpu_preflight(
    *,
    stage: str,
    required_gpus: int,
    excluded_gpus: Iterable[int],
    protected_pids: Iterable[int],
    canary_peak_mib: float,
    project_root: Path,
    gpu_query_command: str = "nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits",
    app_query_command: str = "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits",
    xid_query_command: str = "journalctl -k --since=-10min --no-pager",
) -> dict[str, Any]:
    """Return a process-aware, strict-8 decision without touching processes."""
    excluded = {int(value) for value in excluded_gpus}
    protected = {int(value) for value in protected_pids if int(value) > 0}
    gpu_output, gpu_error = _query(gpu_query_command)
    app_output, app_error = _query(app_query_command)
    xid_output, xid_error = _query(xid_query_command)
    xid_present = bool(re.search(r"NVRM: Xid|GPU has fallen off|Xid \(", xid_output, re.IGNORECASE))
    rows = _gpu_rows(gpu_output)
    apps = _app_rows(app_output)
    by_uuid = {row["uuid"]: row for row in rows}
    minimum_free = float(canary_peak_mib) * 1.5 + 4096.0
    mapped: dict[int, list[dict[str, Any]]] = {}
    for app in apps:
        gpu = by_uuid.get(app["gpu_uuid"], {}).get("index")
        if gpu is not None:
            mapped.setdefault(int(gpu), []).append({**app, "process": _process_info(app["pid"])})
    decisions: list[dict[str, Any]] = []
    safe: list[int] = []
    for row in rows:
        gpu = int(row["index"])
        reasons: list[str] = []
        process_rows = mapped.get(gpu, [])
        process_view: list[dict[str, Any]] = []
        for app in process_rows:
            info = app["process"]
            cmdline = info.get("cmdline") or []
            cwd = info.get("cwd")
            project_owned = _same_path(cwd, project_root) or any(str(project_root) in str(arg) for arg in cmdline)
            protected_process = int(app["pid"]) in protected
            process_view.append({
                "pid": app["pid"], "used_memory_mib": app["used_memory_mib"],
                "uid": info.get("uid"), "command": cmdline, "cwd": cwd,
                "start_ticks": info.get("start_ticks"), "project_owned": project_owned,
                "protected": protected_process, "foreign": not project_owned and not protected_process,
            })
            if protected_process:
                reasons.append("PROTECTED_PROCESS_PRESENT")
            elif project_owned:
                reasons.append("PROJECT_PROCESS_PRESENT")
            else:
                reasons.append("FOREIGN_PROCESS_PRESENT")
        if app_error:
            reasons.append("PROCESS_QUERY_UNKNOWN")
        if xid_present:
            reasons.append("XID_PRESENT")
        if gpu in excluded:
            reasons.append("EXCLUDED_GPU")
        if float(row["memory_free_mib"]) < minimum_free:
            reasons.append("INSUFFICIENT_FREE_MEMORY")
        decision = {**row, "processes": process_view, "safe": not reasons, "reasons": sorted(set(reasons))}
        decisions.append(decision)
        if decision["safe"]:
            safe.append(gpu)
    safe = sorted(set(safe))
    status = "PASS" if not gpu_error and not app_error and not xid_error and not xid_present and len(safe) >= required_gpus else "PRELAUNCH_WAITING_FOR_8_GPUS"
    return {
        "schema": "STAGE_V_GPU_PREFLIGHT_PROCESS_AWARE_V1",
        "stage": stage,
        "status": status,
        "required_gpu_count": required_gpus,
        "safe_gpu_count": len(safe),
        "safe_gpus": safe[:required_gpus],
        "safe_gpu_uuids": [row["uuid"] for row in decisions if row["index"] in safe[:required_gpus]],
        "all_safe_gpus": safe,
        "excluded_gpus": sorted(excluded),
        "protected_pids": sorted(protected),
        "minimum_free_memory_mib": minimum_free,
        "canary_peak_memory_mib": canary_peak_mib,
        "gpu_rows": decisions,
        "gpu_query_error": gpu_error,
        "compute_app_query_error": app_error,
        "xid_query_error": xid_error,
        "xid_present": xid_present,
        "gpu5_touched": 5 in safe,
        "updated_utc": utc_now(),
    }


def _load_object(path: Path, name: str) -> Mapping[str, Any]:
    value = _read(path)
    if not isinstance(value, Mapping):
        raise OrchestratorError(f"{name}_NOT_OBJECT:{path}")
    return value


def _verify_bound_file(path: Path, expected: Any, field: str) -> None:
    if not path.is_file():
        raise OrchestratorError(f"{field}_MISSING:{path}")
    if not isinstance(expected, str) or len(expected) != 64 or sha256_file(path) != expected:
        raise OrchestratorError(f"{field}_SHA256_MISMATCH:{path}")


def _resource_policy(plan: Mapping[str, Any]) -> dict[str, Any]:
    value = plan.get("resource_policy")
    if isinstance(value, Mapping):
        policy = dict(value)
    else:
        legacy = plan.get("gpu_policy")
        if not isinstance(legacy, Mapping):
            return {"resource_kind": "CPU_ONLY", "required_gpu_count": 0, "minimum_gpu_count": 0, "maximum_gpu_count": 0, "strict_gpu_count": False, "excluded_gpus": [], "protected_pids": [], "canary_peak_mib": 0}
        policy = {"resource_kind": "GPU", "required_gpu_count": legacy.get("required_count", 0), "minimum_gpu_count": legacy.get("required_count", 0), "maximum_gpu_count": legacy.get("required_count", 0), "strict_gpu_count": True, **dict(legacy)}
    policy.setdefault("resource_kind", "GPU")
    policy.setdefault("required_gpu_count", policy.get("required_count", 0))
    policy.setdefault("minimum_gpu_count", policy.get("required_gpu_count", 0))
    policy.setdefault("maximum_gpu_count", policy.get("required_gpu_count", 0))
    policy.setdefault("strict_gpu_count", bool(policy.get("required_gpu_count", 0)))
    policy.setdefault("excluded_gpus", [])
    policy.setdefault("protected_pids", [])
    policy.setdefault("canary_peak_mib", 0)
    return policy


def _validate_command_boundary(stage: str, argv: list[str]) -> None:
    rendered = " ".join(argv).lower()
    if UNIVERSAL_FORBIDDEN_COMMAND.search(rendered):
        raise OrchestratorError("PLAN_FORBIDDEN_BOUNDARY_COMMAND")
    if stage != "VIS_SMALL_MATRIX" and PRE_VIS_FORBIDDEN_COMMAND.search(rendered):
        raise OrchestratorError("PLAN_STAGE_FORBIDDEN_COMMAND")


def validate_plan(path: Path, *, source: Mapping[str, str], expected_stage: str | None = None) -> dict[str, Any]:
    plan = dict(_load_object(path, "PLAN"))
    if plan.get("schema") not in {PLAN_SCHEMA, LEGACY_PLAN_SCHEMA}:
        raise OrchestratorError("PLAN_SCHEMA_INVALID")
    stage = str(plan.get("stage", ""))
    if stage not in STAGES or (expected_stage and stage != expected_stage):
        raise OrchestratorError("PLAN_STAGE_INVALID")
    if plan.get("source_commit") != source["commit"] or plan.get("source_tree") != source["tree"]:
        raise OrchestratorError("PLAN_SOURCE_MISMATCH")
    if plan.get("cwd") is None or not Path(str(plan["cwd"])).is_dir():
        raise OrchestratorError("PLAN_CWD_MISSING")
    for path_field, sha_field in (("runner_path", "runner_sha256"), ("auditor_path", "auditor_sha256"), ("config_path", "config_sha256")):
        if not plan.get(path_field):
            raise OrchestratorError(f"PLAN_{path_field.upper()}_MISSING")
        _verify_bound_file(Path(str(plan[path_field])), plan.get(sha_field), path_field)
    inputs = plan.get("input_receipts")
    if not isinstance(inputs, list) or not inputs:
        raise OrchestratorError("PLAN_INPUT_RECEIPTS_MISSING")
    for index, item in enumerate(inputs):
        if not isinstance(item, Mapping):
            raise OrchestratorError(f"PLAN_INPUT_RECEIPT_INVALID:{index}")
        _verify_bound_file(Path(str(item.get("path", ""))), item.get("sha256"), f"input_receipt_{index}")
    parent = plan.get("parent_manifest")
    if not isinstance(parent, Mapping):
        raise OrchestratorError("PLAN_PARENT_MANIFEST_MISSING")
    _verify_bound_file(Path(str(parent.get("path", ""))), parent.get("sha256"), "parent_manifest")
    template = plan.get("output_root_template")
    if not isinstance(template, str) or not Path(template.replace("{commit8}", "x").replace("{utc}", "x")).is_absolute():
        raise OrchestratorError("PLAN_OUTPUT_TEMPLATE_INVALID")
    command = plan.get("command_template")
    audit_command = plan.get("audit_command_template")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        raise OrchestratorError("PLAN_COMMAND_MUST_BE_ARGV_LIST")
    if not isinstance(audit_command, list) or not audit_command or not all(isinstance(item, str) for item in audit_command):
        raise OrchestratorError("PLAN_AUDIT_COMMAND_MUST_BE_ARGV_LIST")
    _validate_command_boundary(stage, command)
    _validate_command_boundary(stage, audit_command)
    for field in FORBIDDEN_BOUNDARY_FIELDS:
        if plan.get(field, 0) != 0:
            raise OrchestratorError(f"PLAN_BOUNDARY_NONZERO:{field}")
    policy = _resource_policy(plan)
    if str(policy.get("resource_kind")) not in {"CPU_ONLY", "GPU"}:
        raise OrchestratorError("PLAN_RESOURCE_KIND_INVALID")
    required = int(policy.get("required_gpu_count", 0))
    minimum = int(policy.get("minimum_gpu_count", required))
    maximum = int(policy.get("maximum_gpu_count", required))
    if required < 0 or minimum < 0 or maximum < minimum:
        raise OrchestratorError("PLAN_RESOURCE_COUNT_INVALID")
    if str(policy.get("resource_kind")) == "CPU_ONLY" and any((required, minimum, maximum)):
        raise OrchestratorError("PLAN_CPU_ONLY_GPU_COUNT_NONZERO")
    if stage in {"C0", "R2A", "R2B"}:
        excluded = {int(x) for x in policy.get("excluded_gpus", [])}
        gpu5_authorized = bool(policy.get("gpu5_authorized"))
        if (str(policy.get("resource_kind")) != "GPU" or required != 8
                or not bool(policy.get("strict_gpu_count"))
                or (5 not in excluded and not gpu5_authorized)):
            raise OrchestratorError("PLAN_GPU_POLICY_NOT_STRICT_8")
    if not isinstance(plan.get("lock_path"), str) or not plan["lock_path"]:
        raise OrchestratorError("PLAN_LOCK_MISSING")
    contract = plan.get("forbidden_boundary_contract")
    if isinstance(contract, Mapping) and any(int(contract.get(field, 0)) != 0 for field in FORBIDDEN_BOUNDARY_FIELDS):
        raise OrchestratorError("PLAN_FORBIDDEN_BOUNDARY_NONZERO")
    return plan


def load_registry(path: Path, *, source: Mapping[str, str]) -> tuple[dict[str, dict[str, Any]], str]:
    registry = dict(_load_object(path, "PLAN_REGISTRY"))
    if registry.get("schema") not in {REGISTRY_SCHEMA, LEGACY_REGISTRY_SCHEMA}:
        raise OrchestratorError("PLAN_REGISTRY_SCHEMA_INVALID")
    if registry.get("source_commit") is not None and registry.get("source_commit") != source["commit"]:
        raise OrchestratorError("PLAN_REGISTRY_SOURCE_MISMATCH")
    if registry.get("source_tree") is not None and registry.get("source_tree") != source["tree"]:
        raise OrchestratorError("PLAN_REGISTRY_TREE_MISMATCH")
    entries = registry.get("plans")
    if not isinstance(entries, list):
        raise OrchestratorError("PLAN_REGISTRY_PLANS_INVALID")
    plans: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise OrchestratorError("PLAN_REGISTRY_ENTRY_INVALID")
        stage = str(entry.get("stage", ""))
        plan_path = Path(str(entry.get("path", ""))).resolve()
        if stage in plans or stage not in STAGES:
            raise OrchestratorError(f"PLAN_REGISTRY_STAGE_INVALID:{stage}")
        expected_sha = entry.get("sha256")
        _verify_bound_file(plan_path, expected_sha, f"plan_{stage}")
        plans[stage] = validate_plan(plan_path, source=source, expected_stage=stage)
        plans[stage]["_path"] = str(plan_path)
        plans[stage]["_sha256"] = sha256_file(plan_path)
    return plans, sha256_file(path)


def _registry_candidate(path: Path) -> Path | None:
    if path.is_file():
        return path.resolve()
    if not path.parent.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for item in path.parent.glob("PLAN_REGISTRY_V*.json"):
        match = re.fullmatch(r"PLAN_REGISTRY_V(\d+)\.json", item.name)
        if match:
            candidates.append((int(match.group(1)), item.resolve()))
    return max(candidates, default=(0, None), key=lambda value: value[0])[1]


def verify_registry_chain(latest_path: Path, *, source: Mapping[str, str]) -> tuple[dict[str, dict[str, Any]], int, str, Path]:
    """Verify an append-only registry chain, newest version first."""
    current = latest_path.resolve()
    seen: set[Path] = set()
    expected_version: int | None = None
    newest_plans: dict[str, dict[str, Any]] | None = None
    newest_sha = ""
    newest_version = 0
    while True:
        if current in seen or not current.is_file():
            raise OrchestratorError("PLAN_REGISTRY_CHAIN_INVALID")
        seen.add(current)
        registry = dict(_load_object(current, "PLAN_REGISTRY_CHAIN"))
        try:
            version = int(registry.get("version"))
        except (TypeError, ValueError) as exc:
            raise OrchestratorError("PLAN_REGISTRY_VERSION_MISSING") from exc
        if version < 1 or (expected_version is not None and version != expected_version - 1):
            raise OrchestratorError("PLAN_REGISTRY_VERSION_GAP")
        if registry.get("schema") == REGISTRY_SCHEMA and current.name != f"PLAN_REGISTRY_V{version:04d}.json":
            raise OrchestratorError("PLAN_REGISTRY_FILENAME_VERSION_MISMATCH")
        if registry.get("schema") == REGISTRY_SCHEMA:
            sidecar = current.with_suffix(current.suffix + ".sha256")
            try:
                sidecar_value = sidecar.read_text(encoding="utf-8").split()
            except (OSError, UnicodeDecodeError) as exc:
                raise OrchestratorError("PLAN_REGISTRY_SHA256_SIDECAR_MISSING") from exc
            if not sidecar_value or sidecar_value[0] != sha256_file(current):
                raise OrchestratorError("PLAN_REGISTRY_SHA256_SIDECAR_MISMATCH")
        if expected_version is None:
            newest_version = version
            newest_sha = sha256_file(current)
        expected_version = version
        plans, _ = load_registry(current, source=source)
        if newest_plans is None:
            newest_plans = plans
        entries = registry.get("plans")
        if registry.get("schema") == REGISTRY_SCHEMA and not isinstance(entries, list):
            raise OrchestratorError("PLAN_REGISTRY_PLANS_INVALID")
        previous = registry.get("previous_registry_path")
        previous_sha = registry.get("previous_registry_sha256")
        if not previous:
            if previous_sha is not None or version != 1 or (isinstance(entries, list) and len(entries) != 1):
                raise OrchestratorError("PLAN_REGISTRY_ROOT_INVALID")
        else:
            previous_path = Path(str(previous)).resolve()
            try:
                previous_value = dict(_load_object(previous_path, "PLAN_REGISTRY_PREVIOUS"))
                previous_entries = previous_value.get("plans")
            except (OSError, OrchestratorError, TypeError, ValueError) as exc:
                raise OrchestratorError("PLAN_REGISTRY_PREVIOUS_INVALID") from exc
            if registry.get("schema") == REGISTRY_SCHEMA:
                if not isinstance(previous_entries, list) or not isinstance(entries, list) or entries[:-1] != previous_entries:
                    raise OrchestratorError("PLAN_REGISTRY_APPEND_PREFIX_MISMATCH")
                newest = entries[-1] if entries else None
                if not isinstance(newest, Mapping) or registry.get("newly_added_stage") != newest.get("stage") or registry.get("new_plan_path") != newest.get("path") or registry.get("new_plan_sha256") != newest.get("sha256"):
                    raise OrchestratorError("PLAN_REGISTRY_APPEND_ENTRY_MISMATCH")
            if not isinstance(previous_sha, str):
                raise OrchestratorError("PLAN_REGISTRY_PREVIOUS_SHA_MISSING")
            try:
                previous_digest = sha256_file(previous_path)
            except OSError as exc:
                raise OrchestratorError("PLAN_REGISTRY_PREVIOUS_MISSING") from exc
            if previous_digest != previous_sha:
                raise OrchestratorError("PLAN_REGISTRY_PREVIOUS_SHA_MISMATCH")
            current = previous_path
            continue
        if registry.get("schema") == REGISTRY_SCHEMA:
            newest = entries[0] if isinstance(entries, list) and entries else None
            if not isinstance(newest, Mapping) or registry.get("newly_added_stage") != newest.get("stage") or registry.get("new_plan_path") != newest.get("path") or registry.get("new_plan_sha256") != newest.get("sha256"):
                raise OrchestratorError("PLAN_REGISTRY_ROOT_ENTRY_MISMATCH")
        break
    return newest_plans or {}, newest_version, newest_sha, latest_path.resolve()


def _format_argv(template: list[str], values: Mapping[str, Any]) -> list[str]:
    try:
        rendered = [item.format(**{key: str(value) for key, value in values.items()}) for item in template]
    except (KeyError, ValueError) as exc:
        raise OrchestratorError(f"COMMAND_TEMPLATE_INVALID:{exc}") from exc
    if any(not item for item in rendered):
        raise OrchestratorError("COMMAND_EMPTY_ARG")
    return rendered


def _output_root(plan: Mapping[str, Any], source_commit: str) -> Path:
    stamp = _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    template = str(plan["output_root_template"])
    value = template.replace("{commit8}", source_commit[:8]).replace("{utc}", stamp)
    root = Path(value).resolve()
    if root.exists():
        raise OrchestratorError("OUTPUT_ROOT_ALREADY_EXISTS")
    root.parent.mkdir(parents=True, exist_ok=True)
    return root


def _event(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"schema": "STAGE_V_R2_ORCHESTRATOR_EVENT_V1", "event": event, "utc": utc_now(), **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class FileLock:
    """fcntl lock on Linux, with a stale-PID fallback for local tests."""

    def __init__(self, path: Path, state_root: Path, payload: Mapping[str, Any]):
        self.path = path.resolve()
        self.state_root = state_root.resolve()
        self.payload = dict(payload)
        self.handle = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        old = _read(self.path, {})
        if os.name != "nt":
            import fcntl
            self.handle = self.path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self.handle.close()
                self.handle = None
                raise DuplicateOrchestrator("DUPLICATE_ORCHESTRATOR") from exc
            if isinstance(old, Mapping) and old.get("pid") and not pid_alive(int(old.get("pid"))):
                atomic_write_json(self.state_root / "STALE_LOCK_AUDIT.json", {
                    "schema": "STAGE_V_R2_STALE_LOCK_AUDIT_V1", "lock_path": str(self.path),
                    "previous": dict(old), "audited_utc": utc_now(),
                })
        else:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                self.handle = self.path.open("a+", encoding="utf-8")
            except FileExistsError as exc:
                if isinstance(old, Mapping) and old.get("pid") and not pid_alive(int(old.get("pid"))):
                    atomic_write_json(self.state_root / "STALE_LOCK_AUDIT.json", {"schema": "STAGE_V_R2_STALE_LOCK_AUDIT_V1", "previous": dict(old), "audited_utc": utc_now()})
                    self.path.unlink(missing_ok=True)
                    return self.acquire()
                raise DuplicateOrchestrator("DUPLICATE_ORCHESTRATOR") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps(self.payload, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        if self.handle is None:
            return
        if os.name != "nt":
            import fcntl
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


def process_identity_matches(receipt: Mapping[str, Any], *, expected_cwd: Path, expected_root: Path) -> bool:
    pid = int(receipt.get("pid", 0) or 0)
    identity = _proc_identity(pid)
    if not identity:
        return False
    if receipt.get("start_ticks") is not None and identity.get("start_ticks") != receipt.get("start_ticks"):
        return False
    if receipt.get("cwd") and not _same_path(identity.get("cwd"), Path(str(receipt["cwd"]))):
        return False
    if not _same_path(identity.get("cwd"), expected_cwd):
        return False
    expected_cmdline = [str(item) for item in receipt.get("cmdline", [])]
    if expected_cmdline and identity.get("cmdline") != expected_cmdline:
        return False
    return str(expected_root) in " ".join(identity.get("cmdline") or []) or receipt.get("output_root") == str(expected_root)


def _completed(root: Path, plan: Mapping[str, Any]) -> bool:
    receipts = plan.get("completion_receipts", [])
    if not isinstance(receipts, list) or not receipts:
        return False
    for item in receipts:
        path = root / str(item) if not Path(str(item)).is_absolute() else Path(str(item))
        value = _read(path)
        if not isinstance(value, Mapping) or value.get("verdict", value.get("status")) not in {
            "PASS", "DONE", "GO", "NO_GO", "R2B_REQUIRED", "R2B_NOT_REQUIRED",
            "STAGE_V_FORMAL_MAP_CLOSED", "STAGE_V2_PASS", "STAGE_O_PASS",
        }:
            return False
    return True


class Orchestrator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = Path(args.repo_root).resolve()
        self.state_root = Path(args.state_root).resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_root / "STAGE_V_R2_ORCHESTRATOR_STATE.json"
        self.heartbeat_path = self.state_root / "STAGE_V_R2_ORCHESTRATOR_HEARTBEAT.json"
        self.events_path = self.state_root / "STAGE_V_R2_ORCHESTRATOR_EVENTS.jsonl"
        self.start_utc = utc_now()
        self.heartbeat_count = 0
        self.plan_sha = None
        self.registry_version = None
        self.registry_path = None
        self.registry_chain_verified = False
        self.previous_plan_sha = None
        self.lock = FileLock(Path(args.lock_path), self.state_root, {
            "schema": "STAGE_V_R2_ORCHESTRATOR_LOCK_V1", "pid": os.getpid(),
            "pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(), "started_utc": self.start_utc,
        })
        self.source = source_binding(self.repo_root)

    def _qualification(self) -> tuple[bool, str]:
        root = Path(self.args.qualification_root).resolve()
        q2_report_path = root / "Q2_CONTROL_QUALIFICATION_REPORT.json"
        if q2_report_path.is_file():
            audit_path = root / "Q2_CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json"
            manifest_path = root / "Q2_PARENT_MANIFEST_A.json"
            if not audit_path.is_file() or not manifest_path.is_file():
                return False, "Q2_QUALIFICATION_INCOMPLETE"
            report = _read(q2_report_path, {})
            audit = _read(audit_path, {})
            manifest = _read(manifest_path, {})
            if not isinstance(report, Mapping) or report.get("status") != "PASS":
                return False, "Q2_QUALIFICATION_REPORT_FAIL"
            if not isinstance(audit, Mapping) or audit.get("verdict") != "PASS":
                return False, "Q2_QUALIFICATION_AUDIT_FAIL"
            if not isinstance(manifest, Mapping) or manifest.get("status") != "PASS" or int(manifest.get("selected_count", -1)) != 40:
                return False, "Q2_QUALIFICATION_MANIFEST_FAIL"
            try:
                qualified = {suite: int((report.get("qualified_by_suite") or {}).get(suite, -1)) for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial")}
                boundary_counts = [int(report.get(field, -1)) for field in ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts", "attack_rollouts")]
            except (TypeError, ValueError):
                return False, "Q2_QUALIFICATION_RECEIPT_INVALID"
            if qualified != {suite: 10 for suite in qualified} or any(boundary_counts):
                return False, "Q2_QUALIFICATION_CLOSURE_COUNT_FAIL"
            return True, "PASS"
        failure_names = ("CONTROL_QUALIFICATION_FAILURE.json", "ABORTED_INCOMPLETE.json", "QUALIFICATION_FAILURE.json")
        if any((root / name).is_file() for name in failure_names):
            return False, "QUALIFICATION_FAILED"
        required = ("CONTROL_QUALIFICATION_REPORT.json", "CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json")
        if not all((root / name).is_file() for name in required):
            return False, "QUALIFICATION_INCOMPLETE"
        report = _read(root / required[0], {})
        audit = _read(root / required[1], {})
        if not isinstance(report, Mapping) or report.get("status") != "PASS":
            return False, "QUALIFICATION_REPORT_FAIL"
        if not isinstance(audit, Mapping) or audit.get("verdict") != "PASS":
            return False, "QUALIFICATION_AUDIT_FAIL"
        try:
            evaluated_rows = int(report.get("evaluated_rows", 0))
            boundary_counts = [int(report.get(field, -1)) for field in FORBIDDEN_BOUNDARY_FIELDS]
        except (TypeError, ValueError):
            return False, "QUALIFICATION_RECEIPT_INVALID"
        if evaluated_rows >= 160 and not any(boundary_counts):
            return True, "PASS"
        dispatcher = _read(root / "DISPATCHER_START.json", {})
        if isinstance(dispatcher, Mapping) and dispatcher.get("dispatcher_pid") and not pid_alive(int(dispatcher["dispatcher_pid"])) and not (root / "DISPATCHER_COMPLETE.json").is_file():
            return False, "QUALIFICATION_DISPATCHER_DIED"
        return False, "QUALIFICATION_CLOSURE_COUNT_FAIL"

    def _observer_healthy(self) -> bool:
        observer_pid = int(getattr(self.args, "observer_pid", 0) or 0)
        return not observer_pid or pid_alive(observer_pid)

    def _load_plans(self) -> tuple[dict[str, dict[str, Any]], str | None, str | None]:
        path = _registry_candidate(Path(self.args.plan_registry))
        if path is None:
            return {}, None, "PLAN_REGISTRY_MISSING"
        try:
            plans, version, registry_sha, registry_path = verify_registry_chain(path, source=self.source)
            previous = _read(self.state_path, {})
            previous_sha = previous.get("last_accepted_registry_sha256") if isinstance(previous, Mapping) else None
            previous_version = previous.get("last_accepted_registry_version") if isinstance(previous, Mapping) else None
            previous_path = previous.get("last_accepted_registry_path") if isinstance(previous, Mapping) else None
            registry = _load_object(registry_path, "PLAN_REGISTRY")
            if previous_sha is not None:
                try:
                    same_version = version == int(previous_version) and registry_sha == previous_sha and str(registry_path) == str(Path(str(previous_path)).resolve())
                    next_version = version == int(previous_version) + 1 and registry.get("previous_registry_sha256") == previous_sha and Path(str(registry.get("previous_registry_path"))).resolve() == Path(str(previous_path)).resolve()
                except (TypeError, ValueError):
                    same_version = next_version = False
                if not same_version and not next_version:
                    return {}, registry_sha, "PLAN_REGISTRY_CHAIN_APPEND_INVALID"
            self.previous_plan_sha = previous_sha
            self.registry_version = version
            self.registry_path = str(registry_path)
            self.registry_chain_verified = True
            return plans, registry_sha, None
        except OrchestratorError as exc:
            return {}, None, str(exc)

    def _write(self, status: str, *, phase: str, reason: str = "", resource: Mapping[str, Any] | None = None, plans: Mapping[str, Any] | None = None) -> None:
        self.heartbeat_count += 1
        payload = {
            "schema": SCHEMA, "status": status, "phase": phase, "reason": reason,
            "control_plane_mode": "LOCAL_AUTONOMOUS", "ssh_is_hard_stop": False,
            "orchestrator_pid": os.getpid(), "orchestrator_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
            "repo_root": str(self.repo_root), "source_commit": self.source["commit"], "source_tree": self.source["tree"],
            "source_status_porcelain": self.source["status_porcelain"], "qualification_root": str(Path(self.args.qualification_root).resolve()),
            "qualification_dispatcher_pid": int(getattr(self.args, "qualification_dispatcher_pid", 0) or 0),
            "qualification_dispatcher_alive": pid_alive(int(getattr(self.args, "qualification_dispatcher_pid", 0) or 0)),
            "observer_pid": int(getattr(self.args, "observer_pid", 0) or 0),
            "observer_alive": self._observer_healthy(),
            "approved_gpus": list((resource or {}).get("safe_gpus", [])), "gpu_preflight_status": (resource or {}).get("status"),
            "planned_stages": list(plans or {}), "plan_registry_sha256": self.plan_sha,
            "last_accepted_registry_version": self.registry_version,
            "last_accepted_registry_sha256": self.plan_sha,
            "last_accepted_registry_path": self.registry_path,
            "registry_chain_verified": self.registry_chain_verified,
            "heartbeat_count": self.heartbeat_count, "ssh_probe_success_count": 0, "ssh_probe_failure_count": 0,
            "longest_ssh_unavailable_interval_seconds": 0, "external_root_process_present": pid_alive(self.args.external_pid),
            "external_root_process_pid": self.args.external_pid, "external_root_process_terminated": False,
            "gpu5_touched": False, "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0,
            "resource": dict(resource or {}), "updated_utc": utc_now(),
        }
        atomic_write_json(self.state_path, payload)
        atomic_write_json(self.heartbeat_path, payload)

    def _materialize(self, stage: str, plan: Mapping[str, Any], resource: Mapping[str, Any]) -> Path:
        dispatchers = active_project_dispatcher_pids(self.repo_root)
        if dispatchers:
            raise OrchestratorError(f"DUPLICATE_DISPATCHER_PRESENT:{','.join(map(str, dispatchers))}")
        output_root = _output_root(plan, self.source["commit"])
        policy = _resource_policy(plan)
        safe_gpus = list(resource.get("safe_gpus", []))
        values = {
            "source_commit": self.source["commit"], "source_tree": self.source["tree"],
            "output_root": output_root, "parent_manifest": plan["parent_manifest"]["path"],
            "parent_manifest_sha256": plan["parent_manifest"]["sha256"],
            "approved_gpus": ",".join(str(x) for x in safe_gpus),
            "approved_gpu_uuids": ",".join(str(x) for x in resource.get("safe_gpu_uuids", [])),
            "stage_root": output_root, "stage": stage,
        }
        command = _format_argv(plan["command_template"], values)
        audit_command = _format_argv(plan["audit_command_template"], values)
        command_path = self.state_root / f"{stage}_COMMAND.json"
        if command_path.exists():
            raise OrchestratorError("COMMAND_ALREADY_MATERIALIZED")
        command_payload = {
            "schema": "STAGE_V_R2_COMMAND_V1", "stage": stage, "plan_sha256": plan["_sha256"],
            "source_commit": self.source["commit"], "source_tree": self.source["tree"],
            "command": command, "audit_command": audit_command, "output_root": str(output_root),
            "input_receipts": plan["input_receipts"], "parent_manifest": plan["parent_manifest"],
            "registry_version": self.registry_version, "registry_sha256": self.plan_sha,
            "gpu_preflight": dict(resource), "created_utc": utc_now(),
        }
        atomic_write_json(command_path, command_payload)
        command_sha = _write_sha(command_path)
        launch_path = self.state_root / f"{stage}_LAUNCH.json"
        if launch_path.exists():
            raise OrchestratorError("LAUNCH_RECEIPT_ALREADY_EXISTS")
        launch_payload = {
            "schema": "STAGE_V_R2_LAUNCH_V1", "stage": stage, "status": "LAUNCHING",
            "command_sha256": command_sha, "plan_sha256": plan["_sha256"],
            "input_receipts": plan["input_receipts"], "parent_manifest": plan["parent_manifest"],
            "output_root": str(output_root), "gpu_assignments": safe_gpus,
            "gpu_assignment_uuids": list(resource.get("safe_gpu_uuids", [])),
            "resource_kind": policy.get("resource_kind"), "registry_version": self.registry_version,
            "registry_sha256": self.plan_sha,
            "cwd": str(Path(str(plan["cwd"])).resolve()), "started_utc": utc_now(),
        }
        atomic_write_json(launch_path, launch_payload)
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in dict(plan.get("env", {})).items()})
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in safe_gpus) if policy.get("resource_kind") == "GPU" else ""
        stdout_path = self.state_root / f"{stage}_STDOUT.log"
        stderr_path = self.state_root / f"{stage}_STDERR.log"
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            process = subprocess.Popen(command, cwd=str(plan["cwd"]), env=env, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
        identity = _proc_identity(process.pid)
        if not identity:
            raise OrchestratorError("LAUNCH_PROCESS_IDENTITY_UNAVAILABLE")
        launch_payload.update({
            "status": "RUNNING", "pid": process.pid,
            "pgid": os.getpgid(process.pid) if hasattr(os, "getpgid") else process.pid,
            "start_ticks": identity.get("start_ticks"), "cmdline": identity.get("cmdline", command),
            "cwd": identity.get("cwd", str(plan["cwd"])), "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path), "updated_utc": utc_now(),
        })
        atomic_write_json(launch_path, launch_payload)
        _event(self.events_path, "STAGE_LAUNCHED", stage=stage, pid=process.pid, output_root=str(output_root), command_sha256=command_sha)
        return output_root

    def _run_audit(self, stage: str, plan: Mapping[str, Any], root: Path) -> None:
        values = {
            "source_commit": self.source["commit"], "source_tree": self.source["tree"],
            "output_root": root, "stage_root": root, "parent_manifest": plan["parent_manifest"]["path"],
            "parent_manifest_sha256": plan["parent_manifest"]["sha256"], "stage": stage,
        }
        command = _format_argv(plan["audit_command_template"], values)
        stdout_path = self.state_root / f"{stage}_AUDIT_STDOUT.log"
        stderr_path = self.state_root / f"{stage}_AUDIT_STDERR.log"
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            completed = subprocess.run(command, cwd=str(plan["cwd"]), env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                        stdout=stdout, stderr=stderr, check=False)
        audit = {
            "schema": "STAGE_V_R2_ORCHESTRATOR_AUDIT_V1", "stage": stage,
            "status": "PASS" if completed.returncode == 0 and _completed(root, plan) else "FAIL",
            "exit_code": completed.returncode, "root": str(root), "command": command,
            "runner_sha256": plan["runner_sha256"], "auditor_sha256": plan["auditor_sha256"],
            "audited_utc": utc_now(), "eval160_reads": 0, "protected_eval_reads": 0,
            "vis_pgd_attack_rollouts": 0,
        }
        atomic_write_json(self.state_root / f"{stage}_AUDIT.json", audit)
        if audit["status"] != "PASS":
            raise OrchestratorError(f"{stage}_AUDIT_FAIL")
        launch_path = self.state_root / f"{stage}_LAUNCH.json"
        launch = dict(_load_object(launch_path, "LAUNCH"))
        launch.update({"status": "AUDITED", "audit_sha256": sha256_file(self.state_root / f"{stage}_AUDIT.json"), "updated_utc": utc_now()})
        atomic_write_json(launch_path, launch)
        _event(self.events_path, "STAGE_AUDITED", stage=stage, output_root=str(root), audit_sha256=launch["audit_sha256"])

    def _reattach_or_audit(self, stage: str, plan: Mapping[str, Any]) -> str | None:
        launch = _read(self.state_root / f"{stage}_LAUNCH.json", {})
        if not isinstance(launch, Mapping):
            return None
        root = Path(str(launch.get("output_root", "")))
        if launch.get("status") == "RUNNING":
            if pid_alive(int(launch.get("pid", 0) or 0)):
                if not process_identity_matches(launch, expected_cwd=Path(str(plan["cwd"])), expected_root=root):
                    raise OrchestratorError("LAUNCH_RECEIPT_PROCESS_IDENTITY_MISMATCH")
                return "RUNNING"
            if _completed(root, plan):
                launch = dict(launch)
                launch.update({"status": "COMPLETE", "updated_utc": utc_now()})
                atomic_write_json(self.state_root / f"{stage}_LAUNCH.json", launch)
                return "COMPLETE"
            raise OrchestratorError("LAUNCH_PROCESS_DIED_BEFORE_COMPLETION")
        if launch.get("status") in {"ABORTED", "FAILED"} or (root / "ABORTED_INCOMPLETE.json").is_file():
            raise OrchestratorError("ABORTED_ROOT_NEVER_RESUMED")
        if launch.get("status") == "COMPLETE":
            return "COMPLETE"
        if launch.get("status") == "AUDITED":
            return "AUDITED"
        return "COMPLETE" if _completed(root, plan) else None

    def _stage_status(self, stage: str, plan: Mapping[str, Any]) -> str | None:
        return self._reattach_or_audit(stage, plan)

    def tick(self) -> str:
        current_source = source_binding(self.repo_root)
        if current_source != self.source:
            self._write(HARD_STOP, phase=HARD_STOP, reason="SOURCE_BINDING_DRIFT")
            return HARD_STOP
        if self.source["status_porcelain"]:
            self._write(HARD_STOP, phase=HARD_STOP, reason="SOURCE_WORKTREE_DIRTY")
            return HARD_STOP
        if not self._observer_healthy():
            self._write(HARD_STOP, phase=HARD_STOP, reason="OBSERVER_PID_NOT_ALIVE")
            return HARD_STOP
        qualification_ok, qualification_reason = self._qualification()
        plans, registry_sha, plan_error = self._load_plans()
        self.plan_sha = registry_sha
        if plan_error and plan_error != "PLAN_REGISTRY_MISSING":
            self._write(HARD_STOP, phase=HARD_STOP, reason=plan_error, plans=plans)
            return HARD_STOP
        if qualification_reason in {"QUALIFICATION_FAILED", "QUALIFICATION_DISPATCHER_DIED"}:
            self._write(HARD_STOP, phase=HARD_STOP, reason=qualification_reason, plans=plans)
            return HARD_STOP
        if not qualification_ok:
            self._write(WAIT_QUALIFICATION, phase=WAIT_QUALIFICATION, reason=qualification_reason, plans=plans)
            return WAIT_QUALIFICATION
        if "C0" not in plans:
            self._write(WAIT_QUALIFICATION, phase=WAIT_QUALIFICATION, reason="C0_PLAN_NOT_REGISTERED", plans=plans)
            return WAIT_QUALIFICATION
        stage = None
        try:
            for candidate in STAGES:
                if candidate not in plans:
                    continue
                plan = plans[candidate]
                status = self._stage_status(candidate, plan)
                if status == "AUDITED":
                    continue
                if status == "COMPLETE":
                    launch = _load_object(self.state_root / f"{candidate}_LAUNCH.json", "LAUNCH")
                    self._run_audit(candidate, plan, Path(str(launch["output_root"])))
                    continue
                stage = candidate
                break
        except OrchestratorError as exc:
            self._write(HARD_STOP, phase=HARD_STOP, reason=str(exc), plans=plans)
            _event(self.events_path, "HARD_STOP", reason=str(exc))
            return HARD_STOP
        if stage is None:
            self._write(WAIT_QUALIFICATION, phase=WAIT_QUALIFICATION, reason="NO_REGISTERED_PLAN", plans=plans)
            return WAIT_QUALIFICATION
        plan = plans[stage]
        policy = _resource_policy(plan)
        if policy["resource_kind"] == "CPU_ONLY":
            resource = {
                "schema": "STAGE_V_CPU_ONLY_PREFLIGHT_V1", "status": "CPU_ONLY", "stage": stage,
                "required_gpu_count": 0, "safe_gpus": [], "safe_gpu_uuids": [], "updated_utc": utc_now(),
            }
        else:
            required = int(policy.get("minimum_gpu_count", policy.get("required_gpu_count", 0)))
            resource = gpu_preflight(
                stage=stage, required_gpus=required, excluded_gpus=policy.get("excluded_gpus", []),
                protected_pids=policy.get("protected_pids", []), canary_peak_mib=float(policy.get("canary_peak_mib", 0)),
                project_root=self.repo_root,
            )
            if not bool(policy.get("strict_gpu_count", False)) and resource.get("status") == "PASS":
                maximum = int(policy.get("maximum_gpu_count", required))
                resource["safe_gpus"] = list(resource.get("all_safe_gpus", resource.get("safe_gpus", [])))[:maximum]
                resource["safe_gpu_uuids"] = [row["uuid"] for row in resource.get("gpu_rows", []) if row.get("index") in resource["safe_gpus"]]
        if policy["resource_kind"] == "GPU":
            atomic_write_json(self.state_root / f"GPU_PREFLIGHT_{stage}.json", resource)
            _write_sha(self.state_root / f"GPU_PREFLIGHT_{stage}.json")
            minimum = int(policy.get("minimum_gpu_count", 0))
            strict_fail = bool(policy.get("strict_gpu_count")) and len(resource.get("safe_gpus", [])) != int(policy.get("required_gpu_count", 0))
            if resource.get("status") != "PASS" or len(resource.get("safe_gpus", [])) < minimum or strict_fail:
                self._write(WAIT_GPUS, phase=WAIT_GPUS, reason="SAFE_GPU_COUNT_BELOW_POLICY", resource=resource, plans=plans)
                return WAIT_GPUS
        try:
            state = self._reattach_or_audit(stage, plan)
            if state == "RUNNING":
                self._write(f"RUN_{stage}", phase=f"RUN_{stage}", resource=resource, plans=plans)
                return f"RUN_{stage}"
            if state == "COMPLETE":
                self._write(f"AUDIT_{stage}", phase=f"AUDIT_{stage}", resource=resource, plans=plans)
                return f"AUDIT_{stage}"
            self._materialize(stage, plan, resource)
            self._write(f"RUN_{stage}", phase=f"RUN_{stage}", resource=resource, plans=plans)
            return f"RUN_{stage}"
        except OrchestratorError as exc:
            self._write(HARD_STOP, phase=HARD_STOP, reason=str(exc), resource=resource, plans=plans)
            _event(self.events_path, "HARD_STOP", reason=str(exc), stage=stage)
            return HARD_STOP

    def run(self) -> int:
        self.lock.acquire()
        try:
            while True:
                status = self.tick()
                if self.args.once or status == HARD_STOP:
                    return 1 if status == HARD_STOP else 0
                time.sleep(max(1.0, float(self.args.poll_seconds)))
        finally:
            self.lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--plan-registry", type=Path, required=True)
    parser.add_argument("--external-pid", type=int, default=1895889)
    parser.add_argument("--observer-pid", type=int, default=0)
    parser.add_argument("--qualification-dispatcher-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=300)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.repo_root.is_dir() or not args.qualification_root.is_dir():
        raise SystemExit("repo-root and qualification-root must exist")
    return Orchestrator(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
