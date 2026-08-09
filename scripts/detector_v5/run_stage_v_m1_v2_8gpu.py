#!/usr/bin/env python3
"""Fail-closed Stage V M1-V2 supervisor (legacy fixed and V2.2 dynamic cohorts)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_PREFIX = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800"
IDENTITY = "libero_10/task_08/state_47"
GPU_IDS = tuple(range(8))
MIN_PRIMARY_CLEAN_GPUS = 4
DYNAMIC_PROTOCOL_SCHEMA = "STAGE_V_M1_VISUAL_DETERMINISM_PROTOCOL_V2_2_DYNAMIC_COHORT_8GPU"
PHASES = {
    "Q1": ("CLEAN_QUALIFICATION", "rep_01"),
    "C1": ("COUNTERFACTUAL_CLEAN_PREFIX", "rep_01"),
    "Q2": ("CLEAN_QUALIFICATION", "rep_02"),
    "C2": ("COUNTERFACTUAL_CLEAN_PREFIX", "rep_02"),
}
LABELS = tuple(PHASES)
BOUNDARIES = (
    "eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts",
    "attack_rollouts", "intervention_applied_steps", "counterfactual_open_steps",
)
AUTHORIZATION_FLAGS = (
    "new_science_rollouts_authorized", "formal_parent_promotion_authorized",
    "vulnerability_label_generation_authorized", "student_training_authorized",
    "protected_evaluation_authorized", "eval160_authorized", "vis_pgd_authorized",
)


class V2Error(RuntimeError):
    pass


def is_dynamic_protocol(protocol: Mapping[str, Any]) -> bool:
    return protocol.get("schema") == DYNAMIC_PROTOCOL_SCHEMA


def dynamic_counts(gpu_ids: tuple[int, ...] | list[int]) -> dict[str, int]:
    n = len(tuple(gpu_ids))
    return {
        "primary_clean_gpu_count": n,
        "r1_run_count": 4 * n,
        "gpu_local_pair_count": 4 * n,
        "cross_gpu_pair_count": 4 * n * (n - 1) // 2,
    }


def canonical_gpu_uuid(value: Any) -> str:
    token = str(value or "").strip().lower()
    return token[4:] if token.startswith("gpu-") else token


def validate_uuid_binding(*, gpu: int, preflight_uuid: Any, canary_uuid: Any,
                          runtime_uuid: Any | None = None, phase: str = "") -> str:
    values = [canonical_gpu_uuid(preflight_uuid), canonical_gpu_uuid(canary_uuid)]
    if runtime_uuid is not None:
        values.append(canonical_gpu_uuid(runtime_uuid))
    if not values[0] or not all(value == values[0] for value in values):
        suffix = f":{phase}" if phase else ""
        raise V2Error(f"HOLD_GPU_UUID_BINDING_MISMATCH:gpu_{gpu:02d}{suffix}")
    return values[0]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise V2Error(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()


def _git_blob_sha(relative_path: str) -> str:
    result = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"HEAD:{relative_path}"], check=True, capture_output=True)
    return hashlib.sha256(result.stdout).hexdigest()


def _number(value: str) -> int | float | None:
    token = value.strip().split()[0] if value.strip() else ""
    try:
        number = float(token)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def _query_nvidia_smi(*args: str) -> str:
    completed = subprocess.run(["nvidia-smi", *args], check=False, capture_output=True, text=True, timeout=20)
    if completed.returncode != 0:
        raise V2Error(f"NVIDIA_SMI_FAILED:{completed.returncode}:{completed.stderr[-300:]}")
    return completed.stdout


def _runtime_gpu_uuid(gpu: int, properties: Any) -> tuple[str, str]:
    torch_uuid = str(getattr(properties, "uuid", "") or "").strip()
    if torch_uuid:
        return torch_uuid, "torch.cuda.get_device_properties.uuid"
    try:
        output = _query_nvidia_smi(
            f"--id={int(gpu)}", "--query-gpu=uuid", "--format=csv,noheader,nounits",
        )
    except (OSError, subprocess.SubprocessError, V2Error):
        return "", "unavailable"
    values = [line.strip() for line in output.splitlines() if line.strip()]
    return (values[0], "nvidia-smi.query-gpu.uuid") if len(values) == 1 else ("", "unavailable")


def _pid_detail(pid: int) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["ps", "-o", "user=,pid=,ppid=,etime=,args=", "-p", str(pid)],
            check=False, capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return {"pid": pid, "owner": None, "command": None}
    line = next((item.strip() for item in result.stdout.splitlines() if item.strip()), "")
    fields = line.split(None, 4)
    return {
        "pid": pid,
        "owner": fields[0] if fields else None,
        "ppid": int(fields[2]) if len(fields) > 2 and fields[2].isdigit() else None,
        "elapsed": fields[3] if len(fields) > 3 else None,
        "command": fields[4] if len(fields) > 4 else None,
    }


def _compute_processes(text: str, uuid_to_gpu: Mapping[str, int]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fields in csv.reader(text.splitlines()):
        if len(fields) < 4 or not fields[1].strip().isdigit():
            continue
        pid = int(fields[1].strip())
        gpu_uuid = fields[0].strip()
        result.append({
            **_pid_detail(pid),
            "gpu_id": uuid_to_gpu.get(canonical_gpu_uuid(gpu_uuid)),
            "gpu_uuid": gpu_uuid,
            "gpu_uuid_canonical": canonical_gpu_uuid(gpu_uuid),
            "process_name": fields[2].strip(),
            "used_memory_mib": _number(fields[3]),
            "kind": "COMPUTE",
        })
    return result


def _pmon_processes(text: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 3 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        result.append({
            **_pid_detail(int(fields[1])),
            "gpu_id": int(fields[0]),
            "kind": fields[2].upper(),
            "process_name": fields[-1],
        })
    return result


def validate_binding_receipt(receipt: Mapping[str, Any], gpu: int, *, require_canonical_uuid: bool = False) -> None:
    expected = {
        "logical_worker_id": f"worker_{gpu}",
        "requested_physical_gpu": gpu,
        "cuda_visible_devices": str(gpu),
        "torch_current_device": 0,
        "mujoco_gl": "egl",
        # CUDA_VISIBLE_DEVICES remaps the selected physical GPU to logical EGL 0.
        "egl_device_identifier": 0,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise V2Error(f"GPU_BINDING_RECEIPT_INVALID:{key}:gpu_{gpu:02d}")
    gpu_uuid = str(receipt.get("gpu_uuid", "")).strip()
    if not gpu_uuid:
        raise V2Error(f"GPU_BINDING_RECEIPT_GPU_UUID_MISSING:gpu_{gpu:02d}")
    if require_canonical_uuid and receipt.get("gpu_uuid_canonical") != canonical_gpu_uuid(gpu_uuid):
        raise V2Error(f"GPU_BINDING_RECEIPT_GPU_UUID_CANONICAL_INVALID:gpu_{gpu:02d}")
    if not isinstance(receipt.get("renderer_device_information"), Mapping):
        raise V2Error(f"GPU_BINDING_RECEIPT_RENDERER_INFO_MISSING:gpu_{gpu:02d}")


def validate_runtime_binding_receipt(receipt: Mapping[str, Any], gpu: int, *, run_set: str, phase: str, source_commit: str, source_tree: str) -> None:
    expected = {
        "schema": receipt.get("schema"),
        "status": "PASS",
        "logical_worker_id": f"worker_{gpu}",
        "requested_physical_gpu": gpu,
        "physical_gpu_index": gpu,
        "cuda_visible_devices": str(gpu),
        "torch_current_device": 0,
        "mujoco_gl": "egl",
        # The installed robosuite binding validates this selector against the
        # physical id in CUDA_VISIBLE_DEVICES; renderer/context ids remain logical 0.
        "mujoco_egl_device_id": str(gpu),
        "env_render_gpu_device_id": 0,
        "run_set": run_set,
        "run_label": phase,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "episode_started": False,
        "receipt_written_before_step_0": True,
    }
    if receipt.get("schema") != "STAGE_V_M1_V2_1_1_RUNTIME_BINDING_RECEIPT_V1":
        raise V2Error(f"RUNTIME_BINDING_RECEIPT_INVALID:schema:gpu_{gpu:02d}:{phase}")
    for key, value in expected.items():
        if key == "schema":
            continue
        if receipt.get(key) != value:
            raise V2Error(f"RUNTIME_BINDING_RECEIPT_INVALID:{key}:gpu_{gpu:02d}:{phase}")
    if not isinstance(receipt.get("pid"), int) or int(receipt["pid"]) <= 0:
        raise V2Error(f"RUNTIME_BINDING_RECEIPT_PID_INVALID:gpu_{gpu:02d}:{phase}")
    torch_uuid = str(receipt.get("torch_device_uuid", "")).strip()
    if not torch_uuid:
        raise V2Error(f"RUNTIME_BINDING_RECEIPT_GPU_UUID_MISSING:gpu_{gpu:02d}:{phase}")
    if receipt.get("torch_device_uuid_canonical") != canonical_gpu_uuid(torch_uuid):
        raise V2Error(f"RUNTIME_BINDING_RECEIPT_GPU_UUID_CANONICAL_INVALID:gpu_{gpu:02d}:{phase}")
    if receipt.get("render_context_observed_device_id") != 0:
        raise V2Error(f"RUNTIME_BINDING_RECEIPT_RENDER_CONTEXT_MISMATCH:gpu_{gpu:02d}:{phase}")
    if not isinstance(receipt.get("renderer_device_information"), Mapping):
        raise V2Error(f"RUNTIME_BINDING_RECEIPT_RENDERER_INFO_MISSING:gpu_{gpu:02d}:{phase}")


def _command_executable(command: Any) -> str | None:
    if not isinstance(command, str) or not command.strip():
        return None
    return command.strip().split()[0]


def _matches_graphics_baseline(process: Mapping[str, Any], contract: Mapping[str, Any]) -> bool:
    return bool(
        contract.get("enabled")
        and process.get("kind") == contract.get("kind")
        and process.get("process_name") == contract.get("process_name")
        and process.get("owner") == contract.get("owner")
        and _command_executable(process.get("command")) == contract.get("executable")
    )


def gpu_preflight(*, gpu_ids: tuple[int, ...] = GPU_IDS, idle_memory_max_mib: int = 1024,
                  system_graphics_baseline: Mapping[str, Any] | None = None,
                  dynamic_cohort: bool = False,
                  minimum_primary_clean_gpus: int = MIN_PRIMARY_CLEAN_GPUS) -> dict[str, Any]:
    """Take one conservative all-GPU snapshot; any telemetry gap is unsafe."""
    gpu_text = _query_nvidia_smi(
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    gpu_rows: list[dict[str, Any]] = []
    for fields in csv.reader(gpu_text.splitlines()):
        if len(fields) < 7 or not fields[0].strip().isdigit():
            continue
        gpu_uuid = fields[1].strip()
        gpu_rows.append({
            "index": int(fields[0]), "uuid": gpu_uuid, "uuid_canonical": canonical_gpu_uuid(gpu_uuid), "name": fields[2].strip(),
            "memory_total_mib": _number(fields[3]), "memory_used_mib": _number(fields[4]),
            "memory_free_mib": _number(fields[5]), "utilization_gpu_percent": _number(fields[6]),
        })
    if {row["index"] for row in gpu_rows} != set(gpu_ids):
        raise V2Error("GPU_INVENTORY_MISMATCH")
    uuid_to_gpu = {str(row["uuid_canonical"]): int(row["index"]) for row in gpu_rows}
    compute = _compute_processes(_query_nvidia_smi(
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ), uuid_to_gpu)
    pmon = _pmon_processes(_query_nvidia_smi("pmon", "-c", "1"))
    by_gpu: dict[int, list[dict[str, Any]]] = {gpu: [] for gpu in gpu_ids}
    seen: set[tuple[int | None, int]] = set()
    unmapped: list[dict[str, Any]] = []
    for process in compute + pmon:
        key = (process.get("gpu_id"), int(process["pid"]))
        if key in seen:
            continue
        seen.add(key)
        if process.get("gpu_id") in by_gpu:
            by_gpu[int(process["gpu_id"])].append(process)
        else:
            unmapped.append(process)
    graphics_candidates = [process for process in pmon if _matches_graphics_baseline(process, system_graphics_baseline or {})]
    candidate_pids = {int(process["pid"]) for process in graphics_candidates}
    compute_pids = {int(process["pid"]) for process in compute}
    candidate_gpus = {int(process["gpu_id"]) for process in graphics_candidates if process.get("gpu_id") in by_gpu}
    graphics_contract = system_graphics_baseline or {}
    consistent_graphics = bool(
        graphics_candidates
        and (not graphics_contract.get("require_all_gpu_coverage") or candidate_gpus == set(gpu_ids))
        and (not graphics_contract.get("require_single_consistent_pid") or len(candidate_pids) == 1)
        and not (graphics_contract.get("compute_processes_never_whitelisted") and candidate_pids & compute_pids)
    )
    baseline_ids = {id(process) for process in graphics_candidates} if consistent_graphics else set()
    baseline_system_graphics = [
        {**process, "classification": "SYSTEM_GRAPHICS_BASELINE"}
        for process in graphics_candidates
        if id(process) in baseline_ids
    ]
    foreign_user_workloads: list[dict[str, Any]] = []
    for process in compute + pmon:
        if id(process) in baseline_ids:
            continue
        if process.get("gpu_id") in by_gpu:
            foreign_user_workloads.append({**process, "classification": "FOREIGN_USER_WORKLOAD"})
    foreign_by_gpu: dict[int, list[dict[str, Any]]] = {gpu: [] for gpu in gpu_ids}
    baseline_by_gpu: dict[int, list[dict[str, Any]]] = {gpu: [] for gpu in gpu_ids}
    for process in foreign_user_workloads:
        foreign_by_gpu[int(process["gpu_id"])].append(process)
    for process in baseline_system_graphics:
        baseline_by_gpu[int(process["gpu_id"])].append(process)
    memory_limit = idle_memory_max_mib
    if system_graphics_baseline is not None and system_graphics_baseline.get("max_memory_used_mib") is not None:
        memory_limit = min(memory_limit, int(system_graphics_baseline["max_memory_used_mib"]))
    decisions: list[dict[str, Any]] = []
    for row in sorted(gpu_rows, key=lambda item: int(item["index"])):
        gpu = int(row["index"])
        reasons: list[str] = []
        used = row.get("memory_used_mib")
        if not isinstance(used, (int, float)) or used > memory_limit:
            reasons.append("MEMORY_NOT_IDLE")
        if foreign_by_gpu[gpu]:
            reasons.append("FOREIGN_PROCESS_PRESENT")
        if unmapped:
            reasons.append("UNMAPPED_PROCESS_TELEMETRY")
        decisions.append({
            **row,
            "compute_pids": [p["pid"] for p in foreign_by_gpu[gpu] if p["kind"] == "COMPUTE"],
            "system_graphics_processes": baseline_by_gpu[gpu],
            "foreign_processes": foreign_by_gpu[gpu],
            "safe": not reasons,
            "primary_clean": not reasons,
            "primary_clean_exclusion_reason": reasons[0] if reasons else None,
            "reasons": reasons,
        })
    all_safe = not unmapped and all(bool(row["safe"]) for row in decisions)
    primary_clean_gpu_set = [int(row["index"]) for row in decisions if row["primary_clean"]]
    if dynamic_cohort:
        if unmapped:
            status = "HOLD_UNMAPPED_PROCESS_TELEMETRY"
        elif len(primary_clean_gpu_set) < minimum_primary_clean_gpus:
            status = "HOLD_PRIMARY_CLEAN_COHORT_BELOW_MINIMUM"
        else:
            status = "PASS"
    else:
        status = "PASS" if all_safe else "HOLD_WAIT_FOR_8GPU_SAFE"
    return {
        "schema": "STAGE_V_M1_V2_2_DYNAMIC_GPU_PREFLIGHT_V1" if dynamic_cohort else ("STAGE_V_M1_V2_1_GPU_PREFLIGHT_V1" if system_graphics_baseline is not None else "STAGE_V_M1_V2_GPU_PREFLIGHT_V1"),
        "status": status,
        "gpu_ids": list(gpu_ids), "all_8_safe": all_safe,
        "primary_clean_gpu_set": primary_clean_gpu_set,
        "primary_clean_gpu_count": len(primary_clean_gpu_set),
        "minimum_primary_clean_gpus": minimum_primary_clean_gpus,
        "dynamic_primary_cohort": dynamic_cohort,
        "uuid_by_gpu": {str(row["index"]): row["uuid_canonical"] for row in gpu_rows},
        "idle_memory_max_mib": idle_memory_max_mib, "graphics_memory_max_mib": memory_limit, "gpu_rows": decisions,
        "unmapped_processes": unmapped, "baseline_system_graphics": baseline_system_graphics,
        "foreign_user_workloads": foreign_user_workloads, "graphics_contract": dict(graphics_contract),
        "captured_utc": _now(), "foreign_processes_touched": False, "gpu5_touched": False,
    }


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = _load(path)
    schema = protocol.get("schema")
    if schema not in {
        "STAGE_V_M1_VISUAL_DETERMINISM_PROTOCOL_V2_8GPU",
        "STAGE_V_M1_VISUAL_DETERMINISM_PROTOCOL_V2_1_8GPU",
        "STAGE_V_M1_VISUAL_DETERMINISM_PROTOCOL_V2_1_1_8GPU",
        DYNAMIC_PROTOCOL_SCHEMA,
    }:
        raise V2Error("V2_PROTOCOL_SCHEMA_INVALID")
    if is_dynamic_protocol(protocol):
        required = {
            "schema": DYNAMIC_PROTOCOL_SCHEMA,
            "status": "FROZEN_DIAGNOSTIC_ONLY_NO_SCIENCE_AUTHORIZATION",
            "gpu_ids": list(GPU_IDS), "workers": "PRIMARY_CLEAN_GPU_COUNT",
            "runs_per_gpu": 4, "total_r1_runs": "4N", "parallelism": "PRIMARY_CLEAN_GPU_COUNT",
            "seed": 7, "worker_gpu_mapping": "WORKER_K_TO_PRIMARY_CLEAN_GPU_SET_SORTED",
            "fresh_subprocess_per_run": True, "lockstep_barriers": True,
            "renderer_binding_canary_required": True, "tolerance_allowed": False,
            "gpu5_authorized": True, "primary_clean_gpu_minimum": MIN_PRIMARY_CLEAN_GPUS,
            "dynamic_count_formulas": {
                "r1_runs": "4N",
                "gpu_local_pairs": "4N",
                "cross_gpu_pairs": "4*C(N,2)",
            },
            "primary_clean_gpu_rule": "ONLY_REGISTERED_SYSTEM_GRAPHICS_BASELINE_NO_FOREIGN_COMPUTE_OR_GRAPHICS_CUDA_EGL_CLOSURE",
            "complete_ownership": "AUDITOR_ONLY",
        }
        for key, expected in required.items():
            if protocol.get(key) != expected:
                raise V2Error(f"V2_2_PROTOCOL_INVALID:{key}")
    else:
        required = {
            "schema": schema,
            "status": "FROZEN_DIAGNOSTIC_ONLY_NO_SCIENCE_AUTHORIZATION",
            "gpu_ids": list(GPU_IDS), "workers": 8, "runs_per_gpu": 4,
            "total_r1_runs": 32, "parallelism": 8, "seed": 7,
            "worker_gpu_mapping": "FIXED_WORKER_I_TO_GPU_I",
            "fresh_subprocess_per_run": True, "lockstep_barriers": True,
            "renderer_binding_canary_required": True, "tolerance_allowed": False,
            "gpu5_authorized": True,
        }
        for key, expected in required.items():
            if protocol.get(key) != expected:
                raise V2Error(f"V2_PROTOCOL_INVALID:{key}")
    if protocol.get("phase_order") != list(PHASES):
        raise V2Error("V2_PHASE_ORDER_INVALID")
    if is_dynamic_protocol(protocol):
        for key, expected in {
            "actual_runtime_binding_receipt_required": True,
            "fresh_preflight_per_gate": True,
            "require_prepare_before_preflight": True,
            "classification_evidence_profile_required": True,
            "independent_classification_check": True,
            "uuid_three_way_closure_required": True,
            "uuid_canonicalization": "LOWERCASE_STRIP_GPU_PREFIX_EXACT",
            "producer_complete_owned_by_auditor": True,
        }.items():
            if protocol.get(key) != expected:
                raise V2Error(f"V2_2_PROTOCOL_INVALID:{key}")
        if _git_blob_sha("configs/stage_v_m1_visual_determinism_protocol_v2_1_1_8gpu.json") != protocol.get("base_v2_1_1_protocol_sha256"):
            raise V2Error("V2_1_1_BASE_PROTOCOL_CHANGED")
        graphics = protocol.get("system_graphics_baseline")
        if not isinstance(graphics, Mapping) or graphics != {
            "enabled": True, "kind": "G", "process_name": "Xorg", "owner": "gdm",
            "executable": "/usr/lib/xorg/Xorg", "max_memory_used_mib": 128,
            "require_all_gpu_coverage": True, "require_single_consistent_pid": True,
            "compute_processes_never_whitelisted": True,
        }:
            raise V2Error("V2_2_GRAPHICS_BASELINE_INVALID")
        if protocol.get("preflight_gates") != [
            "PRE_CANARY", "PRE_R1_Q1", "PRE_R1_C1", "PRE_R1_Q2", "PRE_R1_C2",
            "PRE_R2_Q1", "PRE_R2_C1", "PRE_R2_Q2", "PRE_R2_C2",
        ]:
            raise V2Error("V2_2_PREFLIGHT_GATES_INVALID")
    elif schema in {
        "STAGE_V_M1_VISUAL_DETERMINISM_PROTOCOL_V2_1_8GPU",
        "STAGE_V_M1_VISUAL_DETERMINISM_PROTOCOL_V2_1_1_8GPU",
    }:
        for key, expected in {
            "actual_runtime_binding_receipt_required": True,
            "fresh_preflight_per_gate": True,
            "require_prepare_before_preflight": True,
            "classification_evidence_profile_required": True,
            "independent_classification_check": True,
        }.items():
            if protocol.get(key) != expected:
                raise V2Error(f"V2_1_PROTOCOL_INVALID:{key}")
        if _git_blob_sha("configs/stage_v_m1_visual_determinism_protocol_v2_8gpu.json") != protocol.get("base_v2_protocol_sha256"):
            raise V2Error("V2_BASE_PROTOCOL_CHANGED")
        if schema == "STAGE_V_M1_VISUAL_DETERMINISM_PROTOCOL_V2_1_1_8GPU":
            if _git_blob_sha("configs/stage_v_m1_visual_determinism_protocol_v2_1_8gpu.json") != protocol.get("base_v2_1_protocol_sha256"):
                raise V2Error("V2_1_BASE_PROTOCOL_CHANGED")
            for key, expected in {
                "uuid_three_way_closure_required": True,
                "uuid_canonicalization": "LOWERCASE_STRIP_GPU_PREFIX_EXACT",
                "producer_complete_owned_by_auditor": True,
            }.items():
                if protocol.get(key) != expected:
                    raise V2Error(f"V2_1_1_PROTOCOL_INVALID:{key}")
        graphics = protocol.get("system_graphics_baseline")
        if not isinstance(graphics, Mapping) or graphics != {
            "enabled": True, "kind": "G", "process_name": "Xorg", "owner": "gdm",
            "executable": "/usr/lib/xorg/Xorg", "max_memory_used_mib": 128,
            "require_all_gpu_coverage": True, "require_single_consistent_pid": True,
            "compute_processes_never_whitelisted": True,
        }:
            raise V2Error("V2_1_GRAPHICS_BASELINE_INVALID")
        if protocol.get("preflight_gates") != [
            "PRE_CANARY", "PRE_R1_Q1", "PRE_R1_C1", "PRE_R1_Q2", "PRE_R1_C2",
            "PRE_R2_Q1", "PRE_R2_C1", "PRE_R2_Q2", "PRE_R2_C2",
        ]:
            raise V2Error("V2_1_PREFLIGHT_GATES_INVALID")
    v1_path = REPO_ROOT / "configs/stage_v_m1_visual_determinism_protocol_v1.json"
    v1 = _load(v1_path)
    if _git_blob_sha("configs/stage_v_m1_visual_determinism_protocol_v1.json") != protocol.get("base_v1_visual_protocol_sha256"):
        raise V2Error("V1_VISUAL_PROTOCOL_CHANGED")
    if v1.get("rb1_v1_protocol_sha256") != protocol.get("base_rb1_v1_protocol_sha256"):
        raise V2Error("V1_RB1_PROTOCOL_BINDING_CHANGED")
    if v1.get("gpu5_authorized") is not False or v1.get("rb1_v1_modified") is not False:
        raise V2Error("V1_BOUNDARY_INVALID")
    if set(protocol.get("classification_enum", [])) != {
        "RAW_OBSERVATION_NON_POLICY_DIFFERENCE", "SAME_MODE_RENDER_OR_OBSERVATION_NONDETERMINISM",
        "MODE_PATH_SPECIFIC_VISUAL_DIVERGENCE", "GPU_CONTEXT_DEPENDENT_VISUAL_DIVERGENCE",
        "PROCESSOR_OR_MODEL_INPUT_NONDETERMINISM", "SIMULATOR_RUNTIME_NONDETERMINISM",
        "POLICY_VISUAL_INPUT_NONDETERMINISM_ACTION_STABLE", "HETEROGENEOUS_MULTI_GPU_DIVERGENCE",
        "MULTI_LAYER_NONDETERMINISM", "UNCLASSIFIED",
    }:
        raise V2Error("V2_CLASSIFICATION_ENUM_INVALID")
    return protocol


def _verify_source_binding(manifest: Mapping[str, Any]) -> None:
    if _git("status", "--porcelain"):
        raise V2Error("V2_SOURCE_WORKTREE_DIRTY")
    if manifest.get("source_commit") != _git("rev-parse", "HEAD") or manifest.get("source_tree") != _git("rev-parse", "HEAD^{tree}"):
        raise V2Error("V2_SOURCE_BINDING_MISMATCH")


def _reject_v1_root(root: Path) -> None:
    if "M1_VISUAL_DETERMINISM" in root.name or (root / "M1_MANIFEST.json").exists():
        raise V2Error("V2_MUST_NOT_TOUCH_V1_ROOT")


def validate_manifest_authorization(manifest: Mapping[str, Any]) -> None:
    for field in AUTHORIZATION_FLAGS:
        if manifest.get(field) is not False:
            raise V2Error(f"V2_AUTHORIZATION_BOUNDARY_INVALID:{field}")
    if any(manifest.get(field, 0) != 0 for field in BOUNDARIES):
        raise V2Error("V2_PROTECTED_BOUNDARY_NONZERO")


def prepare_root(root: Path, protocol: Mapping[str, Any], *, source_commit: str, source_tree: str,
                 model_path: str, protocol_path: Path | None = None) -> None:
    if root.exists():
        raise V2Error("V2_ROOT_MUST_BE_NEW")
    if _git("status", "--porcelain"):
        raise V2Error("V2_SOURCE_WORKTREE_DIRTY")
    if _git("rev-parse", "HEAD") != source_commit or _git("rev-parse", "HEAD^{tree}") != source_tree:
        raise V2Error("V2_SOURCE_BINDING_MISMATCH")
    root.mkdir(parents=True)
    protocol_path = (protocol_path or REPO_ROOT / "configs/stage_v_m1_visual_determinism_protocol_v2_1_1_8gpu.json").resolve()
    dynamic = is_dynamic_protocol(protocol)
    manifest_schema = (
        "STAGE_V_M1_V2_2_DYNAMIC_COHORT_MANIFEST_V1"
        if dynamic else (
        "STAGE_V_M1_V2_1_1_8GPU_MANIFEST_V1"
        if protocol.get("schema") == "STAGE_V_M1_VISUAL_DETERMINISM_PROTOCOL_V2_1_1_8GPU"
        else "STAGE_V_M1_V2_1_8GPU_MANIFEST_V1"
        )
    )
    initial_counts = dynamic_counts(()) if dynamic else {"primary_clean_gpu_count": 8, "r1_run_count": 32, "gpu_local_pair_count": 32, "cross_gpu_pair_count": 112}
    manifest = {
        "schema": manifest_schema, "status": "PREPARED_NO_RUNTIME_STARTED",
        "created_utc": _now(), "protocol": protocol.get("protocol_id", "M1_V2_1_1_8GPU"), "protocol_schema": protocol.get("schema"), "protocol_sha256": sha256_file(protocol_path),
        "diagnostic_identity": IDENTITY, "source_commit": source_commit, "source_tree": source_tree,
        "gpu_ids": list(GPU_IDS),
        "primary_clean_gpu_set": [], "primary_clean_gpu_count": initial_counts["primary_clean_gpu_count"],
        "workers": initial_counts["primary_clean_gpu_count"], "runs_per_gpu": 4,
        "total_r1_runs": initial_counts["r1_run_count"],
        "gpu_local_pair_count": initial_counts["gpu_local_pair_count"],
        "cross_gpu_pair_count": initial_counts["cross_gpu_pair_count"],
        "primary_clean_gpu_minimum": int(protocol.get("primary_clean_gpu_minimum", 8 if not dynamic else MIN_PRIMARY_CLEAN_GPUS)),
        "dynamic_primary_cohort": dynamic,
        "phase_order": list(PHASES), "seed": 7, "model_path": model_path,
        "new_science_rollouts_authorized": False, "formal_parent_promotion_authorized": False,
        "eval160_authorized": False, "protected_evaluation_authorized": False,
        "vis_pgd_authorized": False, "student_training_authorized": False,
        "vulnerability_label_generation_authorized": False,
        "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0, "attack_rollouts": 0,
        "intervention_applied_steps": 0, "counterfactual_open_steps": 0,
    }
    _write(root / "M1_V2_MANIFEST.json", manifest)
    _write(root / "M1_V2_STATUS.json", {"schema": "STAGE_V_M1_V2_2_STATUS_V1" if dynamic else "STAGE_V_M1_V2_1_STATUS_V1", "status": "PREPARED_NO_RUNTIME_STARTED", "r1_started": False, "r2_started": False, "classification": "UNCLASSIFIED"})


def _require_manifest(root: Path) -> dict[str, Any]:
    manifest = _load(root / "M1_V2_MANIFEST.json")
    if manifest.get("status") != "PREPARED_NO_RUNTIME_STARTED":
        raise V2Error("V2_ROOT_ALREADY_CONSUMED")
    if manifest.get("diagnostic_identity") != IDENTITY:
        raise V2Error("V2_IDENTITY_MISMATCH")
    validate_manifest_authorization(manifest)
    _verify_source_binding(manifest)
    return manifest


def _preflight_path(root: Path, gate: str) -> Path:
    return root / f"M1_V2_1_GPU_PREFLIGHT_{gate}.json"


def _manifest_cohort(root: Path) -> tuple[int, ...]:
    manifest = _load(root / "M1_V2_MANIFEST.json")
    if not manifest.get("dynamic_primary_cohort"):
        return GPU_IDS
    cohort = tuple(sorted(int(gpu) for gpu in manifest.get("primary_clean_gpu_set", [])))
    if not cohort:
        raise V2Error("PRIMARY_CLEAN_GPU_SET_MISSING")
    return cohort


def _fresh_preflight(root: Path, protocol: Mapping[str, Any], gate: str, *, run_set: str, protocol_path: Path | None = None) -> dict[str, Any]:
    manifest = _require_manifest(root)
    dynamic = is_dynamic_protocol(protocol)
    value = gpu_preflight(
        idle_memory_max_mib=int(protocol["idle_memory_max_mib"]),
        system_graphics_baseline=protocol["system_graphics_baseline"],
        dynamic_cohort=dynamic,
        minimum_primary_clean_gpus=int(protocol.get("primary_clean_gpu_minimum", MIN_PRIMARY_CLEAN_GPUS)),
    )
    value.update({
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file((protocol_path or REPO_ROOT / "configs/stage_v_m1_visual_determinism_protocol_v2_1_1_8gpu.json").resolve()),
        "source_commit": manifest["source_commit"],
        "source_tree": manifest["source_tree"],
        "gate": gate,
        "run_set": run_set,
        "phase_receipt_sha256": None,
        "phase_receipt_path": None,
    })
    if dynamic:
        selected = tuple(sorted(int(gpu) for gpu in value["primary_clean_gpu_set"]))
        if gate == "PRE_CANARY":
            manifest = dict(manifest)
            counts = dynamic_counts(selected)
            manifest.update({
                "primary_clean_gpu_set": list(selected),
                "primary_clean_gpu_count": counts["primary_clean_gpu_count"],
                "workers": counts["primary_clean_gpu_count"],
                "total_r1_runs": counts["r1_run_count"],
                "gpu_local_pair_count": counts["gpu_local_pair_count"],
                "cross_gpu_pair_count": counts["cross_gpu_pair_count"],
                "primary_clean_gpu_minimum": int(protocol["primary_clean_gpu_minimum"]),
            })
            _write(root / "M1_V2_MANIFEST.json", manifest)
        else:
            frozen = tuple(sorted(int(gpu) for gpu in manifest.get("primary_clean_gpu_set", [])))
            value["primary_clean_cohort_frozen"] = list(frozen)
            if selected != frozen:
                value["status"] = "HOLD_PRIMARY_COHORT_DRIFT"
                value["primary_clean_cohort_drift"] = {"frozen": list(frozen), "current": list(selected)}
    path = _preflight_path(root, gate)
    _write(path, value)
    if value.get("status") != "PASS" or (not dynamic and value.get("all_8_safe") is not True):
        status_path = root / "M1_V2_STATUS.json"
        status = _load(status_path)
        status.update({"status": f"HOLD_RESOURCE_DRIFT_{gate}", "preflight_gate": gate, "preflight": value})
        _write(status_path, status)
        raise V2Error(f"HOLD_PRIMARY_CLEAN_COHORT:{gate}" if dynamic else f"HOLD_WAIT_FOR_8GPU_SAFE:{gate}")
    return value


def _bind_preflight(root: Path, gate: str, receipt_path: Path) -> None:
    path = _preflight_path(root, gate)
    value = _load(path)
    value["phase_receipt_path"] = str(receipt_path.relative_to(root).as_posix())
    value["phase_receipt_sha256"] = sha256_file(receipt_path)
    _write(path, value)


def _require_preflight(root: Path, gate: str) -> dict[str, Any]:
    value = _load(_preflight_path(root, gate))
    manifest = _load(root / "M1_V2_MANIFEST.json")
    if value.get("status") != "PASS" or (not manifest.get("dynamic_primary_cohort") and value.get("all_8_safe") is not True):
        raise V2Error(f"HOLD_PRIMARY_CLEAN_COHORT:{gate}" if manifest.get("dynamic_primary_cohort") else f"HOLD_WAIT_FOR_8GPU_SAFE:{gate}")
    if manifest.get("dynamic_primary_cohort") and value.get("primary_clean_gpu_set") != manifest.get("primary_clean_gpu_set"):
        raise V2Error(f"HOLD_PRIMARY_COHORT_DRIFT:{gate}")
    return value


def _run_renderer_canary_child(args: argparse.Namespace) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MUJOCO_GL"] = "egl"
    logical_gpu = 0
    # robosuite's installed binding checks this physical selector against
    # CUDA_VISIBLE_DEVICES before importing; the renderer itself stays logical 0.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import torch
    if not torch.cuda.is_available():
        raise V2Error("CANARY_CUDA_UNAVAILABLE")
    candidate = _load(args.candidate)
    sys.path.insert(0, str(args.upstream_root))
    official_src = args.official_snapshot_root / "src"
    import gripper_attack
    gripper_attack.__path__.append(str(official_src / "gripper_attack"))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite_instance = benchmark.get_benchmark_dict()[args.suite]()
    task = suite_instance.get_task(int(candidate["task_index"]))
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=8, camera_widths=8, has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False, render_gpu_device_id=logical_gpu, horizon=1)
    try:
        observed = None
        sim = getattr(env, "sim", None)
        render_context = getattr(sim, "render_context", None) or getattr(sim, "_render_context_offscreen", None)
        for obj in (env, sim, render_context):
            for name in ("render_gpu_device_id", "gpu_device_id", "device_id"):
                value = getattr(obj, name, None) if obj is not None else None
                if value is not None:
                    observed = int(value)
                    break
            if observed is not None:
                break
        if observed != logical_gpu:
            raise V2Error("EGL_DEVICE_BINDING_MISMATCH")
        properties = torch.cuda.get_device_properties(0)
        gpu_uuid, gpu_uuid_source = _runtime_gpu_uuid(int(args.gpu), properties)
        if not gpu_uuid:
            raise V2Error("CANARY_GPU_UUID_UNAVAILABLE")
        renderer_device_information = {
            "env_class": type(env).__name__,
            "sim_class": type(getattr(env, "sim", None)).__name__,
            "render_context_class": type(render_context).__name__,
            "observed_device_id": observed,
        }
        result = {
            "schema": "STAGE_V_M1_V2_RENDERER_BINDING_CANARY_V1", "status": "PASS",
            "logical_worker_id": f"worker_{int(args.gpu)}", "requested_physical_gpu": int(args.gpu),
            "physical_gpu_index": int(args.gpu), "gpu_uuid": gpu_uuid,
            "gpu_uuid_canonical": canonical_gpu_uuid(gpu_uuid),
            "gpu_uuid_source": gpu_uuid_source,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "torch_current_device": int(torch.cuda.current_device()), "cuda_logical_device": 0,
            "cuda_device_name": torch.cuda.get_device_name(0), "mujoco_gl": os.environ["MUJOCO_GL"],
            "egl_device_identifier": observed, "egl_device_id": observed,
            "egl_binding_source": "OffScreenRenderEnv.render_gpu_device_id",
            "renderer_device_information": renderer_device_information,
            "episode_started": False,
        }
        validate_binding_receipt(result, int(args.gpu), require_canonical_uuid=True)
    finally:
        env.close()
    _write(args.canary_output, result)
    return 0


def run_renderer_canary(root: Path, args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    _fresh_preflight(root, protocol, "PRE_CANARY", run_set="canary", protocol_path=args.protocol)
    cohort = _manifest_cohort(root)
    output_root = root / "renderer_canary"
    output_root.mkdir(parents=True, exist_ok=True)

    def one(gpu: int) -> dict[str, Any]:
        output = output_root / f"gpu_{gpu:02d}.json"
        command = [str(sys.executable), str(Path(__file__).resolve()), "--renderer-canary", "--protocol", str(args.protocol), "--root", str(root), "--gpu", str(gpu), "--canary-output", str(output), "--candidate", str(args.candidate), "--suite", args.suite, "--official-snapshot-root", str(args.official_snapshot_root), "--upstream-root", str(args.upstream_root)]
        result = subprocess.run(command, check=False, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": str(gpu)}, capture_output=True, text=True)
        return {"gpu": gpu, "returncode": result.returncode, "stderr": result.stderr[-1000:], "output": str(output)}

    with ThreadPoolExecutor(max_workers=len(cohort)) as pool:
        results = list(pool.map(one, cohort))
    if any(item["returncode"] != 0 or not Path(item["output"]).is_file() or _load(Path(item["output"])).get("status") != "PASS" for item in results):
        canary_path = root / "M1_V2_RENDERER_CANARY.json"
        _write(canary_path, {"schema": "STAGE_V_M1_V2_1_RENDERER_CANARY_AGGREGATE_V1", "status": "HOLD_RENDER_DEVICE_BINDING_MISMATCH", "results": results})
        _bind_preflight(root, "PRE_CANARY", canary_path)
        raise V2Error("HOLD_RENDER_DEVICE_BINDING_MISMATCH")
    preflight = _require_preflight(root, "PRE_CANARY")
    uuid_by_gpu: dict[str, str] = {}
    try:
        for gpu in cohort:
            canary = _load(output_root / f"gpu_{gpu:02d}.json")
            validate_binding_receipt(canary, gpu, require_canonical_uuid=True)
            uuid_by_gpu[str(gpu)] = validate_uuid_binding(
                gpu=gpu,
                preflight_uuid=preflight["uuid_by_gpu"].get(str(gpu)),
                canary_uuid=canary.get("gpu_uuid_canonical"),
                phase="PRE_CANARY",
            )
    except V2Error as exc:
        canary_path = root / "M1_V2_RENDERER_CANARY.json"
        _write(canary_path, {"schema": "STAGE_V_M1_V2_1_RENDERER_CANARY_AGGREGATE_V1", "status": str(exc).split(":", 1)[0], "uuid_by_gpu": uuid_by_gpu, "results": results})
        _bind_preflight(root, "PRE_CANARY", canary_path)
        raise
    canary_path = root / "M1_V2_RENDERER_CANARY.json"
    _write(canary_path, {"schema": "STAGE_V_M1_V2_2_RENDERER_CANARY_AGGREGATE_V1" if is_dynamic_protocol(protocol) else "STAGE_V_M1_V2_1_RENDERER_CANARY_AGGREGATE_V1", "status": "PASS", "gpu_ids": list(cohort), "uuid_by_gpu": uuid_by_gpu, "results": results})
    _bind_preflight(root, "PRE_CANARY", canary_path)


def _run_one(root: Path, args: argparse.Namespace, gpu: int, label: str, run_set: str) -> dict[str, Any]:
    mode, replicate = PHASES[label]
    base = root / ("runs" if run_set == "r1" else "raw_runs") / f"gpu_{gpu:02d}"
    output = base / label
    if output.exists():
        return {"gpu": gpu, "label": label, "status": "HOLD_PARTIAL_R1_ARTIFACT", "reason": "OUTPUT_ALREADY_EXISTS"}
    log_root = root / "logs" / f"gpu_{gpu:02d}"
    log_root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": str(gpu), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONHASHSEED": "7", "PYTHONUNBUFFERED": "1"}
    command = [str(sys.executable), str(REPO_ROOT / "scripts/detector_v5/run_stage_v_canonical_clean.py"), "--candidate", str(args.candidate), "--contract", str(args.contract), "--output-dir", str(output), "--official-snapshot-root", str(args.official_snapshot_root), "--upstream-root", str(args.upstream_root), "--model-path", str(args.model_path), "--suite", args.suite, "--gpu", str(gpu), "--seed", "7", "--mode", mode, "--source-commit", str(args.source_commit), "--source-tree", str(args.source_tree), "--run-label", label, "--run-set", run_set, "--enable-runtime"]
    if run_set == "r2":
        command.extend(["--raw-capture-plan", str(args.raw_capture_plan)])
    with (log_root / f"{label}.stdout.log").open("w", encoding="utf-8") as stdout, (log_root / f"{label}.stderr.log").open("w", encoding="utf-8") as stderr:
        run = subprocess.run(command, cwd=str(REPO_ROOT), env=env, stdout=stdout, stderr=stderr, check=False)
    if run.returncode != 0:
        return {"gpu": gpu, "label": label, "status": "FAIL", "returncode": run.returncode}
    producer = output / "RB1_PRODUCER_RECEIPT.json"
    independent = output / "RB1_INDEPENDENT_RECEIPT.json"
    audit = subprocess.run([str(sys.executable), str(REPO_ROOT / "scripts/detector_v5/audit_stage_v_rb1_receipt.py"), "--protocol", str(REPO_ROOT / "configs/stage_v_rb1_runtime_equivalence_protocol_v1.json"), "--receipt", str(producer), "--artifact-root", str(output / "trace"), "--core", str(REPO_ROOT / "src/gripper_attack/stage_v_canonical_execution_core.py"), "--output", str(independent), "--repo", str(REPO_ROOT)], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    if audit.returncode != 0 or not independent.is_file():
        return {"gpu": gpu, "label": label, "status": "FAIL_AUDIT", "returncode": audit.returncode, "stderr": audit.stderr[-1000:]}
    try:
        runtime_receipt = _load(output / "M1_V2_RUNTIME_BINDING_RECEIPT.json")
        validate_runtime_binding_receipt(runtime_receipt, gpu, run_set=run_set, phase=label, source_commit=str(args.source_commit), source_tree=str(args.source_tree))
        preflight = _load(_preflight_path(root, f"PRE_{run_set.upper()}_{label}"))
        canary = _load(root / "M1_V2_RENDERER_CANARY.json")
        validate_uuid_binding(
            gpu=gpu,
            preflight_uuid=preflight["uuid_by_gpu"].get(str(gpu)),
            canary_uuid=canary.get("uuid_by_gpu", {}).get(str(gpu)),
            runtime_uuid=runtime_receipt.get("torch_device_uuid_canonical", runtime_receipt.get("torch_device_uuid")),
            phase=f"{run_set.upper()}_{label}",
        )
    except (OSError, ValueError, KeyError, V2Error) as exc:
        if str(exc).startswith("HOLD_GPU_UUID_BINDING_MISMATCH"):
            return {"gpu": gpu, "label": label, "status": "HOLD_GPU_UUID_BINDING_MISMATCH", "reason": str(exc)}
        return {"gpu": gpu, "label": label, "status": "FAIL_RUNTIME_BINDING_RECEIPT", "reason": str(exc)}
    return {"gpu": gpu, "label": label, "status": "PASS", "output": str(output), "mode": mode, "replicate": replicate}


def run_matrix(root: Path, args: argparse.Namespace, run_set: str, protocol: Mapping[str, Any]) -> None:
    manifest = _require_manifest(root)
    if args.source_commit != manifest.get("source_commit") or args.source_tree != manifest.get("source_tree"):
        raise V2Error("V2_SOURCE_BINDING_ARGUMENT_MISMATCH")
    _require_preflight(root, "PRE_CANARY")
    cohort = _manifest_cohort(root)
    if not _load(root / "M1_V2_RENDERER_CANARY.json").get("status") == "PASS":
        raise V2Error("HOLD_RENDER_DEVICE_BINDING_MISMATCH")
    if run_set == "r2" and not args.raw_capture_plan:
        raise V2Error("V2_RAW_CAPTURE_PLAN_REQUIRED")
    status_path = root / "M1_V2_STATUS.json"
    status = _load(status_path)
    status.update({"status": "RUNNING_R1" if run_set == "r1" else "RUNNING_R2", "phase": "M1-R1" if run_set == "r1" else "M1-R2", "source_commit": manifest["source_commit"], "source_tree": manifest["source_tree"], "r1_started": bool(status.get("r1_started")) or run_set == "r1", "r2_started": bool(status.get("r2_started")) or run_set == "r2", "protected_boundaries": {key: 0 for key in BOUNDARIES}})
    _write(status_path, status)
    for label in PHASES:
        gate = f"PRE_{run_set.upper()}_{label}"
        _fresh_preflight(root, protocol, gate, run_set=run_set, protocol_path=args.protocol)
        with ThreadPoolExecutor(max_workers=len(cohort)) as pool:
            results = list(pool.map(lambda gpu: _run_one(root, args, gpu, label, run_set), cohort))
        receipt_path = root / f"M1_V2_{run_set.upper()}_{label}_RECEIPTS.json"
        _write(receipt_path, {"schema": "STAGE_V_M1_V2_1_PHASE_RECEIPTS_V1", "phase": label, "run_set": run_set, "results": results})
        _bind_preflight(root, gate, receipt_path)
        if any(item.get("status") != "PASS" for item in results):
            status.update({"status": "HOLD_GPU_UUID_BINDING_MISMATCH" if any(item.get("status") == "HOLD_GPU_UUID_BINDING_MISMATCH" for item in results) else "HOLD_PARTIAL_R1", "failed_phase": label, "results": results})
            _write(status_path, status)
            if status["status"] == "HOLD_GPU_UUID_BINDING_MISMATCH":
                raise V2Error(f"HOLD_GPU_UUID_BINDING_MISMATCH:{label}")
            raise V2Error(f"HOLD_PARTIAL_R1:{label}")
    status.update({"status": "R1_COMPLETE_PENDING_AUDIT" if run_set == "r1" else "R2_COMPLETE_PENDING_AUDIT", "completed_run_set": run_set})
    _write(status_path, status)
    analysis_command = [str(sys.executable), str(REPO_ROOT / "scripts/detector_v5/analyze_stage_v_m1_v2_multigpu.py"), "--root", str(root)]
    if run_set == "r2":
        analysis_command.append("--final")
    analysis = subprocess.run(analysis_command, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    if analysis.returncode != 0:
        status.update({"status": "HOLD_PRODUCER_ANALYSIS_FAIL", "analysis_stdout": analysis.stdout[-2000:], "analysis_stderr": analysis.stderr[-2000:]})
        _write(status_path, status)
        raise V2Error("HOLD_PRODUCER_ANALYSIS_FAIL")
    audit_command = [str(sys.executable), str(REPO_ROOT / "scripts/detector_v5/audit_stage_v_m1_v2_8gpu.py"), "--root", str(root)]
    if run_set == "r2":
        audit_command.append("--final")
    audit = subprocess.run(audit_command, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    if audit.returncode != 0:
        status.update({"status": "HOLD_AUDIT_DISAGREEMENT", "audit_stdout": audit.stdout[-2000:], "audit_stderr": audit.stderr[-2000:]})
        _write(status_path, status)
        raise V2Error("HOLD_AUDIT_DISAGREEMENT")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-root", action="store_true")
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--run-canary", action="store_true")
    modes.add_argument("--run-r1", action="store_true")
    modes.add_argument("--run-r2", action="store_true")
    modes.add_argument("--renderer-canary", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / "configs/stage_v_m1_visual_determinism_protocol_v2_1_1_8gpu.json")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-tree")
    parser.add_argument("--model-path")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--official-snapshot-root", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--canary-output", type=Path)
    parser.add_argument("--raw-capture-plan", type=Path)
    parser.add_argument("--preflight-gate", default="PRE_CANARY")
    args = parser.parse_args(argv)
    try:
        protocol = validate_protocol(args.protocol.resolve())
        if protocol.get("schema") not in {"STAGE_V_M1_VISUAL_DETERMINISM_PROTOCOL_V2_1_1_8GPU", DYNAMIC_PROTOCOL_SCHEMA}:
            raise V2Error("V2_1_1_PROTOCOL_REQUIRED")
        if args.renderer_canary:
            if args.gpu not in GPU_IDS or not args.canary_output or not args.candidate or not args.official_snapshot_root or not args.upstream_root:
                raise V2Error("CANARY_ARGUMENTS_REQUIRED")
            if sys.prefix != PYTHON_PREFIX:
                raise V2Error(f"V2_PYTHON_PREFIX_MISMATCH:{sys.prefix}")
            canary_root = args.root.resolve()
            _reject_v1_root(canary_root)
            _require_manifest(canary_root)
            return _run_renderer_canary_child(args)
        if sys.prefix != PYTHON_PREFIX:
            raise V2Error(f"V2_PYTHON_PREFIX_MISMATCH:{sys.prefix}")
        root = args.root.resolve()
        _reject_v1_root(root)
        if args.prepare_root:
            if not args.source_commit or not args.source_tree or not args.model_path:
                raise V2Error("PREPARE_ROOT_BINDING_ARGUMENTS_REQUIRED")
            prepare_root(root, protocol, source_commit=args.source_commit, source_tree=args.source_tree, model_path=args.model_path, protocol_path=args.protocol.resolve())
            return 0
        if args.preflight_only:
            gate = str(args.preflight_gate)
            if not gate.startswith("PRE_"):
                raise V2Error("PREFLIGHT_GATE_INVALID")
            _fresh_preflight(root, protocol, gate, run_set="manual", protocol_path=args.protocol)
            return 0
        if not args.candidate or not args.official_snapshot_root or not args.upstream_root or not args.source_commit or not args.source_tree:
            raise V2Error("RUNTIME_ARGUMENTS_REQUIRED")
        if args.run_canary:
            run_renderer_canary(root, args, protocol)
        elif args.run_r1:
            if not args.contract or not args.model_path:
                raise V2Error("R1_ARGUMENTS_REQUIRED")
            run_matrix(root, args, "r1", protocol)
        else:
            if not args.contract or not args.model_path:
                raise V2Error("R2_ARGUMENTS_REQUIRED")
            run_matrix(root, args, "r2", protocol)
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, V2Error) as exc:
        print(json.dumps({"verdict": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
