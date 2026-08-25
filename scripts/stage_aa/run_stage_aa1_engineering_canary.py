#!/usr/bin/env python3
"""Run one AA1 engineering-only canary cell.

The runner reuses the already frozen Stage-Z model/environment loaders, but
implements the AA0 clean eligibility and endpoint guards locally.  The three
canaries are permanently excluded from AA2--AA5; no task success, protected
evaluation, PGD, or scientific parent is read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
HORIZONS = {"libero_10": 520, "libero_object": 280, "libero_spatial": 220}
QUEUE_LENGTH = {"M0_OPENVLA": 1, "M1_OPENVLA_OFT": 8, "M2_PI05_LIBERO": 5}
BOUNDARY = {
    "M0_OPENVLA": "FRESH_PER_STEP",
    "M1_OPENVLA_OFT": "FRESH_OFT_ACTION_QUEUE",
    "M2_PI05_LIBERO": "FRESH_PI05_REPLAN",
}
AA1_SALT = "STAGE_AA_AA1_ENGINEERING_POINT_V1_20260826"
ACTION_DIM = 7
ARM_TOLERANCE = 1e-7
MIN_FREE_MIB = 20_480
DOSES = (3, 5, 10)


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"MODULE_SPEC_UNAVAILABLE:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


Z1 = load_module(ROOT / "scripts/stage_z/run_stage_z_z1_runtime_canary.py", "aa1_z1_runtime")
TAXONOMY = load_module(ROOT / "src/gripper_attack/stage_v_m3_5_physical_taxonomy.py", "aa1_taxonomy")
SEMANTICS = load_module(ROOT / "src/stage_z_preparation/action_semantics.py", "aa1_action_semantics")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def command_open_action(family: str, raw_action: list[float], final_action: list[float], duration: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Minimal local copy of the frozen command-level action contract."""
    if duration not in DOSES or len(raw_action) != ACTION_DIM or len(final_action) != ACTION_DIM:
        raise RuntimeError("AA1_OPEN_ACTION_INPUT_INVALID")
    native_raw = {"M0_OPENVLA": 1.0, "M1_OPENVLA_OFT": 1.0, "M2_PI05_LIBERO": -1.0}.get(family)
    if native_raw is None:
        raise RuntimeError("AA1_UNKNOWN_MODEL_FAMILY")
    opened_raw = tuple(float(value) for value in (*raw_action[:6], native_raw))
    opened_final = tuple(float(value) for value in (*final_action[:6], -1.0))
    if any(abs(opened_raw[index] - float(raw_action[index])) > ARM_TOLERANCE for index in range(6)):
        raise RuntimeError("AA1_RAW_ARM_CHANGED")
    if any(abs(opened_final[index] - float(final_action[index])) > ARM_TOLERANCE for index in range(6)):
        raise RuntimeError("AA1_FINAL_ARM_CHANGED")
    return opened_raw, opened_final


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def gpu_snapshot(gpu_id: int) -> dict[str, Any]:
    query = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    fields = [item.strip() for item in query.split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"GPU_QUERY_INVALID:{query}")
    apps = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu_id), "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    processes = []
    for line in apps.splitlines():
        if line.strip():
            parts = [item.strip() for item in line.split(",")]
            processes.append({"pid": parts[0], "name": parts[1], "used_memory_mib": parts[2]})
    result = {
        "index": int(fields[0]),
        "free_memory_mib": int(fields[1]),
        "used_memory_mib": int(fields[2]),
        "utilization_gpu_percent": int(fields[3]),
        "compute_processes": processes,
    }
    if result["free_memory_mib"] <= MIN_FREE_MIB:
        raise RuntimeError(f"GPU_NOT_ELIGIBLE_FREE_MEMORY_MIB:{result['free_memory_mib']}")
    return result


def require_single_gpu(gpu_id: int) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != str(gpu_id):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES_MUST_BE_SINGLE_GPU:{gpu_id}")


def canary_from_plan(plan: dict[str, Any], key: str) -> dict[str, Any]:
    rows = [row for row in plan.get("canaries", []) if row.get("canonical_parent_key") == key]
    if len(rows) != 1:
        raise RuntimeError(f"AA1_CANARY_NOT_UNIQUE:{key}")
    row = rows[0]
    if row.get("permanent_exclusion") is not True or row.get("scientific_use") is not False:
        raise RuntimeError("AA1_CANARY_EXCLUSION_FIREWALL_INVALID")
    return row


def checkpoint_path(z1_config: dict[str, Any], family: str, suite: str) -> Path:
    spec = z1_config["model_families"][family]
    if family == "M0_OPENVLA":
        return Path(spec["paths"][suite])
    if family == "M1_OPENVLA_OFT":
        return Path(spec["checkpoint_root"]) / suite
    return Path(spec["checkpoint"])


def static_validate(args: argparse.Namespace, protocol: dict[str, Any], plan: dict[str, Any], aa0: dict[str, Any], capacity: dict[str, Any], z1_config: dict[str, Any], key: str) -> dict[str, Any]:
    if protocol.get("status") != "STAGE_AA_AA1_ENGINEERING_RUNTIME_QUALIFICATION_AUTHORIZED":
        raise RuntimeError("AA1_PROTOCOL_NOT_AUTHORIZED")
    if plan.get("status") != "STAGE_AA_AA1_ENGINEERING_CANARY_PLAN_FROZEN":
        raise RuntimeError("AA1_PLAN_NOT_FROZEN")
    if aa0.get("status") != "STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_FROZEN_STOP_FOR_PI":
        raise RuntimeError("AA0_PROTOCOL_NOT_FROZEN")
    if aa0.get("authorization", {}).get("aa1_to_aa5_authorized") is not False:
        raise RuntimeError("AA0_SUPERSEDING_AUTHORITY_INVALID")
    if z1_config.get("status") != "STAGE_Z_Z1_RUNTIME_SOURCE_AUTHORITY_FROZEN":
        raise RuntimeError("Z1_RUNTIME_AUTHORITY_NOT_FROZEN")
    canary = canary_from_plan(plan, key)
    inventory_rows = capacity.get("aa1_engineering_canary_reservation", {}).get("reserved_rows", [])
    inventory_match = [row for row in inventory_rows if row.get("canonical_parent_key") == key]
    if len(inventory_match) != 1 or inventory_match[0].get("selection_rank_sha256") != canary.get("source_selection_rank_sha256"):
        raise RuntimeError("AA1_CANARY_CAPACITY_BINDING_MISMATCH")
    if key in set(capacity.get("analysis_pool_after_aa1_reservation", {}).get("keys", [])):
        raise RuntimeError("AA1_CANARY_REMAINS_IN_AA2_POOL")
    if key in set(load_json(ROOT / "reports/STAGE_Z_Z0R1_SHARED_36_IDENTITY_PANEL_V1.json").get("selected_parent_keys", [])):
        raise RuntimeError("AA1_CANARY_OVERLAPS_STAGE_Z")
    checkpoint = checkpoint_path(z1_config, args.model_family, canary["suite"])
    if not checkpoint.exists():
        raise RuntimeError(f"CHECKPOINT_NOT_MATERIALIZED:{checkpoint}")
    if args.model_family == "M1_OPENVLA_OFT":
        manifest = ROOT / "reports/STAGE_Z_Z0R2_M1_OFT_CHECKPOINT_MANIFESTS_V2.json"
        Z1.verify_m1_materialization(manifest, checkpoint, canary["suite"], str(z1_config["model_families"][args.model_family]["checkpoint_manifests_sha256"]))
    return {"canary": canary, "checkpoint": str(checkpoint), "z1_protocol_sha256": sha256_file(ROOT / "configs/STAGE_Z_Z1_RUNTIME_PROTOCOL_V11.json")}


def bddl_path(env: Any, task: Any) -> Path:
    getter = Z1.make_libero_env.__globals__.get("get_libero_path")
    if getter is None:
        from libero.libero import get_libero_path  # type: ignore
        getter = get_libero_path
    return Path(getter("bddl_files")) / task.problem_folder / task.bddl_file


def model_pairs(infer: Any, obs: dict[str, Any], language: str, family: str, counters: dict[str, int]) -> list[tuple[np.ndarray, np.ndarray]]:
    chunk, meta = infer(obs, language)
    counters["model_inference_calls"] += 1
    raw = np.asarray(meta.get("raw_action_chunk"), dtype=np.float32)
    final = np.asarray(chunk, dtype=np.float32)
    if raw.ndim != 2 or final.ndim != 2 or raw.shape != final.shape or raw.shape[1] != ACTION_DIM:
        raise RuntimeError(f"ACTION_CHUNK_SHAPE_INVALID:{raw.shape}:{final.shape}")
    expected_boundary = BOUNDARY[family]
    if meta.get("fresh_boundary") != expected_boundary:
        raise RuntimeError(f"MODEL_BOUNDARY_INVALID:{family}:{meta.get('fresh_boundary')}")
    length = QUEUE_LENGTH[family]
    if raw.shape[0] < length:
        raise RuntimeError(f"ACTION_CHUNK_TOO_SHORT:{family}:{raw.shape[0]}:{length}")
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(length):
        raw_action = raw[index].copy()
        final_action = final[index].copy()
        if raw_action.size != ACTION_DIM or final_action.size != ACTION_DIM:
            raise RuntimeError("FINAL_ACTION_NOT_EXACTLY_SEVEN")
        check = SEMANTICS.validate_action_pair(family, raw_action.tolist(), final_action.tolist(), raw_gripper=float(raw_action[-1]), final_gripper=float(final_action[-1]))
        if not check.get("accepted"):
            raise RuntimeError(f"ACTION_SEMANTICS_INVALID:{family}:{check.get('reason')}")
        pairs.append((raw_action, final_action))
    return pairs


def telemetry(env: Any, binding: dict[str, Any], target: str, counters: dict[str, int]) -> dict[str, Any]:
    value = TAXONOMY.telemetry_from_env(env, binding, target_object_id=target)
    counters["physical_telemetry_reads"] += 1
    return {key: value.get(key) for key in ("contact_telemetry_valid", "object_identity", "object_position", "eef_position", "object_eef_distance_m", "object_gripper_contact", "object_support_contact")}


def stable(row: dict[str, Any], baseline_z: float, *, require_carry: bool = True) -> bool:
    position = row.get("object_position")
    distance = row.get("object_eef_distance_m")
    if row.get("contact_telemetry_valid") is not True or row.get("object_gripper_contact") is not True or row.get("object_support_contact") is not False:
        return False
    if not isinstance(position, list) or len(position) != 3 or not isinstance(distance, (int, float)):
        return False
    if float(position[2]) - baseline_z < 0.015 or float(distance) > 0.12:
        return False
    if require_carry and float(distance) > 0.04:
        return False
    return True


def select_candidate(rows: list[dict[str, Any]], actions: list[dict[str, Any]], family: str, key: str, anchor_class: str) -> dict[str, Any] | None:
    baseline_z = next((float(row["object_position"][2]) for row in rows if isinstance(row.get("object_position"), list) and len(row["object_position"]) == 3), None)
    if baseline_z is None:
        return None
    horizon = len(rows)
    candidates = []
    for step in range(max(0, horizon - 19)):
        if not actions[step]["boundary"]:
            continue
        window = rows[step:step + 3]
        continuation = rows[step:step + 20]
        if len(continuation) < 20:
            continue
        if anchor_class == "CRITICAL":
            if not all(stable(row, baseline_z) for row in window + continuation):
                continue
        else:
            row = rows[step]
            if row.get("object_gripper_contact") is not False or row.get("object_support_contact") is not False:
                continue
            if not isinstance(row.get("object_eef_distance_m"), (int, float)) or float(row["object_eef_distance_m"]) <= 0.12:
                continue
            if any(stable(previous, baseline_z) for previous in rows[:step]):
                continue
        rank = sha256_bytes(f"{AA1_SALT}|{family}|{key}|{anchor_class}|{step}".encode())
        candidates.append({"step": step, "anchor_class": anchor_class, "selection_rank_sha256": rank, "baseline_z": baseline_z, "boundary": BOUNDARY[family], "candidate_digest": canonical_hash({"step": step, "anchor_class": anchor_class, "row": rows[step]})})
    return min(candidates, key=lambda row: (row["selection_rank_sha256"], row["step"])) if candidates else None


def make_env(config: dict[str, Any], suite: str, task_idx: int, state_id: int, counters: dict[str, int] | None = None):
    env, task_suite, task = Z1.make_libero_env(config, suite, task_idx)
    env.reset()
    initial_states = task_suite.get_task_init_states(task_idx)
    obs = env.set_init_state(initial_states[state_id])
    dummy = [0.0] * 6 + [-1.0]
    for _ in range(int(config["environment"]["dummy_wait_steps"])):
        obs = env.step(dummy)[0]
        if counters is not None:
            counters["env_step_calls"] += 1
    return env, task_suite, task, obs, initial_states


def capture_clean(config: dict[str, Any], family: str, canary: dict[str, Any], infer: Any, counters: dict[str, int]) -> dict[str, Any]:
    suite = str(canary["suite"])
    task_idx = int(canary["task_idx"])
    state_id = int(canary["state_id"])
    env, task_suite, task, obs, _initial_states = make_env(config, suite, task_idx, state_id, counters)
    try:
        binding = TAXONOMY.bind_object_taxonomy(env, bddl_path(env, task))
        if binding.get("status") != "PASS":
            return {"status": "PASS_AA1_PASSIVE_PIPELINE_NO_OBJECT_BINDING", "binding": binding, "rows": [], "actions": [], "candidates": {"critical": [], "noncritical": []}}
        target = str(binding["target_object_ids"][0])
        queue: list[tuple[np.ndarray, np.ndarray]] = []
        rows: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        boundary_states: dict[int, np.ndarray] = {}
        baseline_z: float | None = None
        horizon = HORIZONS[suite]
        for step in range(horizon):
            fresh = not queue
            if fresh:
                boundary_states[step] = Z1.snapshot_state(env)
                queue = model_pairs(infer, obs, str(task.language), family, counters)
            raw_action, final_action = queue.pop(0)
            current = telemetry(env, binding, target, counters)
            if baseline_z is None and isinstance(current.get("object_position"), list):
                baseline_z = float(current["object_position"][2])
            semantics = SEMANTICS.validate_action_pair(family, raw_action.tolist(), final_action.tolist(), raw_gripper=float(raw_action[-1]), final_gripper=float(final_action[-1]))
            if not semantics.get("accepted"):
                raise RuntimeError(f"ACTION_SEMANTICS_INVALID:{semantics.get('reason')}")
            row = {
                "step": step,
                "remaining_horizon": horizon - step,
                "clean_record_valid": True,
                "object_z_baseline_m": baseline_z,
                "model_boundary": fresh,
                "raw_action_7d": raw_action.tolist(),
                "env_action_7d": final_action.tolist(),
                "raw_gripper": float(raw_action[-1]),
                "env_gripper": float(final_action[-1]),
                "boundary_state_sha256": sha256_bytes(boundary_states[step].tobytes()) if fresh else None,
                "action_semantics": semantics,
                **current,
            }
            rows.append(row)
            actions.append({"raw": raw_action.tolist(), "final": final_action.tolist(), "boundary": fresh})
            obs = env.step(final_action.tolist())[0]
            counters["env_step_calls"] += 1
        if len(rows) != horizon:
            raise RuntimeError("AA1_CLEAN_TRAJECTORY_INCOMPLETE")
        critical = select_candidate(rows, actions, family, str(canary["canonical_parent_key"]), "CRITICAL")
        noncritical = select_candidate(rows, actions, family, str(canary["canonical_parent_key"]), "NONCRITICAL")
        return {
            "status": "PASS_AA1_PASSIVE_CLEAN_PIPELINE",
            "binding": binding,
            "target_object": target,
            "horizon": horizon,
            "rows": rows,
            "actions": actions,
            "boundary_states": boundary_states,
            "selected": {"critical": critical, "noncritical": noncritical},
            "candidate_counts": {
                "critical": 1 if critical is not None else 0,
                "noncritical": 0 if noncritical is None else 1,
            },
        }
    finally:
        env.close()


def replay_prefix(config: dict[str, Any], canary: dict[str, Any], actions: list[dict[str, Any]], anchor_step: int, expected_state: np.ndarray, counters: dict[str, int]):
    env, _task_suite, task, obs, _initial_states = make_env(config, str(canary["suite"]), int(canary["task_idx"]), int(canary["state_id"]), counters)
    for index in range(anchor_step):
        obs = env.step(actions[index]["final"])[0]
        counters["env_step_calls"] += 1
    actual = Z1.snapshot_state(env)
    if not np.array_equal(actual, expected_state):
        env.close()
        raise RuntimeError("AA1_ANCHOR_STATE_REPLAY_NOT_EXACT")
    return env, task, obs


def post_record(env: Any, binding: dict[str, Any], target: str, counters: dict[str, int]) -> dict[str, Any]:
    return telemetry(env, binding, target, counters)


def endpoint_label(clean_rows: list[dict[str, Any]], branch_rows: list[dict[str, Any]], dose: int, branch_valid: bool) -> str:
    if not branch_valid:
        return "AA_BRANCH_INVALID"
    required = dose + 10
    if len(branch_rows) < required or len(clean_rows) < required:
        return "AA_ENDPOINT_AMBIGUOUS"
    if any(row.get("post", {}).get("contact_telemetry_valid") is not True for row in branch_rows[:required]):
        return "AA_ENDPOINT_AMBIGUOUS"
    clean_stable = all(
        row.get("post", {}).get("object_gripper_contact") is True
        and row.get("post", {}).get("object_support_contact") is False
        for row in clean_rows[:required]
    )
    if not clean_stable:
        return "AA_NONCAUSAL_TERMINAL_RELEASE"
    for index in range(required - 1):
        current = branch_rows[index]["post"]
        following = branch_rows[index + 1]["post"]
        previous = branch_rows[index].get("pre", {})
        if previous.get("object_gripper_contact") is not True or current.get("object_gripper_contact") is not False or following.get("object_gripper_contact") is not False:
            continue
        if current.get("object_support_contact") is True or following.get("object_support_contact") is True:
            continue
        clean_position = clean_rows[index].get("post", {}).get("object_position")
        treatment_position = current.get("object_position")
        if not isinstance(clean_position, list) or not isinstance(treatment_position, list) or len(clean_position) != 3 or len(treatment_position) != 3:
            return "AA_ENDPOINT_AMBIGUOUS"
        drop = float(clean_position[2]) - float(treatment_position[2])
        displacement = float(np.linalg.norm(np.asarray(clean_position, dtype=float) - np.asarray(treatment_position, dtype=float)))
        if drop >= 0.02 or displacement >= 0.01:
            return "AA_V_PHYS"
    return "AA_NO_PREMATURE_LOSS"


def run_branch(config: dict[str, Any], family: str, canary: dict[str, Any], infer: Any, clean: dict[str, Any], anchor: dict[str, Any], dose: int, arm: str, counters: dict[str, int]) -> dict[str, Any]:
    anchor_step = int(anchor["step"])
    required = int(dose) + 10
    clean_actions = clean["actions"]
    if anchor_step + required > len(clean_actions):
        return {"status": "AA_BRANCH_INVALID", "arm": arm, "dose": dose, "error": "HORIZON_CENSORED", "endpoint_label": "AA_BRANCH_INVALID"}
    expected_state = clean["boundary_states"].get(anchor_step)
    if expected_state is None:
        return {"status": "AA_BRANCH_INVALID", "arm": arm, "dose": dose, "error": "ANCHOR_NOT_FRESH_BOUNDARY", "endpoint_label": "AA_BRANCH_INVALID"}
    env, task, obs = replay_prefix(config, canary, clean_actions, anchor_step, expected_state, counters)
    try:
        binding = clean["binding"]
        target = str(clean["target_object"])
        queue: list[tuple[np.ndarray, np.ndarray]] = []
        rows: list[dict[str, Any]] = []
        action_receipts: list[dict[str, Any]] = []
        action_reference_exact = True
        state_restore_exact = True
        for offset in range(required):
            step = anchor_step + offset
            pre = post_record(env, binding, target, counters)
            if arm == "CLEAN_REFERENCE":
                raw_action = np.asarray(clean_actions[step]["raw"], dtype=np.float32)
                final_action = np.asarray(clean_actions[step]["final"], dtype=np.float32)
            else:
                if not queue:
                    queue = model_pairs(infer, obs, str(task.language), family, counters)
                raw_action, final_action = queue.pop(0)
                reference_final = np.asarray(clean_actions[step]["final"], dtype=np.float32)
                arm_delta = float(np.max(np.abs(reference_final[:6] - final_action[:6])))
                action_reference_exact = action_reference_exact and arm_delta <= ARM_TOLERANCE
            if arm != "CLEAN_REFERENCE" and offset < dose:
                opened_raw, opened_final = command_open_action(family, raw_action.tolist(), final_action.tolist(), duration=dose)
                opened_raw = np.asarray(opened_raw, dtype=np.float32)
                opened_final = np.asarray(opened_final, dtype=np.float32)
                if not np.array_equal(opened_final[:6], final_action[:6]) or float(opened_final[-1]) != -1.0:
                    raise RuntimeError("AA1_OPEN_ACTION_CONTRACT_INVALID")
                action_receipts.append({"step": step, "raw_policy_action": raw_action.tolist(), "opened_raw_action": opened_raw.tolist(), "env_action": opened_final.tolist(), "arm_delta_linf": float(np.max(np.abs(final_action[:6] - opened_final[:6])))})
                action = opened_final
                counters["open_intervention_steps"] += 1
            else:
                action = final_action
            obs = env.step(action.tolist())[0]
            counters["env_step_calls"] += 1
            post = post_record(env, binding, target, counters)
            rows.append({"step": step, "pre": pre, "post": post, "action": action.tolist(), "is_open_step": arm != "CLEAN_REFERENCE" and offset < dose})
        valid = len(rows) == required and action_reference_exact and (len(action_receipts) == dose)
        label = "AA_NO_PREMATURE_LOSS" if arm == "CLEAN_REFERENCE" else endpoint_label(clean["branch_rows"], rows, dose, valid)
        return {
            "status": "PASS_AA1_ENGINEERING_BRANCH" if valid else "AA_BRANCH_INVALID",
            "arm": arm,
            "anchor_class": anchor["anchor_class"],
            "anchor_step": anchor_step,
            "dose": dose,
            "required_horizon_steps": required,
            "state_restore_exact": state_restore_exact,
            "exact_clean_action_reference": action_reference_exact,
            "exact_open_delivery": len(action_receipts) == dose,
            "open_intervention_steps": len(action_receipts),
            "treatment_receipts": action_receipts,
            "rows": rows,
            "endpoint_label": label,
        }
    finally:
        env.close()


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_json(args.protocol)
    plan = load_json(args.plan)
    aa0 = load_json(args.aa0)
    capacity = load_json(args.capacity)
    z1_config = load_json(args.z1_config)
    key = args.canonical_parent_key
    static = static_validate(args, protocol, plan, aa0, capacity, z1_config, key)
    canary = static["canary"]
    counters = {
        "model_inference_calls": 0,
        "env_step_calls": 0,
        "physical_telemetry_reads": 0,
        "open_intervention_steps": 0,
        "pgd_calls": 0,
        "attacked_env_steps": 0,
        "v_phys_reads": 0,
        "task_success_reads": 0,
        "attack_outcome_reads": 0,
        "eval160_reads": 0,
        "protected_reads": 0,
        "scientific_parent_exposure": 0,
        "aa2_exposure": 0,
    }
    receipt: dict[str, Any] = {
        "schema": "STAGE_AA_AA1_ENGINEERING_CANARY_CELL_RECEIPT_V1",
        "status": "RUNNING",
        "gate": protocol["gate"],
        "model_family": args.model_family,
        "canonical_parent_key": key,
        "suite": canary["suite"],
        "task_idx": canary["task_idx"],
        "state_id": canary["state_id"],
        "gpu_id": args.gpu_id,
        "canary_permanent_exclusion": True,
        "scientific_use": False,
        "runtime_counters": counters,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write(args.output, receipt)
    require_single_gpu(args.gpu_id)
    receipt["gpu"] = gpu_snapshot(args.gpu_id)
    Z1.configure_libero(z1_config)
    checkpoint = static["checkpoint"]
    if args.model_family == "M2_PI05_LIBERO":
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    receipt["runtime_environment"] = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "UNSET"),
    }
    if args.model_family == "M0_OPENVLA":
        infer, model, normalization = Z1.load_openvla(checkpoint, oft=False, suite=canary["suite"], return_chunk=True)
    elif args.model_family == "M1_OPENVLA_OFT":
        infer, model, normalization = Z1.load_openvla(checkpoint, oft=True, suite=canary["suite"], return_chunk=True)
    else:
        infer, model = Z1.load_pi05(checkpoint, return_chunk=True)
        normalization = {"checkpoint_mutated": False}
    try:
        clean = capture_clean(z1_config, args.model_family, canary, infer, counters)
        receipt["normalization"] = normalization
        receipt["checkpoint"] = checkpoint
        receipt["passive_clean"] = {key: value for key, value in clean.items() if key not in {"rows", "actions", "boundary_states", "binding"}}
        receipt["runtime_counters"] = counters
        atomic_write(args.output, receipt)
        selected = clean.get("selected", {})
        branches = []
        if selected.get("critical") is not None:
            critical = selected["critical"]
            critical_clean = run_branch(z1_config, args.model_family, canary, infer, {**clean, "branch_rows": []}, critical, 0, "CLEAN_REFERENCE", counters)
            # The clean branch is the counterfactual reference for critical doses.
            clean_with_reference = {**clean, "branch_rows": critical_clean["rows"]}
            branches.append(critical_clean)
            for dose in DOSES:
                branches.append(run_branch(z1_config, args.model_family, canary, infer, clean_with_reference, critical, dose, f"OPEN_T{dose}_CRITICAL", counters))
        if selected.get("noncritical") is not None:
            noncritical = selected["noncritical"]
            branches.append(run_branch(z1_config, args.model_family, canary, infer, {**clean, "branch_rows": clean["rows"][int(noncritical["step"]):]}, noncritical, 5, "OPEN_T5_NONCRITICAL_CONTROL", counters))
        expected = 5 if selected.get("critical") is not None and selected.get("noncritical") is not None else 0
        status = "PASS_AA1_ENGINEERING_CANARY_CELL" if len(branches) == expected == 5 and all(item.get("status") == "PASS_AA1_ENGINEERING_BRANCH" for item in branches) else ("PASS_AA1_PASSIVE_PIPELINE_NO_LEGAL_ENGINEERING_POINT" if not branches else "ENGINEERING_INVALID_AA1_BRANCH")
        receipt.update({
            "status": status,
            "selected_anchors": selected,
            "branch_count": len(branches),
            "expected_branch_count": expected,
            "branches": branches,
            "scientific_claim": "NONE_ENGINEERING_ONLY",
            "claim_boundary": "AA1 runtime/endpoint qualification only; canary permanently excluded from AA2-AA5.",
            "next_legal_action": "STOP_FOR_PI",
        })
        receipt["runtime_counters"] = counters
        atomic_write(args.output, receipt)
        if status == "ENGINEERING_INVALID_AA1_BRANCH":
            raise RuntimeError("AA1_BRANCH_QUALIFICATION_FAILED")
        return receipt
    finally:
        del model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--aa0", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--z1-config", type=Path, required=True)
    parser.add_argument("--canonical-parent-key", required=True)
    parser.add_argument("--model-family", choices=MODELS, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_cell(args)
        print(json.dumps({"status": result["status"], "model_family": args.model_family, "canonical_parent_key": args.canonical_parent_key, "branch_count": result.get("branch_count", 0)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": "STAGE_AA_AA1_ENGINEERING_CANARY_CELL_RECEIPT_V1",
            "status": "ENGINEERING_INVALID_AA1_CELL",
            "model_family": args.model_family,
            "canonical_parent_key": args.canonical_parent_key,
            "gpu_id": args.gpu_id,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "scientific_claim": "NONE_DUE_TO_ENGINEERING_INVALID",
            "next_legal_action": "STOP_FOR_PI",
        }
        atomic_write(args.output, failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
