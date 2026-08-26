#!/usr/bin/env python3
"""Qualify AA1R1 branch machinery on the three consumed engineering canaries."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HORIZONS = {"libero_10": 520, "libero_object": 280, "libero_spatial": 220}
AA1R1_SALT = "STAGE_AA_AA1R1_ENGINEERING_POINT_V1_20260826"
ACTION_DIM = 7
ARM_TOLERANCE = 1e-7
DOSES = (3, 5, 10)
QUEUE_LENGTH = {"M0_OPENVLA": 1, "M1_OPENVLA_OFT": 8, "M2_PI05_LIBERO": 5}
BOUNDARY = {
    "M0_OPENVLA": "FRESH_PER_STEP",
    "M1_OPENVLA_OFT": "FRESH_OFT_ACTION_QUEUE",
    "M2_PI05_LIBERO": "FRESH_PI05_REPLAN",
}


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"MODULE_SPEC_UNAVAILABLE:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AA1 = load_module(ROOT / "scripts/stage_aa/run_stage_aa1_engineering_canary.py", "aa1r1_aa1_runtime")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def step_unpack(result: Any) -> tuple[Any, bool]:
    if not isinstance(result, (tuple, list)) or len(result) < 3:
        raise RuntimeError("AA1R1_ENV_STEP_RETURN_INVALID")
    observation = result[0]
    if len(result) >= 5:
        done = bool(result[2]) or bool(result[3])
    else:
        done = bool(result[2])
    return observation, done


def finite_vector(value: Any, size: int) -> bool:
    if not isinstance(value, list) or len(value) != size:
        return False
    return all(math.isfinite(float(item)) for item in value)


def telemetry_legal(row: dict[str, Any]) -> bool:
    if row.get("contact_telemetry_valid") is not True:
        return False
    if not isinstance(row.get("object_identity"), str) or not row["object_identity"]:
        return False
    if not finite_vector(row.get("object_position"), 3) or not finite_vector(row.get("eef_position"), 3):
        return False
    distance = row.get("object_eef_distance_m")
    return isinstance(distance, (int, float)) and math.isfinite(float(distance))


def enumerate_engineering_points(rows: list[dict[str, Any]], actions: list[dict[str, Any]], boundary_states: dict[int, np.ndarray], family: str, key: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for step in range(max(0, len(rows) - 19)):
        future = rows[step:step + 20]
        if len(future) != 20 or not actions[step].get("boundary") or step not in boundary_states:
            continue
        if any(row.get("terminal_before") is True for row in future):
            continue
        if not all(telemetry_legal(row) for row in future):
            continue
        rank = sha256_bytes(f"{AA1R1_SALT}|{family}|{key}|ENGINEERING_INTERVENTION_POINT_A|{step}".encode())
        candidates.append({
            "step": step,
            "selection_rank_sha256": rank,
            "boundary": BOUNDARY[family],
            "remaining_horizon": int(rows[step]["remaining_horizon"]),
            "clean_boundary_state_sha256": sha256_bytes(boundary_states[step].tobytes()),
            "candidate_digest": canonical_hash({"step": step, "row": rows[step], "future_rows": future}),
        })
    return sorted(candidates, key=lambda row: (row["selection_rank_sha256"], row["step"]))


def choose_points(candidates: list[dict[str, Any]], family: str, key: str) -> dict[str, Any]:
    if not candidates:
        return {"status": "NO_ENGINEERING_INTERVENTION_POINT", "selection_mode": "ZERO_POINT", "point_a": None, "point_b": None, "candidate_count": 0, "candidates": []}
    point_a = dict(candidates[0])
    point_a["point_label"] = "ENGINEERING_INTERVENTION_POINT_A"
    if len(candidates) == 1:
        return {"status": "ENGINEERING_POINT_SELECTED", "selection_mode": "ONE_POINT_ONLY", "point_a": point_a, "point_b": None, "candidate_count": 1, "candidates": candidates}
    ranked_b = sorted(candidates[1:], key=lambda row: (sha256_bytes(f"{AA1R1_SALT}|{family}|{key}|ENGINEERING_INTERVENTION_POINT_B|{row['step']}".encode()), row["step"]))
    point_b = dict(ranked_b[0])
    point_b["point_label"] = "ENGINEERING_INTERVENTION_POINT_B"
    return {"status": "ENGINEERING_POINT_SELECTED", "selection_mode": "TWO_POINTS", "point_a": point_a, "point_b": point_b, "candidate_count": len(candidates), "candidates": candidates}


def capture_clean(config: dict[str, Any], family: str, canary: dict[str, Any], infer: Any, counters: dict[str, int]) -> dict[str, Any]:
    suite = str(canary["suite"])
    env, _task_suite, task, obs, _initial_states = AA1.make_env(config, suite, int(canary["task_idx"]), int(canary["state_id"]), counters)
    try:
        binding = AA1.TAXONOMY.bind_object_taxonomy(env, AA1.bddl_path(env, task))
        if binding.get("status") != "PASS":
            return {"status": "NO_OBJECT_BINDING", "binding": binding, "rows": [], "actions": [], "boundary_states": {}, "points": {}}
        target = str(binding["target_object_ids"][0])
        language = str(task.language)
        queue: list[tuple[np.ndarray, np.ndarray]] = []
        rows: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        boundary_states: dict[int, np.ndarray] = {}
        episode_done = False
        horizon = HORIZONS[suite]
        for step in range(horizon):
            fresh = not queue
            if fresh:
                boundary_states[step] = AA1.Z1.snapshot_state(env)
                queue = AA1.model_pairs(infer, obs, language, family, counters)
            raw_action, final_action = queue.pop(0)
            current = AA1.telemetry(env, binding, target, counters)
            semantics = AA1.SEMANTICS.validate_action_pair(family, raw_action.tolist(), final_action.tolist(), raw_gripper=float(raw_action[-1]), final_gripper=float(final_action[-1]))
            if not semantics.get("accepted"):
                raise RuntimeError(f"ACTION_SEMANTICS_INVALID:{semantics.get('reason')}")
            row = {
                "step": step,
                "remaining_horizon": horizon - step,
                "clean_record_valid": True,
                "terminal_before": episode_done,
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
            obs, episode_done = step_unpack(env.step(final_action.tolist()))
            row["terminal_after"] = episode_done
            counters["env_step_calls"] += 1
        points = enumerate_engineering_points(rows, actions, boundary_states, family, str(canary["canonical_parent_key"]))
        return {
            "status": "PASS_ENGINEERING_CLEAN_TRAJECTORY",
            "binding": binding,
            "target_object": target,
            "language": language,
            "horizon": horizon,
            "rows": rows,
            "actions": actions,
            "boundary_states": boundary_states,
            "points": choose_points(points, family, str(canary["canonical_parent_key"])),
            "clean_trajectory_digest": canonical_hash({"actions": actions, "rows": rows, "boundary_states": {str(k): sha256_bytes(v.tobytes()) for k, v in boundary_states.items()}}),
        }
    finally:
        env.close()


def replay_prefix(config: dict[str, Any], canary: dict[str, Any], actions: list[dict[str, Any]], anchor_step: int, expected_state: np.ndarray, counters: dict[str, int]):
    env, _task_suite, task, obs, _initial_states = AA1.make_env(config, str(canary["suite"]), int(canary["task_idx"]), int(canary["state_id"]), counters)
    episode_done = False
    for index in range(anchor_step):
        if episode_done:
            env.close()
            raise RuntimeError("AA1R1_PREFIX_TERMINAL_BEFORE_POINT")
        obs, episode_done = step_unpack(env.step(actions[index]["final"]))
        counters["env_step_calls"] += 1
    actual = AA1.Z1.snapshot_state(env)
    if not np.array_equal(actual, expected_state):
        env.close()
        raise RuntimeError("AA1R1_SNAPSHOT_RESTORE_NOT_EXACT")
    return env, task, obs, episode_done


def trace_digest(rows: list[dict[str, Any]]) -> str:
    return canonical_hash({"rows": rows})


def run_branch(config: dict[str, Any], family: str, canary: dict[str, Any], infer: Any, clean: dict[str, Any], point: dict[str, Any], dose: int | None, route_name: str, counters: dict[str, int]) -> dict[str, Any]:
    anchor_step = int(point["step"])
    required = 20 if dose is None else int(dose) + 10
    if anchor_step + required > len(clean["actions"]):
        return {"status": "AA1R1_ENGINEERING_BRANCH_INVALID", "route": route_name, "point": point, "error": "HORIZON_CENSORED"}
    expected_state = clean["boundary_states"].get(anchor_step)
    if expected_state is None:
        return {"status": "AA1R1_ENGINEERING_BRANCH_INVALID", "route": route_name, "point": point, "error": "POINT_STATE_MISSING"}
    env, task, obs, episode_done = replay_prefix(config, canary, clean["actions"], anchor_step, expected_state, counters)
    queue: list[tuple[np.ndarray, np.ndarray]] = []
    queue_boundary_steps: list[int] = []
    rows: list[dict[str, Any]] = []
    action_receipts: list[dict[str, Any]] = []
    arm_deltas: list[float] = []
    state_restore_exact = True
    try:
        for offset in range(required):
            step = anchor_step + offset
            if episode_done:
                return {"status": "AA1R1_ENGINEERING_BRANCH_INVALID", "route": route_name, "point": point, "error": "TERMINAL_BEFORE_REQUIRED_HORIZON", "rows": rows, "queue_boundary_steps": queue_boundary_steps}
            pre = AA1.telemetry(env, clean["binding"], str(clean["target_object"]), counters)
            is_clean = dose is None
            if is_clean:
                raw_action = np.asarray(clean["actions"][step]["raw"], dtype=np.float32)
                final_action = np.asarray(clean["actions"][step]["final"], dtype=np.float32)
            else:
                if not queue:
                    queue_boundary_steps.append(step)
                    queue = AA1.model_pairs(infer, obs, clean["language"], family, counters)
                raw_action, final_action = queue.pop(0)
                reference = np.asarray(clean["actions"][step]["final"], dtype=np.float32)
                arm_deltas.append(float(np.max(np.abs(reference[:6] - final_action[:6]))))
            if raw_action.size != ACTION_DIM or final_action.size != ACTION_DIM:
                raise RuntimeError("AA1R1_FINAL_ACTION_NOT_EXACTLY_SEVEN")
            if not np.isfinite(raw_action).all() or not np.isfinite(final_action).all():
                raise RuntimeError("AA1R1_ACTION_NONFINITE")
            if not is_clean and offset < int(dose):
                opened_raw, opened_final = AA1.command_open_action(family, raw_action.tolist(), final_action.tolist(), duration=int(dose))
                opened_raw = np.asarray(opened_raw, dtype=np.float32)
                opened_final = np.asarray(opened_final, dtype=np.float32)
                if not np.array_equal(opened_final[:6], final_action[:6]) or float(opened_final[-1]) != -1.0:
                    raise RuntimeError("AA1R1_OPEN_ACTION_CONTRACT_INVALID")
                action = opened_final
                action_receipts.append({"step": step, "raw_policy_action": raw_action.tolist(), "opened_raw_action": opened_raw.tolist(), "env_action": opened_final.tolist(), "arm_delta_linf": arm_deltas[-1] if arm_deltas else 0.0})
                counters["open_intervention_steps"] += 1
            else:
                action = final_action
            obs, episode_done = step_unpack(env.step(action.tolist()))
            counters["env_step_calls"] += 1
            post = AA1.telemetry(env, clean["binding"], str(clean["target_object"]), counters)
            rows.append({"step": step, "pre": pre, "post": post, "action": action.tolist(), "is_open_step": not is_clean and offset < int(dose or 0), "terminal_after": episode_done})
        expected_boundaries = [] if is_clean else list(range(anchor_step, anchor_step + required, QUEUE_LENGTH[family]))
        queue_reset_verified = True if is_clean else queue_boundary_steps == expected_boundaries
        telemetry_aligned = len(rows) == required and all(isinstance(row.get("pre"), dict) and isinstance(row.get("post"), dict) for row in rows)
        arm_preserved = all(delta <= ARM_TOLERANCE for delta in arm_deltas)
        exact_open_delivery = dose is None or (len(action_receipts) == int(dose) and [row["step"] for row in action_receipts] == list(range(anchor_step, anchor_step + int(dose))))
        valid = state_restore_exact and telemetry_aligned and queue_reset_verified and arm_preserved and exact_open_delivery
        endpoint_output = "AA_ENGINEERING_ENDPOINT_NOT_RUN" if dose is None else AA1.endpoint_label(clean.get("engineering_clean_rows", []), rows, int(dose), valid)
        return {
            "status": "PASS_AA1R1_ENGINEERING_BRANCH" if valid else "AA1R1_ENGINEERING_BRANCH_INVALID",
            "route": route_name,
            "point": point,
            "dose": dose,
            "required_horizon_steps": required,
            "state_restore_exact": state_restore_exact,
            "exact_open_delivery": exact_open_delivery,
            "arm_preserved": arm_preserved,
            "max_arm_delta_linf": max(arm_deltas, default=0.0),
            "queue_reset_verified": queue_reset_verified,
            "queue_boundary_steps": queue_boundary_steps,
            "expected_queue_boundary_steps": expected_boundaries,
            "telemetry_aligned": telemetry_aligned,
            "open_intervention_steps": len(action_receipts),
            "action_receipts": action_receipts,
            "endpoint_state_machine_output": endpoint_output,
            "rows": rows,
            "trace_digest": trace_digest(rows),
        }
    finally:
        env.close()


def clean_determinism(config: dict[str, Any], family: str, canary: dict[str, Any], infer: Any, clean: dict[str, Any], point: dict[str, Any], counters: dict[str, int], label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    first = run_branch(config, family, canary, infer, clean, point, None, f"ENGINEERING_CLEAN_REPLAY_{label}_1", counters)
    second = run_branch(config, family, canary, infer, clean, point, None, f"ENGINEERING_CLEAN_REPLAY_{label}_2", counters)
    deterministic = first.get("status") == "PASS_AA1R1_ENGINEERING_BRANCH" and second.get("status") == "PASS_AA1R1_ENGINEERING_BRANCH" and first.get("trace_digest") == second.get("trace_digest")
    return {"status": "PASS_AA1R1_CLEAN_REPLAY_DETERMINISTIC" if deterministic else "AA1R1_CLEAN_REPLAY_NONDETERMINISTIC", "point": point, "first": first, "second": second, "trace_equal": first.get("trace_digest") == second.get("trace_digest")}, first


def validate_protocol(protocol: dict[str, Any], aa0: dict[str, Any], aa1_protocol: dict[str, Any], plan: dict[str, Any]) -> None:
    if protocol.get("status") != "STAGE_AA_AA1R1_ENGINEERING_BRANCH_QUALIFICATION_AUTHORIZED":
        raise RuntimeError("AA1R1_PROTOCOL_NOT_AUTHORIZED")
    if protocol.get("aa0_scientific_contract_immutable") is not True or protocol.get("aa2_authorized") is not False:
        raise RuntimeError("AA1R1_SCIENTIFIC_FIREWALL_INVALID")
    if aa0.get("status") != "STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_FROZEN_STOP_FOR_PI":
        raise RuntimeError("AA0_PROTOCOL_NOT_FROZEN")
    if aa1_protocol.get("status") != "STAGE_AA_AA1_ENGINEERING_RUNTIME_QUALIFICATION_AUTHORIZED":
        raise RuntimeError("AA1_PROTOCOL_NOT_FROZEN")
    if plan.get("status") != "STAGE_AA_AA1_ENGINEERING_CANARY_PLAN_FROZEN":
        raise RuntimeError("AA1_PLAN_NOT_FROZEN")
    if protocol.get("canaries") != ["libero_10/task_04/state_20", "libero_object/task_02/state_42", "libero_spatial/task_05/state_34"]:
        raise RuntimeError("AA1R1_CANARY_SET_INVALID")


def self_test() -> None:
    rows = []
    actions = []
    for step in range(25):
        rows.append({
            "step": step,
            "remaining_horizon": 25 - step,
            "terminal_before": False,
            "contact_telemetry_valid": True,
            "object_identity": "mock_object",
            "object_position": [0.1, 0.2, 0.3],
            "eef_position": [0.1, 0.2, 0.3],
            "object_eef_distance_m": 0.01,
        })
        actions.append({"boundary": step % 5 == 0, "raw": [0.0] * 7, "final": [0.0] * 6 + [1.0]})
    states = {step: np.zeros(4, dtype=np.float32) for step in (0, 5, 10)}
    candidates = enumerate_engineering_points(rows, actions, states, "M2_PI05_LIBERO", "mock/key")
    selected = choose_points(candidates, "M2_PI05_LIBERO", "mock/key")
    assert selected["status"] == "ENGINEERING_POINT_SELECTED"
    assert selected["point_a"] is not None and selected["point_b"] is not None
    assert telemetry_legal(rows[0])
    rows[0]["object_eef_distance_m"] = float("nan")
    assert not telemetry_legal(rows[0])
    raw = [0.1] * 6 + [0.5]
    final = [0.1] * 6 + [0.5]
    _opened_raw, opened_final = AA1.command_open_action("M0_OPENVLA", raw, final, 3)
    assert opened_final[:6] == tuple(final[:6]) and opened_final[-1] == -1.0
    print(json.dumps({"status": "AA1R1_STATIC_MOCK_PASS", "candidate_count": len(candidates), "selection_mode": selected["selection_mode"]}, sort_keys=True))


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_json(args.protocol)
    aa1_protocol = load_json(args.aa1_protocol)
    plan = load_json(args.plan)
    aa0 = load_json(args.aa0)
    capacity = load_json(args.capacity)
    z1_config = load_json(args.z1_config)
    validate_protocol(protocol, aa0, aa1_protocol, plan)
    legacy_args = SimpleNamespace(model_family=args.model_family, gpu_id=args.gpu_id)
    static = AA1.static_validate(legacy_args, aa1_protocol, plan, aa0, capacity, z1_config, args.canonical_parent_key)
    canary = static["canary"]
    counters = {"model_inference_calls": 0, "env_step_calls": 0, "physical_telemetry_reads": 0, "open_intervention_steps": 0, "pgd_calls": 0, "attacked_env_steps": 0, "v_phys_reads": 0, "task_success_reads": 0, "attack_outcome_reads": 0, "eval160_reads": 0, "protected_reads": 0, "scientific_parent_exposure": 0, "aa2_exposure": 0}
    receipt = {"schema": "STAGE_AA_AA1R1_ENGINEERING_BRANCH_CELL_RECEIPT_V1", "status": "RUNNING", "gate": protocol["gate"], "model_family": args.model_family, "canonical_parent_key": args.canonical_parent_key, "gpu_id": args.gpu_id, "canary_permanent_exclusion": True, "scientific_use": False, "runtime_counters": counters, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_write(args.output, receipt)
    AA1.require_single_gpu(args.gpu_id)
    receipt["gpu"] = AA1.gpu_snapshot(args.gpu_id)
    AA1.Z1.configure_libero(z1_config)
    if args.model_family == "M2_PI05_LIBERO":
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    receipt["runtime_environment"] = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "UNSET")}
    checkpoint = static["checkpoint"]
    if args.model_family == "M0_OPENVLA":
        infer, model, normalization = AA1.Z1.load_openvla(checkpoint, oft=False, suite=canary["suite"], return_chunk=True)
    elif args.model_family == "M1_OPENVLA_OFT":
        infer, model, normalization = AA1.Z1.load_openvla(checkpoint, oft=True, suite=canary["suite"], return_chunk=True)
    else:
        infer, model = AA1.Z1.load_pi05(checkpoint, return_chunk=True)
        normalization = {"checkpoint_mutated": False}
    try:
        clean = capture_clean(z1_config, args.model_family, canary, infer, counters)
        receipt.update({"checkpoint": checkpoint, "normalization": normalization, "engineering_point_selection": clean.get("points"), "clean_trajectory_digest": clean.get("clean_trajectory_digest"), "runtime_counters": counters})
        atomic_write(args.point_ledger, {"schema": "STAGE_AA_AA1R1_ENGINEERING_POINT_LEDGER_CELL_V1", "status": "STAGE_AA_AA1R1_ENGINEERING_POINTS_SEALED_BEFORE_OPEN", "gate": protocol["gate"], "model_family": args.model_family, "canonical_parent_key": args.canonical_parent_key, "selection_salt": AA1R1_SALT, "scientific_anchor_evaluation": "NOT_EVALUATED_BY_DESIGN", "engineering_points": clean.get("points"), "clean_trajectory_digest": clean.get("clean_trajectory_digest"), "runtime_source_checkpoint": checkpoint})
        selected = clean.get("points", {})
        receipt["point_ledger_sealed"] = True
        if selected.get("point_a") is None:
            receipt.update({"status": "AA1R1_ENGINEERING_HOLD_NO_LEGAL_POINT", "branches": [], "branch_count": 0, "next_legal_action": "STOP_FOR_PI", "scientific_claim": "NONE_ENGINEERING_ONLY"})
            receipt["runtime_counters"] = counters
            atomic_write(args.output, receipt)
            return receipt
        determinism: dict[str, Any] = {}
        clean_refs: dict[str, dict[str, Any]] = {}
        for label, point in (("A", selected["point_a"]), ("B", selected.get("point_b"))):
            if point is None:
                continue
            result, clean_ref = clean_determinism(z1_config, args.model_family, canary, infer, clean, point, counters, label)
            determinism[label] = result
            if result["status"] != "PASS_AA1R1_CLEAN_REPLAY_DETERMINISTIC":
                receipt.update({"status": "AA1R1_ENGINEERING_HOLD_CLEAN_REPLAY", "clean_replay": determinism, "branches": [], "branch_count": 0, "next_legal_action": "STOP_FOR_PI", "scientific_claim": "NONE_ENGINEERING_ONLY"})
                receipt["runtime_counters"] = counters
                atomic_write(args.output, receipt)
                return receipt
            clean_refs[label] = clean_ref
        # Endpoint helper consumes clean rows; it is used only as an execution trace, never as a scientific label.
        branches: list[dict[str, Any]] = []
        clean["engineering_clean_rows"] = clean_refs["A"]["rows"]
        for dose in DOSES:
            branches.append(run_branch(z1_config, args.model_family, canary, infer, clean, selected["point_a"], dose, f"ENGINEERING_OPEN_T{dose}_A", counters))
        if selected.get("point_b") is not None:
            clean["engineering_clean_rows"] = clean_refs["B"]["rows"]
            branches.append(run_branch(z1_config, args.model_family, canary, infer, clean, selected["point_b"], 5, "ENGINEERING_OPEN_T5_B", counters))
        all_pass = all(branch.get("status") == "PASS_AA1R1_ENGINEERING_BRANCH" for branch in branches)
        receipt.update({"status": "PASS_AA1R1_ENGINEERING_BRANCH_CELL" if all_pass else "AA1R1_ENGINEERING_HOLD_BRANCH_INVALID", "clean_replay": determinism, "branches": branches, "branch_count": len(branches), "branch_pass_count": sum(branch.get("status") == "PASS_AA1R1_ENGINEERING_BRANCH" for branch in branches), "next_legal_action": "STOP_FOR_PI", "scientific_claim": "NONE_ENGINEERING_ONLY", "claim_boundary": "AA1R1 engineering branch-path qualification only; no scientific result."})
        receipt["runtime_counters"] = counters
        atomic_write(args.output, receipt)
        return receipt
    finally:
        del model


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--aa1-protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--aa0", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--z1-config", type=Path, required=True)
    parser.add_argument("--canonical-parent-key", required=True)
    parser.add_argument("--model-family", choices=("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO"), required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--point-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_cell(args)
        print(json.dumps({"status": result["status"], "model_family": args.model_family, "canonical_parent_key": args.canonical_parent_key, "branch_count": result.get("branch_count", 0)}, sort_keys=True))
        return 0 if result["status"].startswith("PASS_") else 1
    except Exception as exc:
        failure = {"schema": "STAGE_AA_AA1R1_ENGINEERING_BRANCH_CELL_RECEIPT_V1", "status": "AA1R1_ENGINEERING_HOLD_RUNTIME_ERROR", "model_family": args.model_family, "canonical_parent_key": args.canonical_parent_key, "gpu_id": args.gpu_id, "error": {"type": type(exc).__name__, "message": str(exc)}, "scientific_claim": "NONE_DUE_TO_ENGINEERING_HOLD", "next_legal_action": "STOP_FOR_PI"}
        atomic_write(args.output, failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
