#!/usr/bin/env python3
"""Run the V1.4 matched-action primary physical diagnostic for one parent.

Gate B is intentionally separate from the sealed V1.3.4 runner.  It consumes
only Gate-A snapshot packages and canonical clean action windows.  There is no
policy decode or fresh-render observation in the primary window.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from gripper_attack.stage_v_causal_observation_snapshot import (  # noqa: E402
    CausalSnapshotError,
    assert_exact,
    capture_runtime_state,
    capture_simulator_state,
    load_snapshot,
    matched_action,
    restore_rng_state,
)
from gripper_attack.stage_v_canonical_execution_core import canonical_sha256, canonical_value  # noqa: E402
from gripper_attack.stage_v_m3_5_physical_taxonomy import (  # noqa: E402
    bind_object_taxonomy,
    evaluate_treatment_compliance,
    telemetry_from_env,
    v_phys_label,
)
from scripts.detector_v5.run_stage_v_canonical_clean import (  # noqa: E402
    _load_external_modules,
    _load_policy,
    _write_runtime_binding_receipt,
)
from scripts.detector_v5.run_stage_v_m3_5_intervention_parent import (  # noqa: E402
    HORIZONS,
    M35RunnerError,
    _new_env,
)


DOSES = {"T3": 3, "T5": 5, "T10": 10}
H_PHYS = 10
REPETITIONS = 3
EXPECTED_COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}
FAILURE_CLASSES = {"GRIPPER_CONTACT_LOSS", "PREMATURE_OBJECT_RELEASE", "OBJECT_DROP"}
BINARY_CLASSES = {"V_PHYS", "NO_PHYSICAL_VULNERABILITY"}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise M35RunnerError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise M35RunnerError(f"JSONL_OBJECT_REQUIRED:{path}:{line_number}")
        rows.append(dict(value))
    return rows


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes((json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_bytes("".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode("utf-8"))


def _array_sha(value: Any) -> str:
    array = np.asarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _validate_protocol(protocol: Mapping[str, Any], args: argparse.Namespace) -> None:
    if protocol.get("schema") != "STAGE_V_M3_5_DIAGNOSTIC_PROTOCOL_V1_4_GATE_B" or protocol.get("version") != "V1.4-GATE-B":
        raise M35RunnerError("V1_4_GATE_B_PROTOCOL_INVALID")
    if protocol.get("status") != "FROZEN_RUNTIME_AUTHORIZED" or protocol.get("runtime_authorized") is not True:
        raise M35RunnerError("V1_4_GATE_B_PROTOCOL_NOT_FROZEN_OR_AUTHORIZED")
    binding = protocol.get("source_binding", {})
    if str(binding.get("runtime_commit")) != str(args.source_commit) or str(binding.get("runtime_tree")) != str(args.source_tree):
        raise M35RunnerError("V1_4_GATE_B_SOURCE_BINDING_MISMATCH")
    operation = protocol.get("operation", {})
    if operation.get("fresh_render_primary_consumption") != "HARD_STOP" or operation.get("fresh_render_equality_gate_used") is not False:
        raise M35RunnerError("FRESH_RENDER_PRIMARY_CONTRACT_INVALID")
    if operation.get("native_closed_loop_policy_in_primary_window") is not False:
        raise M35RunnerError("NATIVE_POLICY_PRIMARY_WINDOW_FORBIDDEN")
    matrix = protocol.get("matrix", {})
    if matrix.get("repetitions") != REPETITIONS or matrix.get("conditions") != ["CONTROL", "T3", "T5", "T10"]:
        raise M35RunnerError("GATE_B_MATRIX_INVALID")
    if protocol.get("protected_counters") != EXPECTED_COUNTERS:
        raise M35RunnerError("PROTECTED_COUNTER_CONTRACT_INVALID")


def _validate_gate_a(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    receipt_path = root / "M3_5_V1_4_GATE_A_RECEIPT.json"
    if not receipt_path.is_file():
        raise M35RunnerError("GATE_A_RECEIPT_MISSING")
    receipt = _load_json(receipt_path)
    if receipt.get("status") != "PASS" or receipt.get("intervention_executed") is not False or receipt.get("protected_counters") != EXPECTED_COUNTERS:
        raise M35RunnerError("GATE_A_NOT_PASS_OR_NOT_ZERO_TREATMENT")
    if receipt.get("canonical_parent_key") != args.parent_key or int(receipt.get("snapshot_count", -1)) != 24:
        raise M35RunnerError("GATE_A_IDENTITY_OR_SNAPSHOT_COUNT_INVALID")
    audit_path = root / "M3_5_V1_4_GATE_A_INDEPENDENT_AUDIT.json"
    if not audit_path.is_file() or _load_json(audit_path).get("status") != "PASS":
        raise M35RunnerError("GATE_A_INDEPENDENT_AUDIT_NOT_PASS")
    return receipt


def _aperture(obs: Any, env: Any) -> float | None:
    if isinstance(obs, Mapping):
        for key in ("robot0_gripper_qpos", "gripper_qpos"):
            if key in obs:
                try:
                    values = [float(item) for item in obs[key]]
                    if values and all(math.isfinite(item) for item in values):
                        return float(sum(abs(item) for item in values))
                except (TypeError, ValueError):
                    pass
    try:
        values = [float(item) for item in np.asarray(env.sim.data.qpos[-2:]).reshape(-1)]
        return float(sum(abs(item) for item in values)) if values and all(math.isfinite(item) for item in values) else None
    except (AttributeError, TypeError, ValueError):
        return None


def _position(row: Mapping[str, Any]) -> list[float] | None:
    value = row.get("post_object_position")
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _contact_loss(control: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]] | None = None) -> int | None:
    count = 0
    for index, row in enumerate(control):
        if treatment is None:
            lost = row.get("post_object_gripper_contact") is not True
        else:
            lost = control[index].get("post_object_gripper_contact") is True and row.get("post_object_gripper_contact") is not True
        count = count + 1 if lost else 0
        if count >= 2:
            return index - 1
    return None


def _physical_outcome(branch: Mapping[str, Any], *, required_steps: int, reference: Mapping[str, Any] | None = None) -> dict[str, Any]:
    outcome = {"class": "PHYSICAL_AMBIGUITY_ABSTAIN", "failure_latency_steps": None, "required_horizon_steps": int(required_steps), "predicate_evidence": {}}
    if branch.get("status") != "PASS" or branch.get("state_restore_exact") is not True or branch.get("causal_input_binding_pass") is not True or int(branch.get("available_horizon_steps", -1)) < int(required_steps):
        return {**outcome, "class": "HORIZON_CENSORED_ABSTAIN"}
    rows = list(branch.get("rows", []))[:required_steps]
    if len(rows) != required_steps or any(row.get("post_contact_telemetry_valid") is not True for row in rows):
        return outcome
    if reference is None:
        if branch.get("control_action_reference_exact") is not True:
            return {**outcome, "class": "CONTROL_REPLAY_DIVERGENCE_ABSTAIN"}
        loss = _contact_loss(rows)
        return {**outcome, "class": "GRIPPER_CONTACT_LOSS", "failure_latency_steps": loss} if loss is not None else {**outcome, "class": "NO_PHYSICAL_FAILURE"}
    control_rows = list(reference.get("rows", []))[:required_steps]
    if len(control_rows) != required_steps or any(row.get("post_contact_telemetry_valid") is not True for row in control_rows):
        return outcome
    for index, (control, treatment) in enumerate(zip(control_rows, rows)):
        left, right = _position(control), _position(treatment)
        if left is None or right is None:
            return outcome
        displacement = math.sqrt(sum((right[axis] - left[axis]) ** 2 for axis in range(3)))
        z_drop = left[2] - right[2]
        if z_drop >= 0.02 and treatment.get("post_object_gripper_contact") is not True and treatment.get("post_object_support_contact") is not True:
            return {**outcome, "class": "OBJECT_DROP", "failure_latency_steps": index, "predicate_evidence": {"matched_z_drop_m": z_drop}}
        if displacement >= 0.01 and control.get("post_object_gripper_contact") is True and treatment.get("post_object_gripper_contact") is not True:
            return {**outcome, "class": "PREMATURE_OBJECT_RELEASE", "failure_latency_steps": index, "predicate_evidence": {"matched_displacement_m": displacement}}
    loss = _contact_loss(rows, control_rows)
    return {**outcome, "class": "GRIPPER_CONTACT_LOSS", "failure_latency_steps": loss} if loss is not None else {**outcome, "class": "NO_PHYSICAL_FAILURE"}


def _pair_label(control: Mapping[str, Any], treatment: Mapping[str, Any], dose: int) -> dict[str, Any]:
    required = int(dose) + H_PHYS
    control_outcome = _physical_outcome(control, required_steps=required)
    treatment_outcome = _physical_outcome(treatment, required_steps=required, reference=control)
    control_class = str(control_outcome["class"])
    treatment_class = str(treatment_outcome["class"])
    control_valid = control_class == "NO_PHYSICAL_FAILURE"
    treatment_valid = treatment.get("treatment_compliant") is True and treatment_class in FAILURE_CLASSES | {"NO_PHYSICAL_FAILURE"}
    f_control = 1 if control_class in FAILURE_CLASSES else (0 if control_valid else None)
    f_open = 1 if treatment_class in FAILURE_CLASSES else (0 if treatment_valid else None)
    return {
        "control_valid": control_valid,
        "treatment_valid": treatment_valid,
        "control_physical_class": control_class,
        "treatment_physical_class": treatment_class,
        "control_physical_outcome": control_outcome,
        "treatment_physical_outcome": treatment_outcome,
        "f_control": f_control,
        "f_open": f_open,
        "label_class": v_phys_label(control_valid=control_valid, treatment_valid=treatment_valid, f_control=f_control, f_open=f_open),
        "dose_steps": int(dose),
        "H_phys": H_PHYS,
        "required_horizon_steps": required,
    }


def _branch_id(parent_key: str, probe_id: str, repetition: int, arm: str) -> str:
    identity = f"M35_V1_4::{parent_key}::{probe_id}::R{int(repetition)}::{arm}"
    return f"m35-v14-{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


def _restore_probe_env(*, snapshot_root: Path, gate_a_root: Path, OffScreenRenderEnv: Any, bddl: str, horizon: int, init_state: Any, args: argparse.Namespace, output_dir: Path, clean_actions: Sequence[Sequence[float]], model: Any, adapter: Any) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    loaded = load_snapshot(snapshot_root, materialize_torch=True)
    manifest = loaded["manifest"]
    payload = loaded["payload"]
    step = int(payload["probe"]["step"])
    if manifest.get("binding", {}).get("parent_key") != args.parent_key or manifest.get("binding", {}).get("probe_id") != payload["probe"].get("probe_id"):
        raise M35RunnerError("SNAPSHOT_IDENTITY_BINDING_INVALID")
    restore_rng_state(payload["episode_start_rng_state"])
    env, obs = _new_env(OffScreenRenderEnv, bddl, horizon, args.gpu, copy.deepcopy(init_state), args, output_dir)
    for index in range(step):
        if index >= len(clean_actions):
            env.close()
            raise M35RunnerError(f"CLEAN_PREFIX_MISSING:{index}")
        obs, _reward, done, _info = env.step(list(clean_actions[index]))
        if bool(done):
            env.close()
            raise M35RunnerError(f"CLEAN_PREFIX_TERMINATED:{step}")
    simulator = capture_simulator_state(env)
    natural_runtime = capture_runtime_state(env, model=model, adapter=adapter)
    restore_rng_state(payload["required_rng_state"])
    runtime = capture_runtime_state(env, model=model, adapter=adapter)
    try:
        assert_exact(simulator, payload["full_simulator_state"], label="simulator_state")
        assert_exact(runtime, payload["controller_and_wrapper_runtime_state"], label="runtime_state")
    except Exception:
        env.close()
        raise
    return env, obs, payload, {"natural_runtime": natural_runtime, "bound_runtime": runtime}


def _run_branch(*, snapshot_root: Path, gate_a_root: Path, OffScreenRenderEnv: Any, bddl: str, horizon: int, init_state: Any, args: argparse.Namespace, output_dir: Path, clean_actions: Sequence[Sequence[float]], model: Any, adapter: Any, arm: str, dose: int, probe_id: str, repetition: int) -> dict[str, Any]:
    env = None
    rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    treatment_receipts: list[dict[str, Any]] = []
    try:
        env, obs, payload, runtime_receipt = _restore_probe_env(snapshot_root=snapshot_root, gate_a_root=gate_a_root, OffScreenRenderEnv=OffScreenRenderEnv, bddl=bddl, horizon=horizon, init_state=init_state, args=args, output_dir=output_dir, clean_actions=clean_actions, model=model, adapter=adapter)
        reference_window = payload.get("clean_reference_action_window")
        required_steps = int(dose) + H_PHYS if arm != "CONTROL" else max(DOSES.values()) + H_PHYS
        if not isinstance(reference_window, list) or len(reference_window) < required_steps:
            raise M35RunnerError("REFERENCE_ACTION_WINDOW_INCOMPLETE")
        binding = bind_object_taxonomy(env, Path(bddl))
        target_object_id = str(payload["probe"].get("object_identity", ""))
        if binding.get("status") != "PASS" or target_object_id not in binding.get("target_object_ids", []):
            raise M35RunnerError("OBJECT_TAXONOMY_BINDING_INVALID")
        for relative in range(required_steps):
            reference = reference_window[relative]
            forced_open = arm != "CONTROL" and relative < int(dose)
            action = matched_action(reference, forced_open=forced_open)
            actual_raw = list(action["raw_policy_action"])
            actual_env = list(action["env_action"])
            if not forced_open and (actual_raw != list(reference["raw_policy_action"]) or actual_env != list(reference["env_action"])):
                raise M35RunnerError(f"REFERENCE_ACTION_MISMATCH:{relative}")
            if forced_open and (actual_raw[:6] != list(reference["raw_policy_action"])[:6] or actual_env[:6] != list(reference["env_action"])[:6] or float(action["arm_delta_linf"]) != 0.0):
                raise M35RunnerError(f"ARM_ISOLATION_MISMATCH:{relative}")
            pre_aperture = _aperture(obs, env)
            telemetry = telemetry_from_env(env, binding, target_object_id=target_object_id)
            row = {
                "step": int(payload["probe"]["step"]) + relative,
                "relative_step": relative,
                "arm": arm,
                "raw_policy_action": actual_raw,
                "normalized_action": list(action.get("normalized_action", actual_raw)),
                "env_action": actual_env,
                "reference_raw_action": list(reference["raw_policy_action"]),
                "reference_env_action": list(reference["env_action"]),
                "reference_action_sha256": reference.get("action_sha256"),
                "action_sha256": canonical_sha256({"raw": actual_raw, "env": actual_env}),
                "arm_source": "MATCHED_CANONICAL_REFERENCE" if not forced_open else "MATCHED_CANONICAL_ARM_FORCED_OPEN_GRIPPER",
                "arm_delta_linf": float(action.get("arm_delta_linf", 0.0)),
                "gripper_delta_env": float(action.get("gripper_delta_env", 0.0)),
                "pre_aperture": pre_aperture,
                "gripper_aperture": pre_aperture,
                **{key: value for key, value in telemetry.items() if key != "schema"},
            }
            if forced_open:
                treatment_receipts.append({
                    "requested_dose": int(dose),
                    "relative_step": relative,
                    "raw_policy_action": actual_raw,
                    "normalized_action": list(action.get("normalized_action", actual_raw)),
                    "env_action": actual_env,
                    "reference_arm_action": list(reference["env_action"][:6]),
                    "actual_arm_action": actual_env[:6],
                    "arm_delta_linf": float(action.get("arm_delta_linf", math.inf)),
                    "pre_aperture": pre_aperture,
                })
            obs, reward, done, _info = env.step(actual_env)
            post_aperture = _aperture(obs, env)
            post_telemetry = telemetry_from_env(env, binding, target_object_id=target_object_id)
            row.update({f"post_{key}": value for key, value in post_telemetry.items() if key != "schema"})
            row["post_aperture"] = post_aperture
            row["reward"] = float(reward) if isinstance(reward, (int, float)) and math.isfinite(float(reward)) else None
            row["done"] = bool(done)
            if forced_open:
                treatment_receipts[-1]["post_aperture"] = post_aperture
            rows.append(row)
            actions.append({key: value for key, value in action.items()})
            if bool(done) and len(rows) < required_steps:
                raise M35RunnerError("HORIZON_CENSORED_PRIMARY_WINDOW")
        compliance = evaluate_treatment_compliance(treatment_receipts, expected_steps=int(dose)) if arm != "CONTROL" else {"treatment_compliant": True, "compliance_reason": "CONTROL", "delivered_open_steps": 0, "expected_open_steps": 0}
        return {
            "status": "PASS",
            "schema": "STAGE_V_M3_5_V1_4_BRANCH_RESULT_V1",
            "arm": arm,
            "dose_steps": int(dose),
            "repetition": int(repetition),
            "probe_id": probe_id,
            "probe_step": int(payload["probe"]["step"]),
            "state_restore_exact": True,
            "runtime_state_exact": True,
            "causal_input_binding_pass": True,
            "primary_input_authority": "loaded_frozen_canonical_bytes",
            "fresh_render_equality_gate_used": False,
            "fresh_render_primary_consumption": False,
            "native_policy_calls_in_primary_window": 0,
            "reference_action_exact": True,
            "control_action_reference_exact": True,
            "rows": rows,
            "actions": actions,
            "treatment_receipts": treatment_receipts,
            "treatment_compliance": compliance,
            "treatment_compliant": bool(compliance.get("treatment_compliant", False)),
            "available_horizon_steps": len(rows),
            "required_physical_steps": required_steps,
            "target_object_id": target_object_id,
            "runtime_state_sha256": canonical_sha256(canonical_value(runtime_receipt["bound_runtime"])),
            "natural_runtime_state_sha256": canonical_sha256(canonical_value(runtime_receipt["natural_runtime"])),
            "protected_counters": dict(EXPECTED_COUNTERS),
        }
    except Exception as exc:
        return {
            "status": "FAIL",
            "schema": "STAGE_V_M3_5_V1_4_BRANCH_RESULT_V1",
            "arm": arm,
            "dose_steps": int(dose),
            "repetition": int(repetition),
            "probe_id": probe_id,
            "error": f"{type(exc).__name__}:{exc}",
            "state_restore_exact": False,
            "runtime_state_exact": False,
            "causal_input_binding_pass": False,
            "primary_input_authority": "loaded_frozen_canonical_bytes",
            "fresh_render_equality_gate_used": False,
            "fresh_render_primary_consumption": False,
            "native_policy_calls_in_primary_window": 0,
            "rows": rows,
            "actions": actions,
            "treatment_receipts": treatment_receipts,
            "treatment_compliant": False,
            "available_horizon_steps": len(rows),
            "required_physical_steps": int(dose) + H_PHYS if arm != "CONTROL" else max(DOSES.values()) + H_PHYS,
            "protected_counters": dict(EXPECTED_COUNTERS),
        }
    finally:
        if env is not None:
            env.close()


def _record_branch(branch: Mapping[str, Any], *, parent_key: str, probe_id: str, probe_step: int, repetition: int, arm: str, pair: Mapping[str, Any] | None = None, shared_control: Mapping[str, Any] | None = None) -> dict[str, Any]:
    branch_id = _branch_id(parent_key, probe_id, repetition, arm)
    return {
        "schema": "STAGE_V_M3_5_V1_4_PHYSICAL_EXECUTION_V1",
        "canonical_parent_key": parent_key,
        "probe_id": probe_id,
        "probe_step": int(probe_step),
        "repetition": int(repetition),
        "arm": arm,
        "branch_id": branch_id,
        "branch_result_sha256": canonical_sha256(canonical_value(branch)),
        "shared_control_branch_id": shared_control.get("branch_id") if shared_control else None,
        "shared_control_result_sha256": shared_control.get("branch_result_sha256") if shared_control else None,
        "branch": dict(branch),
        "pair": dict(pair) if pair is not None else None,
        "protected_counters": dict(EXPECTED_COUNTERS),
    }


def _seal(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            rows.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": _sha_file(path)})
    (root / "SHA256SUMS").write_bytes("".join(f"{row['sha256']}  {row['path']}\n" for row in rows).encode("utf-8"))
    sums_sha = _sha_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_bytes(f"{sums_sha}  SHA256SUMS\n".encode("utf-8"))
    return {"files": rows, "sha256s_sha256": sums_sha}


def run(args: argparse.Namespace) -> int:
    protocol = _load_json(args.protocol)
    _validate_protocol(protocol, args)
    gate_a_root = args.gate_a_root.resolve()
    gate_a_receipt = _validate_gate_a(gate_a_root, args)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise M35RunnerError(f"REFUSE_OVERWRITE:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol.resolve(), output_dir / "M3_5_V1_4_GATE_B_PROTOCOL.json")
    suite, task_part, state_part = str(args.parent_key).split("/")
    task_index = int(task_part.removeprefix("task_"))
    state_index = int(state_part.removeprefix("state_"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(int(args.gpu))
    get_libero_image, get_processor, get_model, adapter_type, benchmark, libero_runtime = _load_external_modules(args.official_snapshot_root, args.upstream_root)
    get_libero_path, OffScreenRenderEnv = libero_runtime
    suite_obj = benchmark.get_benchmark_dict()[suite]()
    task = suite_obj.get_task(task_index)
    init_state = copy.deepcopy(suite_obj.get_task_init_states(task_index)[state_index])
    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    args.suite = suite
    adapter, model, _processor, _unnorm_key = _load_policy(args, get_processor, get_model, adapter_type)
    clean = _load_json(gate_a_root / "CLEAN_TRAJECTORY_V1_4.json")
    if clean.get("outcomes_read") is not False or not isinstance(clean.get("rows"), list):
        raise M35RunnerError("GATE_A_CLEAN_TRAJECTORY_INVALID")
    clean_actions = [list(row["env_action"]) for row in clean["rows"]]
    plan = _load_json(gate_a_root / "PROBE_PLAN_V1_4.json")
    probes = plan.get("probe_steps")
    if not isinstance(probes, list) or len(probes) != 24:
        raise M35RunnerError("GATE_A_PROBE_PLAN_INVALID")
    binding_env, _binding_obs = _new_env(OffScreenRenderEnv, bddl, int(HORIZONS[suite]), args.gpu, init_state, args, output_dir)
    try:
        _write_runtime_binding_receipt(args, binding_env, output_dir)
    finally:
        binding_env.close()
    branch_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    compliance_rows: list[dict[str, Any]] = []
    isolation_rows: list[dict[str, Any]] = []
    for probe in probes:
        probe_id = str(probe["probe_id"])
        step = int(probe["step"])
        snapshot_root = gate_a_root / "CAUSAL_SNAPSHOTS" / probe_id
        if not snapshot_root.is_dir():
            raise M35RunnerError(f"SNAPSHOT_MISSING:{probe_id}")
        for repetition in range(REPETITIONS):
            control = _run_branch(snapshot_root=snapshot_root, gate_a_root=gate_a_root, OffScreenRenderEnv=OffScreenRenderEnv, bddl=bddl, horizon=int(HORIZONS[suite]), init_state=init_state, args=args, output_dir=output_dir, clean_actions=clean_actions, model=model, adapter=adapter, arm="CONTROL", dose=0, probe_id=probe_id, repetition=repetition)
            control_record = _record_branch(control, parent_key=args.parent_key, probe_id=probe_id, probe_step=step, repetition=repetition, arm="CONTROL")
            branch_rows.append(control_record)
            for arm, dose in DOSES.items():
                treatment = _run_branch(snapshot_root=snapshot_root, gate_a_root=gate_a_root, OffScreenRenderEnv=OffScreenRenderEnv, bddl=bddl, horizon=int(HORIZONS[suite]), init_state=init_state, args=args, output_dir=output_dir, clean_actions=clean_actions, model=model, adapter=adapter, arm=arm, dose=dose, probe_id=probe_id, repetition=repetition)
                pair = _pair_label(control, treatment, dose)
                treatment_record = _record_branch(treatment, parent_key=args.parent_key, probe_id=probe_id, probe_step=step, repetition=repetition, arm=arm, pair=pair, shared_control=control_record)
                branch_rows.append(treatment_record)
                observations.append({
                    "schema": "STAGE_V_M3_5_V1_4_TREATMENT_REPETITION_OBSERVATION_V1",
                    "canonical_parent_key": args.parent_key,
                    "probe_id": probe_id,
                    "probe_step": step,
                    "repetition": repetition,
                    "dose": arm,
                    "label_class": pair["label_class"],
                    "control_valid": pair["control_valid"],
                    "treatment_valid": pair["treatment_valid"],
                    "f_control": pair["f_control"],
                    "f_open": pair["f_open"],
                    "control_physical_class": pair["control_physical_class"],
                    "treatment_physical_class": pair["treatment_physical_class"],
                    "treatment_compliant": treatment.get("treatment_compliant") is True,
                    "delivered_open_steps": treatment.get("treatment_compliance", {}).get("delivered_open_steps"),
                    "required_horizon_steps": pair["required_horizon_steps"],
                    "treatment_branch_id": treatment_record["branch_id"],
                    "treatment_result_sha256": treatment_record["branch_result_sha256"],
                    "shared_control_branch_id": control_record["branch_id"],
                    "shared_control_result_sha256": control_record["branch_result_sha256"],
                    "protected_counters": dict(EXPECTED_COUNTERS),
                })
                compliance_rows.append({"probe_id": probe_id, "repetition": repetition, "dose": arm, **dict(treatment.get("treatment_compliance", {}))})
                isolation_rows.extend({"probe_id": probe_id, "repetition": repetition, "dose": arm, "relative_step": row.get("relative_step"), "arm_delta_linf": row.get("arm_delta_linf"), "passed": float(row.get("arm_delta_linf", math.inf)) == 0.0} for row in treatment.get("treatment_receipts", []))
    collapsed: list[dict[str, Any]] = []
    for probe in probes:
        for arm in DOSES:
            rows = sorted([row for row in observations if row["probe_id"] == probe["probe_id"] and row["dose"] == arm], key=lambda row: int(row["repetition"]))
            classes = [row["label_class"] for row in rows]
            repeat_pass = len(rows) == 3 and len(set(classes)) == 1 and not classes[0].endswith("_ABSTAIN") and classes[0] in BINARY_CLASSES and all(row["treatment_compliant"] is True for row in rows)
            collapsed.append({
                "schema": "STAGE_V_M3_5_V1_4_COLLAPSED_PROBE_DOSE_LABEL_V1",
                "canonical_parent_key": args.parent_key,
                "probe_id": probe["probe_id"],
                "probe_step": probe["step"],
                "dose": arm,
                "label_class": classes[0] if repeat_pass else (classes[0] if classes and len(set(classes)) == 1 else "HOLD_STOCHASTIC_INTERVENTION_OUTCOME"),
                "binary_label_consumable": repeat_pass,
                "repeatability_status": "PASS_REPEATABILITY_3_OF_3" if repeat_pass else "HOLD_STOCHASTIC_INTERVENTION_OUTCOME",
                "repetitions": rows,
                "protected_counters": dict(EXPECTED_COUNTERS),
            })
    _write_jsonl(output_dir / "COUNTERFACTUAL_BRANCHES_V1_4.jsonl", branch_rows)
    _write_jsonl(output_dir / "TREATMENT_REPETITION_OBSERVATIONS_V1_4.jsonl", observations)
    _write_jsonl(output_dir / "COLLAPSED_PROBE_DOSE_LABELS_V1_4.jsonl", collapsed)
    treatment_pass = len(compliance_rows) == 216 and all(row.get("treatment_compliant") is True for row in compliance_rows)
    expected_open_receipts = len(probes) * REPETITIONS * sum(DOSES.values())
    isolation_pass = len(isolation_rows) == expected_open_receipts and all(row["passed"] for row in isolation_rows)
    repeatability_pass = len(collapsed) == 72 and all(row["repeatability_status"] == "PASS_REPEATABILITY_3_OF_3" for row in collapsed)
    taxonomy_pass = all(row["label_class"] in BINARY_CLASSES for row in collapsed if row["binary_label_consumable"])
    causal_pass = len(branch_rows) == 288 and all(row.get("branch", {}).get("status") == "PASS" and row.get("branch", {}).get("causal_input_binding_pass") is True for row in branch_rows)
    parent_pass = all((treatment_pass, isolation_pass, repeatability_pass, taxonomy_pass, causal_pass))
    status = "PASS_PARENT_DIAGNOSTIC" if parent_pass else "HOLD_SEALED"
    _write_json(output_dir / "TREATMENT_COMPLIANCE_AUDIT.json", {"schema": "STAGE_V_M3_5_V1_4_TREATMENT_COMPLIANCE_AUDIT_V1", "status": "PASS" if treatment_pass else "FAIL", "rows": compliance_rows, "protected_counters": dict(EXPECTED_COUNTERS)})
    _write_json(output_dir / "ARM_ISOLATION_AUDIT.json", {"schema": "STAGE_V_M3_5_V1_4_ARM_ISOLATION_AUDIT_V1", "status": "PASS" if isolation_pass else "FAIL", "rows": isolation_rows, "protected_counters": dict(EXPECTED_COUNTERS)})
    _write_json(output_dir / "REPEATABILITY_AUDIT.json", {"schema": "STAGE_V_M3_5_V1_4_REPEATABILITY_AUDIT_V1", "status": "PASS" if repeatability_pass else "FAIL", "rows": collapsed, "protected_counters": dict(EXPECTED_COUNTERS)})
    _write_json(output_dir / "PHYSICAL_TAXONOMY_AUDIT.json", {"schema": "STAGE_V_M3_5_V1_4_PHYSICAL_TAXONOMY_AUDIT_V1", "status": "PASS" if taxonomy_pass else "FAIL", "binary_label_count": sum(int(row["binary_label_consumable"]) for row in collapsed), "protected_counters": dict(EXPECTED_COUNTERS)})
    receipt = {
        "schema": "STAGE_V_M3_5_V1_4_GATE_B_RECEIPT_V1",
        "status": status,
        "M3_5_LABEL_VALIDATION": "PENDING_FOUR_SUITE_COVERAGE" if parent_pass else "HOLD",
        "canonical_parent_key": args.parent_key,
        "suite": suite,
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "runner_sha256": _sha_file(Path(__file__)),
        "protocol_sha256": _sha_file(args.protocol.resolve()),
        "gate_a_receipt_sha256": _sha_file(gate_a_root / "M3_5_V1_4_GATE_A_RECEIPT.json"),
        "gate_a_root": str(gate_a_root),
        "matrix": {"probes": len(probes), "physical_branches": len(branch_rows), "treatment_observations": len(observations), "collapsed_labels": len(collapsed), "control_repetitions": 3, "treatment_repetitions_each": 3},
        "gates": {"causal_snapshot": causal_pass, "treatment_compliance": treatment_pass, "arm_isolation": isolation_pass, "repeatability": repeatability_pass, "physical_taxonomy": taxonomy_pass, "four_suite_coverage": False},
        "primary_window": "MATCHED_CANONICAL_ACTION_T_PLUS_H_PHYS",
        "native_policy_calls_in_primary_window": 0,
        "fresh_render_equality_gate_used": False,
        "fresh_render_primary_consumption": False,
        "protected_counters": dict(EXPECTED_COUNTERS),
    }
    _write_json(output_dir / "M3_5_V1_4_GATE_B_RECEIPT.json", receipt)
    _write_json(output_dir / "PARENT_RESULT.json", receipt)
    _seal(output_dir)
    auditor = REPO_ROOT / "scripts/detector_v5/audit_stage_v_m3_5_v1_4_gate_b.py"
    audit_process = subprocess.run(
        [
            sys.executable,
            str(auditor),
            "--root", str(output_dir),
            "--parent-key", str(args.parent_key),
            "--source-commit", str(args.source_commit),
            "--source-tree", str(args.source_tree),
        ],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    audit_path = output_dir / "M3_5_V1_4_GATE_B_INDEPENDENT_AUDIT.json"
    audit = _load_json(audit_path) if audit_path.is_file() else {"status": "FAIL", "error": audit_process.stderr[-1000:]}
    receipt["independent_audit_status"] = audit.get("status")
    receipt["independent_audit_sha256"] = _sha_file(audit_path) if audit_path.is_file() else None
    if audit.get("status") != "PASS_PARENT_INDEPENDENT":
        receipt["status"] = "HOLD_SEALED"
        receipt["M3_5_LABEL_VALIDATION"] = "HOLD"
    _write_json(output_dir / "M3_5_V1_4_GATE_B_RECEIPT.json", receipt)
    _write_json(output_dir / "PARENT_RESULT.json", receipt)
    _seal(output_dir)
    return 0 if receipt["status"] == "PASS_PARENT_DIAGNOSTIC" and audit.get("status") == "PASS_PARENT_INDEPENDENT" and audit_process.returncode == 0 else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--gate-a-root", type=Path, required=True)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-snapshot-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--enable-runtime", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.enable_runtime:
            raise M35RunnerError("RUNTIME_DISABLED_UNTIL_FROZEN_V1_4_GATE_B_AUTHORIZATION")
        return run(args)
    except (OSError, KeyError, ValueError, CausalSnapshotError, M35RunnerError) as exc:
        print(json.dumps({"status": "FAIL", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
