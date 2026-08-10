#!/usr/bin/env python3
"""Run one outcome-blind, parent-atomic M3.5 diagnostic bundle.

The runner creates one fresh clean trajectory, derives the 24 probes before
reading any branch result, then runs three matched control repetitions and
three T3/T5/T10 treatment repetitions for each probe.  The runtime guard is
deliberately fail-closed: a prospective protocol must bind this script and
explicitly authorize the launch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from gripper_attack.stage_v_m3_5_phase_classifier import classify_trajectory  # noqa: E402
from gripper_attack.stage_v_m3_5_physical_taxonomy import (  # noqa: E402
    build_forced_open_action,
    evaluate_treatment_compliance,
    repeatability_receipt,
    telemetry_from_env,
    v_phys_label,
    aperture_metric,
)


NUM_STEPS_WAIT = 10
REPETITIONS = 3
DOSES = {"T3": 3, "T5": 5, "T10": 10}
HORIZONS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
EXPECTED_PROTECTED_COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}


class M35RunnerError(RuntimeError):
    """Raised when the parent bundle cannot satisfy the frozen contract."""


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise M35RunnerError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _array_hash(value: Any) -> str:
    array = np.asarray(value, dtype=np.float64)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _finite_action(value: Any) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32).reshape(-1)
    if action.size != 7 or not np.isfinite(action).all():
        raise M35RunnerError("ACTION_VECTOR_INVALID")
    return action


def _action_semantics_valid(raw: np.ndarray, env: np.ndarray) -> bool:
    raw_gripper = float(raw[-1])
    env_gripper = float(env[-1])
    if abs(raw_gripper - 0.5) <= 1e-6:
        return False
    expected = -1.0 if raw_gripper > 0.5 else 1.0
    return abs(env_gripper - expected) <= 1e-6


def _tokens(meta: Any) -> list[int]:
    if not isinstance(meta, Mapping):
        return []
    value = meta.get("captured_action_token_ids", meta.get("tokens", []))
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return []


def _aperture(obs: Any, env: Any) -> float | None:
    if isinstance(obs, Mapping):
        for key in ("robot0_gripper_qpos", "gripper_qpos"):
            if key in obs:
                metric = aperture_metric(obs[key])
                if metric is not None:
                    return metric
    try:
        return aperture_metric(np.asarray(env.sim.data.qpos[-2:], dtype=np.float64).tolist())
    except Exception:
        return None


def _prefix(prefix: str, telemetry: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in telemetry.items() if key != "schema"}


def _model_binding_receipt(args: argparse.Namespace, env: Any, output_dir: Path) -> None:
    """Bind physical GPU, logical CUDA 0, and logical EGL 0 before step 0."""
    import torch

    render_id = getattr(env, "render_gpu_device_id", None)
    try:
        render_id = int(render_id)
    except (TypeError, ValueError):
        render_id = None
    if render_id != 0:
        raise M35RunnerError("RUNTIME_EGL_LOGICAL_DEVICE_UNVERIFIED")
    if not torch.cuda.is_available() or int(torch.cuda.current_device()) != 0:
        raise M35RunnerError("RUNTIME_CUDA_LOGICAL_DEVICE_UNVERIFIED")
    properties = torch.cuda.get_device_properties(0)
    uuid = str(getattr(properties, "uuid", "")).strip()
    if not uuid:
        raise M35RunnerError("RUNTIME_GPU_UUID_UNAVAILABLE")
    receipt = {
        "schema": "STAGE_V_M3_5_RUNTIME_BINDING_RECEIPT_V1",
        "status": "PASS",
        "physical_gpu_index": int(args.gpu),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_current_device": int(torch.cuda.current_device()),
        "torch_device_uuid": uuid,
        "torch_device_name": torch.cuda.get_device_name(0),
        "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID", ""),
        "env_render_gpu_device_id": render_id,
        "physical_to_logical_mapping": "CUDA_VISIBLE_DEVICES[0]=requested physical GPU; CUDA/EGL logical device=0",
        "runtime_python": sys.executable,
        "source_commit": str(args.source_commit),
        "source_tree": str(args.source_tree),
        "parent_key": str(args.parent_key),
        "worker_pid": os.getpid(),
        "episode_started": False,
        "receipt_written_before_policy_step_0": True,
    }
    _write_json(output_dir / "M35_RUNTIME_BINDING_RECEIPT.json", receipt)


def _new_env(OffScreenRenderEnv: Any, bddl: str, horizon: int, gpu: int, init_state: Any, args: argparse.Namespace, output_dir: Path, *, write_binding: bool = False) -> tuple[Any, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=256,
        camera_widths=256,
        camera_names=["agentview"],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        control_freq=20,
        render_gpu_device_id=0,
        horizon=horizon + NUM_STEPS_WAIT,
    )
    if write_binding:
        _model_binding_receipt(args, env, output_dir)
    env.seed(0)
    obs = env.reset()
    obs = env.set_init_state(copy.deepcopy(init_state))
    for _ in range(NUM_STEPS_WAIT):
        obs, _reward, done, _info = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        if bool(done):
            env.close()
            raise M35RunnerError("DUMMY_WAIT_TERMINATED")
    return env, obs


def _policy_action(adapter: Any, get_libero_image: Any, obs: Any, task_label: str) -> dict[str, Any]:
    image = get_libero_image(obs, 224)
    raw_value, _generation, meta = adapter.predict_action_with_scores(image, task_label)
    raw = _finite_action(raw_value)
    env = _finite_action(adapter.postprocess(raw))
    if not _action_semantics_valid(raw, env):
        raise M35RunnerError("POLICY_ACTION_SEMANTICS_INVALID")
    return {"raw_policy_action": raw.tolist(), "env_action": env.tolist(), "token_ids": _tokens(meta), "meta": {"single_generation": bool(meta.get("single_generation", False)) if isinstance(meta, Mapping) else False}}


def _clean_step_row(step: int, horizon: int, action: Mapping[str, Any], telemetry: Mapping[str, Any], aperture: float | None, baseline_z: float | None) -> tuple[dict[str, Any], float | None]:
    position = telemetry.get("object_position")
    if baseline_z is None and isinstance(position, list) and len(position) == 3:
        baseline_z = float(position[2])
    raw = list(action["raw_policy_action"])
    env = list(action["env_action"])
    row = {
        "step": int(step),
        "clean_record_valid": bool(telemetry.get("contact_telemetry_valid") is True),
        "clean_terminal": False,
        "remaining_horizon": int(horizon - step),
        "raw_action": raw,
        "env_action": env,
        "raw_gripper": float(raw[-1]),
        "env_gripper": float(env[-1]),
        "token_ids": list(action.get("token_ids", [])),
        "single_generation": bool(action.get("meta", {}).get("single_generation", False)),
        "gripper_aperture": aperture,
        "object_z_baseline_m": baseline_z,
        **{key: value for key, value in telemetry.items() if key != "schema"},
    }
    return row, baseline_z


def _replay_to_probe(
    OffScreenRenderEnv: Any,
    bddl: str,
    horizon: int,
    init_state: Any,
    clean_actions: list[list[float]],
    probe_step: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[Any, Any, np.ndarray]:
    env, obs = _new_env(OffScreenRenderEnv, bddl, horizon, args.gpu, init_state, args, output_dir)
    for absolute in range(int(probe_step)):
        if absolute >= len(clean_actions):
            env.close()
            raise M35RunnerError(f"CLEAN_PREFIX_MISSING:{absolute}")
        obs, _reward, done, _info = env.step(clean_actions[absolute])
        if bool(done):
            env.close()
            raise M35RunnerError(f"CLEAN_PREFIX_TERMINATED:{probe_step}")
    state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
    return env, obs, state


def _branch_telemetry_row(
    *,
    absolute_step: int,
    relative_step: int,
    action: Mapping[str, Any],
    obs: Any,
    env: Any,
    binding: Mapping[str, Any],
    arm: str,
    pre_aperture: float | None,
) -> dict[str, Any]:
    telemetry = telemetry_from_env(env, binding)
    raw = list(action["raw_policy_action"])
    env_action = list(action["env_action"])
    return {
        "step": int(absolute_step),
        "relative_step": int(relative_step),
        "arm": arm,
        "raw_policy_action": raw,
        "env_action": env_action,
        "token_ids": list(action.get("token_ids", [])),
        "gripper_aperture": pre_aperture,
        "pre_aperture": pre_aperture,
        "raw_gripper": float(raw[-1]),
        "env_gripper": float(env_action[-1]),
        **{key: value for key, value in telemetry.items() if key != "schema"},
    }


def _run_branch(
    *,
    OffScreenRenderEnv: Any,
    bddl: str,
    horizon: int,
    init_state: Any,
    clean_actions: list[list[float]],
    snapshot: np.ndarray,
    probe_step: int,
    args: argparse.Namespace,
    output_dir: Path,
    binding: Mapping[str, Any],
    adapter: Any,
    get_libero_image: Any,
    task_label: str,
    arm: str,
    dose_steps: int,
    control_actions: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    env, obs, restored = _replay_to_probe(OffScreenRenderEnv, bddl, horizon, init_state, clean_actions, probe_step, args, output_dir)
    restore_delta = float(np.max(np.abs(restored - snapshot))) if restored.size else 0.0
    restore_exact = bool(np.array_equal(restored, snapshot))
    if not restore_exact:
        env.close()
        raise M35RunnerError(f"STATE_RESTORE_NOT_EXACT:{probe_step}:{restore_delta:.9g}")
    rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    treatment_receipts: list[dict[str, Any]] = []
    done = False
    task_success = False
    termination = "HORIZON_CENSORED"
    try:
        for relative in range(max(0, horizon - int(probe_step))):
            absolute = int(probe_step) + relative
            if arm == "CONTROL":
                action = _policy_action(adapter, get_libero_image, obs, task_label)
                action["arm_source"] = "OWN_CLOSED_LOOP_POLICY"
            elif relative < int(dose_steps):
                if control_actions is None or relative >= len(control_actions):
                    termination = "TREATMENT_DELIVERY_INCOMPLETE"
                    break
                action = build_forced_open_action(
                    control_actions[relative]["raw_policy_action"],
                    control_actions[relative]["env_action"],
                )
                action["token_ids"] = list(control_actions[relative].get("token_ids", []))
                action["arm_source"] = "MATCHED_CONTROL_ARM"
            else:
                action = _policy_action(adapter, get_libero_image, obs, task_label)
                action["arm_source"] = "OWN_CLOSED_LOOP_POLICY_AFTER_TREATMENT"
            pre_aperture = _aperture(obs, env)
            row = _branch_telemetry_row(
                absolute_step=absolute,
                relative_step=relative,
                action=action,
                obs=obs,
                env=env,
                binding=binding,
                arm=arm,
                pre_aperture=pre_aperture,
            )
            row["arm_source"] = str(action.get("arm_source", ""))
            if arm != "CONTROL" and relative < int(dose_steps):
                treatment_receipts.append({
                    "raw_policy_action": list(action["raw_policy_action"]),
                    "env_action": list(action["env_action"]),
                    "arm_delta_linf": float(action.get("arm_delta_linf", math.inf)),
                    "pre_aperture": pre_aperture,
                })
            obs, reward, done, info = env.step(list(action["env_action"]))
            post_aperture = _aperture(obs, env)
            post_telemetry = telemetry_from_env(env, binding)
            row.update(_prefix("post_", post_telemetry))
            row["post_aperture"] = post_aperture
            row["reward"] = float(reward) if isinstance(reward, (int, float)) and math.isfinite(float(reward)) else None
            row["done"] = bool(done)
            try:
                task_success = bool(env.check_success())
            except Exception:
                task_success = False
            row["task_success"] = task_success
            if arm != "CONTROL" and relative < int(dose_steps):
                treatment_receipts[-1]["post_aperture"] = post_aperture
            rows.append(row)
            actions.append({key: value for key, value in action.items() if key != "meta"})
            if task_success:
                termination = "SUCCESS"
                break
            if bool(done):
                termination = "DONE_WITHOUT_SUCCESS"
                break
        else:
            termination = "POLICY_HORIZON"
    except Exception as exc:
        termination = "ERROR"
        return {
            "status": "FAIL",
            "error": f"{type(exc).__name__}:{exc}",
            "probe_step": int(probe_step),
            "arm": arm,
            "dose_steps": int(dose_steps),
            "state_restore_exact": restore_exact,
            "state_restore_max_abs_delta": restore_delta,
            "rows": rows,
            "actions": actions,
            "treatment_receipts": treatment_receipts,
            "task_success": task_success,
            "termination": termination,
        }
    finally:
        env.close()
    compliance = evaluate_treatment_compliance(treatment_receipts) if arm != "CONTROL" else {"treatment_compliant": True, "compliance_reason": "CONTROL"}
    return {
        "status": "PASS",
        "probe_step": int(probe_step),
        "arm": arm,
        "dose_steps": int(dose_steps),
        "state_restore_exact": restore_exact,
        "state_restore_max_abs_delta": restore_delta,
        "rows": rows,
        "actions": actions,
        "treatment_receipts": treatment_receipts,
        "treatment_compliance": compliance,
        "treatment_compliant": bool(compliance.get("treatment_compliant", False)),
        "physical_horizon_complete": len(rows) >= 10,
        "task_success": task_success,
        "termination": termination,
    }


def _contact_loss(rows: list[Mapping[str, Any]], required: int = 2) -> bool:
    count = 0
    for row in rows:
        if row.get("post_object_gripper_contact") is True:
            count = 0
        else:
            count += 1
            if count >= int(required):
                return True
    return False


def _position(row: Mapping[str, Any], prefix: str = "post_") -> list[float] | None:
    value = row.get(f"{prefix}object_position")
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _physical_class(branch: Mapping[str, Any], *, contact_eligible: bool, reference: Mapping[str, Any] | None = None) -> str:
    if branch.get("status") != "PASS" or branch.get("state_restore_exact") is not True or branch.get("physical_horizon_complete") is not True:
        return "HORIZON_CENSORED_ABSTAIN"
    if not contact_eligible:
        return "NO_CONTACT_ELIGIBILITY_ABSTAIN"
    rows = list(branch.get("rows", []))
    if _contact_loss(rows):
        return "GRIPPER_CONTACT_LOSS"
    if reference is not None:
        ref_rows = list(reference.get("rows", []))
        for left, right in zip(ref_rows[:10], rows[:10]):
            ref_pos = _position(left)
            pos = _position(right)
            if ref_pos is None or pos is None:
                return "PHYSICAL_AMBIGUITY_ABSTAIN"
            displacement = math.sqrt(sum((pos[index] - ref_pos[index]) ** 2 for index in range(3)))
            if displacement >= 0.01 and left.get("post_object_support_contact") is True and right.get("post_object_support_contact") is not True:
                return "PREMATURE_OBJECT_RELEASE"
            if ref_pos[2] - pos[2] >= 0.02 and right.get("post_object_support_contact") is not True:
                return "OBJECT_DROP"
    return "NO_PHYSICAL_FAILURE"


def _pair_label(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    control_rows = list(control.get("rows", []))
    contact_eligible = bool(control_rows and control_rows[0].get("object_gripper_contact") is True)
    control_class = _physical_class(control, contact_eligible=contact_eligible)
    treatment_class = _physical_class(treatment, contact_eligible=contact_eligible, reference=control)
    failure_classes = {"GRIPPER_CONTACT_LOSS", "PREMATURE_OBJECT_RELEASE", "OBJECT_DROP"}
    control_valid = control_class == "NO_PHYSICAL_FAILURE"
    treatment_valid = treatment.get("treatment_compliant") is True and treatment_class in (failure_classes | {"NO_PHYSICAL_FAILURE"})
    f_control = 1 if control_class in failure_classes else (0 if control_valid else None)
    f_open = 1 if treatment_class in failure_classes else (0 if treatment_valid else None)
    return {
        "control_valid": control_valid,
        "treatment_valid": treatment_valid,
        "control_physical_class": control_class,
        "treatment_physical_class": treatment_class,
        "f_control": f_control,
        "f_open": f_open,
        "label_class": v_phys_label(control_valid=control_valid, treatment_valid=treatment_valid, f_control=f_control, f_open=f_open),
        "control_contact_eligible": contact_eligible,
    }


def _seal(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            rows.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": _sha256_file(path)})
    sums = "".join(f"{row['sha256']}  {row['path']}\n" for row in rows)
    (root / "SHA256SUMS").write_text(sums, encoding="utf-8")
    sums_sha = _sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {"files": rows, "sha256s_sha256": sums_sha}


def _selected_parent(selection: Mapping[str, Any], parent_key: str) -> dict[str, Any]:
    for row in selection.get("selected_parents", []):
        if isinstance(row, Mapping) and str(row.get("canonical_parent_key")) == parent_key:
            return dict(row)
    raise M35RunnerError("PARENT_NOT_IN_FROZEN_SELECTION")


def _verify_runtime_contract(args: argparse.Namespace, protocol: Mapping[str, Any], selection: Mapping[str, Any]) -> dict[str, Any]:
    if protocol.get("runtime_authorized") is not True or protocol.get("runtime_prerequisites", {}).get("intervention_runner_status") != "PASS":
        raise M35RunnerError("RUNTIME_NOT_AUTHORIZED_OR_RUNNER_NOT_BOUND")
    expected_python = str(protocol.get("source_binding", {}).get("runtime_python", ""))
    if expected_python and Path(sys.executable).as_posix() != Path(expected_python).as_posix():
        raise M35RunnerError("RUNTIME_PYTHON_MISMATCH")
    actual_commit = _git(REPO_ROOT, "rev-parse", "HEAD")
    actual_tree = _git(REPO_ROOT, "rev-parse", "HEAD^{tree}")
    if actual_commit != str(args.source_commit) or actual_tree != str(args.source_tree):
        raise M35RunnerError("SOURCE_COMMIT_OR_TREE_MISMATCH")
    runner_binding = protocol.get("runner_binding", {})
    runner_path = REPO_ROOT / str(runner_binding.get("module_path", ""))
    if not runner_path.is_file() or _sha256_file(runner_path) != str(runner_binding.get("sha256")):
        raise M35RunnerError("RUNNER_SHA_BINDING_MISMATCH")
    taxonomy_binding = protocol.get("physical_taxonomy", {})
    taxonomy_path = REPO_ROOT / str(taxonomy_binding.get("module_path", ""))
    if not taxonomy_path.is_file() or _sha256_file(taxonomy_path) != str(taxonomy_binding.get("sha256")):
        raise M35RunnerError("PHYSICAL_TAXONOMY_SHA_BINDING_MISMATCH")
    counters = dict(selection.get("protected_counters", {}))
    if counters != EXPECTED_PROTECTED_COUNTERS:
        raise M35RunnerError("SELECTION_PROTECTED_COUNTERS_NONZERO")
    return {"actual_source_commit": actual_commit, "actual_source_tree": actual_tree}


def run_parent(args: argparse.Namespace) -> int:
    protocol = _load_json(args.protocol)
    selection_path = Path(protocol["diagnostic_parent_selection"]["path"])
    selection = _load_json(selection_path)
    if _sha256_file(selection_path) != str(protocol["diagnostic_parent_selection"]["sha256"]):
        raise M35RunnerError("SELECTION_MANIFEST_SHA_MISMATCH")
    parent = _selected_parent(selection, args.parent_key)
    _verify_runtime_contract(args, protocol, selection)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise M35RunnerError(f"REFUSE_OVERWRITE:{output_dir}")
    output_dir.mkdir(parents=True)

    suite, task_part, state_part = str(args.parent_key).split("/")
    task_index = int(task_part.removeprefix("task_"))
    state_index = int(state_part.removeprefix("state_"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    from scripts.detector_v5.run_stage_v_canonical_clean import _load_external_modules, _load_policy

    get_libero_image, get_processor, get_model, adapter_type, benchmark, libero_runtime = _load_external_modules(args.official_snapshot_root, args.upstream_root)
    get_libero_path, OffScreenRenderEnv = libero_runtime
    suite_obj = benchmark.get_benchmark_dict()[suite]()
    task = suite_obj.get_task(task_index)
    initial_states = suite_obj.get_task_init_states(task_index)
    if state_index < 0 or state_index >= len(initial_states):
        raise M35RunnerError("STATE_INDEX_OUT_OF_RANGE")
    init_state = copy.deepcopy(initial_states[state_index])
    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    args.suite = suite
    adapter, model, _processor, unnorm_key = _load_policy(args, get_processor, get_model, adapter_type)
    del model, unnorm_key
    horizon = int(HORIZONS[suite])

    first_env, first_obs = _new_env(OffScreenRenderEnv, bddl, horizon, args.gpu, init_state, args, output_dir, write_binding=True)
    try:
        from gripper_attack.stage_v_m3_5_physical_taxonomy import bind_object_taxonomy

        binding = bind_object_taxonomy(first_env, Path(bddl))
        if binding.get("status") != "PASS":
            raise M35RunnerError(f"OBJECT_TAXONOMY_BINDING_{binding.get('reason', 'ABSTAIN')}")
        clean_rows: list[dict[str, Any]] = []
        clean_actions: list[list[float]] = []
        clean_snapshots: dict[int, np.ndarray] = {}
        baseline_z: float | None = None
        task_success = False
        for step in range(horizon):
            clean_snapshots[step] = np.asarray(first_env.get_sim_state(), dtype=np.float64).copy()
            action = _policy_action(adapter, get_libero_image, first_obs, str(task.language))
            raw = np.asarray(action["raw_policy_action"], dtype=np.float32)
            env_action = np.asarray(action["env_action"], dtype=np.float32)
            telemetry = telemetry_from_env(first_env, binding)
            row, baseline_z = _clean_step_row(step, horizon, action, telemetry, _aperture(first_obs, first_env), baseline_z)
            row["state_sha256"] = _array_hash(clean_snapshots[step])
            clean_rows.append(row)
            clean_actions.append(env_action.tolist())
            first_obs, _reward, done, _info = first_env.step(env_action.tolist())
            try:
                task_success = bool(first_env.check_success())
            except Exception:
                task_success = False
            if task_success or bool(done):
                row["clean_terminal"] = True
                break
    finally:
        first_env.close()

    clean_labels = classify_trajectory(clean_rows)
    for row, label in zip(clean_rows, clean_labels):
        row.update(label)
    clean_trajectory_sha = _sha256_json(clean_rows)
    _write_json(output_dir / "CLEAN_TRAJECTORY.json", {"schema": "STAGE_V_M3_5_CLEAN_TRAJECTORY_V1", "outcomes_read": False, "rows": clean_rows, "trajectory_sha256": clean_trajectory_sha, "task_success": task_success})

    from scripts.detector_v5.build_stage_v_m3_5_probe_plan import select_probe_steps

    probe_plan = select_probe_steps(clean_rows, args.parent_key)
    _write_json(output_dir / "PROBE_PLAN.json", probe_plan)
    probe_steps = [dict(row) for row in probe_plan["probe_steps"]]
    branch_rows: list[dict[str, Any]] = []
    repeatability_rows: list[dict[str, Any]] = []
    for probe in probe_steps:
        probe_step = int(probe["step"])
        snapshot = clean_snapshots[probe_step]
        for repetition in range(REPETITIONS):
            control = _run_branch(
                OffScreenRenderEnv=OffScreenRenderEnv, bddl=bddl, horizon=horizon, init_state=init_state,
                clean_actions=clean_actions, snapshot=snapshot, probe_step=probe_step, args=args,
                output_dir=output_dir, binding=binding, adapter=adapter, get_libero_image=get_libero_image,
                task_label=str(task.language), arm="CONTROL", dose_steps=0,
            )
            branch_rows.append({"probe_step": probe_step, "repetition": repetition, "arm": "CONTROL", "branch": control})
            control_actions = list(control.get("actions", []))
            for arm, dose in DOSES.items():
                treatment = _run_branch(
                    OffScreenRenderEnv=OffScreenRenderEnv, bddl=bddl, horizon=horizon, init_state=init_state,
                    clean_actions=clean_actions, snapshot=snapshot, probe_step=probe_step, args=args,
                    output_dir=output_dir, binding=binding, adapter=adapter, get_libero_image=get_libero_image,
                    task_label=str(task.language), arm=arm, dose_steps=dose, control_actions=control_actions,
                )
                pair = _pair_label(control, treatment)
                branch_rows.append({"probe_step": probe_step, "repetition": repetition, "arm": arm, "branch": treatment, "pair": pair})
                repeatability_rows.append({"probe_step": probe_step, "dose": arm, "repetition": repetition, "outcome_class": pair["label_class"], "treatment_compliant": treatment.get("treatment_compliant") is True})
        for arm in DOSES:
            rows = [row for row in repeatability_rows if row["probe_step"] == probe_step and row["dose"] == arm]
            summary = repeatability_receipt(rows)
            summary.update({"probe_step": probe_step, "dose": arm})
            repeatability_rows.append({"probe_step": probe_step, "dose": arm, "repetition": "SUMMARY", **summary})

    (output_dir / "COUNTERFACTUAL_BRANCHES.jsonl").write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in branch_rows), encoding="utf-8")
    _write_json(output_dir / "REPEATABILITY_SUMMARY.json", {"schema": "STAGE_V_M3_5_REPEATABILITY_SUMMARY_V1", "rows": repeatability_rows, "control_repetitions": 3, "treatment_repetitions_each": 3})
    result = {
        "schema": "STAGE_V_M3_5_PARENT_RESULT_V1",
        "status": "PASS",
        "canonical_parent_key": args.parent_key,
        "suite": suite,
        "task_index": task_index,
        "state_index": state_index,
        "source_commit": str(args.source_commit),
        "source_tree": str(args.source_tree),
        "runner_sha256": _sha256_file(Path(__file__)),
        "parent_atomic": True,
        "gpu": int(args.gpu),
        "model_path": str(args.model_path),
        "bddl_file": bddl,
        "clean_steps": len(clean_rows),
        "clean_success": bool(task_success),
        "probe_count": len(probe_steps),
        "expected_physical_branches": 24 * 4 * 3,
        "actual_physical_branches": len(branch_rows),
        "expected_treatment_label_rows": 24 * 3 * 3,
        "actual_treatment_label_rows": len([row for row in branch_rows if row["arm"] != "CONTROL"]),
        "selection_outcomes_read": False,
        "branch_outcomes_read_before_probe_selection": False,
        "protected_counters": dict(EXPECTED_PROTECTED_COUNTERS),
        "taxonomy_binding": binding,
        "state_restore_api": "fresh_env_set_init_state_dummy_wait_then_exact_clean_prefix_replay",
        "post_treatment_mode": "own_closed_loop_policy_after_matched_control_arm",
    }
    _write_json(output_dir / "PARENT_RESULT.json", result)
    result["artifact_seal"] = _seal(output_dir)
    _write_json(output_dir / "PARENT_RESULT.json", result)
    _seal(output_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-snapshot-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--run-label", default="M3_5")
    parser.add_argument("--run-set", default="M3_5_PARENT")
    parser.add_argument("--enable-runtime", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.enable_runtime:
            raise M35RunnerError("RUNTIME_DISABLED_UNTIL_V1_2_AUTHORIZATION")
        return run_parent(args)
    except (OSError, KeyError, ValueError, M35RunnerError) as exc:
        print(json.dumps({"status": "FAIL", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
