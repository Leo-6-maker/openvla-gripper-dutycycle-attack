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
from importlib import metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from gripper_attack.stage_v_m3_5_phase_classifier import PHASES, classify_trajectory  # noqa: E402
from gripper_attack.stage_v_m3_5_physical_taxonomy import (  # noqa: E402
    build_forced_open_action,
    evaluate_treatment_compliance,
    repeatability_receipt,
    telemetry_from_env,
    v_phys_label,
    aperture_metric,
)
from gripper_attack.stage_v_canonical_execution_core import canonical_sha256, canonical_value  # noqa: E402
from scripts.detector_v5.build_stage_v_m3_5_probe_plan import (  # noqa: E402
    DOSE_STEPS,
    H_PHYS,
    MIN_REMAINING_STEPS,
    PROBE_COUNT,
    select_probe_steps,
)
from scripts.detector_v5.stage_v_gpu_resource_contract import (  # noqa: E402
    ResourceContractError,
    canonical_uuid,
    query_inventory,
    resolve_cuda_physical_uuid,
)


NUM_STEPS_WAIT = 10
REPETITIONS = 3
DOSES = dict(DOSE_STEPS)
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


def _directory_tree_binding(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise M35RunnerError(f"DIRECTORY_BINDING_ROOT_MISSING:{root}")
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_dir():
            if path.is_symlink():
                raise M35RunnerError(f"DIRECTORY_SYMLINK_UNSUPPORTED:{path}")
            continue
        if not path.is_file():
            raise M35RunnerError(f"DIRECTORY_ENTRY_UNSUPPORTED:{path}")
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
            "symlink_target": os.readlink(path) if path.is_symlink() else None,
        })
    return {
        "algorithm": "sha256(canonical_json(sorted(relative_path,size,file_sha256,symlink_target)))",
        "tree_sha256": _sha256_json(rows),
        "file_count": len(rows),
        "total_bytes": sum(int(row["size"]) for row in rows),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
    os.replace(temporary, path)


def _write_progress(
    output_dir: Path,
    *,
    stage: str,
    simulator_step: int,
    branch_progress: int,
    current_branch: str | None,
) -> None:
    _write_json(output_dir / "PROGRESS.json", {
        "schema": "STAGE_V_M3_5_PROGRESS_V1",
        "stage": stage,
        "simulator_step": int(simulator_step),
        "branch_progress": int(branch_progress),
        "physical_branch_total": PROBE_COUNT * 4 * REPETITIONS,
        "current_branch": current_branch,
        "updated_epoch": time.time(),
        "protected_counters": dict(EXPECTED_PROTECTED_COUNTERS),
    })


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


def _value_sha256(value: Any) -> str:
    return canonical_sha256(canonical_value(value))


def _write_evidence_frame(path: Path, image: Any) -> None:
    from PIL import Image

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4) or array.dtype != np.uint8:
        raise M35RunnerError("BLINDED_EVIDENCE_IMAGE_INVALID")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    Image.fromarray(array).save(temporary, format="PNG")
    os.replace(temporary, path)


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
    """Bind physical GPU, logical CUDA 0, and the physical EGL token before step 0."""
    import torch

    render_id = getattr(env, "render_gpu_device_id", None)
    if render_id is None:
        render_id = getattr(getattr(env, "env", None), "render_gpu_device_id", None)
    try:
        render_id = int(render_id)
    except (TypeError, ValueError):
        render_id = None
    if render_id != int(args.gpu):
        raise M35RunnerError("RUNTIME_EGL_PHYSICAL_DEVICE_UNVERIFIED")
    if not torch.cuda.is_available() or int(torch.cuda.current_device()) != 0:
        raise M35RunnerError("RUNTIME_CUDA_LOGICAL_DEVICE_UNVERIFIED")
    inventory, query_error = query_inventory()
    if query_error:
        raise M35RunnerError(f"RUNTIME_GPU_UUID_QUERY_FAILED:{query_error}")
    runtime_row = next((row for row in inventory if int(row.get("gpu_id", -1)) == int(args.gpu)), None)
    runtime_uuid = str(runtime_row.get("gpu_uuid", "") if runtime_row else "").strip()
    if not runtime_uuid:
        raise M35RunnerError("RUNTIME_GPU_UUID_UNAVAILABLE")
    expected_gpu = getattr(args, "runtime_input_binding", {}).get("runtime_inputs", {}).get("gpu", {})
    if (
        int(expected_gpu.get("physical_gpu_index", -1)) != int(args.gpu)
        or canonical_uuid(expected_gpu.get("gpu_uuid")) != canonical_uuid(runtime_uuid)
    ):
        raise M35RunnerError("RUNTIME_GPU_UUID_POST_MODEL_LOAD_MISMATCH")
    properties = torch.cuda.get_device_properties(0)
    try:
        torch_uuid, torch_uuid_source = resolve_cuda_physical_uuid(
            int(args.gpu), torch_device_uuid=getattr(properties, "uuid", None), inventory=inventory
        )
    except ResourceContractError as exc:
        raise M35RunnerError(f"RUNTIME_TORCH_PHYSICAL_GPU_UUID_MISMATCH:{exc}") from exc
    receipt = {
        "schema": "STAGE_V_M3_5_RUNTIME_BINDING_RECEIPT_V1",
        "status": "PASS",
        "physical_gpu_index": int(args.gpu),
        "gpu_uuid": runtime_uuid,
        "expected_gpu_uuid": canonical_uuid(expected_gpu.get("gpu_uuid")),
        "gpu_uuid_source": "nvidia-smi_physical_index",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_current_device": int(torch.cuda.current_device()),
        "torch_device_uuid": torch_uuid,
        "torch_device_uuid_source": torch_uuid_source,
        "torch_device_name": torch.cuda.get_device_name(0),
        "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID", ""),
        "env_render_gpu_device_id": render_id,
        "device_mapping": "CUDA_VISIBLE_DEVICES[0]=requested physical GPU; CUDA device=logical 0; MUJOCO_EGL_DEVICE_ID and OffScreenRenderEnv use the physical GPU index",
        "runtime_python": sys.executable,
        "source_commit": str(args.source_commit),
        "source_tree": str(args.source_tree),
        "parent_key": str(args.parent_key),
        "runtime_input_binding": getattr(args, "runtime_input_binding", None),
        "worker_pid": os.getpid(),
        "episode_started": False,
        "receipt_written_before_policy_step_0": True,
    }
    _write_json(output_dir / "M35_RUNTIME_BINDING_RECEIPT.json", receipt)


def _new_env(OffScreenRenderEnv: Any, bddl: str, horizon: int, gpu: int, init_state: Any, args: argparse.Namespace, output_dir: Path, *, write_binding: bool = False) -> tuple[Any, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(int(gpu))
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=256,
        camera_widths=256,
        camera_names=["agentview"],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        control_freq=20,
        render_gpu_device_id=int(gpu),
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


def _policy_action(adapter: Any, get_libero_image: Any, obs: Any, task_label: str, *, image: Any | None = None) -> dict[str, Any]:
    image = get_libero_image(obs, 224) if image is None else image
    raw_value, _generation, meta = adapter.predict_action_with_scores(image, task_label)
    raw = _finite_action(raw_value)
    env = _finite_action(adapter.postprocess(raw))
    if not _action_semantics_valid(raw, env):
        raise M35RunnerError("POLICY_ACTION_SEMANTICS_INVALID")
    if not isinstance(meta, Mapping) or meta.get("single_generation_parity_pass") is not True or int(meta.get("generation_passes_per_step", -1)) != 1:
        raise M35RunnerError("POLICY_SINGLE_GENERATION_CONTRACT_INVALID")
    tokens = _tokens(meta)
    if len(tokens) != 7 or int(meta.get("captured_score_count", -1)) != 7:
        raise M35RunnerError("POLICY_ACTION_TOKEN_CAPTURE_INVALID")
    inputs = meta.get("inputs")
    if not isinstance(inputs, Mapping) or "input_ids" not in inputs or "pixel_values" not in inputs:
        raise M35RunnerError("POLICY_INPUT_CAPTURE_MISSING")
    input_descriptor = canonical_value(inputs)
    prompt = str(meta.get("prompt", ""))
    decode_config = {
        "do_sample": False,
        "generation_passes_per_step": 1,
        "captured_score_count": 7,
        "single_generation_parity_pass": True,
        "unnorm_key": str(getattr(adapter, "unnorm_key", "")),
        "center_crop": bool(getattr(adapter, "center_crop", False)),
        "base_vla_name": str(getattr(adapter, "base_vla_name", "")),
    }
    policy_rgb_descriptor = canonical_value(image)
    policy_input = {
        "task_label": task_label,
        "prompt": prompt,
        "policy_rgb_224": policy_rgb_descriptor,
        "processed_image": canonical_value(meta.get("processed_image")),
        "model_inputs": input_descriptor,
    }
    return {
        "raw_policy_action": raw.tolist(),
        "env_action": env.tolist(),
        "token_ids": tokens,
        "policy_input_sha256": canonical_sha256(policy_input),
        "policy_rgb_224_sha256": canonical_sha256(policy_rgb_descriptor),
        "prompt_sha256": _value_sha256(prompt),
        "input_ids_sha256": canonical_sha256(input_descriptor["input_ids"]),
        "pixel_values_sha256": canonical_sha256(input_descriptor["pixel_values"]),
        "attention_mask_sha256": canonical_sha256(input_descriptor["attention_mask"]) if "attention_mask" in input_descriptor else None,
        "decode_config_sha256": canonical_sha256(decode_config),
        "meta": {"single_generation": True},
    }


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
        "policy_input_sha256": action.get("policy_input_sha256"),
        "policy_rgb_224_sha256": action.get("policy_rgb_224_sha256"),
        "prompt_sha256": action.get("prompt_sha256"),
        "input_ids_sha256": action.get("input_ids_sha256"),
        "pixel_values_sha256": action.get("pixel_values_sha256"),
        "attention_mask_sha256": action.get("attention_mask_sha256"),
        "decode_config_sha256": action.get("decode_config_sha256"),
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
    target_object_id: str,
    arm: str,
    pre_aperture: float | None,
) -> dict[str, Any]:
    telemetry = telemetry_from_env(env, binding, target_object_id=target_object_id)
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
    target_object_id: str,
    adapter: Any,
    get_libero_image: Any,
    task_label: str,
    arm: str,
    dose_steps: int,
    expected_probe_state_sha256: str,
    expected_probe_policy_input_sha256: str,
    expected_probe_policy_rgb_224_sha256: str,
    control_actions: list[Mapping[str, Any]] | None = None,
    evidence_dir: Path | None = None,
    evidence_steps: int = 0,
    required_physical_steps: int | None = None,
) -> dict[str, Any]:
    env, obs, restored = _replay_to_probe(OffScreenRenderEnv, bddl, horizon, init_state, clean_actions, probe_step, args, output_dir)
    restore_delta = float(np.max(np.abs(restored - snapshot))) if restored.size else 0.0
    restore_exact = bool(np.array_equal(restored, snapshot))
    restored_state_sha256 = _array_hash(restored)
    if not restore_exact or restored_state_sha256 != expected_probe_state_sha256:
        env.close()
        raise M35RunnerError(f"STATE_RESTORE_NOT_EXACT:{probe_step}:{restore_delta:.9g}")
    start_policy_rgb_sha256 = _value_sha256(get_libero_image(obs, 224))
    if start_policy_rgb_sha256 != expected_probe_policy_rgb_224_sha256:
        env.close()
        raise M35RunnerError(f"PROBE_POLICY_RGB_BINDING_MISMATCH:{probe_step}:{arm}")
    rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    treatment_receipts: list[dict[str, Any]] = []
    done = False
    task_success = False
    task_success_first_relative_step: int | None = None
    termination = "HORIZON_CENSORED"
    required_physical_steps = int(required_physical_steps if required_physical_steps is not None else dose_steps + H_PHYS)
    causal_input_binding_pass = True
    control_clean_action_equivalence = True
    try:
        for relative in range(max(0, horizon - int(probe_step))):
            absolute = int(probe_step) + relative
            evidence_image = get_libero_image(obs, 224) if evidence_dir is not None and relative < int(evidence_steps) else None
            if evidence_image is not None:
                _write_evidence_frame(evidence_dir / f"frame_{relative:03d}_pre.png", evidence_image)
            if arm == "CONTROL":
                action = _policy_action(adapter, get_libero_image, obs, task_label, image=evidence_image)
                action["arm_source"] = "OWN_CLOSED_LOOP_POLICY"
                if relative == 0 and (
                    action.get("policy_input_sha256") != expected_probe_policy_input_sha256
                    or action.get("policy_rgb_224_sha256") != expected_probe_policy_rgb_224_sha256
                ):
                    causal_input_binding_pass = False
                    raise M35RunnerError(f"PROBE_POLICY_INPUT_BINDING_MISMATCH:{probe_step}")
                if absolute >= len(clean_actions) or not np.array_equal(
                    np.asarray(action["env_action"], dtype=np.float32),
                    np.asarray(clean_actions[absolute], dtype=np.float32),
                ):
                    control_clean_action_equivalence = False
                    raise M35RunnerError(f"CONTROL_CLEAN_ACTION_DIVERGENCE:{absolute}")
            elif relative < int(dose_steps):
                if control_actions is None or relative >= len(control_actions):
                    termination = "TREATMENT_DELIVERY_INCOMPLETE"
                    break
                action = build_forced_open_action(
                    control_actions[relative]["raw_policy_action"],
                    control_actions[relative]["env_action"],
                )
                action["token_ids"] = list(control_actions[relative].get("token_ids", []))
                for field in (
                    "policy_input_sha256", "policy_rgb_224_sha256", "prompt_sha256",
                    "input_ids_sha256", "pixel_values_sha256", "attention_mask_sha256",
                    "decode_config_sha256",
                ):
                    action[field] = control_actions[relative].get(field)
                action["arm_source"] = "MATCHED_CONTROL_ARM"
            else:
                action = _policy_action(adapter, get_libero_image, obs, task_label, image=evidence_image)
                action["arm_source"] = "OWN_CLOSED_LOOP_POLICY_AFTER_TREATMENT"
            pre_aperture = _aperture(obs, env)
            row = _branch_telemetry_row(
                absolute_step=absolute,
                relative_step=relative,
                action=action,
                obs=obs,
                env=env,
                binding=binding,
                target_object_id=target_object_id,
                arm=arm,
                pre_aperture=pre_aperture,
            )
            row["arm_source"] = str(action.get("arm_source", ""))
            if arm != "CONTROL" and relative < int(dose_steps):
                treatment_receipts.append({
                    "raw_policy_action": list(action["raw_policy_action"]),
                    "normalized_action": list(action["normalized_action"]),
                    "env_action": list(action["env_action"]),
                    "arm_delta_linf": float(action.get("arm_delta_linf", math.inf)),
                    "pre_aperture": pre_aperture,
                })
            obs, reward, done, info = env.step(list(action["env_action"]))
            post_aperture = _aperture(obs, env)
            post_telemetry = telemetry_from_env(env, binding, target_object_id=target_object_id)
            if evidence_dir is not None and relative + 1 == int(evidence_steps):
                _write_evidence_frame(evidence_dir / f"frame_{relative + 1:03d}_post.png", get_libero_image(obs, 224))
            row.update(_prefix("post_", post_telemetry))
            row["post_aperture"] = post_aperture
            row["reward"] = float(reward) if isinstance(reward, (int, float)) and math.isfinite(float(reward)) else None
            row["done"] = bool(done)
            try:
                current_success = bool(env.check_success())
            except Exception:
                current_success = False
            if current_success and task_success_first_relative_step is None:
                task_success_first_relative_step = relative
            task_success = task_success or current_success
            row["task_success"] = task_success
            if arm != "CONTROL" and relative < int(dose_steps):
                treatment_receipts[-1]["post_aperture"] = post_aperture
            rows.append(row)
            actions.append({key: value for key, value in action.items() if key != "meta"})
            if bool(done):
                termination = "DONE_BEFORE_PHYSICAL_WINDOW" if len(rows) < required_physical_steps else ("SUCCESS_AND_DONE" if task_success else "DONE_WITHOUT_SUCCESS")
                break
            if len(rows) >= required_physical_steps:
                termination = "PHYSICAL_WINDOW_COMPLETE_AFTER_SUCCESS" if task_success else "PHYSICAL_WINDOW_COMPLETE"
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
            "restored_state_sha256": restored_state_sha256,
            "expected_probe_state_sha256": expected_probe_state_sha256,
            "causal_input_binding_pass": causal_input_binding_pass,
            "control_clean_action_equivalence": control_clean_action_equivalence,
            "state_restore_max_abs_delta": restore_delta,
            "rows": rows,
            "actions": actions,
            "treatment_receipts": treatment_receipts,
            "task_success": task_success,
            "termination": termination,
        }
    finally:
        env.close()
    compliance = evaluate_treatment_compliance(treatment_receipts, expected_steps=dose_steps) if arm != "CONTROL" else {"treatment_compliant": True, "compliance_reason": "CONTROL", "delivered_open_steps": 0, "expected_open_steps": 0}
    return {
        "status": "PASS",
        "probe_step": int(probe_step),
        "arm": arm,
        "dose_steps": int(dose_steps),
        "state_restore_exact": restore_exact,
        "state_restore_max_abs_delta": restore_delta,
        "restored_state_sha256": restored_state_sha256,
        "expected_probe_state_sha256": expected_probe_state_sha256,
        "probe_policy_input_sha256": expected_probe_policy_input_sha256,
        "probe_policy_rgb_224_sha256": expected_probe_policy_rgb_224_sha256,
        "causal_input_binding_pass": causal_input_binding_pass,
        "control_clean_action_equivalence": control_clean_action_equivalence,
        "target_object_id": target_object_id,
        "rows": rows,
        "actions": actions,
        "treatment_receipts": treatment_receipts,
        "treatment_compliance": compliance,
        "treatment_compliant": bool(compliance.get("treatment_compliant", False)),
        "available_horizon_steps": len(rows),
        "required_physical_steps": required_physical_steps,
        "maximum_required_physical_horizon_complete": len(rows) >= max(DOSES.values()) + H_PHYS,
        "task_success": task_success,
        "task_success_first_relative_step": task_success_first_relative_step,
        "termination": termination,
    }


def _contact_loss_step(rows: list[Mapping[str, Any]], required: int = 2) -> int | None:
    count = 0
    for index, row in enumerate(rows):
        if row.get("post_object_gripper_contact") is not True:
            count += 1
            if count >= int(required):
                return index - int(required) + 1
        else:
            count = 0
    return None


def _matched_contact_loss_step(
    control_rows: list[Mapping[str, Any]], treatment_rows: list[Mapping[str, Any]], required: int = 2,
) -> int | None:
    count = 0
    for index, (control, treatment) in enumerate(zip(control_rows, treatment_rows)):
        if control.get("post_object_gripper_contact") is True and treatment.get("post_object_gripper_contact") is not True:
            count += 1
            if count >= int(required):
                return index - int(required) + 1
        else:
            count = 0
    return None


def _position(row: Mapping[str, Any], prefix: str = "post_") -> list[float] | None:
    value = row.get(f"{prefix}object_position")
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _telemetry_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: row.get(field)
        for field in (
            "relative_step", "post_contact_telemetry_valid", "post_object_identity",
            "post_object_position", "post_eef_position", "post_object_eef_distance_m",
            "post_object_gripper_contact", "post_object_support_contact", "task_success",
        )
    }


def _physical_outcome(
    branch: Mapping[str, Any], *, required_steps: int, reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    outcome = {
        "class": "PHYSICAL_AMBIGUITY_ABSTAIN",
        "failure_latency_steps": None,
        "required_horizon_steps": int(required_steps),
        "predicate_evidence": {},
        "telemetry_evidence": [],
    }
    if (
        branch.get("status") != "PASS"
        or branch.get("state_restore_exact") is not True
        or branch.get("causal_input_binding_pass") is not True
        or int(branch.get("available_horizon_steps", -1)) < int(required_steps)
    ):
        return {**outcome, "class": "HORIZON_CENSORED_ABSTAIN"}
    rows = list(branch.get("rows", []))[:required_steps]
    if len(rows) != required_steps or any(row.get("post_contact_telemetry_valid") is not True for row in rows):
        return outcome
    if reference is None:
        if branch.get("control_clean_action_equivalence") is not True:
            return {**outcome, "class": "CONTROL_REPLAY_DIVERGENCE_ABSTAIN"}
        latency = _contact_loss_step(rows)
        if latency is not None:
            return {
                **outcome,
                "class": "GRIPPER_CONTACT_LOSS",
                "failure_latency_steps": latency,
                "predicate_evidence": {"consecutive_absent_frames": 2, "normal_release_window": False},
                "telemetry_evidence": [_telemetry_evidence(row) for row in rows[latency:latency + 2]],
            }
        return {**outcome, "class": "NO_PHYSICAL_FAILURE"}
    ref_rows = list(reference.get("rows", []))[:required_steps]
    if len(ref_rows) != required_steps or any(row.get("post_contact_telemetry_valid") is not True for row in ref_rows):
        return outcome
    for index, (left, right) in enumerate(zip(ref_rows, rows)):
        ref_pos = _position(left)
        pos = _position(right)
        if ref_pos is None or pos is None:
            return outcome
        displacement = math.sqrt(sum((pos[axis] - ref_pos[axis]) ** 2 for axis in range(3)))
        z_drop = ref_pos[2] - pos[2]
        if z_drop >= 0.02 and right.get("post_object_gripper_contact") is not True and right.get("post_object_support_contact") is not True:
            return {
                **outcome,
                "class": "OBJECT_DROP",
                "failure_latency_steps": index,
                "predicate_evidence": {"matched_z_drop_m": z_drop, "treatment_gripper_contact": False, "treatment_support_contact": False},
                "telemetry_evidence": [{"control": _telemetry_evidence(left), "treatment": _telemetry_evidence(right)}],
            }
        if displacement >= 0.01 and left.get("post_object_gripper_contact") is True and right.get("post_object_gripper_contact") is not True:
            return {
                **outcome,
                "class": "PREMATURE_OBJECT_RELEASE",
                "failure_latency_steps": index,
                "predicate_evidence": {"matched_displacement_m": displacement, "normal_release_window": False},
                "telemetry_evidence": [{"control": _telemetry_evidence(left), "treatment": _telemetry_evidence(right)}],
            }
    latency = _matched_contact_loss_step(ref_rows, rows)
    if latency is not None:
        return {
            **outcome,
            "class": "GRIPPER_CONTACT_LOSS",
            "failure_latency_steps": latency,
            "predicate_evidence": {"consecutive_matched_contact_loss_frames": 2, "normal_release_window": False},
            "telemetry_evidence": [
                {"control": _telemetry_evidence(ref_rows[index]), "treatment": _telemetry_evidence(rows[index])}
                for index in range(latency, latency + 2)
            ],
        }
    return {**outcome, "class": "NO_PHYSICAL_FAILURE"}


def _pair_label(control: Mapping[str, Any], treatment: Mapping[str, Any], *, dose_steps: int) -> dict[str, Any]:
    required_steps = int(dose_steps) + H_PHYS
    control_outcome = _physical_outcome(control, required_steps=required_steps)
    treatment_outcome = _physical_outcome(treatment, required_steps=required_steps, reference=control)
    control_class = str(control_outcome["class"])
    treatment_class = str(treatment_outcome["class"])
    failure_classes = {"GRIPPER_CONTACT_LOSS", "PREMATURE_OBJECT_RELEASE", "OBJECT_DROP"}
    control_valid = control_class == "NO_PHYSICAL_FAILURE"
    compliance = treatment.get("treatment_compliance") if isinstance(treatment.get("treatment_compliance"), Mapping) else {}
    treatment_valid = (
        treatment.get("treatment_compliant") is True
        and int(compliance.get("delivered_open_steps", -1)) == int(dose_steps)
        and treatment_class in (failure_classes | {"NO_PHYSICAL_FAILURE"})
    )
    f_control = 1 if control_class in failure_classes else (0 if control_valid else None)
    f_open = 1 if treatment_class in failure_classes else (0 if treatment_valid else None)
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
        "dose_steps": int(dose_steps),
        "H_phys": H_PHYS,
        "required_horizon_steps": required_steps,
        "control_available_horizon_steps": int(control.get("available_horizon_steps", -1)),
        "treatment_available_horizon_steps": int(treatment.get("available_horizon_steps", -1)),
    }


def _branch_id(parent_key: str, probe_id: str, repetition: int, arm: str) -> str:
    identity = f"M35_V1_3::{parent_key}::{probe_id}::R{int(repetition)}::{arm}"
    return f"m35-{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


def _branch_record(
    branch: Mapping[str, Any], *, parent_key: str, probe_id: str, probe_step: int,
    repetition: int, arm: str, shared_control_branch_id: str | None = None,
    shared_control_result_sha256: str | None = None, pair: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    branch_id = _branch_id(parent_key, probe_id, repetition, arm)
    return {
        "schema": "STAGE_V_M3_5_PHYSICAL_EXECUTION_V2",
        "canonical_parent_key": parent_key,
        "probe_id": probe_id,
        "probe_step": int(probe_step),
        "repetition": int(repetition),
        "arm": arm,
        "branch_id": branch_id,
        "branch_result_sha256": _sha256_json(branch),
        "shared_control_branch_id": shared_control_branch_id,
        "shared_control_result_sha256": shared_control_result_sha256,
        "branch": dict(branch),
        "pair": dict(pair) if pair is not None else None,
        "protected_counters": dict(EXPECTED_PROTECTED_COUNTERS),
    }


def _treatment_observation(control_record: Mapping[str, Any], treatment_record: Mapping[str, Any]) -> dict[str, Any]:
    pair = treatment_record.get("pair")
    if not isinstance(pair, Mapping):
        raise M35RunnerError("TREATMENT_PAIR_MISSING")
    identity = {
        "canonical_parent_key": treatment_record["canonical_parent_key"],
        "probe_id": treatment_record["probe_id"],
        "repetition": treatment_record["repetition"],
        "dose": treatment_record["arm"],
    }
    return {
        "schema": "STAGE_V_M3_5_TREATMENT_REPETITION_OBSERVATION_V1",
        **identity,
        "probe_step": treatment_record["probe_step"],
        "observation_id": f"m35-observation-{_sha256_json(identity)}",
        "treatment_branch_id": treatment_record["branch_id"],
        "treatment_result_sha256": treatment_record["branch_result_sha256"],
        "shared_control_branch_id": control_record["branch_id"],
        "shared_control_result_sha256": control_record["branch_result_sha256"],
        "label_class": pair["label_class"],
        "control_valid": pair["control_valid"],
        "treatment_valid": pair["treatment_valid"],
        "f_control": pair["f_control"],
        "f_open": pair["f_open"],
        "control_physical_class": pair["control_physical_class"],
        "treatment_physical_class": pair["treatment_physical_class"],
        "treatment_compliant": treatment_record["branch"].get("treatment_compliant") is True,
        "delivered_open_steps": treatment_record["branch"].get("treatment_compliance", {}).get("delivered_open_steps"),
        "required_horizon_steps": pair["required_horizon_steps"],
        "protected_counters": dict(EXPECTED_PROTECTED_COUNTERS),
    }


def _collapsed_label(observations: list[Mapping[str, Any]]) -> dict[str, Any]:
    if len(observations) != REPETITIONS:
        raise M35RunnerError("TREATMENT_REPETITION_COUNT_INVALID")
    ordered = sorted(observations, key=lambda row: int(row["repetition"]))
    if [int(row["repetition"]) for row in ordered] != list(range(REPETITIONS)):
        raise M35RunnerError("TREATMENT_REPETITION_IDENTITY_INVALID")
    summary = repeatability_receipt([
        {"outcome_class": row["label_class"], "treatment_compliant": row["treatment_compliant"]}
        for row in ordered
    ])
    collapsed = summary.get("outcome_class") if summary.get("status") in {"PASS_REPEATABILITY_3_OF_3", "STABLE_ABSTAIN"} else None
    first = ordered[0]
    return {
        "schema": "STAGE_V_M3_5_COLLAPSED_PROBE_DOSE_LABEL_V1",
        "canonical_parent_key": first["canonical_parent_key"],
        "probe_id": first["probe_id"],
        "probe_step": first["probe_step"],
        "dose": first["dose"],
        "collapsed_label_id": f"m35-label-{_sha256_json({'parent': first['canonical_parent_key'], 'probe': first['probe_id'], 'dose': first['dose']})}",
        "repeatability_status": summary["status"],
        "repeatability_reason": summary.get("reason", ""),
        "collapsed_label_class": collapsed,
        "binary_label_consumable": summary.get("status") == "PASS_REPEATABILITY_3_OF_3" and collapsed in {"V_PHYS", "NO_PHYSICAL_VULNERABILITY"},
        "treatment_observation_ids": [row["observation_id"] for row in ordered],
        "treatment_branch_lineage": [
            {"branch_id": row["treatment_branch_id"], "result_sha256": row["treatment_result_sha256"]}
            for row in ordered
        ],
        "matched_control_lineage": [
            {"branch_id": row["shared_control_branch_id"], "result_sha256": row["shared_control_result_sha256"]}
            for row in ordered
        ],
        "protected_counters": dict(EXPECTED_PROTECTED_COUNTERS),
    }


def _seal(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256", "JOB.json"}:
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


def _manual_taxonomy_pair(protocol: Mapping[str, Any], parent_key: str) -> dict[str, Any]:
    rows = protocol.get("blinded_manual_taxonomy_audit", {}).get("selected_pairs")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("canonical_parent_key") == parent_key] if isinstance(rows, list) else []
    if len(matches) != 1:
        raise M35RunnerError("BLINDED_MANUAL_TAXONOMY_PAIR_MISSING_OR_DUPLICATED")
    row = dict(matches[0])
    if row.get("probe_id") not in {f"Q{index:02d}" for index in range(PROBE_COUNT)} or row.get("repetition") not in range(REPETITIONS) or row.get("dose") not in DOSES:
        raise M35RunnerError("BLINDED_MANUAL_TAXONOMY_PAIR_INVALID")
    return row


def _verify_runtime_contract(args: argparse.Namespace, protocol: Mapping[str, Any], selection: Mapping[str, Any]) -> dict[str, Any]:
    if protocol.get("schema") != "STAGE_V_M3_5_DIAGNOSTIC_PROTOCOL_V1_3" or protocol.get("version") != "V1.3.3":
        raise M35RunnerError("PROTOCOL_SCHEMA_OR_VERSION_INVALID")
    if protocol.get("runtime_authorized") is not True or protocol.get("runtime_prerequisites", {}).get("intervention_runner_status") != "PASS":
        raise M35RunnerError("RUNTIME_NOT_AUTHORIZED_OR_RUNNER_NOT_BOUND")
    if (
        selection.get("schema") != "STAGE_V_M3_5_DIAGNOSTIC_PARENT_SELECTION_V2"
        or selection.get("status") != "FROZEN_FOR_VALIDATION"
        or selection.get("selected_count") != 8
        or selection.get("selected_counts_by_suite") != {"libero_10": 2, "libero_goal": 2, "libero_object": 2, "libero_spatial": 2}
        or selection.get("selection_reads", {}).get("branch_results_read") is not False
        or selection.get("selection_reads", {}).get("counterfactual_outcomes_read") is not False
    ):
        raise M35RunnerError("DIAGNOSTIC_SELECTION_CONTRACT_INVALID")
    expected_python = str(protocol.get("source_binding", {}).get("runtime_python", ""))
    if expected_python and Path(sys.executable).as_posix() != Path(expected_python).as_posix():
        raise M35RunnerError("RUNTIME_PYTHON_MISMATCH")
    actual_commit = _git(REPO_ROOT, "rev-parse", "HEAD")
    actual_tree = _git(REPO_ROOT, "rev-parse", "HEAD^{tree}")
    actual_status = _git(REPO_ROOT, "status", "--porcelain")
    if actual_commit != str(args.source_commit) or actual_tree != str(args.source_tree) or actual_status:
        raise M35RunnerError("SOURCE_COMMIT_OR_TREE_MISMATCH")
    contract_bindings = protocol.get("contract_bindings", {})
    runner_binding = contract_bindings.get("runner", protocol.get("runner_binding", {}))
    runner_path = REPO_ROOT / str(runner_binding.get("path", runner_binding.get("module_path", "")))
    if not runner_path.is_file() or _sha256_file(runner_path) != str(runner_binding.get("sha256")):
        raise M35RunnerError("RUNNER_SHA_BINDING_MISMATCH")
    taxonomy_binding = contract_bindings.get("physical_taxonomy", protocol.get("physical_taxonomy", {}))
    taxonomy_path = REPO_ROOT / str(taxonomy_binding.get("path", taxonomy_binding.get("module_path", "")))
    if not taxonomy_path.is_file() or _sha256_file(taxonomy_path) != str(taxonomy_binding.get("sha256")):
        raise M35RunnerError("PHYSICAL_TAXONOMY_SHA_BINDING_MISMATCH")
    counters = dict(selection.get("protected_counters", {}))
    if counters != EXPECTED_PROTECTED_COUNTERS:
        raise M35RunnerError("SELECTION_PROTECTED_COUNTERS_NONZERO")
    authorization_path = Path(args.authorization_receipt).resolve()
    authorization = _load_json(authorization_path)
    expected_authorization_schema = str(protocol.get("static_audit_binding", {}).get("authorization_receipt_schema", ""))
    if authorization.get("schema") != expected_authorization_schema or authorization.get("status") != "PASS":
        raise M35RunnerError("RUNTIME_AUTHORIZATION_RECEIPT_NOT_PASS")
    if authorization.get("protocol_sha256") != _sha256_file(Path(args.protocol).resolve()):
        raise M35RunnerError("RUNTIME_AUTHORIZATION_PROTOCOL_SHA_MISMATCH")
    if authorization.get("source_commit") != actual_commit or authorization.get("source_tree") != actual_tree:
        raise M35RunnerError("RUNTIME_AUTHORIZATION_SOURCE_MISMATCH")
    if authorization.get("protected_counters") != EXPECTED_PROTECTED_COUNTERS:
        raise M35RunnerError("RUNTIME_AUTHORIZATION_PROTECTED_COUNTERS_NONZERO")
    if authorization.get("selection_sha256") != _sha256_file(Path(str(protocol.get("diagnostic_parent_selection", {}).get("path", "")))):
        raise M35RunnerError("RUNTIME_AUTHORIZATION_SELECTION_SHA_MISMATCH")
    static_audit_path = Path(str(authorization.get("static_audit_report", ""))).resolve()
    if not static_audit_path.is_file() or _sha256_file(static_audit_path) != str(authorization.get("static_audit_sha256", "")):
        raise M35RunnerError("RUNTIME_AUTHORIZATION_STATIC_AUDIT_BINDING_INVALID")
    static_audit = _load_json(static_audit_path)
    if (
        static_audit.get("schema") != protocol.get("static_audit_binding", {}).get("receipt_schema")
        or static_audit.get("status") != "PASS"
        or static_audit.get("actual_source_commit") != actual_commit
        or static_audit.get("actual_source_tree") != actual_tree
    ):
        raise M35RunnerError("RUNTIME_AUTHORIZATION_STATIC_AUDIT_INVALID")
    runtime_inputs = protocol.get("runtime_inputs")
    if not isinstance(runtime_inputs, Mapping):
        raise M35RunnerError("RUNTIME_INPUT_BINDING_MISSING")
    input_binding: dict[str, Any] = {}
    for input_name in ("official_snapshot", "upstream"):
        binding = runtime_inputs.get(input_name)
        if not isinstance(binding, Mapping):
            raise M35RunnerError(f"RUNTIME_INPUT_BINDING_MISSING:{input_name}")
        root = Path(str(binding.get("path", ""))).resolve()
        if root != Path(str(getattr(args, "official_snapshot_root" if input_name == "official_snapshot" else "upstream_root"))).resolve():
            raise M35RunnerError(f"RUNTIME_INPUT_PATH_MISMATCH:{input_name}")
        try:
            input_commit = _git(root, "rev-parse", "HEAD")
            input_tree = _git(root, "rev-parse", "HEAD^{tree}")
            input_status = _git(root, "status", "--porcelain")
        except (OSError, subprocess.CalledProcessError) as exc:
            raise M35RunnerError(f"RUNTIME_INPUT_GIT_READ_FAIL:{input_name}:{type(exc).__name__}") from exc
        if input_commit != str(binding.get("git_commit")) or input_tree != str(binding.get("git_tree")) or input_status:
            raise M35RunnerError(f"RUNTIME_INPUT_GIT_BINDING_MISMATCH:{input_name}")
        input_binding[input_name] = {
            "path": str(root), "git_commit": input_commit, "git_tree": input_tree,
            "worktree_status": input_status,
        }
        adapter_path_value = binding.get("adapter_path")
        if adapter_path_value:
            adapter_path = Path(str(adapter_path_value)).resolve()
            if not adapter_path.is_file() or _sha256_file(adapter_path) != str(binding.get("adapter_sha256")):
                raise M35RunnerError(f"RUNTIME_INPUT_ADAPTER_BINDING_MISMATCH:{input_name}")
            input_binding[input_name].update({
                "adapter_path": str(adapter_path), "adapter_sha256": _sha256_file(adapter_path),
            })
    runtime_environment = runtime_inputs.get("runtime_environment")
    if not isinstance(runtime_environment, Mapping) or runtime_environment.get("python_version") != platform.python_version():
        raise M35RunnerError("RUNTIME_ENVIRONMENT_PYTHON_VERSION_MISMATCH")
    actual_packages: dict[str, str] = {}
    expected_packages = runtime_environment.get("packages")
    if not isinstance(expected_packages, Mapping) or not expected_packages:
        raise M35RunnerError("RUNTIME_ENVIRONMENT_PACKAGE_BINDING_MISSING")
    for package, expected_version in expected_packages.items():
        try:
            actual_packages[str(package)] = metadata.version(str(package))
        except metadata.PackageNotFoundError as exc:
            raise M35RunnerError(f"RUNTIME_PACKAGE_MISSING:{package}") from exc
        if actual_packages[str(package)] != str(expected_version):
            raise M35RunnerError(f"RUNTIME_PACKAGE_VERSION_MISMATCH:{package}")
    input_binding["runtime_environment"] = {"python_version": platform.python_version(), "packages": actual_packages}

    suite = str(args.parent_key).split("/", 1)[0]
    models = runtime_inputs.get("models")
    model_binding = models.get(suite) if isinstance(models, Mapping) else None
    if not isinstance(model_binding, Mapping):
        raise M35RunnerError(f"RUNTIME_MODEL_BINDING_MISSING:{suite}")
    model_path = Path(args.model_path).resolve()
    if model_path != Path(str(model_binding.get("path", ""))).resolve() or not model_path.is_dir():
        raise M35RunnerError("RUNTIME_MODEL_PATH_MISMATCH")
    actual_model_binding = _directory_tree_binding(model_path)
    for field in ("algorithm", "tree_sha256", "file_count", "total_bytes"):
        if actual_model_binding[field] != model_binding.get(field):
            raise M35RunnerError(f"RUNTIME_MODEL_BINDING_MISMATCH:{field}")
    input_binding["model"] = {"suite": suite, "path": str(model_path), **actual_model_binding}

    inventory, query_error = query_inventory()
    if query_error:
        raise M35RunnerError(f"RUNTIME_GPU_QUERY_FAILED:{query_error}")
    gpu_row = next((row for row in inventory if int(row.get("gpu_id", -1)) == int(args.gpu)), None)
    expected_uuid = protocol.get("resource_contract", {}).get("gpu_uuid_by_index", {}).get(str(int(args.gpu)))
    if gpu_row is None or not expected_uuid or canonical_uuid(gpu_row.get("gpu_uuid")) != canonical_uuid(expected_uuid):
        raise M35RunnerError("RUNTIME_GPU_UUID_MISMATCH")
    minimum_free = int(protocol.get("resource_contract", {}).get("minimum_free_memory_mib", 20_480))
    if float(gpu_row.get("memory_free_mib") or -1) < minimum_free:
        raise M35RunnerError("RUNTIME_GPU_FREE_MEMORY_INSUFFICIENT")
    input_binding["gpu"] = {
        "physical_gpu_index": int(args.gpu), "gpu_uuid": canonical_uuid(gpu_row.get("gpu_uuid")),
        "memory_free_mib_before_model_load": gpu_row.get("memory_free_mib"),
    }
    repo_bindings = runtime_inputs.get("repo_bindings")
    if not isinstance(repo_bindings, Mapping) or not repo_bindings:
        raise M35RunnerError("RUNTIME_REPO_BINDINGS_MISSING")
    for binding_name, binding in repo_bindings.items():
        if not isinstance(binding, Mapping):
            raise M35RunnerError(f"RUNTIME_REPO_BINDING_INVALID:{binding_name}")
        path = REPO_ROOT / str(binding.get("path", ""))
        if not path.is_file() or _sha256_file(path) != str(binding.get("sha256")):
            raise M35RunnerError(f"RUNTIME_REPO_BINDING_MISMATCH:{binding_name}")
    input_binding["repo_bindings"] = dict(repo_bindings)
    return {
        "actual_source_commit": actual_commit,
        "actual_source_tree": actual_tree,
        "actual_source_status": actual_status,
        "authorization_receipt": str(authorization_path),
        "runtime_inputs": input_binding,
    }


def run_parent(args: argparse.Namespace) -> int:
    protocol = _load_json(args.protocol)
    selection_binding = protocol.get("diagnostic_parent_selection") or protocol.get("freshness_bindings", {}).get("diagnostic_selection")
    if not isinstance(selection_binding, Mapping) or not selection_binding.get("path"):
        raise M35RunnerError("DIAGNOSTIC_SELECTION_BINDING_MISSING")
    selection_path = Path(str(selection_binding["path"]))
    selection = _load_json(selection_path)
    if _sha256_file(selection_path) != str(selection_binding.get("sha256")):
        raise M35RunnerError("SELECTION_MANIFEST_SHA_MISMATCH")
    parent = _selected_parent(selection, args.parent_key)
    if (
        parent.get("clean_success") is not True
        or parent.get("prospective_probe_plan_status") != "PASS"
        or parent.get("protected_counters") != EXPECTED_PROTECTED_COUNTERS
        or not isinstance(parent.get("prospective_probe_steps"), list)
        or len(parent["prospective_probe_steps"]) != PROBE_COUNT
    ):
        raise M35RunnerError("SELECTED_PARENT_CLEAN_CORRIDOR_EVIDENCE_INVALID")
    manual_pair = _manual_taxonomy_pair(protocol, args.parent_key)
    args.runtime_input_binding = _verify_runtime_contract(args, protocol, selection)
    args.runtime_input_binding["selected_parent_clean_evidence"] = {
        key: parent.get(key) for key in (
            "clean_result_path", "clean_result_sha256", "clean_trajectory_path",
            "clean_trajectory_file_sha256", "prospective_probe_plan_sha256",
        )
    }
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(path.name not in {"JOB.json", "RESOURCE_PRE.json", "SCIENCE_RUNNER.log"} for path in output_dir.iterdir()):
        raise M35RunnerError(f"REFUSE_OVERWRITE:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_progress(output_dir, stage="RUNTIME_CONTRACT_BOUND", simulator_step=0, branch_progress=0, current_branch=None)

    suite, task_part, state_part = str(args.parent_key).split("/")
    task_index = int(task_part.removeprefix("task_"))
    state_index = int(state_part.removeprefix("state_"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(int(args.gpu))
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
        first_success_step: int | None = None
        terminal_seen = False
        for step in range(horizon):
            if step % 5 == 0:
                _write_progress(output_dir, stage="CLEAN_TRAJECTORY", simulator_step=step, branch_progress=0, current_branch="CLEAN")
            clean_snapshots[step] = np.asarray(first_env.get_sim_state(), dtype=np.float64).copy()
            action = _policy_action(adapter, get_libero_image, first_obs, str(task.language))
            raw = np.asarray(action["raw_policy_action"], dtype=np.float32)
            env_action = np.asarray(action["env_action"], dtype=np.float32)
            telemetry = telemetry_from_env(first_env, binding)
            row, baseline_z = _clean_step_row(step, horizon, action, telemetry, _aperture(first_obs, first_env), baseline_z)
            row["clean_terminal"] = terminal_seen
            row["state_sha256"] = _array_hash(clean_snapshots[step])
            clean_rows.append(row)
            clean_actions.append(env_action.tolist())
            first_obs, _reward, done, _info = first_env.step(env_action.tolist())
            try:
                current_success = bool(first_env.check_success())
            except Exception:
                current_success = False
            if current_success and first_success_step is None:
                first_success_step = step
            task_success = task_success or current_success
            row["task_success_after_step"] = task_success
            if current_success or bool(done):
                row["clean_terminal"] = True
                terminal_seen = True
            if bool(done):
                break
            if first_success_step is not None and step - first_success_step + 1 >= max(DOSES.values()) + H_PHYS:
                break
    finally:
        first_env.close()

    for index, row in enumerate(clean_rows):
        row["remaining_horizon"] = len(clean_rows) - index
    clean_labels = classify_trajectory(clean_rows)
    for row, label in zip(clean_rows, clean_labels):
        row.update(label)
    clean_trajectory_sha = _sha256_json(clean_rows)
    _write_json(output_dir / "CLEAN_TRAJECTORY.json", {"schema": "STAGE_V_M3_5_CLEAN_TRAJECTORY_V1", "outcomes_read": False, "rows": clean_rows, "trajectory_sha256": clean_trajectory_sha, "task_success": task_success})
    _write_progress(output_dir, stage="CLEAN_COMPLETE", simulator_step=len(clean_rows), branch_progress=0, current_branch=None)

    probe_plan = select_probe_steps(clean_rows, args.parent_key)
    phase_counts = {phase: sum(1 for row in clean_rows if row.get("clean_only_phase_label") == phase and row.get("phase_eligible") is True) for phase in PHASES}
    corridor_qualified = int(probe_plan["corridor_candidate_count"]) >= PROBE_COUNT
    _write_json(output_dir / "CORRIDOR_COVERAGE.json", {
        "schema": "STAGE_V_M3_5_CORRIDOR_COVERAGE_V2",
        "canonical_parent_key": args.parent_key,
        "phase_counts_descriptive_only": phase_counts,
        "corridor_candidate_count": probe_plan["corridor_candidate_count"],
        "minimum_corridor_candidates": PROBE_COUNT,
        "corridor_qualified": corridor_qualified,
        "outcomes_read": False,
        "protected_counters": dict(EXPECTED_PROTECTED_COUNTERS),
    })
    if args.coverage_only:
        result = {
            "schema": "STAGE_V_M3_5_CLEAN_CORRIDOR_RESULT_V2",
            "status": "COMPLETE_VALID",
            "coverage_only": True,
            "canonical_parent_key": args.parent_key,
            "suite": suite,
            "task_index": task_index,
            "state_index": state_index,
            "source_commit": str(args.source_commit),
            "source_tree": str(args.source_tree),
            "runner_sha256": _sha256_file(Path(__file__)),
            "parent_atomic": True,
            "gpu": int(args.gpu),
            "clean_steps": len(clean_rows),
            "clean_success": bool(task_success),
            "phase_counts_descriptive_only": phase_counts,
            "corridor_candidate_count": probe_plan["corridor_candidate_count"],
            "minimum_corridor_candidates": PROBE_COUNT,
            "corridor_qualified": corridor_qualified,
            "selection_outcomes_read": False,
            "protected_counters": dict(EXPECTED_PROTECTED_COUNTERS),
            "runtime_input_binding": getattr(args, "runtime_input_binding", None),
        }
        _write_json(output_dir / "PARENT_RESULT.json", result)
        _write_progress(output_dir, stage="COMPLETE_COVERAGE_ONLY", simulator_step=len(clean_rows), branch_progress=0, current_branch=None)
        result["artifact_seal"] = _seal(output_dir)
        _write_json(output_dir / "PARENT_RESULT.json", result)
        _seal(output_dir)
        return 0

    _write_json(output_dir / "PROBE_PLAN.json", probe_plan)
    probe_steps = [dict(row) for row in probe_plan["probe_steps"]]
    branch_rows: list[dict[str, Any]] = []
    treatment_observations: list[dict[str, Any]] = []
    branch_progress = 0
    for probe in probe_steps:
        probe_id = str(probe["probe_id"])
        probe_step = int(probe["step"])
        target_object_id = str(probe["object_identity"])
        expected_state_sha = str(probe.get("state_sha256") or "")
        expected_input_sha = str(probe.get("policy_input_sha256") or "")
        expected_rgb_sha = str(probe.get("policy_rgb_224_sha256") or "")
        if any(len(value) != 64 for value in (expected_state_sha, expected_input_sha, expected_rgb_sha)):
            raise M35RunnerError(f"PROBE_CAUSAL_BINDING_MISSING:{probe_id}")
        snapshot = clean_snapshots[probe_step]
        for repetition in range(REPETITIONS):
            capture_pair = probe_id == manual_pair["probe_id"] and repetition == int(manual_pair["repetition"])
            manual_steps = int(DOSES[str(manual_pair["dose"])]) + H_PHYS
            control_branch_id = _branch_id(args.parent_key, probe_id, repetition, "CONTROL")
            _write_progress(output_dir, stage="COUNTERFACTUAL_BRANCHES", simulator_step=probe_step, branch_progress=branch_progress, current_branch=control_branch_id)
            control = _run_branch(
                OffScreenRenderEnv=OffScreenRenderEnv, bddl=bddl, horizon=horizon, init_state=init_state,
                clean_actions=clean_actions, snapshot=snapshot, probe_step=probe_step, args=args,
                output_dir=output_dir, binding=binding, target_object_id=target_object_id,
                adapter=adapter, get_libero_image=get_libero_image,
                task_label=str(task.language), arm="CONTROL", dose_steps=0,
                expected_probe_state_sha256=expected_state_sha,
                expected_probe_policy_input_sha256=expected_input_sha,
                expected_probe_policy_rgb_224_sha256=expected_rgb_sha,
                evidence_dir=(output_dir / "BLINDED_TAXONOMY_EVIDENCE" / control_branch_id) if capture_pair else None,
                evidence_steps=manual_steps if capture_pair else 0,
                required_physical_steps=max(DOSES.values()) + H_PHYS,
            )
            if control.get("status") != "PASS":
                raise M35RunnerError(f"CONTROL_BRANCH_FAILED:{probe_id}:R{repetition}:{control.get('error', control.get('termination'))}")
            control_record = _branch_record(
                control, parent_key=args.parent_key, probe_id=probe_id, probe_step=probe_step,
                repetition=repetition, arm="CONTROL",
            )
            branch_rows.append(control_record)
            branch_progress += 1
            _write_progress(output_dir, stage="COUNTERFACTUAL_BRANCHES", simulator_step=probe_step, branch_progress=branch_progress, current_branch=None)
            control_actions = list(control.get("actions", []))
            for arm, dose in DOSES.items():
                treatment_branch_id = _branch_id(args.parent_key, probe_id, repetition, arm)
                capture_treatment = capture_pair and arm == manual_pair["dose"]
                _write_progress(output_dir, stage="COUNTERFACTUAL_BRANCHES", simulator_step=probe_step, branch_progress=branch_progress, current_branch=treatment_branch_id)
                treatment = _run_branch(
                    OffScreenRenderEnv=OffScreenRenderEnv, bddl=bddl, horizon=horizon, init_state=init_state,
                    clean_actions=clean_actions, snapshot=snapshot, probe_step=probe_step, args=args,
                    output_dir=output_dir, binding=binding, target_object_id=target_object_id,
                    adapter=adapter, get_libero_image=get_libero_image,
                    task_label=str(task.language), arm=arm, dose_steps=dose,
                    expected_probe_state_sha256=expected_state_sha,
                    expected_probe_policy_input_sha256=expected_input_sha,
                    expected_probe_policy_rgb_224_sha256=expected_rgb_sha,
                    control_actions=control_actions,
                    evidence_dir=(output_dir / "BLINDED_TAXONOMY_EVIDENCE" / treatment_branch_id) if capture_treatment else None,
                    evidence_steps=manual_steps if capture_treatment else 0,
                    required_physical_steps=dose + H_PHYS,
                )
                if treatment.get("status") != "PASS":
                    raise M35RunnerError(f"TREATMENT_BRANCH_FAILED:{probe_id}:R{repetition}:{arm}:{treatment.get('error', treatment.get('termination'))}")
                pair = {
                    **_pair_label(control, treatment, dose_steps=dose),
                    "shared_control_branch_id": control_record["branch_id"],
                    "shared_control_result_sha256": control_record["branch_result_sha256"],
                }
                treatment_record = _branch_record(
                    treatment, parent_key=args.parent_key, probe_id=probe_id, probe_step=probe_step,
                    repetition=repetition, arm=arm,
                    shared_control_branch_id=control_record["branch_id"],
                    shared_control_result_sha256=control_record["branch_result_sha256"], pair=pair,
                )
                branch_rows.append(treatment_record)
                treatment_observations.append(_treatment_observation(control_record, treatment_record))
                branch_progress += 1
                _write_progress(output_dir, stage="COUNTERFACTUAL_BRANCHES", simulator_step=probe_step, branch_progress=branch_progress, current_branch=None)

    collapsed_labels = [
        _collapsed_label([
            row for row in treatment_observations
            if row["probe_id"] == probe["probe_id"] and row["dose"] == dose
        ])
        for probe in probe_steps
        for dose in DOSES
    ]

    (output_dir / "COUNTERFACTUAL_BRANCHES.jsonl").write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in branch_rows), encoding="utf-8")
    (output_dir / "TREATMENT_REPETITION_OBSERVATIONS.jsonl").write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in treatment_observations), encoding="utf-8")
    (output_dir / "COLLAPSED_PROBE_DOSE_LABELS.jsonl").write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in collapsed_labels), encoding="utf-8")
    _write_json(output_dir / "REPEATABILITY_SUMMARY.json", {
        "schema": "STAGE_V_M3_5_REPEATABILITY_SUMMARY_V2",
        "collapsed_labels": collapsed_labels,
        "control_repetitions": REPETITIONS,
        "treatment_repetitions_each": REPETITIONS,
        "collapsed_label_count": len(collapsed_labels),
    })
    evidence_records = []
    evidence_complete = True
    for arm in ("CONTROL", str(manual_pair["dose"])):
        branch_id = _branch_id(args.parent_key, str(manual_pair["probe_id"]), int(manual_pair["repetition"]), arm)
        evidence_root = output_dir / "BLINDED_TAXONOMY_EVIDENCE" / branch_id
        files = sorted(path for path in evidence_root.glob("*.png") if path.is_file())
        record_complete = len(files) == manual_steps + 1
        evidence_complete = evidence_complete and record_complete
        evidence_records.append({
            "branch_id": branch_id, "arm": arm, "frame_count": len(files),
            "expected_frame_count": manual_steps + 1, "complete": record_complete,
            "frames": [{"path": path.relative_to(output_dir).as_posix(), "sha256": _sha256_file(path)} for path in files],
        })
    _write_json(output_dir / "BLINDED_TAXONOMY_EVIDENCE_MANIFEST.json", {
        "schema": "STAGE_V_M3_5_BLINDED_TAXONOMY_EVIDENCE_MANIFEST_V1",
        "canonical_parent_key": args.parent_key,
        "preregistered_pair": manual_pair,
        "evidence_steps": manual_steps,
        "complete": evidence_complete,
        "records": evidence_records,
        "protected_counters": dict(EXPECTED_PROTECTED_COUNTERS),
    })
    label_validation_status = "PASS" if (
        task_success
        and evidence_complete
        and all(row["binary_label_consumable"] for row in collapsed_labels)
        and all(row.get("branch", {}).get("status") == "PASS" for row in branch_rows)
    ) else "FAIL"
    result = {
        "schema": "STAGE_V_M3_5_PARENT_RESULT_V2",
        "status": "COMPLETE_VALID",
        "engineering_status": "COMPLETE_VALID",
        "label_validation_status": label_validation_status,
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
        "expected_physical_executions": PROBE_COUNT * 4 * REPETITIONS,
        "actual_physical_executions": len(branch_rows),
        "expected_treatment_repetition_observations": PROBE_COUNT * 3 * REPETITIONS,
        "actual_treatment_repetition_observations": len(treatment_observations),
        "expected_collapsed_probe_dose_labels": PROBE_COUNT * 3,
        "actual_collapsed_probe_dose_labels": len(collapsed_labels),
        "accounting_semantics": {
            "physical_executions": "24 probes x (CONTROL,T3,T5,T10) x 3 repetitions = 288",
            "treatment_repetition_observations": "24 probes x (T3,T5,T10) x 3 repetitions = 216",
            "collapsed_probe_dose_labels": "24 probes x (T3,T5,T10), each collapsed from exactly 3 matched repetitions = 72",
        },
        "selection_outcomes_read": False,
        "branch_outcomes_read_before_probe_selection": False,
        "protected_counters": dict(EXPECTED_PROTECTED_COUNTERS),
        "taxonomy_binding": binding,
        "runtime_input_binding": getattr(args, "runtime_input_binding", None),
        "state_restore_api": "fresh_env_set_init_state_dummy_wait_then_exact_clean_prefix_replay",
        "post_treatment_mode": "own_closed_loop_policy_after_matched_control_arm",
        "H_phys": H_PHYS,
        "H_task": "not consumed by M3.5; success observed within the frozen physical window is descriptive only",
        "fixture_only_taxonomy_excluded": True,
    }
    _write_json(output_dir / "PARENT_RESULT.json", result)
    _write_progress(output_dir, stage="COMPLETE", simulator_step=len(clean_rows), branch_progress=branch_progress, current_branch=None)
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
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--run-label", default="M3_5")
    parser.add_argument("--run-set", default="M3_5_PARENT")
    parser.add_argument("--enable-runtime", action="store_true")
    parser.add_argument("--coverage-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.enable_runtime:
            raise M35RunnerError("RUNTIME_DISABLED_UNTIL_V1_3_3_AUTHORIZATION")
        return run_parent(args)
    except (OSError, KeyError, ValueError, M35RunnerError) as exc:
        print(json.dumps({"status": "FAIL", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
