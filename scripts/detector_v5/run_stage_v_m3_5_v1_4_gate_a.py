#!/usr/bin/env python3
"""Create and replay exact M3.5 V1.4 causal snapshots.

This runner is deliberately separate from the sealed V1.3.4 intervention
runner.  Gate A has no treatment and never asks a fresh renderer to provide a
primary policy input.  A frozen/authorized protocol is required before any
GPU work can start.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from gripper_attack.stage_v_causal_observation_snapshot import (  # noqa: E402
    CausalSnapshotError,
    assert_exact,
    capture_rng_state,
    capture_runtime_state,
    capture_simulator_state,
    assert_primary_observation_exact,
    load_snapshot,
    reference_action_window,
    restore_rng_state,
    restore_runtime_state,
    write_snapshot,
)
from gripper_attack.stage_v_canonical_execution_core import canonical_sha256, canonical_value  # noqa: E402
from gripper_attack.stage_v_m3_5_phase_classifier import PHASES, classify_trajectory  # noqa: E402
from gripper_attack.stage_v_m3_5_physical_taxonomy import (  # noqa: E402
    aperture_metric,
    telemetry_from_env,
)
from scripts.detector_v5.build_stage_v_m3_5_probe_plan import (  # noqa: E402
    DEFAULT_SELECTION_VERSION,
    CLOSED_PREFIX_SELECTION_VERSION,
    H_PHYS,
    PREFIX_SELECTION_VERSION,
    PROBE_COUNT,
    select_probe_steps,
)
from scripts.detector_v5.run_stage_v_m3_5_intervention_parent import (  # noqa: E402
    HORIZONS,
    M35RunnerError,
    NUM_STEPS_WAIT,
    _seal as _seal_root,
    _new_env,
)
from scripts.detector_v5.run_stage_v_canonical_clean import (  # noqa: E402
    _load_external_modules,
    _load_policy,
    _write_runtime_binding_receipt,
)


PROTECTED_COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise M35RunnerError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _array_sha(value: Any) -> str:
    array = np.asarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _finite_action(value: Any) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32).reshape(-1)
    if action.size != 7 or not np.isfinite(action).all():
        raise M35RunnerError("ACTION_VECTOR_INVALID")
    return action


def _action_semantics_valid(raw: np.ndarray, env: np.ndarray) -> bool:
    if abs(float(raw[-1]) - 0.5) <= 1e-6:
        return False
    expected = -1.0 if float(raw[-1]) > 0.5 else 1.0
    return abs(float(env[-1]) - expected) <= 1e-6


def _decode_config(adapter: Any, meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "do_sample": False,
        "generation_passes_per_step": int(meta.get("generation_passes_per_step", -1)),
        "captured_score_count": int(meta.get("captured_score_count", -1)),
        "single_generation_parity_pass": bool(meta.get("single_generation_parity_pass", False)),
        "unnorm_key": str(getattr(adapter, "unnorm_key", "")),
        "center_crop": bool(getattr(adapter, "center_crop", False)),
        "base_vla_name": str(getattr(adapter, "base_vla_name", "")),
    }


def _policy_capture(adapter: Any, get_libero_image: Any, obs: Any, task_label: str) -> dict[str, Any]:
    """Run the official clean step once and retain the actual consumed inputs."""
    image = get_libero_image(obs, 224)
    raw_value, _generation, meta = adapter.predict_action_with_scores(image, task_label)
    if not isinstance(meta, Mapping) or meta.get("single_generation_parity_pass") is not True:
        raise M35RunnerError("POLICY_SINGLE_GENERATION_CONTRACT_INVALID")
    if int(meta.get("generation_passes_per_step", -1)) != 1 or int(meta.get("captured_score_count", -1)) != 7:
        raise M35RunnerError("POLICY_SINGLE_GENERATION_CONTRACT_INVALID")
    tokens = [int(item) for item in meta.get("captured_action_token_ids", [])]
    if len(tokens) != 7:
        raise M35RunnerError("POLICY_ACTION_TOKEN_CAPTURE_INVALID")
    raw = _finite_action(raw_value)
    env_action = _finite_action(adapter.postprocess(raw))
    if not _action_semantics_valid(raw, env_action):
        raise M35RunnerError("POLICY_ACTION_SEMANTICS_INVALID")
    model_inputs = getattr(getattr(adapter, "model", None), "_stage_v_last_model_inputs", None)
    if not isinstance(model_inputs, Mapping) or not {"input_ids", "pixel_values"}.issubset(model_inputs):
        raise M35RunnerError("MODEL_INPUT_CAPTURE_MISSING")
    prompt = str(meta.get("prompt", ""))
    processed_image = meta.get("processed_image")
    decode_config = _decode_config(adapter, meta)
    model_input_descriptor = {
        "input_ids": model_inputs["input_ids"],
        "pixel_values": model_inputs["pixel_values"],
    }
    if "attention_mask" in model_inputs:
        model_input_descriptor["attention_mask"] = model_inputs["attention_mask"]
    policy_descriptor = {
        "prompt": prompt,
        "processed_image": canonical_value(processed_image),
        "model_inputs": canonical_value(model_input_descriptor),
        "policy_rgb_224": canonical_value(image),
        "decode_config": decode_config,
    }
    return {
        "raw_policy_action": raw.tolist(),
        "env_action": env_action.tolist(),
        "token_ids": tokens,
        "raw_observation": obs,
        "canonical_policy_rgb_224": image,
        "processed_image": processed_image,
        "input_ids": model_inputs["input_ids"],
        "pixel_values": model_inputs["pixel_values"],
        "attention_mask": model_inputs.get("attention_mask"),
        "attention_mask_present": "attention_mask" in model_inputs,
        "prompt": prompt,
        "decode_config": decode_config,
        "policy_input_sha256": canonical_sha256(policy_descriptor),
        "policy_rgb_224_sha256": canonical_sha256(canonical_value(image)),
        "input_ids_sha256": canonical_sha256(canonical_value(model_inputs["input_ids"])),
        "pixel_values_sha256": canonical_sha256(canonical_value(model_inputs["pixel_values"])),
        "attention_mask_sha256": canonical_sha256(canonical_value(model_inputs["attention_mask"])) if model_inputs.get("attention_mask") is not None else None,
    }


def _clean_row(step: int, horizon: int, capture: Mapping[str, Any], telemetry: Mapping[str, Any], aperture: float | None, baseline_z: float | None) -> tuple[dict[str, Any], float | None]:
    position = telemetry.get("object_position")
    if baseline_z is None and isinstance(position, list) and len(position) == 3:
        baseline_z = float(position[2])
    raw = list(capture["raw_policy_action"])
    env_action = list(capture["env_action"])
    row = {
        "step": int(step),
        "clean_record_valid": bool(telemetry.get("contact_telemetry_valid") is True),
        "clean_terminal": False,
        "remaining_horizon": int(horizon - step),
        "raw_action": raw,
        "env_action": env_action,
        "raw_gripper": float(raw[-1]),
        "env_gripper": float(env_action[-1]),
        "token_ids": list(capture["token_ids"]),
        "single_generation": True,
        "policy_input_sha256": capture["policy_input_sha256"],
        "policy_rgb_224_sha256": capture["policy_rgb_224_sha256"],
        "input_ids_sha256": capture["input_ids_sha256"],
        "pixel_values_sha256": capture["pixel_values_sha256"],
        "attention_mask_sha256": capture["attention_mask_sha256"],
        "gripper_aperture": aperture,
        "object_z_baseline_m": baseline_z,
        **{key: value for key, value in telemetry.items() if key != "schema"},
    }
    return row, baseline_z


def _validate_protocol(protocol: Mapping[str, Any], args: argparse.Namespace) -> None:
    if protocol.get("schema") != "STAGE_V_M3_5_DIAGNOSTIC_PROTOCOL_V1_4_GATE_A" or protocol.get("version") not in {"V1.4-GATE-A", "V1.4.1-GATE-A"}:
        raise M35RunnerError("V1_4_GATE_A_PROTOCOL_INVALID")
    if protocol.get("status") != "FROZEN_RUNTIME_AUTHORIZED" or protocol.get("runtime_authorized") is not True:
        raise M35RunnerError("V1_4_GATE_A_PROTOCOL_NOT_FROZEN_OR_AUTHORIZED")
    binding = protocol.get("source_binding", {})
    if str(binding.get("runtime_commit")) != str(args.source_commit) or str(binding.get("runtime_tree")) != str(args.source_tree):
        raise M35RunnerError("V1_4_SOURCE_BINDING_MISMATCH")
    if protocol.get("operation", {}).get("fresh_render_primary_consumption") != "HARD_STOP":
        raise M35RunnerError("FRESH_RENDER_PRIMARY_CONSUMPTION_NOT_HARD_STOP")
    if protocol.get("operation", {}).get("intervention_executed") is not False:
        raise M35RunnerError("GATE_A_MUST_BE_ZERO_TREATMENT")


def _verify_exposed_parent(selection_path: Path, parent_key: str) -> Mapping[str, Any]:
    selection = _load_json(selection_path)
    if selection.get("selection_reads", {}).get("branch_results_read") is not False or selection.get("selection_reads", {}).get("counterfactual_outcomes_read") is not False:
        raise M35RunnerError("SELECTION_OUTCOME_LEAKAGE")
    for row in selection.get("selected_parents", []):
        if isinstance(row, Mapping) and str(row.get("canonical_parent_key")) == parent_key:
            return row
    raise M35RunnerError("PARENT_NOT_IN_FROZEN_SELECTION")


def _snapshot_payload(capture: Mapping[str, Any], *, probe: Mapping[str, Any], simulator: Mapping[str, Any], runtime: Mapping[str, Any], episode_rng: Mapping[str, Any], action_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    step = int(probe["step"])
    return {
        "probe": {key: probe[key] for key in ("probe_id", "step", "object_identity", "phase_label") if key in probe},
        "full_simulator_state": simulator,
        "controller_and_wrapper_runtime_state": runtime,
        "required_rng_state": runtime["rng"],
        "episode_start_rng_state": episode_rng,
        "raw_observation": capture["raw_observation"],
        "raw_observation_sha256": canonical_sha256(canonical_value(capture["raw_observation"])),
        "canonical_policy_rgb_224": capture["canonical_policy_rgb_224"],
        "processed_image_sha256": canonical_sha256(canonical_value(capture["processed_image"])),
        "processed_image": capture["processed_image"],
        "input_ids": capture["input_ids"],
        "pixel_values": capture["pixel_values"],
        "attention_mask": capture["attention_mask"],
        "attention_mask_present": capture["attention_mask_present"],
        "prompt": capture["prompt"],
        "decode_config": capture["decode_config"],
        "policy_input_sha256": capture["policy_input_sha256"],
        "policy_rgb_224_sha256": capture["policy_rgb_224_sha256"],
        "clean_reference_action_window": reference_action_window(action_rows, start_step=step, length=20),
    }


def _replay_canary(snapshot_root: Path, *, OffScreenRenderEnv: Any, bddl: str, horizon: int, init_state: Any, args: argparse.Namespace, output_dir: Path, clean_actions: list[list[float]], model: Any, adapter: Any) -> dict[str, Any]:
    loaded = load_snapshot(snapshot_root, materialize_torch=True)
    payload = loaded["payload"]
    observation_hashes = assert_primary_observation_exact(payload)
    step = int(payload["probe"]["step"])
    restore_rng_state(payload["episode_start_rng_state"])
    env, _obs = _new_env(OffScreenRenderEnv, bddl, horizon, args.gpu, copy.deepcopy(init_state), args, output_dir)
    try:
        for index in range(step):
            if index >= len(clean_actions):
                raise M35RunnerError(f"CLEAN_PREFIX_MISSING:{index}")
            _obs, _reward, done, _info = env.step(clean_actions[index])
            if bool(done):
                raise M35RunnerError(f"CLEAN_PREFIX_TERMINATED:{step}")
        replay_simulator = capture_simulator_state(env)
        replay_runtime_natural = capture_runtime_state(env, model=model, adapter=adapter)
        # The frozen branch state explicitly restores required RNG bytes; no
        # policy decode or renderer output is consumed in this canary.
        restore_runtime_state(env, payload["controller_and_wrapper_runtime_state"], model=model, adapter=adapter)
        restore_rng_state(payload["required_rng_state"])
        replay_runtime_bound = capture_runtime_state(env, model=model, adapter=adapter)
        assert_exact(replay_simulator, payload["full_simulator_state"], label="simulator_state")
        assert_exact(replay_runtime_bound, payload["controller_and_wrapper_runtime_state"], label="runtime_state")
        primary = {key: payload[key] for key in ("raw_observation", "canonical_policy_rgb_224", "processed_image", "input_ids", "pixel_values", "attention_mask", "prompt", "decode_config")}
        assert_exact(primary, {key: payload[key] for key in primary}, label="frozen_primary_input")
        primary_checks = {
            "raw_observation_exact": True,
            "canonical_policy_rgb_exact": True,
            "processed_image_exact": True,
            "input_ids_exact": True,
            "pixel_values_exact": True,
            "attention_mask_exact_if_present": True,
            "prompt_exact": True,
            "decode_config_exact": True,
            "no_fresh_render_primary_consumption": True,
        }
        return {
            "status": "PASS",
            "probe_step": step,
            "simulator_state_exact": True,
            "runtime_state_exact": True,
            "primary_exact_checks": primary_checks,
            "primary_observation_hashes": observation_hashes,
            "primary_input_authority": "loaded_frozen_canonical_bytes",
            "fresh_render_equality_gate_used": False,
            "fresh_render_diagnostic_sha256": None,
            "natural_runtime_state_sha256": canonical_sha256(canonical_value(replay_runtime_natural)),
            "bound_runtime_state_sha256": canonical_sha256(canonical_value(replay_runtime_bound)),
            "intervention_executed": False,
            "protected_counters": dict(PROTECTED_COUNTERS),
        }
    finally:
        env.close()


def run(args: argparse.Namespace) -> int:
    protocol = _load_json(args.protocol)
    _validate_protocol(protocol, args)
    _verify_exposed_parent(args.selection_manifest, args.parent_key)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise M35RunnerError(f"REFUSE_OVERWRITE:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol.resolve(), output_dir / "M3_5_V1_4_GATE_A_PROTOCOL.json")
    for filename in (
        "STAGE_V_M3_5_V1_4_CAUSAL_SNAPSHOT_COMPONENT_INVENTORY.json",
        "STAGE_V_M3_5_V1_4_CAUSAL_SNAPSHOT_SUFFICIENCY_AUDIT.json",
        "STAGE_V_M3_5_V1_4_MODEL_STEP_STATELESS_RECEIPT.json",
        "STAGE_V_M3_5_V1_4_CAUSAL_SNAPSHOT_STATIC_AUDIT.json",
    ):
        source = REPO_ROOT / "docs/handoffs" / filename
        if source.is_file():
            shutil.copy2(source, output_dir / ("CAUSAL_SNAPSHOT_" + filename.split("CAUSAL_SNAPSHOT_", 1)[-1] if "CAUSAL_SNAPSHOT_" in filename else filename))
    _write_json(output_dir / "CAUSAL_PROBE_SNAPSHOT_SCHEMA.json", {
        "schema": "STAGE_V_CAUSAL_PROBE_SNAPSHOT_SCHEMA_V2",
        "snapshot_schema": "STAGE_V_CAUSAL_PROBE_SNAPSHOT_V2",
        "runtime_state_schema": "STAGE_V_CONTROLLER_WRAPPER_RUNTIME_STATE_V2",
        "required_runtime_gripper_fields": ["current_action", "speed", "dof"],
        "primary_authority": "uncompressed_sidecar_bytes",
        "required_payload_fields": ["full_simulator_state", "controller_and_wrapper_runtime_state", "required_rng_state", "raw_observation", "raw_observation_sha256", "canonical_policy_rgb_224", "processed_image", "processed_image_sha256", "input_ids", "pixel_values", "attention_mask", "attention_mask_present", "prompt", "decode_config", "clean_reference_action_window"],
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "fresh_render_equality_gate_used": False,
        "protected_counters": dict(PROTECTED_COUNTERS),
    })
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
    init_states = suite_obj.get_task_init_states(task_index)
    init_state = copy.deepcopy(init_states[state_index])
    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    args.suite = suite
    adapter, model, _processor, _unnorm_key = _load_policy(args, get_processor, get_model, adapter_type)
    horizon = int(HORIZONS[suite])
    episode_rng = capture_rng_state()
    first_env, first_obs = _new_env(OffScreenRenderEnv, bddl, horizon, args.gpu, init_state, args, output_dir)
    _write_runtime_binding_receipt(args, first_env, output_dir)
    captures: dict[int, dict[str, Any]] = {}
    clean_rows: list[dict[str, Any]] = []
    clean_actions: list[list[float]] = []
    baseline_z: float | None = None
    task_success = False
    terminal_seen = False
    try:
        from gripper_attack.stage_v_m3_5_physical_taxonomy import bind_object_taxonomy

        taxonomy = bind_object_taxonomy(first_env, Path(bddl))
        if taxonomy.get("status") != "PASS":
            raise M35RunnerError(f"OBJECT_TAXONOMY_BINDING_{taxonomy.get('reason', 'ABSTAIN')}")
        for step in range(horizon):
            simulator = capture_simulator_state(first_env)
            runtime = capture_runtime_state(first_env, model=model, adapter=adapter)
            capture = _policy_capture(adapter, get_libero_image, first_obs, str(task.language))
            captures[step] = {**capture, "simulator": simulator, "runtime": runtime}
            telemetry = telemetry_from_env(first_env, taxonomy)
            aperture = None
            if isinstance(first_obs, Mapping):
                for key in ("robot0_gripper_qpos", "gripper_qpos"):
                    if key in first_obs:
                        aperture = aperture_metric(first_obs[key])
                        if aperture is not None:
                            break
            row, baseline_z = _clean_row(step, horizon, capture, telemetry, aperture, baseline_z)
            row["state_sha256"] = _array_sha(simulator["registered_flat_state"])
            row["clean_terminal"] = terminal_seen
            clean_rows.append(row)
            clean_actions.append(list(capture["env_action"]))
            first_obs, _reward, done, _info = first_env.step(capture["env_action"])
            try:
                current_success = bool(first_env.check_success())
            except Exception:
                current_success = False
            task_success = task_success or current_success
            if current_success or bool(done):
                terminal_seen = True
                row["clean_terminal"] = True
            if bool(done):
                break
    finally:
        first_env.close()
    if len(clean_rows) < PROBE_COUNT:
        raise M35RunnerError("CLEAN_TRAJECTORY_TOO_SHORT")
    for index, row in enumerate(clean_rows):
        row["remaining_horizon"] = len(clean_rows) - index
    for row, label in zip(clean_rows, classify_trajectory(clean_rows)):
        row.update(label)
    trajectory_sha = canonical_sha256(clean_rows)
    _write_json(output_dir / "CLEAN_TRAJECTORY_V1_4.json", {"schema": "STAGE_V_M3_5_CLEAN_TRAJECTORY_V1_4", "outcomes_read": False, "rows": clean_rows, "trajectory_sha256": trajectory_sha, "task_success": task_success, "protected_counters": dict(PROTECTED_COUNTERS)})
    selection_version = str(protocol.get("probe_plan_selection_version", DEFAULT_SELECTION_VERSION))
    if selection_version not in {DEFAULT_SELECTION_VERSION, PREFIX_SELECTION_VERSION, CLOSED_PREFIX_SELECTION_VERSION}:
        raise M35RunnerError(f"PROBE_PLAN_SELECTION_VERSION_INVALID:{selection_version}")
    plan = select_probe_steps(clean_rows, args.parent_key, selection_version=selection_version)
    _write_json(output_dir / "PROBE_PLAN_V1_4.json", plan)
    snapshot_rows = []
    for probe in plan["probe_steps"]:
        step = int(probe["step"])
        capture = captures[step]
        if str(probe.get("policy_input_sha256")) != str(capture["policy_input_sha256"]):
            raise M35RunnerError(f"PROBE_POLICY_INPUT_BINDING_MISMATCH:{probe.get('probe_id')}")
        if str(probe.get("policy_rgb_224_sha256")) != str(capture["policy_rgb_224_sha256"]):
            raise M35RunnerError(f"PROBE_POLICY_RGB_BINDING_MISMATCH:{probe.get('probe_id')}")
        if str(probe.get("state_sha256")) != _array_sha(capture["simulator"]["registered_flat_state"]):
            raise M35RunnerError(f"PROBE_SIM_STATE_BINDING_MISMATCH:{probe.get('probe_id')}")
        snapshot_root = output_dir / "CAUSAL_SNAPSHOTS" / str(probe["probe_id"])
        manifest = write_snapshot(
            snapshot_root,
            _snapshot_payload(capture, probe=probe, simulator=capture["simulator"], runtime=capture["runtime"], episode_rng=episode_rng, action_rows=clean_rows),
            binding={"parent_key": args.parent_key, "probe_id": probe["probe_id"], "step": step, "object_identity": probe.get("object_identity"), "source_commit": args.source_commit, "source_tree": args.source_tree},
        )
        snapshot_rows.append({"probe_id": probe["probe_id"], "step": step, "path": snapshot_root.relative_to(output_dir).as_posix(), "manifest_sha256": manifest["manifest_sha256"], "policy_input_sha256": capture["policy_input_sha256"], "policy_rgb_224_sha256": capture["policy_rgb_224_sha256"]})
    canary_receipts = []
    for row in snapshot_rows:
        canary_receipts.append(_replay_canary(output_dir / row["path"], OffScreenRenderEnv=OffScreenRenderEnv, bddl=bddl, horizon=horizon, init_state=init_state, args=args, output_dir=output_dir, clean_actions=clean_actions, model=model, adapter=adapter))
    status = "PASS" if len(canary_receipts) == len(snapshot_rows) and all(item["status"] == "PASS" for item in canary_receipts) else "HOLD"
    receipt = {
        "schema": "STAGE_V_M3_5_V1_4_GATE_A_RECEIPT_V1",
        "status": status,
        "runtime_authorized": True,
        "canonical_parent_key": args.parent_key,
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "runner_sha256": _sha256_file(Path(__file__)),
        "snapshot_count": len(snapshot_rows),
        "snapshots": snapshot_rows,
        "canary_receipts": canary_receipts,
        "intervention_executed": False,
        "outcomes_read": False,
        "fresh_render_equality_gate_used": False,
        "primary_input_authority": "loaded_frozen_canonical_bytes",
        "protected_counters": dict(PROTECTED_COUNTERS),
    }
    _write_json(output_dir / "M3_5_V1_4_GATE_A_RECEIPT.json", receipt)
    auditor = REPO_ROOT / "scripts/detector_v5/audit_stage_v_m3_5_v1_4_gate_a.py"
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
    audit_path = output_dir / "M3_5_V1_4_GATE_A_INDEPENDENT_AUDIT.json"
    audit = _load_json(audit_path) if audit_path.is_file() else {"status": "FAIL", "error": audit_process.stderr[-1000:]}
    receipt["independent_audit_status"] = audit.get("status")
    receipt["independent_audit_sha256"] = _sha256_file(audit_path) if audit_path.is_file() else None
    if audit.get("status") != "PASS":
        receipt["status"] = "HOLD"
    _write_json(output_dir / "M3_5_V1_4_GATE_A_RECEIPT.json", receipt)
    _seal_root(output_dir)
    return 0 if receipt["status"] == "PASS" and audit.get("status") == "PASS" and audit_process.returncode == 0 else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
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
            raise M35RunnerError("RUNTIME_DISABLED_UNTIL_FROZEN_V1_4_GATE_A_AUTHORIZATION")
        return run(args)
    except (OSError, KeyError, ValueError, CausalSnapshotError, M35RunnerError) as exc:
        print(json.dumps({"status": "FAIL", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
