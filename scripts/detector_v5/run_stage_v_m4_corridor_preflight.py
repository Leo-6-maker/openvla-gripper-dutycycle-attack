#!/usr/bin/env python3
"""Run the clean-only, outcome-blind M4 corridor qualification."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from gripper_attack.stage_v_causal_observation_snapshot import (  # noqa: E402
    capture_rng_state,
    capture_runtime_state,
    capture_simulator_state,
)
from gripper_attack.stage_v_m3_5_physical_taxonomy import (  # noqa: E402
    aperture_metric,
    bind_object_taxonomy,
    telemetry_from_env,
)
from gripper_attack.stage_v_m3_5_phase_classifier import classify_trajectory  # noqa: E402
from scripts.detector_v5.build_stage_v_m3_5_probe_plan import (  # noqa: E402
    ProbePlanError,
    select_probe_steps,
)
from scripts.detector_v5.run_stage_v_m3_5_v1_4_gate_a import (  # noqa: E402
    _clean_row,
    _policy_capture,
)
from scripts.detector_v5.run_stage_v_m4_matched_parent import (  # noqa: E402
    COUNTERS,
    HORIZONS,
)
from scripts.detector_v5.run_stage_v_canonical_clean import (  # noqa: E402
    _load_external_modules,
    _load_policy,
    _write_runtime_binding_receipt,
)
from scripts.detector_v5.run_stage_v_m3_5_intervention_parent import _new_env  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validate(protocol: Mapping[str, Any], authorization: Mapping[str, Any], args: argparse.Namespace) -> None:
    if protocol.get("schema") != "STAGE_V_M4_CORRIDOR_QUALIFICATION_PROTOCOL_V1" or protocol.get("status") != "FROZEN_RUNTIME_AUTHORIZED" or protocol.get("runtime_authorized") is not True:
        raise ValueError("M4_CORRIDOR_PROTOCOL_NOT_AUTHORIZED")
    source = protocol.get("source_binding", {})
    if source.get("runtime_commit") != args.source_commit or source.get("runtime_tree") != args.source_tree:
        raise ValueError("M4_CORRIDOR_SOURCE_BINDING_MISMATCH")
    if authorization.get("status") != "PASS" or authorization.get("protocol_sha256") != _sha(args.protocol):
        raise ValueError("M4_CORRIDOR_AUTHORIZATION_BINDING_INVALID")
    if protocol.get("protected_counters") != COUNTERS:
        raise ValueError("M4_CORRIDOR_PROTECTED_BOUNDARY_INVALID")
    split = Path(str(protocol["inputs"]["formal_parent_split_path"])).resolve()
    if _sha(split) != protocol["inputs"].get("formal_parent_split_sha256"):
        raise ValueError("M4_CORRIDOR_SPLIT_BINDING_INVALID")
    value = _load(split)
    if value.get("schema") != "STAGE_V_TRAIN_VAL_TEST_PARENT_SPLIT_V1" or value.get("status") != "FROZEN" or len(value.get("parents", [])) != 40:
        raise ValueError("M4_CORRIDOR_SPLIT_INVALID")


def _run_clean(args: argparse.Namespace, parent: Mapping[str, Any], output: Path) -> dict[str, Any]:
    key = str(parent["canonical_parent_key"])
    suite = str(parent["suite"])
    task_index = int(parent["task_index"])
    state_index = int(parent["state_index"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
    get_image, get_processor, get_model, adapter_type, benchmark, libero_runtime = _load_external_modules(args.official_snapshot_root, args.upstream_root)
    get_libero_path, OffScreenRenderEnv = libero_runtime
    suite_obj = benchmark.get_benchmark_dict()[suite]()
    task = suite_obj.get_task(task_index)
    init_state = copy.deepcopy(suite_obj.get_task_init_states(task_index)[state_index])
    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    args.suite = suite
    adapter, model, _processor, _unnorm_key = _load_policy(args, get_processor, get_model, adapter_type)
    horizon = int(HORIZONS[suite])
    episode_rng = capture_rng_state()
    env, obs = _new_env(OffScreenRenderEnv, bddl, horizon, args.gpu, init_state, args, output)
    _write_runtime_binding_receipt(args, env, output)
    captures: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    baseline_z: float | None = None
    task_success = False
    terminal_seen = False
    try:
        taxonomy = bind_object_taxonomy(env, Path(bddl))
        if taxonomy.get("status") != "PASS":
            raise RuntimeError(f"OBJECT_TAXONOMY_BINDING_{taxonomy.get('reason', 'ABSTAIN')}")
        for step in range(horizon):
            simulator = capture_simulator_state(env)
            runtime = capture_runtime_state(env, model=model, adapter=adapter)
            capture = _policy_capture(adapter, get_image, obs, str(task.language))
            captures[step] = {**capture, "simulator": simulator, "runtime": runtime}
            aperture = None
            if isinstance(obs, Mapping):
                for field in ("robot0_gripper_qpos", "gripper_qpos"):
                    if field in obs:
                        aperture = aperture_metric(obs[field])
                        if aperture is not None:
                            break
            telemetry = telemetry_from_env(env, taxonomy)
            row, baseline_z = _clean_row(step, horizon, capture, telemetry, aperture, baseline_z)
            row["state_sha256"] = hashlib.sha256(np.asarray(simulator["registered_flat_state"]).tobytes(order="C")).hexdigest()
            row["clean_terminal"] = terminal_seen
            rows.append(row)
            obs, _reward, done, _info = env.step(capture["env_action"])
            try:
                current_success = bool(env.check_success())
            except Exception:
                current_success = False
            task_success = task_success or current_success
            if current_success or bool(done):
                terminal_seen = True
                rows[-1]["clean_terminal"] = True
            if bool(done):
                break
    finally:
        env.close()
    if not task_success:
        receipt = {"schema": "STAGE_V_M4_CORRIDOR_PREFLIGHT_V1", "status": "CLEAN_FAILURE", "canonical_parent_key": key, "replicate": args.replicate, "suite": suite, "task_index": task_index, "state_index": state_index, "source_commit": args.source_commit, "source_tree": args.source_tree, "clean_success": False, "corridor_candidate_count": 0, "probe_count": 0, "m4_probe_eligible": False, "m4_primary_horizon_complete": False, "outcomes_read": False, "old_artifacts_reused": False, "source_artifacts_modified": False, "protected_counters": dict(COUNTERS)}
        _write(output / "M4_CORRIDOR_PREFLIGHT.json", receipt)
        return receipt
    for index, row in enumerate(rows):
        row["remaining_horizon"] = len(rows) - index
    for row, label in zip(rows, classify_trajectory(rows)):
        row.update(label)
    _write(output / "CLEAN_TRAJECTORY_V1_4.json", {"schema": "STAGE_V_M4_CLEAN_TRAJECTORY_V1", "outcomes_read": False, "rows": rows, "task_success": True, "protected_counters": dict(COUNTERS)})
    try:
        plan = select_probe_steps(rows, key)
        status = "PASS"
        reason = "M4_CORRIDOR_24_EXACT"
        corridor = int(plan["corridor_candidate_count"])
        probe_count = len(plan["probe_steps"])
        _write(output / "PROBE_PLAN_V1_4.json", plan)
    except ProbePlanError as exc:
        status = "INELIGIBLE"
        reason = str(exc)
        corridor = sum(1 for row in rows if row.get("phase_eligible") is True and row.get("contact_telemetry_valid") is True and row.get("object_gripper_contact") is True and row.get("clean_terminal") is not True and isinstance(row.get("object_identity"), str) and row.get("object_identity") and int(row.get("remaining_horizon", 0)) >= 20)
        probe_count = 0
    receipt = {"schema": "STAGE_V_M4_CORRIDOR_PREFLIGHT_V1", "status": status, "reason": reason, "canonical_parent_key": key, "replicate": args.replicate, "suite": suite, "task_index": task_index, "state_index": state_index, "source_commit": args.source_commit, "source_tree": args.source_tree, "clean_success": True, "corridor_candidate_count": corridor, "probe_count": probe_count, "m4_probe_eligible": status == "PASS", "m4_primary_horizon_complete": True, "trajectory_sha256": _sha(output / "CLEAN_TRAJECTORY_V1_4.json"), "outcomes_read": False, "old_artifacts_reused": False, "source_artifacts_modified": False, "protected_counters": dict(COUNTERS)}
    _write(output / "M4_CORRIDOR_PREFLIGHT.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-snapshot-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--replicate", choices=("A", "B"), default="A")
    args = parser.parse_args(argv)
    try:
        protocol = _load(args.protocol.resolve())
        authorization = _load(args.authorization.resolve())
        _validate(protocol, authorization, args)
        split = _load(Path(str(protocol["inputs"]["formal_parent_split_path"])).resolve())
        parent = next(row for row in split["parents"] if str(row["canonical_parent_key"]) == args.parent_key)
        output = args.output_dir.resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"REFUSE_OVERWRITE:{output}")
        output.mkdir(parents=True, exist_ok=False)
        result = _run_clean(args, parent, output)
        return 0
    except (OSError, KeyError, ValueError, RuntimeError, StopIteration) as exc:
        print(json.dumps({"status": "FAIL", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
