#!/usr/bin/env python3
"""R10.4-R4B: replay sealed clean actions through official LIBERO only.

This script never imports OpenVLA, never creates a detector, and never
generates or mutates an action.  It is intentionally FIT-only and stops with
HOLD_SOURCE_INCOMPLETE when any measured source component is unavailable.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
from gripper_attack.r10_4_runtime import (
    ACTION_DIM,
    FEATURE_ABS_TOLERANCE,
    FEATURE_NAMES,
    FEATURE_ORDER_SHA256,
    HoldSourceIncomplete,
    NUM_STEPS_WAIT,
    R10_4ContractError,
    canonical_json_sha,
    pickle4_sha,
    sha256_file,
    verify_checksum_manifest,
    verify_legacy_artifact_manifest,
)
from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2


SUITE_BENCHMARK = {
    "libero_object": "LIBERO_OBJECT",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_goal": "LIBERO_GOAL",
    "libero_10": "LIBERO_10",
}
FIXED_INDICES = (0, 99, 199)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _selection(clean_root: Path) -> dict[str, Any]:
    suite_root = clean_root / "libero_10"
    identities = sorted(
        f"libero_10/{task.name}/{state.name}"
        for task in suite_root.iterdir()
        if task.is_dir()
        for state in task.iterdir()
        if state.is_dir() and state.name.startswith("state_") and int(state.name.split("_")[1]) < 20
    )
    if len(identities) != 200 or len(set(identities)) != 200:
        raise HoldSourceIncomplete(f"MULTI_OBJECT_FIT_DIRECTORY_CLOSURE:{len(identities)}")
    selected = [identities[index] for index in FIXED_INDICES]
    return {
        "schema": "R10_4_REPLAY_SELECTION_V1",
        "selection_status": "FROZEN_BEFORE_PARITY_OUTCOME",
        "selection_rule": "sorted eligible multi-object FIT identity directories; fixed zero-based indices [0,99,199]",
        "eligible_identity_count": len(identities),
        "eligible_identity_sha256": canonical_json_sha(identities),
        "selected_indices": list(FIXED_INDICES),
        "selected_identities": selected,
    }


def _suite_benchmark(suite: str) -> Any:
    from libero.libero import benchmark

    constructors = benchmark.get_benchmark_dict()
    try:
        return constructors[suite]()
    except KeyError as exc:
        raise HoldSourceIncomplete(f"LIBERO_SUITE_MISSING:{suite}") from exc


def _source(clean_root: Path, s1_root: Path, identity: str) -> dict[str, Any]:
    parts = identity.split("/")
    if len(parts) != 3:
        raise HoldSourceIncomplete(f"IDENTITY_FORMAT:{identity}")
    suite, task_name, state_name = parts
    task_idx = int(task_name.split("_")[1])
    state_id = int(state_name.split("_")[1])
    clean_episode = clean_root.joinpath(*parts)
    s1_episode = s1_root.joinpath(*parts)
    clean_meta = _json(clean_episode / "episode_metadata.json")
    if clean_meta.get("split") != "FIT" or clean_meta.get("condition") != "CLEAN":
        raise HoldSourceIncomplete(f"SOURCE_SPLIT_OR_CONDITION:{identity}")
    if clean_meta.get("attack_enabled") is not False:
        raise R10_4ContractError(f"SOURCE_ATTACK_FLAG:{identity}")
    raw_seal = verify_legacy_artifact_manifest(clean_episode)
    s1_seal = verify_checksum_manifest(s1_episode)
    raw_rows = _jsonl(clean_episode / "step_records.jsonl")
    s1_rows = _jsonl(s1_episode / "student_input_records.jsonl")
    if not raw_rows or len(raw_rows) != len(s1_rows):
        raise HoldSourceIncomplete(f"STEP_CLOSURE:{identity}")
    if any(int(row.get("step", -1)) != index for index, row in enumerate(raw_rows)):
        raise R10_4ContractError(f"RAW_STEP_SEQUENCE:{identity}")
    if any(int(row.get("step", -1)) != index for index, row in enumerate(s1_rows)):
        raise R10_4ContractError(f"S1_STEP_SEQUENCE:{identity}")
    actions: list[list[float]] = []
    for row in raw_rows:
        action = row.get("action_env")
        if not isinstance(action, list) or len(action) != ACTION_DIM:
            raise HoldSourceIncomplete(f"RAW_ACTION_MISSING:{identity}")
        values = np.asarray(action, dtype=np.float32)
        if not np.isfinite(values).all():
            raise R10_4ContractError(f"RAW_ACTION_NONFINITE:{identity}")
        actions.append([float(value) for value in action])
    if any(row.get("feature_order_sha256") not in {None, FEATURE_ORDER_SHA256} for row in s1_rows):
        raise R10_4ContractError(f"S1_FEATURE_ORDER:{identity}")

    benchmark = _suite_benchmark(suite)
    task = benchmark.get_task(task_idx)
    metadata_task_name = str(clean_meta.get("task_name", ""))
    if metadata_task_name and metadata_task_name != str(task.name):
        raise R10_4ContractError(f"TASK_NAME_BINDING:{identity}")
    states = benchmark.get_task_init_states(task_idx)
    initial_state = states[state_id]
    initial_sha = pickle4_sha(initial_state)
    if initial_sha != clean_meta.get("initial_state_sha256"):
        raise HoldSourceIncomplete(f"INITIAL_STATE_SHA_MISMATCH:{identity}")
    from libero.libero import get_libero_path

    bddl_path = Path(get_libero_path("bddl_files")) / str(task.problem_folder) / str(task.bddl_file)
    if not bddl_path.is_file():
        raise HoldSourceIncomplete(f"BDDL_MISSING:{identity}")
    return {
        "identity": identity,
        "suite": suite,
        "task_idx": task_idx,
        "state_id": state_id,
        "task_name": str(task.name),
        "task_language": str(task.language),
        "clean_episode": str(clean_episode),
        "s1_episode": str(s1_episode),
        "raw_artifact_recursive_sha256": raw_seal["recursive_sha256"],
        "s1_sha256sums_sha256": s1_seal["sha256sums_sha256"],
        "initial_state_sha256": initial_sha,
        "bddl_path": str(bddl_path),
        "bddl_sha256": sha256_file(bddl_path),
        "official_libero_task_name": str(task.name),
        "official_libero_task_language": str(task.language),
        "official_libero_upstream_git_commit_declared": clean_meta.get("libero_upstream_git_commit_actual"),
        "step_count": len(raw_rows),
        "action_sequence_sha256": canonical_json_sha(actions),
        "raw_rows": raw_rows,
        "s1_rows": s1_rows,
        "actions": actions,
        "initial_state": initial_state,
    }


def _eef_position(env: Any, observation: Any) -> np.ndarray:
    try:
        site_id = env.sim.model.site_name2id("gripper0_grip_site")
        return np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float32).reshape(3).copy()
    except Exception:
        value = np.asarray(observation.get("robot0_eef_pos", []), dtype=np.float32).reshape(-1)
        if value.size != 3:
            raise R10_4ContractError("EEF_SOURCE_MISSING")
        return value.copy()


def _step_env(env: Any, action: list[float]) -> tuple[Any, bool]:
    result = env.step(action)
    if len(result) != 4:
        raise R10_4ContractError("LIBERO_STEP_RETURN_ARITY")
    observation, _reward, done, _info = result
    return observation, bool(done)


def replay_one(source: dict[str, Any], render_gpu_device_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env, observation = build_v4_exact_env(
        source["bddl_path"], render_gpu_device_id, source["step_count"], NUM_STEPS_WAIT
    )
    adapter = SC5StreamingFeatureAdapterV2()
    per_step: list[dict[str, Any]] = []
    previous_eef: np.ndarray | None = None
    try:
        observation = env.set_init_state(copy.deepcopy(source["initial_state"]))
        for _ in range(NUM_STEPS_WAIT):
            observation, _done = _step_env(env, [0, 0, 0, 0, 0, 0, -1])
        for index, (raw, sealed) in enumerate(zip(source["raw_rows"], source["s1_rows"])):
            action = np.asarray(source["actions"][index], dtype=np.float32)
            qpos = np.asarray(observation.get("robot0_gripper_qpos", []), dtype=np.float32).reshape(-1)
            if qpos.size < 2:
                raise R10_4ContractError(f"GRIPPER_QPOS_SOURCE_MISSING:{source['identity']}:{index}")
            eef = _eef_position(env, observation)
            velocity = np.zeros(3, dtype=np.float32) if previous_eef is None else eef - previous_eef
            previous_eef = eef.copy()
            raw_gripper = float(raw["action_raw"][-1])
            stream = adapter.update(
                step_id=index,
                raw_gripper=raw_gripper,
                env_gripper=float(action[-1]),
                gripper_qpos=float(qpos[:2].sum()),
                gripper_opening_proxy=float(np.abs(qpos[:2]).sum()),
                eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]),
                eef_vx=float(velocity[0]), eef_vy=float(velocity[1]), eef_vz=float(velocity[2]),
                action_dx=float(action[0]), action_dy=float(action[1]), action_dz=float(action[2]),
                action_gripper=raw_gripper,
            )
            online = np.asarray([stream["features"][name] for name in FEATURE_NAMES], dtype=np.float32)
            expected = np.asarray(sealed["features_25d"], dtype=np.float32)
            if expected.shape != (25,):
                raise R10_4ContractError(f"S1_FEATURE_SHAPE:{source['identity']}:{index}")
            diff = np.abs(online - expected)
            per_step.append(
                {
                    "identity": source["identity"],
                    "step": index,
                    "online_valid": bool(stream.get("valid")),
                    "sealed_valid": bool(sealed.get("valid")),
                    "valid_match": bool(stream.get("valid") is bool(sealed.get("valid"))),
                    "max_abs_feature_error": float(diff.max()),
                    "feature_pass": bool(float(diff.max()) <= FEATURE_ABS_TOLERANCE),
                    "close_onset_error": float(diff[16]),
                    "action_source_unchanged": True,
                }
            )
            observation, done = _step_env(env, source["actions"][index])
            if done and index + 1 < source["step_count"]:
                raise R10_4ContractError(f"EARLY_ENV_DONE:{source['identity']}:{index}")
    finally:
        env.close()
    errors = np.asarray([row["max_abs_feature_error"] for row in per_step], dtype=np.float64)
    summary = {
        "identity": source["identity"],
        "status": "PASS" if per_step and bool(np.all(errors <= FEATURE_ABS_TOLERANCE)) and all(row["valid_match"] for row in per_step) else "FAIL",
        "step_count": len(per_step),
        "sealed_step_count": source["step_count"],
        "valid_match_count": sum(row["valid_match"] for row in per_step),
        "feature_pass_count": sum(row["feature_pass"] for row in per_step),
        "max_abs_feature_error": float(errors.max()) if len(errors) else None,
        "mean_abs_feature_error": float(errors.mean()) if len(errors) else None,
        "feature_abs_tolerance": FEATURE_ABS_TOLERANCE,
        "action_sequence_sha256": source["action_sequence_sha256"],
        "action_mutated": False,
        "event_reset_state": "NOT_PRESENT_IN_S1_SOURCE",
    }
    return summary, per_step


def _seal_root(staging: Path) -> dict[str, Any]:
    files = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file() or path.name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            continue
        files.append({"path": path.relative_to(staging).as_posix(), "sha256": sha256_file(path), "size": path.stat().st_size})
    sums = staging / "SHA256SUMS"
    sums.write_text("".join(f"{row['sha256']}  {row['path']}\n" for row in files), encoding="utf-8")
    sums_sha = sha256_file(sums)
    (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {"file_count": len(files), "sha256sums_sha256": sums_sha, "files": files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", required=True, type=Path)
    parser.add_argument("--s1-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--render-gpu-device-id", required=True, type=int)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit("OUTPUT_ROOT_ALREADY_EXISTS")
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_root.name}.{uuid.uuid4().hex}.staging.", dir=args.output_root.parent))
    selection = _selection(args.clean_root)
    _write_json(staging / "selection_manifest.json", selection)
    protocol = {
        "schema": "R10_4_R4_LIBERO_REPLAY_PARITY_PROTOCOL_V1",
        "scope": "FIT_ONLY_REPLAY_NO_OPENVLA_NO_DETECTOR_NO_ACTION_MUTATION",
        "selected_indices": list(FIXED_INDICES),
        "feature_order_sha256": FEATURE_ORDER_SHA256,
        "feature_abs_tolerance": FEATURE_ABS_TOLERANCE,
        "action_dim": ACTION_DIM,
        "num_steps_wait": NUM_STEPS_WAIT,
        "action_rule": "pass sealed action_env directly to env.step",
        "protected_splits_read": 0,
        "openvla_model_loaded": False,
        "detector_executed": False,
    }
    _write_json(staging / "protocol.json", protocol)
    summaries: list[dict[str, Any]] = []
    per_step_rows: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    status = "PASS"
    try:
        for identity in selection["selected_identities"]:
            source = _source(args.clean_root, args.s1_root, identity)
            bindings.append({key: value for key, value in source.items() if key not in {"raw_rows", "s1_rows", "actions", "initial_state"}})
            summary, rows = replay_one(source, args.render_gpu_device_id)
            summaries.append(summary)
            per_step_rows.extend(rows)
            if summary["status"] != "PASS":
                status = "FAIL"
    except HoldSourceIncomplete as exc:
        status = "HOLD_SOURCE_INCOMPLETE"
        _write_json(staging / "failure.json", {"status": status, "reason": str(exc)})
    except Exception as exc:
        status = "FAIL"
        _write_json(staging / "failure.json", {"status": status, "reason": f"{type(exc).__name__}:{exc}"})
    _write_json(staging / "source_binding.json", {"status": status, "bindings": bindings})
    _write_json(staging / "parity_summary.json", {"status": status, "episodes": summaries, "protocol": protocol})
    with (staging / "parity_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in summaries for key in row}) or ["identity", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    with (staging / "per_step_parity.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_step_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write_json(staging / "MANIFEST.json", {
        "schema": "R10_4_R4_REPLAY_EVIDENCE_V1",
        "status": status,
        "selection_manifest": selection,
        "source_artifact_mutation": 0,
        "openvla_model_loaded": False,
        "detector_executed": False,
        "action_mutated": False,
    })
    _write_json(staging / "runtime_audit.json", {
        "status": status,
        "openvla_model_loaded": False,
        "detector_executed": False,
        "action_mutated": False,
        "source_artifact_mutation": 0,
    })
    seal = _seal_root(staging)
    os.replace(staging, args.output_root)
    print(json.dumps({"status": status, "output_root": str(args.output_root), "sha256sums_sha256": seal["sha256sums_sha256"]}, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
