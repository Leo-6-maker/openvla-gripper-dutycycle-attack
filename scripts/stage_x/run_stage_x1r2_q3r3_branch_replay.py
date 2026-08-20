#!/usr/bin/env python3
"""Run one-suite Q3R3 reference-clean and fixed-action branch replay qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "stage_x"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
import run_stage_x1r_t1d1_screening_clean as clean

from gripper_attack.stage_x_q3r3_branch_replay import BranchReplay, ReferenceClean, compare_branch_state
from gripper_attack.stage_v_causal_observation_snapshot import capture_runtime_state, capture_simulator_state

PROTOCOL = REPO / "configs/STAGE_X_X1R2_Q3R3_BRANCH_REPLAY_PROTOCOL_V1.json"
AUTHORITY = REPO / "configs/STAGE_X_X1R2_Q3R2_RUNTIME_AUTHORITY_V1.json"
POOL = REPO / "reports/STAGE_X_X1R2_Q3R2_ENGINEERING_FIXTURE_POOL_V1.json"
CONTRACT = REPO / "configs/STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1.json"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
HORIZONS = {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}
EXPOSED = {"libero_10/task_08/state_44", "libero_goal/task_02/state_37", "libero_object/task_01/state_34", "libero_spatial/task_09/state_29"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True, stderr=subprocess.STDOUT).strip()


def gpu_receipt(physical_gpu: int, *, require_free: bool = True) -> dict[str, Any]:
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits", "-i", str(physical_gpu)], text=True).strip()
    fields = [item.strip() for item in query.split(",")]
    receipt = {"physical_gpu": int(fields[0]), "gpu_uuid": fields[1], "free_memory_mib": int(fields[2]), "used_memory_mib": int(fields[3]), "utilization_gpu_percent": int(fields[4]), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader", "-i", str(physical_gpu)], text=True).strip()
    receipt["compute_apps"] = [line.strip() for line in apps.splitlines() if line.strip()]
    if require_free and receipt["free_memory_mib"] <= 20480:
        raise RuntimeError(f"GPU_RESOURCE_GATE_FAIL:{receipt}")
    return receipt


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


SOURCE_FILES = (
    "configs/STAGE_X_X1R2_Q3R3_BRANCH_REPLAY_PROTOCOL_V1.json",
    "scripts/stage_x/run_stage_x1r2_q3r3_branch_replay.py",
    "src/gripper_attack/stage_x_q3r3_branch_replay.py",
    "configs/STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1.json",
    "reports/STAGE_X_X1R2_Q3R2_ENGINEERING_FIXTURE_POOL_V1.json",
)


def source_receipt(source_commit: str, source_tree: str) -> dict[str, Any]:
    observed_head = git("rev-parse", "HEAD")
    observed_tree = git("rev-parse", "HEAD^{tree}")
    if source_commit != observed_head or source_tree != observed_tree:
        raise RuntimeError(f"SOURCE_BINDING_MISMATCH:expected={source_commit}/{source_tree}:observed={observed_head}/{observed_tree}")
    return {
        "commit": source_commit,
        "tree": source_tree,
        "repository_observed_head": observed_head,
        "repository_observed_tree": observed_tree,
        "status_porcelain": git("status", "--porcelain"),
        "runtime_file_blobs": {path: git("rev-parse", f"HEAD:{path}") for path in SOURCE_FILES},
    }


def seed_for(key: str) -> int:
    return int(hashlib.sha256(f"STAGE_X1R2_Q3R3_REFERENCE_CLEAN_V1_20260820|{key}".encode()).hexdigest()[:8], 16)


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    protocol = load_json(PROTOCOL)
    authority = load_json(AUTHORITY)
    if protocol.get("status") != "FROZEN_ENGINEERING_BRANCH_REPLAY_ONLY":
        raise RuntimeError("Q3R3_C_PROTOCOL_NOT_FROZEN")
    if authority.get("status") != "STAGE_X_X1R2_Q3R2_RUNTIME_AUTHORITY_PASS":
        raise RuntimeError("Q3R2_RUNTIME_AUTHORITY_NOT_PASS")
    if protocol.get("reference_clean", {}).get("t5") != 5 or protocol.get("reference_clean", {}).get("h_phys") != 10:
        raise RuntimeError("Q3R3_C_TIMING_BINDING_MISMATCH")
    branch = protocol.get("branch_replay", {})
    if branch.get("repeat_count") != 2 or branch.get("prebranch_openvla_calls") != 0 or branch.get("prebranch_student_calls") != 0:
        raise RuntimeError("Q3R3_C_BRANCH_CONTRACT_MISMATCH")
    if git("rev-parse", "HEAD:reports/STAGE_X_X1R2_Q3R2_ENGINEERING_FIXTURE_POOL_V1.json") != protocol["q3r2_pool"]["git_blob_sha256"]:
        raise RuntimeError("Q3R2_POOL_GIT_BLOB_MISMATCH")
    pool = load_json(POOL).get("selected", [])
    if len(pool) != 48 or any(not row.get("permanent_exclusion") or row.get("scientific_use") or row.get("outcome_read") for row in pool):
        raise RuntimeError("Q3R2_POOL_SCOPE_INVALID")
    contract = load_json(CONTRACT)
    if contract.get("status") != "FROZEN_FOR_CLEAN_PARITY_ONLY" or contract.get("scientific_authority") != "X1R_NOT_AUTHORIZED":
        raise RuntimeError("SUITE_VICTIM_CONTRACT_SCOPE_INVALID")
    if git("rev-parse", "HEAD:configs/STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1.json") != authority["victim_contract"]["git_blob_sha"]:
        raise RuntimeError("SUITE_VICTIM_CONTRACT_GIT_BLOB_MISMATCH")
    if set(protocol.get("already_exposed_q3r2_keys", [])) != EXPOSED:
        raise RuntimeError("Q3R2_EXPOSED_KEY_BINDING_MISMATCH")
    return protocol, authority, contract, pool


def student_paths(authority: Mapping[str, Any]) -> dict[str, Path]:
    paths = {}
    for row in authority["student_binding"]["artifacts"]:
        if row["name"] in {"checkpoint", "normalization", "thresholds"}:
            paths[row["name"]] = Path(str(row["path"]))
    return paths


def verify_student_source_binding(authority: Mapping[str, Any]) -> None:
    for artifact in authority["student_binding"]["artifacts"]:
        raw_path = Path(str(artifact["path"]))
        path = raw_path if raw_path.is_absolute() else REPO / raw_path
        if not path.is_file() or sha256_file(path) != str(artifact["sha256"]):
            raise RuntimeError(f"STUDENT_SOURCE_BINDING_MISMATCH:{artifact['name']}:{path}")


def verify_model_identity(suite_cfg: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(suite_cfg["model_path"]))
    if not path.is_dir():
        raise RuntimeError(f"MODEL_DIRECTORY_MISSING:{path}")
    observed = clean.file_tree_digest(path)
    expected = suite_cfg["model_identity"]
    for key in ("file_count", "bytes", "tree_sha256"):
        if observed[key] != expected[key]:
            raise RuntimeError(f"MODEL_IDENTITY_MISMATCH:{key}:{observed[key]}!={expected[key]}")
    key_files: dict[str, str] = {}
    for relative, expected_sha in expected.get("key_files", {}).items():
        key_path = path / relative
        if not key_path.is_file():
            raise RuntimeError(f"MODEL_KEY_FILE_MISSING:{relative}")
        actual = sha256_file(key_path)
        key_files[relative] = actual
        if actual != expected_sha:
            raise RuntimeError(f"MODEL_KEY_FILE_SHA_MISMATCH:{relative}")
    return {"path": str(path), "identity": observed, "key_files": key_files}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parent_env(parent: Mapping[str, Any], physical_gpu: int):
    from libero.libero import benchmark, get_libero_path
    from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env

    suite = str(parent["suite"])
    task_idx = int(parent["task_idx"])
    state_id = int(parent["state_id"])
    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_idx)
    initial_states = task_suite.get_task_init_states(task_idx)
    if state_id >= len(initial_states):
        raise RuntimeError(f"STATE_ID_OUT_OF_RANGE:{parent['canonical_parent_key']}:{len(initial_states)}")
    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    env, obs = build_v4_exact_env(bddl, physical_gpu, HORIZONS[suite], 10)
    obs = env.set_init_state(initial_states[state_id])
    env, obs = apply_dummy_wait(env, obs, 10)
    return env, obs, str(task.language), initial_states[state_id]


def capture_branch_state(env: Any, parent: Mapping[str, Any], authority: Mapping[str, Any], suite_cfg: Mapping[str, Any], seed: int, step: int) -> dict[str, Any]:
    sim = _plain(capture_simulator_state(env))
    runtime = _plain(capture_runtime_state(env))
    data = sim.get("data", {})
    sim_state = sim.get("sim_state", {})
    environment = runtime.get("environment", {})
    wrapper_step = environment.get("timestep", environment.get("cur_time"))
    if wrapper_step is None:
        raise RuntimeError("BRANCH_WRAPPER_STEP_INDEX_MISSING")
    required_data = ("qpos", "qvel", "act", "ctrl")
    if any(field not in data for field in required_data) or "time" not in sim_state:
        raise RuntimeError("BRANCH_SIMULATOR_STATE_INCOMPLETE")
    return {
        "model_identity": str(suite_cfg["model_identity"]["tree_sha256"]),
        "suite_task_state_identity": str(parent["canonical_parent_key"]),
        "seed_and_dummy_wait": [int(seed), 10],
        "wrapper_step_index": _plain(wrapper_step),
        "qpos": data["qpos"],
        "qvel": data["qvel"],
        "act": data["act"],
        "ctrl": data["ctrl"],
        "time": sim_state["time"],
        "mocap_state": {"mocap_pos": data.get("mocap_pos"), "mocap_quat": data.get("mocap_quat")},
        "task_object_state": {"registered_flat_state": sim.get("registered_flat_state"), "udd_state": sim_state.get("udd_state"), "qacc": data.get("qacc"), "qacc_warmstart": data.get("qacc_warmstart"), "qfrc_applied": data.get("qfrc_applied"), "xfrc_applied": data.get("xfrc_applied")},
        "controller_state": {"robots": runtime.get("robots", []), "environment": {key: value for key, value in environment.items() if key not in {"np_random", "_np_random", "rng", "_rng", "random_state"}}},
    }


def reference_rollout(parent: Mapping[str, Any], model: Any, processor: Any, device: str, student: tuple[Any, np.ndarray, np.ndarray, float, float], suite_cfg: Mapping[str, Any], physical_gpu: int, out: Path) -> dict[str, Any]:
    key = str(parent["canonical_parent_key"])
    seed = seed_for(key)
    seed_all(seed)
    env = None
    rows: list[dict[str, Any]] = []
    image_cache: dict[int, bytes] = {}
    clean_success = False
    try:
        env, obs, instruction, initial_state = build_parent_env(parent, physical_gpu)
        previous_eef = None
        from gripper_attack.d8_streaming_features_v3 import D8StreamingFeatureAdapterV3, FEATURE_NAMES

        adapter = D8StreamingFeatureAdapterV3()
        for step in range(HORIZONS[parent["suite"]]):
            raw_image = np.asarray(obs["agentview_image"]).copy()
            if raw_image.dtype != np.uint8:
                raw_image = np.clip(raw_image, 0, 255).astype(np.uint8)
            image_cache[step] = raw_image.tobytes()
            decoded = clean.decode_clean(model, processor, raw_image, instruction, str(suite_cfg["unnorm_key"]), device, 7)
            qpos = np.asarray(obs.get("robot0_gripper_qpos", []), dtype=np.float64).reshape(-1)
            eef = np.asarray(obs.get("robot0_eef_pos", []), dtype=np.float64).reshape(-1)
            velocity = np.zeros(3, dtype=np.float64) if previous_eef is None else eef - previous_eef
            feature = adapter.update(step_id=step, raw_gripper=float(decoded["raw_action_7d"][6]), env_gripper=float(decoded["env_action_7d"][6]), gripper_qpos=float(qpos[0] + qpos[1]), gripper_opening_proxy=float(abs(qpos[0]) + abs(qpos[1])), eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]), eef_vx=float(velocity[0]), eef_vy=float(velocity[1]), eef_vz=float(velocity[2]), action_dx=float(decoded["raw_action_7d"][0]), action_dy=float(decoded["raw_action_7d"][1]), action_dz=float(decoded["raw_action_7d"][2]), action_gripper=float(decoded["env_action_7d"][6]))
            valid = bool(feature.get("valid", False))
            rows.append({"step": step, "canonical_parent_key": key, "raw_agentview_sha256": sha256_bytes(raw_image.tobytes()), "raw_agentview_shape": list(raw_image.shape), "processor_input_ids_sha256": decoded["raw_input_hashes"].get("input_ids", ""), "processor_pixel_values_sha256": decoded["raw_input_hashes"].get("pixel_values", ""), "generation_input_ids_sha256": decoded["generation_input_ids_sha256"], "direct_generated_token_ids": decoded["tokens"], "raw_action_7d": decoded["raw_action_7d"], "action_env_7d": decoded["env_action_7d"], "robot0_gripper_qpos": qpos.tolist(), "robot0_eef_pos": eef.tolist(), "robot0_eef_velocity": velocity.tolist(), "features_25d": [float(feature["features"][name]) for name in FEATURE_NAMES] if valid else [], "feature_valid": valid, "candidate_close": bool(valid and float(decoded["raw_action_7d"][6]) < 0.5)})
            obs, reward, done, _ = env.step(decoded["env_action_7d"])
            rows[-1].update({"reward": float(reward), "done_after_env_step": bool(done)})
            previous_eef = eef
            if done:
                clean_success = True
                break
        valid_features = bool(rows) and all(bool(row["feature_valid"]) for row in rows)
        student_model, mean, std, physical_threshold, closing_threshold = student
        predictions = clean.student_trace(student_model, [row["features_25d"] for row in rows], mean, std) if valid_features else []
        schedule = clean.schedule(predictions, [bool(row["candidate_close"]) for row in rows], HORIZONS[parent["suite"]], physical_threshold, closing_threshold) if valid_features else {"first_emit_step": None, "traces": []}
        for row, prediction, trace in zip(rows, predictions, schedule["traces"]):
            row["student_probabilities"] = prediction
            row["student_scheduler_trace"] = trace
        emit = schedule.get("first_emit_step")
        legal = emit is not None and int(emit) + 5 + 10 <= HORIZONS[parent["suite"]]
        receipt = {"schema": "STAGE_X1R2_Q3R3_REFERENCE_CLEAN_RECEIPT_V1", "status": "PASS_REFERENCE_CLEAN", "canonical_parent_key": key, "fixture_id": parent["fixture_id"], "suite": parent["suite"], "task_idx": int(parent["task_idx"]), "state_id": int(parent["state_id"]), "seed": seed, "policy_horizon": HORIZONS[parent["suite"]], "policy_steps_executed": len(rows), "clean_success": clean_success, "valid_feature_stream": valid_features, "first_emit_step": emit, "first_emit_legal": legal, "student_forward_calls": 1 if valid_features else 0, "protected_boundary": {"model_inference_calls": len(rows), "env_step_calls": len(rows) + 10, "student_calls": 1 if valid_features else 0, "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "attacked_env_steps": 0, "protected_reads": 0, "eval160": "UNREAD", "protected_evaluation": "UNREAD"}}
        write_json(out / "reference_receipt.json", receipt)
        write_json(out / "reference_telemetry.json", {"rows": rows})
        if emit is not None and legal and clean_success and valid_features:
            seed_all(seed)
            ref_env, _, _, _ = build_parent_env(parent, physical_gpu)
            try:
                for row in rows[: int(emit)]:
                    ref_env.step(row["action_env_7d"])
                ref_state = capture_branch_state(ref_env, parent, {}, suite_cfg, seed, int(emit))
            finally:
                ref_env.close()
            image_path = out / "reference_observation_t_emit.bin"
            image_path.write_bytes(image_cache[int(emit)])
            prefix = [{"step": int(row["step"]), "action_env_7d": row["action_env_7d"], "direct_generated_token_ids": row["direct_generated_token_ids"]} for row in rows[: int(emit) + 1]]
            write_json(out / "reference_action_prefix.json", {"t_emit": int(emit), "rows": prefix})
            write_json(out / "reference_branch_state.json", ref_state)
            return {"receipt": receipt, "rows": rows, "instruction": instruction, "initial_state": initial_state, "branch_state": ref_state, "reference_image": image_cache[int(emit)], "emit_row": rows[int(emit)], "action_rows": rows}
        return {"receipt": receipt, "rows": rows}
    finally:
        if env is not None:
            env.close()


def branch_repeat(parent: Mapping[str, Any], reference: Mapping[str, Any], model: Any, processor: Any, device: str, suite_cfg: Mapping[str, Any], physical_gpu: int, out: Path, repeat: int) -> dict[str, Any]:
    receipt = reference["receipt"]
    emit = int(receipt["first_emit_step"])
    key = str(parent["canonical_parent_key"])
    seed = int(receipt["seed"])
    image = np.frombuffer(reference["reference_image"], dtype=np.uint8).reshape((256, 256, 3)).copy()
    ref_record = {"status": "PASS_REFERENCE_CLEAN", "clean_success": True, "initial_state": reference["initial_state"], "dummy_wait_steps": 10, "policy_horizon": HORIZONS[parent["suite"]], "first_emit_step": emit, "t5": 5, "h_phys": 10, "student_calls": 1, "env_actions": [row["action_env_7d"] for row in reference["action_rows"]], "observation_bytes": [b""] * emit + [reference["reference_image"]]}
    branch = BranchReplay(ReferenceClean.from_record(ref_record), "CLEAN")
    env = None
    prefix_steps = 0
    post_steps = 0
    clean_tokens_match = False
    try:
        seed_all(seed)
        env, live_obs, instruction, _ = build_parent_env(parent, physical_gpu)
        events: list[int] = []
        prefix_steps = branch.replay_prefix(lambda step, action: (env.step(action), events.append(step)))
        candidate_state = capture_branch_state(env, parent, {}, suite_cfg, seed, emit)
        state_audit = compare_branch_state(reference["branch_state"], candidate_state)
        observe = getattr(env, "_get_observations", None)
        if callable(observe):
            live_obs = observe()
        live_image = np.asarray(live_obs["agentview_image"]).copy()
        if live_image.dtype != np.uint8:
            live_image = np.clip(live_image, 0, 255).astype(np.uint8)
        branch.validate_first_decision(emit, reference["reference_image"])
        decoded = clean.decode_clean(model, processor, image, instruction, str(suite_cfg["unnorm_key"]), device, 7)
        clean_tokens_match = [int(value) for value in decoded["tokens"]] == [int(value) for value in reference["emit_row"]["direct_generated_token_ids"]]
        if not state_audit["equal"]:
            raise RuntimeError(f"BRANCH_STATE_MISMATCH:{state_audit}")
        if not clean_tokens_match:
            raise RuntimeError("BRANCH_CLEAN_DIRECT_TOKENS_MISMATCH")
        obs, reward, done, _ = env.step(decoded["env_action_7d"])
        post_steps = 1
        for row in reference["action_rows"][emit + 1 : emit + 15]:
            if done:
                break
            obs, reward, done, _ = env.step(row["action_env_7d"])
            post_steps += 1
        result = {"schema": "STAGE_X1R2_Q3R3_BRANCH_REPLAY_RECEIPT_V1", "status": "PASS_BRANCH_REPLAY", "repeat": repeat, "canonical_parent_key": key, "fixture_id": parent["fixture_id"], "t_emit": emit, "prefix_steps": prefix_steps, "prefix_step_ids": events, "post_branch_steps": post_steps, "branch_env_step_calls": prefix_steps + post_steps, "live_branch_observation_sha256": sha256_bytes(live_image.tobytes()), "reference_observation_sha256": sha256_bytes(reference["reference_image"]), "state_audit": state_audit, "clean_direct_tokens_match": clean_tokens_match, "prebranch_openvla_calls": 0, "prebranch_student_calls": 0, "branch_openvla_calls": 1, "branch_student_calls": 0, "protected_boundary": {"model_inference_calls": 1, "env_step_calls": prefix_steps + post_steps, "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "attacked_env_steps": 0, "eval160": "UNREAD", "protected_evaluation": "UNREAD"}}
        write_json(out / f"branch_repeat_{repeat}.json", result)
        return result
    except Exception as exc:
        result = {"schema": "STAGE_X1R2_Q3R3_BRANCH_REPLAY_RECEIPT_V1", "status": "HOLD_BRANCH_REPLAY", "repeat": repeat, "canonical_parent_key": key, "fixture_id": parent["fixture_id"], "error": f"{type(exc).__name__}:{exc}", "prefix_steps": prefix_steps, "post_branch_steps": post_steps, "branch_env_step_calls": prefix_steps + post_steps, "clean_direct_tokens_match": clean_tokens_match, "prebranch_openvla_calls": 0, "prebranch_student_calls": 0, "protected_boundary": {"model_inference_calls": 1 if prefix_steps >= emit else 0, "env_step_calls": prefix_steps + post_steps, "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "attacked_env_steps": 0, "protected_reads": 0, "eval160": "UNREAD", "protected_evaluation": "UNREAD"}}
        write_json(out / f"branch_repeat_{repeat}.json", result)
        return result
    finally:
        if env is not None:
            env.close()


def run_suite(args: argparse.Namespace) -> int:
    protocol, authority, contract, pool = load_inputs()
    suite = args.suite
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    source = source_receipt(args.source_commit, args.source_tree)
    mount = gpu_receipt(args.physical_gpu)
    root = Path(str(protocol["resource"]["durable_output_root"]))
    root.mkdir(parents=True, exist_ok=True)
    if int(os.statvfs(root).f_bavail * os.statvfs(root).f_frsize) <= int(protocol["resource"]["minimum_free_bytes"]):
        raise RuntimeError("HOLD_DURABLE_STORAGE")
    contract_suite = contract["suites"][suite]
    model_identity_observed = verify_model_identity(contract_suite)
    paths = student_paths(authority)
    verify_student_source_binding(authority)
    import torch

    torch.set_num_threads(1)
    model, processor, device, action_dim = clean.load_openvla(Path(str(contract_suite["model_path"])), str(contract_suite["unnorm_key"]))
    student = clean.load_student({}, paths)
    candidates = [row for row in pool if row["suite"] == suite and row["canonical_parent_key"] not in set(protocol["already_exposed_q3r2_keys"])]
    scan: list[dict[str, Any]] = []
    selected = None
    selected_ref: dict[str, Any] | None = None
    suite_root = root / suite
    suite_root.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        out = suite_root / str(candidate["fixture_id"])
        if out.exists():
            raise RuntimeError(f"Q3R3_C_OUTPUT_EXISTS:{out}")
        out.mkdir(parents=True)
        result = reference_rollout(candidate, model, processor, device, student, contract_suite, args.physical_gpu, out)
        rec = result["receipt"]
        scan.append({"fixture_id": candidate["fixture_id"], "canonical_parent_key": candidate["canonical_parent_key"], "status": rec["status"], "clean_success": rec["clean_success"], "valid_feature_stream": rec["valid_feature_stream"], "first_emit_step": rec["first_emit_step"], "first_emit_legal": rec["first_emit_legal"]})
        if rec["clean_success"] and rec["valid_feature_stream"] and rec["first_emit_step"] is not None and rec["first_emit_legal"]:
            selected = candidate
            selected_ref = result
            break
    branches: list[dict[str, Any]] = []
    status = "HOLD_NO_CURRENT_RUNTIME_QUALIFIED_REFERENCE_CLEAN"
    if selected is not None and selected_ref is not None:
        selected_root = suite_root / str(selected["fixture_id"])
        branches = [branch_repeat(selected, selected_ref, model, processor, device, contract_suite, args.physical_gpu, selected_root, repeat) for repeat in range(2)]
        status = "PASS_SUITE_BRANCH_REPLAY" if all(item.get("status") == "PASS_BRANCH_REPLAY" and item.get("state_audit", {}).get("equal") and item.get("clean_direct_tokens_match") for item in branches) else "HOLD_BRANCH_REPLAY"
    report = {"schema": "STAGE_X1R2_Q3R3_BRANCH_REPLAY_SUITE_REPORT_V1", "status": status, "suite": suite, "source": source, "model_identity_observed": model_identity_observed, "mount_gpu": mount, "scan": scan, "selected_fixture": selected["fixture_id"] if selected else None, "selected_parent_key": selected["canonical_parent_key"] if selected else None, "branch_receipts": branches, "protected_boundary": {"pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "attacked_env_steps": 0, "protected_reads": 0, "protected_evaluation": "UNREAD", "eval160": "UNREAD"}, "scientific_authority": False, "next_gate": "STAGE_X1R2_Q3R3_FOUR_SUITE_BRANCH_REPLAY_PASS"}
    write_json(suite_root / "SUITE_BRANCH_REPLAY_REPORT_V1.json", report)
    return 0 if status == "PASS_SUITE_BRANCH_REPLAY" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    return run_suite(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
