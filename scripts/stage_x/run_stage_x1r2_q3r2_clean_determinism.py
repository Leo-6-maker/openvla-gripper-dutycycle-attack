#!/usr/bin/env python3
"""Run the authorized current-runtime clean-prefix determinism qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "stage_x"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
import run_stage_x1r_t1d1_screening_clean as clean

PROTOCOL = REPO / "configs/STAGE_X_X1R2_Q3R2_CLEAN_DETERMINISM_PROTOCOL_V1.json"
AUTHORITY = REPO / "configs/STAGE_X_X1R2_Q3R2_RUNTIME_AUTHORITY_V1.json"
POOL = REPO / "reports/STAGE_X_X1R2_Q3R2_ENGINEERING_FIXTURE_POOL_V1.json"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
HORIZONS = {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}
MIN_FREE_MIB = 20480
TRACE_FIELDS = ("raw_agentview_sha256", "processor_input_ids_sha256", "processor_pixel_values_sha256", "generation_input_ids_sha256", "direct_generated_token_ids", "raw_action_7d", "action_env_7d", "robot0_gripper_qpos", "robot0_eef_pos", "robot0_eef_velocity", "features_25d", "student_probabilities", "student_scheduler_trace", "candidate_close", "done_after_env_step")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=clean._json_default)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=clean._json_default) + "\n", encoding="utf-8")


def gpu_receipt(physical_gpu: int, require_free: bool = True) -> dict[str, Any]:
    fields = [item.strip() for item in subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits", "-i", str(physical_gpu)], text=True).strip().split(",")]
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader", "-i", str(physical_gpu)], text=True).strip()
    receipt = {"physical_gpu": int(fields[0]), "gpu_uuid": fields[1], "free_memory_mib": int(fields[2]), "used_memory_mib": int(fields[3]), "utilization_gpu_percent": int(fields[4]), "compute_apps": [line.strip() for line in apps.splitlines() if line.strip()], "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    if require_free and receipt["free_memory_mib"] <= MIN_FREE_MIB:
        raise RuntimeError(f"GPU_RESOURCE_GATE_FAIL:{receipt}")
    return receipt


def load_protocol() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    protocol = clean.load_json(PROTOCOL)
    authority = clean.load_json(AUTHORITY)
    if authority.get("status") != "STAGE_X_X1R2_Q3R2_RUNTIME_AUTHORITY_PASS":
        raise RuntimeError("Q3R2_RUNTIME_AUTHORITY_NOT_PASS")
    if sha256_file(POOL) != protocol["pool"]["sha256"]:
        raise RuntimeError("Q3R2_ENGINEERING_POOL_SHA_MISMATCH")
    pool = clean.load_json(POOL)["selected"]
    return protocol, authority, pool


def seed_for(protocol: dict[str, Any], key: str) -> int:
    salt = protocol["seed"]["salt"]
    return int(hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()[:8], 16)


def student_paths(authority: dict[str, Any]) -> dict[str, Path]:
    paths = {}
    for row in authority["student_binding"]["artifacts"]:
        if row["name"] in {"checkpoint", "normalization", "thresholds"}:
            paths[row["name"]] = Path(str(row["path"]))
    return paths


def run_episode(parent: dict[str, Any], protocol: dict[str, Any], contract: dict[str, Any], model: Any, processor: Any, device: str, action_dim: int, student: tuple[Any, np.ndarray, np.ndarray, float, float], physical_gpu: int, root: Path, repeat: int) -> dict[str, Any]:
    from gripper_attack.d8_streaming_features_v3 import D8StreamingFeatureAdapterV3, FEATURE_NAMES
    from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
    from libero.libero import benchmark, get_libero_path
    key = str(parent["canonical_parent_key"])
    suite, task_text, state_text = key.split("/")
    task_idx, state_id = int(task_text.split("_")[1]), int(state_text.split("_")[1])
    seed = seed_for(protocol, key)
    out = root / suite / str(parent["fixture_id"]) / f"repeat_{repeat}"
    if out.exists():
        raise RuntimeError(f"EXPOSED_FIXTURE_OUTPUT_EXISTS:{out}")
    out.mkdir(parents=True)
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_idx)
    initial_states = task_suite.get_task_init_states(task_idx)
    if state_id >= len(initial_states):
        raise RuntimeError(f"STATE_ID_OUT_OF_RANGE:{key}:{len(initial_states)}")
    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    horizon = HORIZONS[suite]
    rows: list[dict[str, Any]] = []
    env = None
    first_policy_decision = False
    task_success = False
    try:
        env, obs = clean.build_v4_exact_env(bddl, physical_gpu, horizon, 10) if hasattr(clean, "build_v4_exact_env") else build_v4_exact_env(bddl, physical_gpu, horizon, 10)
        obs = env.set_init_state(initial_states[state_id])
        env, obs = apply_dummy_wait(env, obs, 10)
        adapter = D8StreamingFeatureAdapterV3()
        previous_eef = None
        suite_cfg = contract["suites"][suite]
        for step in range(horizon):
            raw_image = np.asarray(obs["agentview_image"]).copy()
            if raw_image.dtype != np.uint8:
                raw_image = np.clip(raw_image, 0, 255).astype(np.uint8)
            decoded = clean.decode_clean(model, processor, raw_image, str(task.language), str(suite_cfg["unnorm_key"]), device, action_dim)
            first_policy_decision = True
            qpos = np.asarray(obs.get("robot0_gripper_qpos", []), dtype=np.float64).reshape(-1)
            eef = np.asarray(obs.get("robot0_eef_pos", []), dtype=np.float64).reshape(-1)
            velocity = np.zeros(3, dtype=np.float64) if previous_eef is None else eef - previous_eef
            raw_action = np.asarray(decoded["raw_action_7d"], dtype=np.float64)
            env_action = np.asarray(decoded["env_action_7d"], dtype=np.float64)
            feature = adapter.update(step_id=step, raw_gripper=float(raw_action[6]), env_gripper=float(env_action[6]), gripper_qpos=float(qpos[0] + qpos[1]), gripper_opening_proxy=float(abs(qpos[0]) + abs(qpos[1])), eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]), eef_vx=float(velocity[0]), eef_vy=float(velocity[1]), eef_vz=float(velocity[2]), action_dx=float(raw_action[0]), action_dy=float(raw_action[1]), action_dz=float(raw_action[2]), action_gripper=float(env_action[6]))
            valid = bool(feature.get("valid", False))
            values = [float(feature["features"][name]) for name in FEATURE_NAMES] if valid else []
            obs, reward, done, _ = env.step(env_action.tolist())
            rows.append({"step": step, "canonical_parent_key": key, "raw_agentview_sha256": sha256_bytes(raw_image.tobytes()), "processor_input_ids_sha256": decoded["raw_input_hashes"].get("input_ids", ""), "processor_pixel_values_sha256": decoded["raw_input_hashes"].get("pixel_values", ""), "generation_input_ids_sha256": decoded["generation_input_ids_sha256"], "direct_generated_token_ids": decoded["tokens"], "raw_action_7d": decoded["raw_action_7d"], "action_env_7d": decoded["env_action_7d"], "robot0_gripper_qpos": qpos.tolist(), "robot0_eef_pos": eef.tolist(), "robot0_eef_velocity": velocity.tolist(), "feature_valid": valid, "features_25d": values, "candidate_close": bool(valid and raw_action[6] < 0.5), "reward": float(reward), "done_after_env_step": bool(done)})
            previous_eef = eef
            if done:
                task_success = True
                break
        valid_features = bool(rows) and all(bool(row["feature_valid"]) for row in rows)
        student_model, mean, std, physical_threshold, closing_threshold = student
        predictions = clean.student_trace(student_model, [row["features_25d"] for row in rows], mean, std) if valid_features else []
        schedule_result = clean.schedule(predictions, [bool(row["candidate_close"]) for row in rows], horizon, physical_threshold, closing_threshold) if valid_features else {"first_emit_step": None, "emitted_count": 0, "traces": []}
        for row, prediction, trace in zip(rows, predictions, schedule_result["traces"]):
            row["student_probabilities"] = prediction
            row["student_scheduler_trace"] = trace
        for row in rows[len(predictions):]:
            row["student_probabilities"] = {}
            row["student_scheduler_trace"] = {}
        receipt = {"schema": "STAGE_X_X1R2_Q3R2_CLEAN_REPEAT_RECEIPT_V1", "status": "PASS_CLEAN_REPEAT" if first_policy_decision else "RUNTIME_INVALID", "fixture_id": parent["fixture_id"], "canonical_parent_key": key, "suite": suite, "task_idx": task_idx, "state_id": state_id, "repeat": repeat, "seed": seed, "policy_horizon": horizon, "policy_steps_executed": len(rows), "clean_success": task_success, "first_emit_step": schedule_result["first_emit_step"], "first_emit_legal": schedule_result["first_emit_step"] is not None and int(schedule_result["first_emit_step"]) + 5 + 10 <= horizon, "valid_feature_stream": valid_features, "rows": len(rows), "trace_sha256": sha256_bytes(canonical([{field: row.get(field) for field in TRACE_FIELDS} for row in rows]).encode()), "runtime_source": clean.source_receipt(), "gpu": gpu_receipt(physical_gpu, require_free=False), "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "attacked_env_steps": 0, "protected_reads": 0}}
        write_json(out / "parent_receipt.json", receipt)
        (out / "step_telemetry.jsonl").write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")
        return {"receipt": receipt, "rows": rows}
    except Exception as exc:
        failure = {"schema": "STAGE_X_X1R2_Q3R2_CLEAN_REPEAT_RECEIPT_V1", "status": "RUNTIME_INVALID", "fixture_id": parent["fixture_id"], "canonical_parent_key": key, "suite": suite, "repeat": repeat, "seed": seed, "first_policy_decision": first_policy_decision, "error": f"{type(exc).__name__}:{exc}", "runtime_source": clean.source_receipt(), "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "attacked_env_steps": 0, "protected_reads": 0}}
        write_json(out / "parent_receipt.json", failure)
        return {"receipt": failure, "rows": rows}
    finally:
        if env is not None:
            env.close()


def compare_prefix(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    a, b = first["receipt"], second["receipt"]
    emit = a.get("first_emit_step")
    prefix_len = int(emit) + 1 if emit is not None else 0
    first_rows = [{field: row.get(field) for field in TRACE_FIELDS} for row in first["rows"][:prefix_len]]
    second_rows = [{field: row.get(field) for field in TRACE_FIELDS} for row in second["rows"][:prefix_len]]
    full_a = [{field: row.get(field) for field in TRACE_FIELDS} for row in first["rows"]]
    full_b = [{field: row.get(field) for field in TRACE_FIELDS} for row in second["rows"]]
    return {"prefix_steps": prefix_len, "prefix_sha256_repeat0": sha256_bytes(canonical(first_rows).encode()), "prefix_sha256_repeat1": sha256_bytes(canonical(second_rows).encode()), "prefix_exact": first_rows == second_rows and len(first_rows) == prefix_len and len(second_rows) == prefix_len, "full_sha256_repeat0": sha256_bytes(canonical(full_a).encode()), "full_sha256_repeat1": sha256_bytes(canonical(full_b).encode()), "full_exact": full_a == full_b, "clean_success_equal": a.get("clean_success") == b.get("clean_success"), "first_emit_equal": a.get("first_emit_step") == b.get("first_emit_step")}


def run_suite(args: argparse.Namespace) -> None:
    protocol, authority, pool = load_protocol()
    suite = args.suite
    candidates = [row for row in pool if row["suite"] == suite]
    if len(candidates) != 12:
        raise RuntimeError(f"POOL_SUITE_COUNT_INVALID:{suite}:{len(candidates)}")
    physical_gpu = int(args.physical_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    mount = gpu_receipt(physical_gpu)
    root = Path(protocol["resource"]["durable_output_root"])
    root.mkdir(parents=True, exist_ok=True)
    if int(os.statvfs(root).f_bavail * os.statvfs(root).f_frsize) <= int(protocol["resource"]["minimum_free_bytes"]):
        raise RuntimeError("HOLD_DURABLE_STORAGE")
    contract = clean.load_json(REPO / "configs/STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1.json")
    paths = student_paths(authority)
    import torch
    torch.set_num_threads(1)
    model, processor, device, action_dim = clean.load_openvla(Path(str(contract["suites"][suite]["model_path"])), str(contract["suites"][suite]["unnorm_key"]))
    student = clean.load_student({}, paths)
    scan = []
    selected = None
    for candidate in candidates:
        result = run_episode(candidate, protocol, contract, model, processor, device, action_dim, student, physical_gpu, root, 0)
        rec = result["receipt"]
        scan.append({"fixture_id": candidate["fixture_id"], "canonical_parent_key": candidate["canonical_parent_key"], "status": rec.get("status"), "clean_success": rec.get("clean_success"), "first_emit_step": rec.get("first_emit_step"), "first_emit_legal": rec.get("first_emit_legal"), "trace_sha256": rec.get("trace_sha256"), "error": rec.get("error", "")})
        if rec.get("status") == "RUNTIME_INVALID":
            break
        if rec.get("clean_success") and rec.get("valid_feature_stream") and rec.get("first_emit_step") is not None and rec.get("first_emit_legal"):
            selected = candidate
            break
    comparison = None
    if selected is not None and scan[-1]["status"] != "RUNTIME_INVALID":
        first = run_episode(selected, protocol, contract, model, processor, device, action_dim, student, physical_gpu, root, 1)
        comparison = compare_prefix({"receipt": clean.load_json(root / suite / selected["fixture_id"] / "repeat_0" / "parent_receipt.json"), "rows": [json.loads(line) for line in (root / suite / selected["fixture_id"] / "repeat_0" / "step_telemetry.jsonl").read_text().splitlines()]}, first)
        selected_status = "PASS_SUITE_CLEAN_PREFIX_DETERMINISM" if comparison["prefix_exact"] and comparison["clean_success_equal"] and comparison["first_emit_equal"] else "OWNER_REVIEW_Q3R2_CLEAN_PREFIX_DETERMINISM_NOT_ESTABLISHED"
    elif scan and scan[-1]["status"] == "RUNTIME_INVALID":
        selected_status = "OWNER_REVIEW_Q3R2_RUNTIME_INVALID"
    else:
        selected_status = "HOLD_NO_CURRENT_RUNTIME_QUALIFIED_FIXTURE"
    report = {"schema": "STAGE_X_X1R2_Q3R2_CLEAN_DETERMINISM_SUITE_REPORT_V1", "status": selected_status, "suite": suite, "scan": scan, "selected_fixture": selected["fixture_id"] if selected else None, "selected_parent_key": selected["canonical_parent_key"] if selected else None, "comparison": comparison, "mount_gpu": mount, "runtime_authority": {"path": str(AUTHORITY), "status": authority.get("status"), "runtime_code_commit": authority["source_binding"]["runtime_code_commit"], "runtime_code_tree": authority["source_binding"]["runtime_code_tree"]}, "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "attacked_env_steps": 0, "protected_reads": 0}}
    write_json(root / suite / "SUITE_CLEAN_DETERMINISM_REPORT_V1.json", report)
    if selected_status != "PASS_SUITE_CLEAN_PREFIX_DETERMINISM":
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
