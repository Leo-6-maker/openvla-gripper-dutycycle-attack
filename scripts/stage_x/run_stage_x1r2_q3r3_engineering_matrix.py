#!/usr/bin/env python3
"""Run one Q3R3-D five-arm engineering fixture on one physical GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "stage_x"))

import run_stage_x1r2_q3r3_branch_replay as q3r3_c
import run_stage_x1r_primary_matrix as primary
from run_stage_x1r_t1d1_screening_clean import load_openvla
from gripper_attack.failure_evidence import write_failure_receipt

PROTOCOL = ROOT / "configs/STAGE_X_X1R2_Q3R3_ENGINEERING_MATRIX_PROTOCOL_V1.json"
VICTIM_CONTRACT = ROOT / "configs/STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1.json"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
ARM_LABELS = (
    "CLEAN_ENGINEERING",
    "TRUE_PGD_T5_ENGINEERING",
    "RAND_UNIFORM_T5_ENGINEERING",
    "SHUFFLED_GRAD_T5_ENGINEERING",
    "TRUE_RANDOM_TIME_T5_ENGINEERING",
)
HORIZONS = primary.HORIZONS


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
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT).strip()


def source_receipt() -> dict[str, Any]:
    return {
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "status_porcelain": git("status", "--porcelain"),
    }


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def gpu_receipt(physical_gpu: int, *, require_free: bool = True) -> dict[str, Any]:
    fields = [field.strip() for field in subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits", "-i", str(physical_gpu)
    ], text=True).strip().split(",")]
    if len(fields) != 5:
        raise RuntimeError(f"GPU_QUERY_INVALID:{fields}")
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader", "-i", str(physical_gpu)
    ], text=True).strip()
    receipt = {
        "physical_gpu": int(fields[0]),
        "gpu_uuid": fields[1],
        "free_memory_mib": int(fields[2]),
        "used_memory_mib": int(fields[3]),
        "utilization_gpu_percent": int(fields[4]),
        "compute_apps": [line.strip() for line in apps.splitlines() if line.strip()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "strict_gate": "free_memory_mib > 20480",
    }
    if require_free and receipt["free_memory_mib"] <= 20480:
        raise RuntimeError(f"GPU_RESOURCE_GATE_FAIL:{receipt}")
    return receipt


def durable_preflight(root: Path, minimum_free_bytes: int) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(root)
    free_bytes = int(stat.f_bavail * stat.f_frsize)
    if free_bytes <= int(minimum_free_bytes):
        raise RuntimeError(f"HOLD_DURABLE_STORAGE:{free_bytes}")
    probe = root / f".d_probe_{os.getpid()}"
    probe.write_bytes(b"STAGE_X1R2_Q3R3_D_DURABLE_PROBE_V1\n")
    probe_sha = sha256_file(probe)
    probe.unlink()
    return {"root": str(root), "free_bytes": free_bytes, "write_probe_sha256": probe_sha}


def normalize_image(value: Any) -> np.ndarray:
    image = np.asarray(value).copy()
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape != (256, 256, 3):
        raise RuntimeError(f"AGENTVIEW_SHAPE_INVALID:{list(image.shape)}")
    return image


def model_parent(fixture: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "suite": str(fixture["suite"]),
        "fixture_id": str(fixture["fixture_id"]),
        "canonical_parent_key": str(fixture["canonical_parent_key"]),
        "task_idx": int(receipt["task_idx"]),
        "state_id": int(receipt["state_id"]),
        "policy_horizon": int(receipt["policy_horizon"]),
        "first_emit_step": int(receipt["first_emit_step"]),
        "ordinal": 0,
    }


def validate_c_root(protocol: Mapping[str, Any]) -> None:
    binding = protocol["q3r3_c_binding"]
    root = Path(str(binding["durable_root"]))
    seal = root / str(binding["root_seal"]["path"])
    if not seal.is_file() or sha256_file(seal) != str(binding["root_seal"]["sha256"]):
        raise RuntimeError("Q3R3_C_ROOT_SEAL_MISMATCH")


def load_fixture(protocol: Mapping[str, Any], fixture: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(protocol["q3r3_c_binding"]["durable_root"]))
    suite = str(fixture["suite"])
    directory = root / suite / str(fixture["fixture_id"])
    report_path = root / suite / "SUITE_BRANCH_REPLAY_REPORT_V1.json"
    required = [
        report_path,
        directory / "reference_receipt.json",
        directory / "reference_telemetry.json",
        directory / "reference_action_prefix.json",
        directory / "reference_branch_state.json",
        directory / "reference_observation_t_emit.bin",
    ]
    if any(not path.is_file() for path in required):
        raise RuntimeError(f"Q3R3_C_REFERENCE_FILE_MISSING:{suite}")
    observation_manifest_path = directory / "reference_observations_manifest.json"
    if protocol.get("reference_observation_source_required") and not observation_manifest_path.is_file():
        raise RuntimeError(f"Q3R3_C_REFERENCE_OBSERVATION_MANIFEST_MISSING:{suite}")
    if sha256_file(report_path) != str(fixture["report_sha256"]):
        raise RuntimeError(f"Q3R3_C_SUITE_REPORT_SHA_MISMATCH:{suite}")
    report = load_json(report_path)
    if report.get("status") != "PASS_SUITE_BRANCH_REPLAY" or report.get("selected_fixture") != fixture["fixture_id"] or report.get("selected_parent_key") != fixture["canonical_parent_key"]:
        raise RuntimeError(f"Q3R3_C_SUITE_REPORT_INVALID:{suite}")
    receipt = load_json(directory / "reference_receipt.json")
    if receipt.get("status") != "PASS_REFERENCE_CLEAN" or receipt.get("clean_success") is not True or receipt.get("valid_feature_stream") is not True or receipt.get("first_emit_legal") is not True:
        raise RuntimeError(f"Q3R3_C_REFERENCE_RECEIPT_INVALID:{suite}")
    if int(receipt["first_emit_step"]) != int(fixture["t_emit"]):
        raise RuntimeError(f"Q3R3_C_T_EMIT_MISMATCH:{suite}")
    telemetry = load_json(directory / "reference_telemetry.json").get("rows", [])
    if not telemetry or [int(row.get("step", -1)) for row in telemetry] != list(range(len(telemetry))):
        raise RuntimeError(f"Q3R3_C_TELEMETRY_SEQUENCE_INVALID:{suite}")
    for row in telemetry:
        if len(row.get("action_env_7d", [])) != 7 or len(row.get("direct_generated_token_ids", [])) != 7:
            raise RuntimeError(f"Q3R3_C_ACTION_ROW_INVALID:{suite}:{row.get('step')}")
    emit = int(fixture["t_emit"])
    if len(telemetry) <= emit + 14:
        raise RuntimeError(f"Q3R3_C_REFERENCE_WINDOW_UNAVAILABLE:{suite}")
    prefix = load_json(directory / "reference_action_prefix.json")
    prefix_rows = prefix.get("rows", [])
    if int(prefix.get("t_emit", -1)) != emit or len(prefix_rows) != emit + 1:
        raise RuntimeError(f"Q3R3_C_PREFIX_INVALID:{suite}")
    for left, right in zip(prefix_rows, telemetry[: emit + 1]):
        if left.get("step") != right.get("step") or left.get("action_env_7d") != right.get("action_env_7d") or left.get("direct_generated_token_ids") != right.get("direct_generated_token_ids"):
            raise RuntimeError(f"Q3R3_C_PREFIX_TELEMETRY_MISMATCH:{suite}")
    image_bytes = (directory / "reference_observation_t_emit.bin").read_bytes()
    if len(image_bytes) != 256 * 256 * 3:
        raise RuntimeError(f"Q3R3_C_REFERENCE_IMAGE_INVALID:{suite}")
    state = load_json(directory / "reference_branch_state.json")
    model_identity = str(contract["suites"][suite]["model_identity"]["tree_sha256"])
    if state.get("model_identity") != model_identity:
        raise RuntimeError(f"Q3R3_C_MODEL_IDENTITY_MISMATCH:{suite}")
    observation_manifest = load_json(observation_manifest_path) if observation_manifest_path.is_file() else None
    if protocol.get("reference_observation_source_required"):
        rows_by_step = {int(row["step"]): row for row in observation_manifest.get("rows", [])}
        if set(rows_by_step) != {int(row["step"]) for row in telemetry}:
            raise RuntimeError(f"Q3R3_C_REFERENCE_OBSERVATION_MANIFEST_INCOMPLETE:{suite}")
    return {
        "fixture": dict(fixture),
        "parent": model_parent(fixture, receipt),
        "report": report,
        "receipt": receipt,
        "rows": telemetry,
        "branch_state": state,
        "reference_image": image_bytes,
        "reference_image_sha256": sha256_bytes(image_bytes),
        "reference_observations_manifest": observation_manifest,
        "root": str(directory),
    }


def load_reference_observation(data: Mapping[str, Any], step: int) -> tuple[bytes, dict[str, Any]]:
    manifest = data.get("reference_observations_manifest")
    if not isinstance(manifest, Mapping):
        raise RuntimeError("Q3R3_C_REFERENCE_OBSERVATION_SOURCE_MISSING")
    entries = {int(row["step"]): row for row in manifest.get("rows", [])}
    entry = entries.get(int(step))
    if entry is None:
        raise RuntimeError(f"Q3R3_C_REFERENCE_OBSERVATION_STEP_MISSING:{step}")
    path = Path(str(data["root"])) / str(entry["path"])
    payload = path.read_bytes()
    if sha256_bytes(payload) != str(entry["sha256"]):
        raise RuntimeError(f"Q3R3_C_REFERENCE_OBSERVATION_SHA_MISMATCH:{step}")
    if len(payload) != 256 * 256 * 3:
        raise RuntimeError(f"Q3R3_C_REFERENCE_OBSERVATION_SIZE_INVALID:{step}")
    return payload, dict(entry)


def random_time_start(protocol: Mapping[str, Any], parent: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> tuple[int, str, list[int]]:
    emit = int(parent["first_emit_step"])
    last = int(rows[-1]["step"])
    done_steps = {int(row["step"]) for row in rows if bool(row.get("done_after_env_step"))}
    candidates = [
        start for start in range(0, last - 14 + 1)
        if not (emit <= start <= emit + 4) and not any(done < start for done in done_steps)
    ]
    if not candidates:
        raise RuntimeError("NO_LEGAL_RANDOM_TIME_ENGINEERING_START")
    salt = str(protocol["random_time"]["salt"])
    rank = hashlib.sha256(f"{salt}|{parent['canonical_parent_key']}".encode()).hexdigest()
    return candidates[int(rank[:8], 16) % len(candidates)], rank, candidates


def replay_to_start(parent: Mapping[str, Any], suite_cfg: Mapping[str, Any], rows: list[Mapping[str, Any]], start: int, physical_gpu: int) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    seed = int(hashlib.sha256(f"STAGE_X1R2_Q3R3_REFERENCE_CLEAN_V1_20260820|{parent['canonical_parent_key']}".encode()).hexdigest()[:8], 16)
    primary.set_seed(seed)
    env = None
    try:
        env, obs, _instruction, _task_idx, _state_id, _bddl = primary.build_parent_env(parent, suite_cfg, physical_gpu)
        for row in rows[:start]:
            obs, _reward, done, _info = env.step(list(row["action_env_7d"]))
            if done:
                raise RuntimeError(f"REFERENCE_PREFIX_TERMINATED:{start}:{row['step']}")
        state = q3r3_c.capture_branch_state(env, parent, {}, suite_cfg, seed, start)
        image = normalize_image(obs["agentview_image"])
        return state, image, {"seed": seed, "prefix_steps": start}
    finally:
        if env is not None:
            env.close()


def compare_image_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    return left.dtype == right.dtype and left.shape == right.shape and bool(np.array_equal(left, right))


def build_attack(condition: str, seed: int, model: Any, processor: Any, device: str) -> Any:
    from gripper_attack.attack_adapter import OpenVLAVisualAttacker

    config = primary.build_attack_config(condition, seed, arm_isolation_candidate_policy="STRICT_CANDIDATE_AUDIT_V1")
    return OpenVLAVisualAttacker(
        model,
        processor,
        config,
        seed=seed,
        preprocess_kwargs={"center_crop": True, "resize_size": 224, "libero_preprocess_backend": "official_pil_lanczos", "postprocess_gripper": True},
        device=device,
    )


def audit_selective_route(attack_summary: Mapping[str, Any], clean_semantics: Mapping[str, Any], executed_semantics: Mapping[str, Any], direct_audit: Mapping[str, Any], step: int) -> None:
    route = attack_summary.get("route")
    if not isinstance(route, Mapping):
        raise RuntimeError(f"D_ATTACK_ROUTE_MISSING:{step}")
    if route.get("strict_route") is not True or route.get("allow_fallback") is not False or route.get("fallback_used") is not False:
        raise RuntimeError(f"D_STRICT_ROUTE_INVALID:{step}")
    if route.get("target_execution_class") != "NATIVE_OPEN" or int(route.get("target_token_id", -1)) != 31745:
        raise RuntimeError(f"D_TARGET_AUTHORITY_INVALID:{step}")
    if float(route.get("pixel_budget_adv_inputs_linf", 1.0)) > 0.03 + 1e-6:
        raise RuntimeError(f"D_PROCESSOR_LINF_INVALID:{step}")
    if direct_audit.get("arm_token_ids_equal") is not True:
        raise RuntimeError(f"D_ARM_ISOLATION_INVALID:{step}")
    if clean_semantics.get("execution_class") == "NATIVE_OPEN" or executed_semantics.get("execution_class") != "NATIVE_OPEN":
        raise RuntimeError(f"D_GRIPPER_TRANSITION_CLASS_INVALID:{step}")
    if direct_audit.get("gripper_token_changed") is not True:
        raise RuntimeError(f"D_GRIPPER_TOKEN_NOT_CHANGED:{step}")
    selected_index = route.get("selected_candidate_index")
    audits = route.get("arm_isolation_candidate_audit")
    if selected_index is None or not isinstance(audits, list):
        raise RuntimeError(f"D_SELECTIVE_CANDIDATE_MISSING:{step}")
    selected = [item for item in audits if item.get("candidate_index") == selected_index]
    if len(selected) != 1 or selected[0].get("clean_gripper_is_native_open") is not False or selected[0].get("gripper_token_changed") is not True or selected[0].get("direct_generated_gripper_is_native_open") is not True:
        raise RuntimeError(f"D_SELECTIVE_CANDIDATE_INVALID:{step}")


def attack_random(clean_inputs: Mapping[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    adv_inputs, linf, delta_sha, correction_count = primary.random_adv_inputs(clean_inputs, seed)
    return adv_inputs, {
        "route": "m3_controls.sample_processor_delta",
        "seed": int(seed),
        "epsilon": 0.03,
        "step_size": 0.006,
        "num_steps": 0,
        "pixel_budget_adv_inputs_linf": float(linf),
        "delta_sha256": delta_sha,
        "projection_correction_count": int(correction_count),
        "strict_route": "control_not_pgd",
    }


def run_arm(data: Mapping[str, Any], arm: Mapping[str, Any], model: Any, processor: Any, device: str, contract: Mapping[str, Any], physical_gpu: int, suite_output: Path, random_material: Mapping[str, Any] | None) -> dict[str, Any]:
    parent = data["parent"]
    suite = str(parent["suite"])
    rows = list(data["rows"])
    emit = int(parent["first_emit_step"])
    is_random = str(arm["label"]) == "TRUE_RANDOM_TIME_T5_ENGINEERING"
    start = int(random_material["start"]) if is_random and random_material is not None else emit
    common_image = np.frombuffer((random_material["image_bytes"] if is_random and random_material is not None else data["reference_image"]), dtype=np.uint8).reshape((256, 256, 3)).copy()
    expected_row = rows[start]
    expected_tokens = [int(value) for value in expected_row["direct_generated_token_ids"]]
    seed = int(hashlib.sha256(f"STAGE_X1R2_Q3R3_ENGINEERING_D|{parent['canonical_parent_key']}|{arm['label']}".encode()).hexdigest()[:8], 16)
    output = suite_output / str(arm["label"])
    output.mkdir(parents=True, exist_ok=False)
    telemetry_path = output / "step_telemetry.jsonl"
    counters = {"model_inference_calls": 0, "env_reset_calls": 0, "env_step_calls": 0, "pgd_calls": 0, "adversarial_images": 0, "attacked_env_steps": 0, "attack_backward_calls": 0, "loss_forward_count": 0, "vphys_reads": 0, "physical_interventions": 0, "attack_outcome_reads": 0, "protected_reads": 0, "eval160_reads": 0}
    attack_kind = str(arm["kind"])
    attacker = build_attack("SHUFFLED_GRAD_T5" if attack_kind == "shuffled_pgd" else "TRUE_PGD_T5", seed, model, processor, device) if attack_kind in {"true_pgd", "shuffled_pgd"} else None
    if attacker is not None:
        attacker.reset_temporal_state()
    env = None
    branch_rows: list[dict[str, Any]] = []
    start_time = time.time()
    try:
        primary.set_seed(seed)
        env, obs, instruction, _task_idx, _state_id, _bddl = primary.build_parent_env(parent, contract["suites"][suite], physical_gpu)
        counters["env_reset_calls"] = 1
        for row in rows[:start]:
            obs, _reward, done, _info = env.step(list(row["action_env_7d"]))
            counters["env_step_calls"] += 1
            if done:
                raise RuntimeError(f"D_PREFIX_TERMINATED:{start}:{row['step']}")
        branch_state = q3r3_c.capture_branch_state(env, parent, {}, contract["suites"][suite], int(hashlib.sha256(f"STAGE_X1R2_Q3R3_REFERENCE_CLEAN_V1_20260820|{parent['canonical_parent_key']}".encode()).hexdigest()[:8], 16), start)
        state_audit = q3r3_c.compare_branch_state(data["branch_state"] if not is_random else random_material["state"], branch_state)
        if not state_audit.get("equal"):
            raise RuntimeError(f"D_BRANCH_STATE_MISMATCH:{state_audit}")
        prefix_live_image = normalize_image(obs["agentview_image"])
        attack_rows = 0
        for offset in range(15):
            step = start + offset
            raw_image = common_image if offset == 0 else normalize_image(obs["agentview_image"])
            clean_prepared = primary.prepare_generation(model, processor, raw_image, instruction, suite, device)
            clean_decoded = primary.decode_tokens(model, clean_prepared["tokens"], suite)
            clean_decoded.update({"generated": clean_prepared["generated"], "inputs": clean_prepared["inputs"], "prompt_len": int(clean_prepared["inputs"]["input_ids"].shape[1]), "raw_hashes": clean_prepared["raw_hashes"]})
            counters["model_inference_calls"] += 1
            clean_tokens = [int(value) for value in clean_decoded["tokens"]]
            if len(clean_tokens) != 7:
                raise RuntimeError(f"D_CLEAN_TOKEN_COUNT_INVALID:{step}")
            if offset == 0 and clean_tokens != expected_tokens:
                raise RuntimeError(f"D_CLEAN_REFERENCE_TOKEN_MISMATCH:{step}")
            executed = clean_decoded
            attack_summary: dict[str, Any] = {"attack_executed": False, "kind": "clean"}
            attack_tensor = None
            attacked = attack_kind != "clean" and offset < 5
            if attacked:
                if attack_kind == "random_uniform":
                    draw_seed = int(hashlib.sha256(f"{seed}|random|{offset}".encode()).hexdigest()[:8], 16)
                    adv_inputs, route = attack_random(clean_decoded["inputs"], draw_seed)
                    attack_summary = {"attack_executed": True, "kind": "random_uniform", **route}
                else:
                    trace: dict[str, Any] = {}
                    try:
                        result = attacker.attack(raw_image, instruction, clean_action=np.asarray(clean_decoded["raw_action_7d"], dtype=np.float32), target_action=np.asarray(clean_decoded["raw_action_7d"], dtype=np.float32), clean_model_output=clean_decoded["generated"], unnorm_key=suite, execution_trace=trace)
                    finally:
                        attack_summary = {"attack_executed": True, "kind": attack_kind, "execution_trace": dict(trace)}
                    adv_inputs = result.debug["adv_inputs"]
                    route_summary = primary.summarize_attack(result)
                    attack_summary["route"] = route_summary
                    counters["pgd_calls"] += 1
                    counters["attack_backward_calls"] += int(route_summary.get("num_backwards") or 0)
                    counters["loss_forward_count"] += int(route_summary.get("num_loss_forwards") or 0)
                if not np.array_equal(clean_decoded["inputs"]["input_ids"].detach().cpu().numpy(), adv_inputs["input_ids"].detach().cpu().numpy()):
                    raise RuntimeError(f"D_INPUT_IDS_CHANGED:{step}")
                executed = primary.decode_from_inputs(model, adv_inputs, clean_decoded["prompt_len"], suite)
                attack_summary["processor_input_ids_sha256"] = primary.clean_tensor_sha256(adv_inputs["input_ids"])
                attack_tensor = primary.persist_attack_tensor(output, step, str(arm["label"]), clean_decoded["inputs"], adv_inputs)
                attack_summary["processor_tensor"] = attack_tensor
                attack_summary["attack_decoded_directly_from_adv_inputs"] = True
                counters["adversarial_images"] += 1
                attack_rows += 1
            executed_tokens = [int(value) for value in executed["tokens"]]
            direct_audit = primary.audit_direct_action_tokens(clean_tokens, executed_tokens)
            clean_semantics = primary.classify_gripper(model, suite, clean_tokens[6])
            executed_semantics = primary.classify_gripper(model, suite, executed_tokens[6])
            if attacked:
                if direct_audit.get("arm_token_ids_equal") is not True:
                    raise RuntimeError(f"D_ARM_ISOLATION_INVALID:{step}")
                if attack_kind in {"true_pgd", "shuffled_pgd"}:
                    audit_selective_route(attack_summary, clean_semantics, executed_semantics, direct_audit, step)
            env_action = np.asarray(executed["env_action_7d"], dtype=np.float32)
            obs_after, reward, done, info = env.step(env_action.tolist())
            counters["env_step_calls"] += 1
            if attacked:
                counters["attacked_env_steps"] += 1
            row = {
                "step": step,
                "arm": str(arm["label"]),
                "attack_offset": offset if attacked else None,
                "branch_start": start,
                "first_decision_common_observation": offset == 0,
                "common_observation_sha256": sha256_bytes(common_image.tobytes()),
                "live_observation_sha256": sha256_bytes(raw_image.tobytes()),
                "prefix_live_observation_sha256": sha256_bytes(prefix_live_image.tobytes()) if offset == 0 else None,
                "clean_processor_input_ids_sha256": clean_decoded["raw_hashes"].get("input_ids", ""),
                "clean_processor_pixel_values_sha256": clean_decoded["raw_hashes"].get("pixel_values", ""),
                "clean_token_ids": clean_tokens,
                "executed_token_ids": executed_tokens,
                "clean_action_7d": clean_decoded["env_action_7d"],
                "executed_action_7d": executed["env_action_7d"],
                "env_step_action_7d": env_action.tolist(),
                "direct_action_equals_env_step": env_action.tolist() == [float(value) for value in executed["env_action_7d"]],
                "direct_action_audit": direct_audit,
                "clean_gripper_semantics": clean_semantics,
                "executed_gripper_semantics": executed_semantics,
                "attack": attack_summary,
                "reward": float(reward),
                "done_after_env_step": bool(done),
            }
            branch_rows.append(row)
            telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            with telemetry_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            obs = obs_after
            if done and offset < 14:
                raise RuntimeError(f"D_BRANCH_TERMINATED_BEFORE_15_ACTIONS:{step}")
        if len(branch_rows) != 15 or attack_rows != (0 if attack_kind == "clean" else 5):
            raise RuntimeError(f"D_WINDOW_ACCOUNTING_INVALID:{len(branch_rows)}:{attack_rows}")
        receipt = {
            "schema": "STAGE_X1R2_Q3R3_ENGINEERING_ARM_RECEIPT_V1",
            "status": "PASS_Q3R3_D_ENGINEERING_ARM",
            "structural_valid": True,
            "scientific_use": False,
            "suite": suite,
            "fixture_id": parent["fixture_id"],
            "canonical_parent_key": parent["canonical_parent_key"],
            "arm": str(arm["label"]),
            "kind": attack_kind,
            "branch_start_step": start,
            "student_emit_step": emit,
            "prefix_steps_replayed": start,
            "required_actions": 15,
            "attack_rows": attack_rows,
            "state_audit": state_audit,
            "reference_row_at_branch": expected_row,
            "common_observation_sha256": sha256_bytes(common_image.tobytes()),
            "prefix_live_observation_sha256": sha256_bytes(prefix_live_image.tobytes()),
            "telemetry": {"path": str(telemetry_path), "sha256": sha256_file(telemetry_path), "rows": len(branch_rows)},
            "counters": counters,
            "gpu": gpu_receipt(physical_gpu, require_free=False),
            "official_environment": "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800",
            "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "protected_reads": 0, "eval160_reads": 0},
            "elapsed_seconds": time.time() - start_time,
            "source": source_receipt(),
        }
        write_json(output / "arm_receipt.json", receipt)
        return receipt
    except Exception as exc:
        failure = {
            "schema": "STAGE_X1R2_Q3R3_ENGINEERING_ARM_RECEIPT_V1",
            "status": "HOLD_Q3R3_D_ENGINEERING_ARM",
            "structural_valid": False,
            "scientific_use": False,
            "suite": suite,
            "fixture_id": parent["fixture_id"],
            "canonical_parent_key": parent["canonical_parent_key"],
            "arm": str(arm["label"]),
            "branch_start_step": start,
            "student_emit_step": emit,
            "rows_materialized": len(branch_rows),
            "error": f"{type(exc).__name__}:{exc}",
            "counters": counters,
            "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "protected_reads": 0, "eval160_reads": 0},
            "retry_authorized": False,
            "source": source_receipt(),
        }
        write_failure_receipt(output / "arm_receipt.json", failure, exc, attacker)
        raise
    finally:
        if env is not None:
            env.close()


def run_suite(args: argparse.Namespace) -> int:
    protocol = load_json(args.protocol)
    contract = load_json(VICTIM_CONTRACT)
    if protocol.get("status") != "FROZEN_ENGINEERING_ONLY_PRE_GPU" or protocol.get("scientific_authority") is not False:
        raise SystemExit("D_PROTOCOL_NOT_FROZEN")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != str(args.physical_gpu):
        raise SystemExit("CUDA_VISIBLE_DEVICES_MUST_BIND_SINGLE_PHYSICAL_GPU")
    source = source_receipt()
    if source["status_porcelain"]:
        raise SystemExit("WORKTREE_NOT_CLEAN")
    validate_c_root(protocol)
    contract_path = ROOT / str(protocol["attack_contract"]["path"])
    if sha256_file(contract_path).lower() != str(protocol["attack_contract"]["file_sha256"]).lower() or git("rev-parse", f"HEAD:{contract_path.relative_to(ROOT).as_posix()}") != str(protocol["attack_contract"]["git_blob_sha"]):
        raise SystemExit("ATTACK_CONTRACT_BINDING_MISMATCH")
    durable = durable_preflight(Path(str(protocol["resource"]["durable_output_root"])), int(protocol["resource"]["minimum_free_bytes"]))
    gpu = gpu_receipt(args.physical_gpu, require_free=True)
    fixture_rows = [row for row in protocol["fixtures"] if row["suite"] == args.suite]
    if len(fixture_rows) != 1:
        raise SystemExit("D_SUITE_NOT_FROZEN")
    data = load_fixture(protocol, fixture_rows[0], contract)
    suite = str(args.suite)
    suite_cfg = contract["suites"][suite]
    primary.verify_model_identity(contract, suite)
    random_material = None
    random_start, random_rank, candidates = random_time_start(protocol, data["parent"], data["rows"])
    random_state_a, random_image_a, random_replay_audit = replay_to_start(data["parent"], suite_cfg, data["rows"], random_start, args.physical_gpu)
    random_state_b, random_image_b, random_replay_b = replay_to_start(data["parent"], suite_cfg, data["rows"], random_start, args.physical_gpu)
    random_state_audit = q3r3_c.compare_branch_state(random_state_a, random_state_b)
    if not random_state_audit.get("equal"):
        raise RuntimeError(f"D_RANDOM_TIME_REPLAY_MISMATCH:{random_state_audit}")
    if protocol.get("reference_observation_source_required"):
        random_reference_bytes, random_reference_entry = load_reference_observation(data, random_start)
        reference_observation_source = "Q3R3_C_REFERENCE_CLEAN"
    else:
        if not compare_image_bytes(random_image_a, random_image_b):
            raise RuntimeError(f"D_RANDOM_TIME_REPLAY_OBSERVATION_MISMATCH:{random_start}")
        random_reference_bytes = random_image_a.tobytes()
        random_reference_entry = {"step": random_start, "sha256": sha256_bytes(random_reference_bytes), "source": "two_replay_observation"}
        reference_observation_source = "Q3R3_D_TWO_REPLAY_OBSERVATION"
    random_dir = Path(str(protocol["resource"]["durable_output_root"])) / suite / str(data["parent"]["fixture_id"])
    random_dir.mkdir(parents=True, exist_ok=False)
    random_obs_path = random_dir / "random_reference_observation.bin"
    random_obs_path.write_bytes(random_reference_bytes)
    random_material = {"start": random_start, "rank": random_rank, "candidates": candidates, "state": random_state_a, "image_bytes": random_reference_bytes, "replay_audit": random_state_audit, "replay_a": random_replay_audit, "replay_b": random_replay_b, "replay_image_a_sha256": sha256_bytes(random_image_a.tobytes()), "replay_image_b_sha256": sha256_bytes(random_image_b.tobytes()), "reference_observation_source": reference_observation_source, "reference_observation_entry": random_reference_entry, "observation_path": str(random_obs_path), "observation_sha256": sha256_file(random_obs_path)}
    if random_start >= int(data["parent"]["first_emit_step"]) and random_start <= int(data["parent"]["first_emit_step"]) + 4:
        raise RuntimeError("D_RANDOM_TIME_OVERLAPS_STUDENT_EMIT_WINDOW")
    write_json(random_dir / "random_time_materialization.json", {key: value for key, value in random_material.items() if key != "image_bytes"})
    model, processor, device, _action_dim = load_openvla(Path(str(suite_cfg["model_path"])), suite)
    arm_specs = [
        {"label": "CLEAN_ENGINEERING", "kind": "clean"},
        {"label": "TRUE_PGD_T5_ENGINEERING", "kind": "true_pgd"},
        {"label": "RAND_UNIFORM_T5_ENGINEERING", "kind": "random_uniform"},
        {"label": "SHUFFLED_GRAD_T5_ENGINEERING", "kind": "shuffled_pgd"},
        {"label": "TRUE_RANDOM_TIME_T5_ENGINEERING", "kind": "true_pgd"},
    ]
    arm_receipts = []
    for arm in arm_specs:
        arm_receipts.append(run_arm(data, arm, model, processor, device, contract, args.physical_gpu, random_dir, random_material if arm["label"] == "TRUE_RANDOM_TIME_T5_ENGINEERING" else None))
    suite_receipt = {
        "schema": "STAGE_X1R2_Q3R3_ENGINEERING_SUITE_RECEIPT_V1",
        "status": "PASS_Q3R3_D_ENGINEERING_SUITE",
        "structural_valid": True,
        "scientific_use": False,
        "suite": suite,
        "fixture_id": data["parent"]["fixture_id"],
        "canonical_parent_key": data["parent"]["canonical_parent_key"],
        "source": source,
        "c_source": protocol["q3r3_c_binding"],
        "durable_storage": durable,
        "gpu_before_model_load": gpu,
        "gpu_after": gpu_receipt(args.physical_gpu, require_free=False),
        "random_time": {key: value for key, value in random_material.items() if key != "image_bytes"},
        "arms": [{"label": row["arm"], "status": row["status"], "path": str(random_dir / row["arm"] / "arm_receipt.json")} for row in arm_receipts],
        "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "protected_reads": 0, "eval160_reads": 0},
    }
    write_json(random_dir / "suite_receipt.json", suite_receipt)
    print(json.dumps({"status": suite_receipt["status"], "suite": suite, "fixture_id": data["parent"]["fixture_id"], "random_start": random_start, "arms": len(arm_receipts)}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
