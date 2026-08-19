#!/usr/bin/env python3
"""Run one atomic Stage-X X1R primary-matrix parent.

The runner deliberately has no replacement, retry, protected-outcome, or
protected-evaluation path. One invocation owns one frozen parent and all four
conditions for that parent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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

PROTOCOL = ROOT / "configs/STAGE_X_X1R_PRIMARY_MATRIX_PROTOCOL_V1.json"
COHORT = ROOT / "reports/STAGE_X_X1R_T1D1M1_FINAL_ATTACK_COHORT_V1.json"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
HORIZONS = {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}
CONDITIONS = ("CLEAN_EVAL", "TRUE_PGD_T5", "RAND_UNIFORM_T5", "SHUFFLED_GRAD_T5")
TARGET_TOKEN = 31745
TARGET_CLASS = "NATIVE_OPEN"
EPSILON = 0.03
STEP_SIZE = 0.006
ATTACK_STEPS = 5
ATTACK_WINDOW = 5
H_PHYS = 10
PROBE_ID = "PRIMARY_EMIT_T5"
MIN_FREE_MIB = 20480
MIN_FREE_BYTES = 4 * 1024**3

COUNTER_NAMES = (
    "openvla_model_inference_calls",
    "model_inference_calls",
    "student_forward_calls",
    "env_reset_calls",
    "env_step_calls",
    "pgd_calls",
    "attack_backward_calls",
    "adversarial_images",
    "attacked_env_steps",
    "physical_interventions",
    "vphys_reads",
    "attack_outcome_reads",
    "eval160_reads",
    "protected_reads",
    "policy_action_materialized_count",
    "rows_materialized",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


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
    return {"branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"), "status_porcelain": git("status", "--porcelain")}


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def seed_for(namespace: str, parent_key: str, probe_id: str = PROBE_ID) -> int:
    return int(hashlib.sha256(f"{namespace}|{parent_key}|{probe_id}".encode()).hexdigest()[:8], 16)


def primary_seed_values(protocol: Mapping[str, Any], parent_key: str) -> dict[str, int]:
    contract = protocol["seed_contract"]
    return {
        "eval_seed": seed_for(str(contract["eval_seed_namespace"]), parent_key),
        "perturb_seed": seed_for(str(contract["perturb_seed_namespace"]), parent_key),
    }


def arm_order(parent_key: str, protocol: Mapping[str, Any]) -> list[str]:
    contract = protocol["seed_contract"]
    base = list(contract["arm_order_base"])
    rotation = int(hashlib.sha256(f"{contract['arm_order_namespace']}|{parent_key}|{PROBE_ID}".encode()).hexdigest()[:2], 16) % 4
    return base[rotation:] + base[:rotation]


def gpu_receipt(physical_gpu: int, *, require_free: bool = True) -> dict[str, Any]:
    fields = [field.strip() for field in subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits", "-i", str(physical_gpu)
    ], text=True).strip().split(",")]
    if len(fields) != 5:
        raise RuntimeError(f"GPU_QUERY_INVALID:{fields}")
    apps_text = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader", "-i", str(physical_gpu)
    ], text=True).strip()
    receipt = {
        "physical_gpu": int(fields[0]),
        "gpu_uuid": fields[1],
        "free_memory_mib": int(fields[2]),
        "used_memory_mib": int(fields[3]),
        "utilization_gpu_percent": int(fields[4]),
        "compute_apps": [line.strip() for line in apps_text.splitlines() if line.strip()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if require_free and receipt["free_memory_mib"] <= MIN_FREE_MIB:
        raise RuntimeError(f"GPU_RESOURCE_GATE_FAIL:{receipt}")
    return receipt


def model_tree_digest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rows.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)})
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return {"file_count": len(rows), "bytes": sum(row["size"] for row in rows), "tree_sha256": sha256_bytes(canonical)}


def verify_model_identity(contract: Mapping[str, Any], suite: str) -> dict[str, Any]:
    cfg = contract["suites"][suite]
    path = Path(str(cfg["model_path"]))
    if not path.is_dir():
        raise RuntimeError(f"MODEL_DIR_MISSING:{suite}:{path}")
    observed = model_tree_digest(path)
    expected = cfg["model_identity"]
    for key in ("file_count", "bytes", "tree_sha256"):
        if observed[key] != expected[key]:
            raise RuntimeError(f"MODEL_IDENTITY_MISMATCH:{suite}:{key}:{observed[key]}!={expected[key]}")
    for relative, expected_sha in expected.get("key_files", {}).items():
        actual = sha256_file(path / relative)
        if actual != expected_sha:
            raise RuntimeError(f"MODEL_KEY_SHA_MISMATCH:{suite}:{relative}:{actual}!={expected_sha}")
    return {"path": str(path), "observed": observed, "expected": expected}


def prepare_generation(model: Any, processor: Any, image: np.ndarray, instruction: str, unnorm_key: str, device: str) -> dict[str, Any]:
    import torch

    from gripper_attack.openvla_libero_exec_spec import official_prompt
    from gripper_attack.openvla_preprocess import prepare_openvla_image
    from gripper_attack.v3_generation_parity import extract_exact_new_tokens

    processed = prepare_openvla_image(image, center_crop=True, resize_size=224, libero_preprocess_backend="official_pil_lanczos")
    inputs = dict(processor(official_prompt(instruction), processed, return_tensors="pt"))
    raw_hashes = {key: clean_tensor_sha256(value) for key, value in inputs.items() if hasattr(value, "detach")}
    inputs.pop("attention_mask", None)
    if not torch.all(inputs["input_ids"][:, -1] == 29871):
        inputs["input_ids"] = torch.cat((inputs["input_ids"], torch.full_like(inputs["input_ids"][:, :1], 29871)), dim=1)
    model_dtype = next(model.parameters()).dtype
    model_inputs = {
        key: value.to(device=device, dtype=model_dtype) if torch.is_floating_point(value) else value.to(device=device)
        for key, value in inputs.items()
    }
    prompt_len = int(model_inputs["input_ids"].shape[1])
    action_dim = int(model.get_action_dim(unnorm_key))
    with torch.inference_mode():
        generated = model.generate(**model_inputs, max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
    tokens = extract_exact_new_tokens(generated.sequences, prompt_len=prompt_len, expected_new_tokens=action_dim)
    return {"processed": processed, "inputs": model_inputs, "raw_hashes": raw_hashes, "generated": generated, "tokens": [int(x) for x in tokens]}


def clean_tensor_sha256(value: Any) -> str:
    import io
    import torch

    buffer = io.BytesIO()
    torch.save(value.detach().cpu(), buffer)
    return sha256_bytes(buffer.getvalue())


def decode_tokens(model: Any, tokens: list[int], unnorm_key: str) -> dict[str, Any]:
    from gripper_attack.openvla_libero_exec_spec import raw_gripper_to_env_gripper

    token_ids = np.asarray(tokens, dtype=np.int64)
    vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    centers = np.asarray(model.bin_centers.detach().cpu() if hasattr(model.bin_centers, "detach") else model.bin_centers)
    discretized = np.clip(vocab_size - token_ids - 1, 0, centers.shape[0] - 1)
    normalized = centers[discretized]
    stats = model.get_action_stats(unnorm_key)
    mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)), dtype=bool)
    low, high = np.asarray(stats["q01"], dtype=np.float32), np.asarray(stats["q99"], dtype=np.float32)
    raw = np.where(mask, 0.5 * (normalized + 1.0) * (high - low) + low, normalized).astype(np.float32)
    env = np.clip(raw, -1.0, 1.0).astype(np.float32)
    env[-1] = raw_gripper_to_env_gripper(float(raw[-1]))
    return {"tokens": [int(x) for x in tokens], "raw_action_7d": raw.tolist(), "env_action_7d": np.clip(env, -1.0, 1.0).tolist(), "raw_gripper": float(raw[-1]), "env_gripper": float(env[-1])}


def decode_from_inputs(model: Any, inputs: Mapping[str, Any], prompt_len: int, unnorm_key: str) -> dict[str, Any]:
    import torch

    from gripper_attack.v3_generation_parity import extract_exact_new_tokens

    action_dim = int(model.get_action_dim(unnorm_key))
    with torch.inference_mode():
        generated = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
    tokens = extract_exact_new_tokens(generated.sequences, prompt_len=int(prompt_len), expected_new_tokens=action_dim)
    return {"generated": generated, **decode_tokens(model, [int(x) for x in tokens], unnorm_key)}


def build_attack_config(condition: str, seed: int) -> dict[str, Any]:
    return {"attack_optimizer": {
        "method": "token_prefix_pgd",
        "strict_route": True,
        "allow_fallback": False,
        "objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        "target_token_id": TARGET_TOKEN,
        "target_execution_class": TARGET_CLASS,
        "epsilon": EPSILON,
        "step_size": STEP_SIZE,
        "num_steps": ATTACK_STEPS,
        "cw_margin": 5.0,
        "gripper_margin": 5.0,
        "random_start": False,
        "temporal_init": "none",
        "temporal_smooth_lambda": 0.0,
        "prefix_refresh_interval": 1,
        "surrogate_score_path": "cached_autoregressive_generate_v1",
        "gradient_transform": "permute" if condition == "SHUFFLED_GRAD_T5" else "none",
        "gradient_transform_seed": int(seed),
        "arm_preserve_weight": 0.1,
    }}


def summarize_attack(result: Any) -> dict[str, Any]:
    debug = getattr(result, "debug", {}) or {}
    return {
        "attack_method": str(getattr(result, "attack_method", "")),
        "directional_loss_available": bool(getattr(result, "directional_loss_available", False)),
        "epsilon": float(getattr(result, "epsilon", 0.0)),
        "step_size": float(getattr(result, "step_size", 0.0)),
        "num_attack_steps": int(getattr(result, "num_attack_steps", 0)),
        "strict_route": debug.get("strict_route"),
        "allow_fallback": debug.get("allow_fallback"),
        "fallback_used": debug.get("fallback_used"),
        "resolved_adapter_class": debug.get("resolved_adapter_class"),
        "requested_objective": debug.get("requested_objective"),
        "resolved_objective": debug.get("resolved_objective"),
        "target_token_id": debug.get("target_token_id"),
        "target_execution_class": debug.get("target_execution_class"),
        "num_backwards": debug.get("num_backwards"),
        "num_loss_forwards": debug.get("num_loss_forwards"),
        "num_adv_decodes": debug.get("num_adv_decodes"),
        "pixel_space": debug.get("pixel_space"),
        "pixel_budget_adv_inputs_linf": debug.get("pixel_budget_adv_inputs_linf"),
        "pixel_budget_master_linf": debug.get("pixel_budget_master_linf"),
        "gradient_transform": debug.get("gradient_transform"),
        "arm_prefix_match_count": debug.get("arm_prefix_match_count"),
        "arm_prefix_match_denominator": debug.get("arm_prefix_match_denominator"),
        "target_token_objective_loss_trajectory": debug.get("target_token_objective_loss_trajectory"),
        "target_token_arm_preservation_loss_trajectory": debug.get("target_token_arm_preservation_loss_trajectory"),
        "gradient_norm_trajectory": debug.get("gradient_norm_trajectory"),
        "generated_arm_prefix_trajectory": debug.get("generated_arm_prefix_trajectory"),
        "delta_final_sha256": debug.get("delta_final_sha256"),
        "processor_input_sha256": debug.get("processor_input_sha256"),
    }


def classify_gripper(model: Any, suite: str, token_id: int) -> dict[str, Any]:
    from gripper_attack.execution_target import classify_execution_token

    stats = model.get_action_stats(suite)
    centers = model.bin_centers.detach().cpu().numpy() if hasattr(model.bin_centers, "detach") else np.asarray(model.bin_centers)
    result = classify_execution_token(token_id, vocab_eff=int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of), n_bins=len(centers), bin_centers=centers, action_stats=stats)
    return {"token_id": int(token_id), "execution_class": result.execution_class, "decoded_raw_gripper": result.decoded_raw_gripper, "executed_env_gripper": result.executed_env_gripper}


def random_adv_inputs(clean_inputs: Mapping[str, Any], seed: int) -> tuple[dict[str, Any], float, str, int]:
    import torch

    from gripper_attack.m3_controls import project_and_cast_processor_values, sample_processor_delta

    original = clean_inputs["pixel_values"]
    delta = sample_processor_delta(tuple(original.shape), epsilon=EPSILON, seed=int(seed), dtype=torch.float32, device=original.device)
    adv, correction_count = project_and_cast_processor_values(original, delta, epsilon=EPSILON, candidate_is_delta=True)
    diff = (adv.float() - original.float()).detach()
    return {"input_ids": clean_inputs["input_ids"], "pixel_values": adv}, float(diff.abs().max().detach().cpu()), clean_tensor_sha256(diff), int(correction_count)


def persist_attack_tensor(output: Path, step: int, condition: str, clean_inputs: Mapping[str, Any], adv_inputs: Mapping[str, Any]) -> dict[str, Any]:
    clean = clean_inputs["pixel_values"].detach().float().cpu().numpy()
    adv = adv_inputs["pixel_values"].detach().float().cpu().numpy()
    delta = adv - clean
    path = output / "attack_tensors" / f"{condition}_{int(step):04d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, clean_pixel_values=clean, adv_pixel_values=adv, delta=delta)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "linf": float(np.abs(delta).max()) if delta.size else 0.0,
        "l2": float(np.linalg.norm(delta.reshape(-1))) if delta.size else 0.0,
    }


def append_telemetry(path: Path, row: Mapping[str, Any]) -> None:
    line = json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def initial_exposure() -> dict[str, Any]:
    return {
        "policy_action_materialized": False,
        "first_env_step_executed": False,
        "model_inference_calls": 0,
        "rows_materialized": 0,
    }


def mark_policy_action_materialized(exposure: dict[str, Any], counters: dict[str, int]) -> None:
    exposure["policy_action_materialized"] = True
    counters["policy_action_materialized_count"] += 1


def mark_env_step_executed(exposure: dict[str, Any]) -> None:
    exposure["first_env_step_executed"] = True


def set_seed(seed: int) -> None:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_parent_env(parent: Mapping[str, Any], suite_cfg: Mapping[str, Any], physical_gpu: int):
    from libero.libero import benchmark, get_libero_path
    from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env

    key = str(parent["canonical_parent_key"])
    suite, task_text, state_text = key.split("/")
    task_idx = int(task_text.split("_")[1])
    state_id = int(state_text.split("_")[1])
    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_idx)
    initial_states = task_suite.get_task_init_states(task_idx)
    if state_id >= len(initial_states):
        raise RuntimeError(f"STATE_ID_OUT_OF_RANGE:{key}:{len(initial_states)}")
    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    env, obs = build_v4_exact_env(bddl, physical_gpu, HORIZONS[suite], 10)
    obs = env.set_init_state(initial_states[state_id])
    env, obs = apply_dummy_wait(env, obs, 10)
    return env, obs, str(task.language), task_idx, state_id, bddl


def update_feature(adapter: Any, step: int, obs: Mapping[str, Any], decoded: Mapping[str, Any], previous_eef: np.ndarray | None) -> tuple[dict[str, Any], np.ndarray]:
    from gripper_attack.d8_streaming_features_v3 import FEATURE_NAMES

    qpos = np.asarray(obs.get("robot0_gripper_qpos", []), dtype=np.float64).reshape(-1)
    eef = np.asarray(obs.get("robot0_eef_pos", []), dtype=np.float64).reshape(-1)
    if qpos.size != 2 or eef.size != 3 or not np.isfinite(qpos).all() or not np.isfinite(eef).all():
        raise RuntimeError(f"TELEMETRY_FIELDS_INVALID:step={step}")
    velocity = np.zeros(3, dtype=np.float64) if previous_eef is None else eef - previous_eef
    raw = np.asarray(decoded["raw_action_7d"], dtype=np.float64)
    env = np.asarray(decoded["env_action_7d"], dtype=np.float64)
    result = adapter.update(step_id=step, raw_gripper=float(raw[6]), env_gripper=float(env[6]), gripper_qpos=float(qpos[0] + qpos[1]), gripper_opening_proxy=float(abs(qpos[0]) + abs(qpos[1]),), eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]), eef_vx=float(velocity[0]), eef_vy=float(velocity[1]), eef_vz=float(velocity[2]), action_dx=float(raw[0]), action_dy=float(raw[1]), action_dz=float(raw[2]), action_gripper=float(env[6]))
    feature = [float(result["features"][name]) for name in FEATURE_NAMES] if result.get("valid") else []
    return {"qpos": qpos.tolist(), "eef": eef.tolist(), "velocity": velocity.tolist(), "feature_valid": bool(result.get("valid")), "feature_error": str(result.get("error", "")), "features_25d": feature, "candidate_close": bool(result.get("valid") and raw[6] < 0.5)}, eef


def run_condition(parent: Mapping[str, Any], condition: str, model: Any, processor: Any, device: str, contract: Mapping[str, Any], protocol: Mapping[str, Any], physical_gpu: int, output: Path, arm_index: int) -> dict[str, Any]:
    from gripper_attack.route_contract import route_config_from_attack_config, validate_attack_request, validate_true_pgd_attack_result
    from gripper_attack.d8_streaming_features_v3 import D8StreamingFeatureAdapterV3
    from gripper_attack.attack_adapter import TokenPrefixPGDAttacker

    suite = str(parent["suite"])
    key = str(parent["canonical_parent_key"])
    expected_emit = int(parent["first_emit_step"])
    horizon = int(parent["policy_horizon"])
    counters = {name: 0 for name in COUNTER_NAMES}
    seeds = primary_seed_values(protocol, key)
    set_seed(seeds["eval_seed"])
    exposure = initial_exposure()
    env = None
    writer = None
    rows: list[dict[str, Any]] = []
    previous_eef: np.ndarray | None = None
    clean_emit_verified = False
    attack_delivered = 0
    terminal_success = False
    terminal_success_step: int | None = None
    perturb_draw_index = 0
    perturb_rng = random.Random(seeds["perturb_seed"])
    start = time.time()
    telemetry_path = output / "step_telemetry.jsonl"
    try:
        env, obs, instruction, task_idx, state_id, bddl = build_parent_env(parent, contract["suites"][suite], physical_gpu)
        counters["env_reset_calls"] = 1
        telemetry_path.touch(exist_ok=True)
        import imageio.v2 as imageio

        video_path = output / "rollout.mp4"
        writer = imageio.get_writer(video_path, fps=20, codec="libx264", macro_block_size=1, quality=7, ffmpeg_log_level="error")
        adapter = D8StreamingFeatureAdapterV3()
        attacker = None
        if condition in {"TRUE_PGD_T5", "SHUFFLED_GRAD_T5"}:
            attacker = TokenPrefixPGDAttacker(model, processor, build_attack_config(condition, seeds["perturb_seed"]), seed=seeds["perturb_seed"], preprocess_kwargs={"center_crop": True, "resize_size": 224, "libero_preprocess_backend": "official_pil_lanczos", "postprocess_gripper": True}, device=device)
            attacker.reset_temporal_state()

        for step in range(horizon):
            raw_image = np.asarray(obs["agentview_image"]).copy()
            if raw_image.dtype != np.uint8:
                raw_image = np.clip(raw_image, 0, 255).astype(np.uint8)
            writer.append_data(raw_image)
            clean_prepared = prepare_generation(model, processor, raw_image, instruction, suite, device)
            counters["openvla_model_inference_calls"] += 1
            counters["model_inference_calls"] += 1
            exposure["model_inference_calls"] += 1
            clean_decoded = decode_tokens(model, clean_prepared["tokens"], suite)
            clean_decoded.update({"generated": clean_prepared["generated"], "inputs": clean_prepared["inputs"], "prompt_len": int(clean_prepared["inputs"]["input_ids"].shape[1]), "raw_hashes": clean_prepared["raw_hashes"]})
            if len(clean_decoded["tokens"]) != 7:
                raise RuntimeError(f"CLEAN_ACTION_TOKEN_COUNT_INVALID:{len(clean_decoded['tokens'])}")
            mark_policy_action_materialized(exposure, counters)
            telemetry, previous_eef = update_feature(adapter, step, obs, clean_decoded, previous_eef)
            if step <= expected_emit:
                if not telemetry["feature_valid"]:
                    if not terminal_success:
                        raise RuntimeError(f"CLEAN_FEATURE_INVALID_BEFORE_EMIT:{step}")
            if step == expected_emit:
                clean_emit_verified = True

            executed = clean_decoded
            attack_summary: dict[str, Any] = {"condition": "CLEAN", "attack_executed": False}
            attack_tensor: dict[str, Any] | None = None
            if clean_emit_verified and expected_emit <= step <= expected_emit + ATTACK_WINDOW - 1 and condition != "CLEAN_EVAL":
                if condition == "RAND_UNIFORM_T5":
                    draw_seed = perturb_rng.getrandbits(32)
                    adv_inputs, linf, delta_sha, correction_count = random_adv_inputs(clean_decoded["inputs"], draw_seed)
                    perturb_draw_index += 1
                    executed = decode_from_inputs(model, adv_inputs, clean_decoded["prompt_len"], suite)
                    attack_summary = {"condition": condition, "attack_executed": True, "route": "m3_controls.sample_processor_delta", "seed": seeds["perturb_seed"], "draw_seed": draw_seed, "draw_index": perturb_draw_index, "epsilon": EPSILON, "step_size": STEP_SIZE, "num_steps": 0, "temporal_attack_budget_frames": ATTACK_WINDOW, "optimizer_steps": 0, "pixel_linf": linf, "delta_sha256": delta_sha, "projection_correction_count": correction_count, "gradient_used": False}
                else:
                    result = attacker.attack(raw_image, instruction, clean_action=np.asarray(clean_decoded["raw_action_7d"], dtype=np.float32), target_action=np.asarray(clean_decoded["raw_action_7d"], dtype=np.float32), clean_model_output=clean_decoded["generated"], unnorm_key=suite)
                    route = route_config_from_attack_config(build_attack_config(condition, seeds["perturb_seed"]))
                    validate_attack_request(route, target_action_present=True)
                    validate_true_pgd_attack_result(result, route)
                    adv_inputs = result.debug["adv_inputs"]
                    executed = decode_from_inputs(model, adv_inputs, clean_decoded["prompt_len"], suite)
                    counters["pgd_calls"] += 1
                    counters["attack_backward_calls"] += int((result.debug or {}).get("num_backwards", 0))
                    attack_summary = {"condition": condition, "attack_executed": True, "seed": seeds["perturb_seed"], "temporal_attack_budget_frames": ATTACK_WINDOW, "optimizer_steps": ATTACK_STEPS, "route": summarize_attack(result)}
                attack_tensor = persist_attack_tensor(output, step, condition, clean_decoded["inputs"], adv_inputs)
                attack_summary["processor_tensor"] = attack_tensor
                counters["adversarial_images"] += 1
                counters["attacked_env_steps"] += 1
                attack_delivered += 1

            clean_tokens = [int(x) for x in clean_decoded["tokens"]]
            executed_tokens = [int(x) for x in executed["tokens"]]
            if len(executed_tokens) != 7:
                raise RuntimeError(f"EXECUTED_ACTION_TOKEN_COUNT_INVALID:{len(executed_tokens)}")
            arm_equal = executed_tokens[:6] == clean_tokens[:6]
            gripper_semantics = classify_gripper(model, suite, int(executed_tokens[-1]))
            if attack_summary.get("attack_executed") and not arm_equal:
                raise RuntimeError(f"ARM_TOKEN_ISOLATION_FAIL:{step}")
            env_action = np.asarray(executed["env_action_7d"], dtype=np.float32)
            obs, reward, done, info = env.step(env_action.tolist())
            counters["env_step_calls"] += 1
            mark_env_step_executed(exposure)
            next_rows_materialized = int(exposure["rows_materialized"]) + 1
            row = {
                "step": step,
                "condition": condition,
                "canonical_parent_key": key,
                "arm_index": arm_index,
                "expected_first_emit_step": expected_emit,
                "raw_agentview_sha256": sha256_bytes(raw_image.tobytes()),
                "raw_agentview_shape": list(raw_image.shape),
                "clean_processor_input_ids_sha256": clean_decoded["raw_hashes"].get("input_ids", ""),
                "clean_processor_pixel_values_sha256": clean_decoded["raw_hashes"].get("pixel_values", ""),
                "clean_token_ids": clean_tokens,
                "executed_token_ids": executed_tokens,
                "clean_action_7d": clean_decoded["env_action_7d"],
                "executed_action_7d": executed["env_action_7d"],
                "clean_raw_gripper": clean_decoded["raw_gripper"],
                "executed_raw_gripper": executed["raw_gripper"],
                "arm_token_ids_equal": arm_equal,
                "executed_gripper_semantics": gripper_semantics,
                "student_feature_valid": telemetry["feature_valid"],
                "candidate_close": telemetry["candidate_close"],
                "robot0_gripper_qpos": telemetry["qpos"],
                "robot0_eef_pos": telemetry["eef"],
                "attack": attack_summary,
                "seeds": {"eval_seed": seeds["eval_seed"], "perturb_seed": seeds["perturb_seed"]},
                "reward": float(reward),
                "done_after_env_step": bool(done),
                "policy_action_materialized": True,
                "first_env_step_executed": True,
                "counters": dict(counters),
                "rows_materialized": next_rows_materialized,
            }
            rows.append(row)
            exposure["rows_materialized"] = next_rows_materialized
            counters["rows_materialized"] = next_rows_materialized
            append_telemetry(telemetry_path, row)
            if done:
                terminal_success = True
                terminal_success_step = step
                break

        if not clean_emit_verified and not terminal_success:
            raise RuntimeError("CLEAN_EMIT_NOT_REACHED")
        if writer is not None:
            writer.close()
            writer = None
        requested_attack_steps = ATTACK_WINDOW if condition != "CLEAN_EVAL" else 0
        requested_followup_end = expected_emit + ATTACK_WINDOW + H_PHYS - 1
        last_step = len(rows) - 1
        physical_followup_complete = bool(clean_emit_verified and last_step >= requested_followup_end)
        attack_fully_delivered = bool(condition == "CLEAN_EVAL" or attack_delivered == requested_attack_steps)
        official_horizon_reached = bool(not terminal_success and len(rows) >= horizon)
        physical_followup_censored = bool(terminal_success and not physical_followup_complete)
        attack_censored = bool(terminal_success and not attack_fully_delivered)
        attack_compliant = attack_fully_delivered
        receipt = {
            "schema": "STAGE_X_X1R_PRIMARY_MATRIX_BRANCH_RECEIPT_V1",
            "status": "PASS_PRIMARY_MATRIX_BRANCH",
            "structural_valid": True,
            "canonical_parent_key": key,
            "review_id": parent["review_id"],
            "ordinal": parent["ordinal"],
            "suite": suite,
            "task_idx": task_idx,
            "state_id": state_id,
            "condition": condition,
            "arm_index": arm_index,
            "arm_order": arm_order(key, protocol),
            "probe_id": PROBE_ID,
            "eval_seed": seeds["eval_seed"],
            "perturb_seed": seeds["perturb_seed"],
            "first_emit_step": expected_emit,
            "attack_window": [expected_emit, expected_emit + ATTACK_WINDOW - 1],
            "physical_followup": [expected_emit + ATTACK_WINDOW, expected_emit + ATTACK_WINDOW + H_PHYS - 1],
            "policy_steps_executed": len(rows),
            "requested_observation_end_step": requested_followup_end,
            "physical_followup_complete": physical_followup_complete,
            "physical_followup_censored_by_terminal_success": physical_followup_censored,
            "attack_requested_steps": requested_attack_steps,
            "attack_delivered_steps": attack_delivered,
            "attack_fully_delivered": attack_fully_delivered,
            "attack_censored_by_terminal_success": attack_censored,
            "attack_compliant": attack_compliant,
            "official_task_success": bool(terminal_success),
            "terminal_success_step": terminal_success_step,
            "official_horizon_reached": official_horizon_reached,
            "final_policy_steps_executed": len(rows),
            "policy_action_materialized": bool(exposure["policy_action_materialized"]),
            "first_env_step_executed": bool(exposure["first_env_step_executed"]),
            "model_inference_calls": int(exposure["model_inference_calls"]),
            "rows_materialized": int(exposure["rows_materialized"]),
            "raw_rollout_video": {"path": str(output / "rollout.mp4"), "sha256": sha256_file(output / "rollout.mp4"), "bytes": (output / "rollout.mp4").stat().st_size},
            "telemetry": {"path": str(telemetry_path), "sha256": sha256_file(telemetry_path), "rows": len(rows)},
            "runtime_source": source_receipt(),
            "official_environment": "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800",
            "gpu": gpu_receipt(physical_gpu, require_free=False),
            "counters": counters,
            "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "vphys_reads": 0, "attack_outcome_reads": 0, "physical_interventions": 0, "eval160_reads": 0, "protected_reads": 0},
            "elapsed_seconds": time.time() - start,
        }
        write_json(output / "branch_receipt.json", receipt)
        return receipt
    except Exception as exc:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        failure = {
            "schema": "STAGE_X_X1R_PRIMARY_MATRIX_BRANCH_RECEIPT_V1",
            "status": "RUNTIME_INVALID_AFTER_FIRST_POLICY_DECISION" if exposure["policy_action_materialized"] else "RUNTIME_INVALID_BEFORE_FIRST_POLICY_DECISION",
            "structural_valid": False,
            "canonical_parent_key": key,
            "review_id": parent["review_id"],
            "ordinal": parent["ordinal"],
            "suite": suite,
            "condition": condition,
            "first_policy_decision": bool(exposure["policy_action_materialized"]),
            "error": f"{type(exc).__name__}:{exc}",
            "policy_action_materialized": bool(exposure["policy_action_materialized"]),
            "first_env_step_executed": bool(exposure["first_env_step_executed"]),
            "model_inference_calls": int(exposure["model_inference_calls"]),
            "rows_materialized": int(exposure["rows_materialized"]),
            "eval_seed": seeds["eval_seed"],
            "perturb_seed": seeds["perturb_seed"],
            "runtime_source": source_receipt(),
            "counters": counters,
            "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "vphys_reads": 0, "attack_outcome_reads": 0, "physical_interventions": 0, "eval160_reads": 0, "protected_reads": 0},
            "retry_authorized": False,
        }
        write_json(output / "branch_receipt.json", failure)
        (output / "step_telemetry.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n" for row in rows), encoding="utf-8")
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def claim_parent(root: Path, parent: Mapping[str, Any]) -> Path:
    claim = root / "claims" / f"{int(parent['ordinal']):03d}_{safe_name(str(parent['canonical_parent_key']))}.claim"
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"PARENT_ALREADY_CLAIMED:{claim}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"canonical_parent_key": parent["canonical_parent_key"], "ordinal": parent["ordinal"], "pid": os.getpid(), "host": socket.gethostname(), "source": source_receipt()}, handle, sort_keys=True)
        handle.write("\n")
    return claim


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-index", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = read_json(PROTOCOL)
    cohort = read_json(COHORT)
    contract = read_json(ROOT / "configs/STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1.json")
    if protocol.get("status") != "FROZEN_PRE_LABEL_INGESTION" or cohort.get("status") != "FROZEN_PRE_ATTACK_IMPLEMENTATION_AUDIT":
        raise SystemExit("PRIMARY_MATRIX_AUTHORITY_NOT_FROZEN")
    rows = list(cohort.get("rows", []))
    if not rows or args.parent_index < 0 or args.parent_index >= len(rows):
        raise SystemExit("PARENT_INDEX_OUT_OF_RANGE")
    parent = rows[args.parent_index]
    if parent.get("suite") not in SUITES or parent.get("legal_horizon") is not True:
        raise SystemExit("PARENT_NOT_ATTACK_ELIGIBLE")
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    if os.statvfs(root).f_bavail * os.statvfs(root).f_frsize <= MIN_FREE_BYTES:
        raise SystemExit("HOLD_DURABLE_STORAGE")
    mount_gpu = gpu_receipt(args.physical_gpu, require_free=True)
    claim = claim_parent(root, parent)
    suite = str(parent["suite"])
    suite_identity = verify_model_identity(contract, suite)
    from scripts.stage_x.run_stage_x1r_t1d1_screening_clean import load_openvla

    suite_cfg = contract["suites"][suite]
    model, processor, device, _action_dim = load_openvla(Path(str(suite_cfg["model_path"])), suite)
    parent_root = root / "parents" / f"{int(parent['ordinal']):03d}_{safe_name(str(parent['canonical_parent_key']))}"
    if parent_root.exists() and any(parent_root.iterdir()):
        raise SystemExit(f"PARENT_OUTPUT_EXISTS:{parent_root}")
    parent_root.mkdir(parents=True, exist_ok=False)
    run_manifest = {
        "schema": "STAGE_X_X1R_PRIMARY_MATRIX_PARENT_MANIFEST_V1",
        "status": "RUNNING",
        "parent": parent,
        "claim": str(claim),
        "source": source_receipt(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "cohort_sha256": sha256_file(COHORT),
        "seed_contract": primary_seed_values(protocol, str(parent["canonical_parent_key"])),
        "arm_order": arm_order(str(parent["canonical_parent_key"]), protocol),
        "suite_model_identity": suite_identity,
        "gpu_before_model_load": mount_gpu,
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "official_environment": "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800",
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    write_json(parent_root / "parent_manifest.json", run_manifest)
    branch_receipts = []
    try:
        for arm_index, condition in enumerate(arm_order(str(parent["canonical_parent_key"]), protocol)):
            branch = parent_root / condition
            branch.mkdir(parents=True, exist_ok=False)
            branch_receipts.append(run_condition(parent, condition, model, processor, device, contract, protocol, args.physical_gpu, branch, arm_index))
        if not all(receipt.get("structural_valid") for receipt in branch_receipts):
            raise RuntimeError("PARENT_STRUCTURAL_INVALID")
        write_json(parent_root / "parent_receipt.json", {
            "schema": "STAGE_X_X1R_PRIMARY_MATRIX_PARENT_RECEIPT_V1",
            "status": "PASS_PRIMARY_MATRIX_PARENT",
            "structural_valid": True,
            "parent": parent,
            "arm_order": arm_order(str(parent["canonical_parent_key"]), protocol),
            "branch_receipts": [{"condition": r["condition"], "status": r["status"], "path": str(parent_root / r["condition"] / "branch_receipt.json")} for r in branch_receipts],
            "source": source_receipt(),
            "gpu_after": gpu_receipt(args.physical_gpu, require_free=False),
            "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "vphys_reads": 0, "attack_outcome_reads": 0, "physical_interventions": 0, "eval160_reads": 0, "protected_reads": 0},
        })
        run_manifest["status"] = "PASS_PRIMARY_MATRIX_PARENT"
        run_manifest["branch_receipts"] = branch_receipts
        write_json(parent_root / "parent_manifest.json", run_manifest)
        print(json.dumps({"status": "PASS_PRIMARY_MATRIX_PARENT", "parent": parent["canonical_parent_key"], "conditions": arm_order(str(parent["canonical_parent_key"]), protocol)}, sort_keys=True))
        return 0
    except Exception as exc:
        write_json(parent_root / "parent_receipt.json", {"schema": "STAGE_X_X1R_PRIMARY_MATRIX_PARENT_RECEIPT_V1", "status": "HOLD_PARENT_STRUCTURAL_FAILURE", "structural_valid": False, "parent": parent, "branch_receipts": branch_receipts, "error": f"{type(exc).__name__}:{exc}", "source": source_receipt(), "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "vphys_reads": 0, "attack_outcome_reads": 0, "physical_interventions": 0, "eval160_reads": 0, "protected_reads": 0}})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
