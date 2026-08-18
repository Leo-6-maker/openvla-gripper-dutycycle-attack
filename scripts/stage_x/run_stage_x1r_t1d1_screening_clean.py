#!/usr/bin/env python3
"""Run the authorized Stage-X T1-D1 clean screening census.

This entrypoint deliberately contains no attacker, PGD, intervention, V_phys,
Eval160, or protected-evaluation path.  It consumes only the frozen 39-parent
ledger and writes clean screening evidence below the durable root in the
protocol.  One process may run one suite shard so the OpenVLA checkpoint is
loaded once per worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

PROTOCOL = REPO / "configs/STAGE_X_X1R_T1D1_SCREENING_CLEAN_PROTOCOL_V1.json"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
HORIZONS = {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}
PARENT_REL = "reports/STAGE_X_X1R_T1D0R2_PARENT_SEED_INVARIANCE_V1.json"
COUNTER_NAMES = (
    "openvla_weight_loads", "openvla_model_inference_calls",
    "prospective_parent_student_forward_calls", "prospective_parent_clean_rollouts",
    "env_reset_for_prospective_parent", "env_step_calls", "pgd_calls",
    "attack_backward_calls", "adversarial_images", "physical_interventions",
    "vphys_reads", "attack_outcome_reads", "eval160_reads", "protected_reads",
    "attacked_env_steps",
)


def load_json(path: Path) -> Any:
    value = json.loads(path.read_text(encoding="utf-8"))
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default).encode("utf-8"))


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True, stderr=subprocess.STDOUT).strip()


def source_receipt() -> dict[str, Any]:
    return {
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "status_porcelain": git("status", "--porcelain"),
    }


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def stat_free_bytes(path: Path) -> int:
    return int(os.statvfs(path).f_bavail * os.statvfs(path).f_frsize)


def tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return sha256_bytes(value.numpy().tobytes())


def file_tree_digest(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rows.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)})
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {"file_count": len(rows), "bytes": sum(int(row["size"]) for row in rows), "tree_sha256": sha256_bytes(canonical), "rows": rows}


def verify_model_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    errors: list[str] = []
    for suite in SUITES:
        cfg = contract["suites"][suite]
        path = Path(str(cfg["model_path"]))
        if not path.is_dir():
            errors.append(f"MODEL_DIR_MISSING:{suite}:{path}")
            continue
        digest = file_tree_digest(path)
        expected = cfg["model_identity"]
        for key in ("file_count", "bytes", "tree_sha256"):
            if digest[key] != expected[key]:
                errors.append(f"MODEL_IDENTITY_MISMATCH:{suite}:{key}:{digest[key]}!={expected[key]}")
        key_hashes: dict[str, str] = {}
        for relative, expected_sha in expected.get("key_files", {}).items():
            key_path = path / relative
            if not key_path.is_file():
                errors.append(f"MODEL_KEY_MISSING:{suite}:{relative}")
                continue
            actual = sha256_file(key_path)
            key_hashes[relative] = actual
            if actual != expected_sha:
                errors.append(f"MODEL_KEY_SHA_MISMATCH:{suite}:{relative}")
        observed[suite] = {"path": str(path), "identity": digest, "key_files": key_hashes}
    return {"status": "PASS" if not errors else "HOLD_MODEL_PROVENANCE", "errors": errors, "suites": observed}


def load_parent_rows(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = REPO / str(protocol["parent_population"]["seed_report"])
    if sha256_file(path) != protocol["parent_population"]["seed_report_sha256"]:
        raise RuntimeError("PARENT_SEED_REPORT_SHA_MISMATCH")
    report = load_json(path)
    rows = sorted(report.get("rows", []), key=lambda row: int(row["ordinal"]))
    if len(rows) != int(protocol["parent_population"]["count"]):
        raise RuntimeError("PARENT_COUNT_MISMATCH")
    if report.get("status") != "PASS_D0R1_INVARIANTS":
        raise RuntimeError("PARENT_SEED_REPORT_NOT_PASS")
    for row in rows:
        key = str(row["canonical_parent_key"])
        if not row.get("seed_match") or not row.get("ledger_key_match"):
            raise RuntimeError(f"PARENT_SEED_ROW_NOT_CLOSED:{key}")
        suite, task_part, state_part = key.split("/")
        if suite not in SUITES or not task_part.startswith("task_") or not state_part.startswith("state_"):
            raise RuntimeError(f"PARENT_KEY_INVALID:{key}")
    return rows


def load_suite_contract(protocol: Mapping[str, Any]) -> dict[str, Any]:
    path = REPO / str(protocol["suite_contract"]["path"])
    if sha256_file(path) != protocol["suite_contract"]["sha256"]:
        raise RuntimeError("SUITE_CONTRACT_SHA_MISMATCH")
    contract = load_json(path)
    if contract.get("status") != protocol["suite_contract"]["status_required"] or contract.get("scientific_authority") != protocol["suite_contract"]["scientific_authority_required"]:
        raise RuntimeError("SUITE_CONTRACT_SCOPE_INVALID")
    return contract


def student_paths(protocol: Mapping[str, Any]) -> dict[str, Path]:
    cfg = protocol["student"]
    receipt = load_json(Path(str(load_json(REPO / "configs/STAGE_X_X1R_T1D0R2_CLEAN_RUNTIME_AUTHORITY_V1.json")["historical_t1_receipt"]["path"])))
    sealed = receipt["sealed_detector"]
    paths = {
        "checkpoint": Path(str(cfg["checkpoint"])),
        "normalization": Path(str(sealed["normalization"]["path"])),
        "thresholds": Path(str(sealed["thresholds"]["path"])),
    }
    expected = {"checkpoint": cfg["checkpoint_sha256"], "normalization": cfg["normalization_sha256"], "thresholds": cfg["thresholds_sha256"]}
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected[name]:
            raise RuntimeError(f"STUDENT_ARTIFACT_SHA_MISMATCH:{name}:{path}")
    for relative, expected_sha in ((cfg["source"], cfg["source_raw_sha256"]), (cfg["feature_source"], cfg["feature_source_sha256"]), (cfg["adapter_source"], cfg["adapter_source_sha256"])):
        path = REPO / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"STUDENT_SOURCE_SHA_MISMATCH:{relative}")
    return paths


def gpu_receipt(physical_gpu: int) -> dict[str, Any]:
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits", "-i", str(physical_gpu)],
        text=True,
    ).strip()
    fields = [field.strip() for field in query.split(",")]
    if len(fields) != 5:
        raise RuntimeError(f"GPU_QUERY_INVALID:{query}")
    receipt = {"physical_gpu": int(fields[0]), "gpu_uuid": fields[1], "free_memory_mib": int(fields[2]), "used_memory_mib": int(fields[3]), "utilization_gpu_percent": int(fields[4])}
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader", "-i", str(physical_gpu)],
        text=True,
    ).strip()
    receipt["compute_apps"] = [line.strip() for line in apps.splitlines() if line.strip()]
    if receipt["free_memory_mib"] <= 20480:
        raise RuntimeError(f"GPU_RESOURCE_GATE_FAIL:{receipt}")
    return receipt


def durable_preflight(protocol: Mapping[str, Any], *, verify_models: bool = True) -> dict[str, Any]:
    storage = protocol["durable_storage"]
    root = Path(str(storage["root"]))
    root.mkdir(parents=True, exist_ok=True)
    free = stat_free_bytes(root)
    required = int(storage["required_bytes"])
    probe = root / ".d1_write_probe"
    probe.write_bytes(b"STAGE_X_X1R_T1D1_DURABLE_STORAGE_PROBE_V1\n")
    probe_sha = sha256_file(probe)
    probe.unlink()
    video_probe = root / ".d1_video_probe.mp4"
    video_status = "PASS"
    video_error = ""
    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(video_probe, fps=int(storage["video"]["fps"]), codec=str(storage["video"]["codec"]), macro_block_size=1, quality=7, ffmpeg_log_level="error")
        writer.append_data(np.zeros((16, 16, 3), dtype=np.uint8))
        writer.close()
    except Exception as exc:
        video_status = "HOLD"
        video_error = f"{type(exc).__name__}:{exc}"
    finally:
        if video_probe.exists():
            video_probe.unlink()
    model_report = {"status": "SKIPPED", "errors": [], "suites": {}}
    if verify_models:
        model_report = verify_model_contract(load_suite_contract(protocol))
    result = {
        "schema": "STAGE_X_X1R_T1D1_DURABLE_STORAGE_PREFLIGHT_V1",
        "status": "PASS_DURABLE_STORAGE" if free > required and video_status == "PASS" and model_report["status"] == "PASS" else "HOLD_DURABLE_STORAGE",
        "root": str(root),
        "mount": {"free_bytes": free, "required_bytes": required, "margin_bytes": free - required, "free_gib": round(free / 2**30, 3), "required_gib": round(required / 2**30, 3)},
        "write_probe": {"status": "PASS", "sha256": probe_sha},
        "video_probe": {"status": video_status, "error": video_error, "codec": storage["video"]["codec"], "fps": storage["video"]["fps"]},
        "model_contract": model_report,
        "source": source_receipt(),
        "host": socket.gethostname(),
        "timestamp_unix": time.time(),
        "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "attack_outcome_reads": 0, "vphys_reads": 0, "physical_interventions": 0, "pgd_calls": 0},
    }
    write_json(root / "preflight" / "D1_DURABLE_STORAGE_PREFLIGHT.json", result)
    if result["status"] != "PASS_DURABLE_STORAGE":
        raise RuntimeError(result["status"])
    return result


def register_openvla() -> None:
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    registrations = (
        (AutoConfig.register, "openvla", OpenVLAConfig),
        (AutoImageProcessor.register, OpenVLAConfig, PrismaticImageProcessor),
        (AutoProcessor.register, OpenVLAConfig, PrismaticProcessor),
        (AutoModelForVision2Seq.register, OpenVLAConfig, OpenVLAForActionPrediction),
    )
    for register, key, value in registrations:
        try:
            register(key, value)
        except ValueError:
            pass


def load_openvla(model_path: Path, unnorm_key: str) -> tuple[Any, Any, str, int]:
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    register_openvla()
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True, use_fast=False)
    model = AutoModelForVision2Seq.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager", device_map={"": 0})
    model.eval()
    if unnorm_key not in getattr(model, "norm_stats", {}):
        raise RuntimeError(f"UNNORM_KEY_MISSING:{unnorm_key}")
    action_dim = int(model.get_action_dim(unnorm_key))
    if action_dim != 7:
        raise RuntimeError(f"ACTION_DIM_INVALID:{action_dim}")
    return model, processor, str(next(model.parameters()).device), action_dim


def decode_clean(model: Any, processor: Any, image: Any, instruction: str, unnorm_key: str, device: str, action_dim: int) -> dict[str, Any]:
    import torch
    from gripper_attack.openvla_libero_exec_spec import official_prompt, raw_gripper_to_env_gripper
    from gripper_attack.openvla_preprocess import prepare_openvla_image
    from gripper_attack.v3_generation_parity import extract_exact_new_tokens

    processed = prepare_openvla_image(image, center_crop=True, resize_size=224, libero_preprocess_backend="official_pil_lanczos")
    prompt = official_prompt(instruction)
    inputs = processor(prompt, processed, return_tensors="pt")
    raw_input_hashes = {"input_ids": tensor_sha256(inputs["input_ids"]), "pixel_values": tensor_sha256(inputs["pixel_values"])}
    if "attention_mask" in inputs:
        raw_input_hashes["attention_mask"] = tensor_sha256(inputs["attention_mask"])
    inputs = dict(inputs)
    inputs.pop("attention_mask", None)
    if not torch.all(inputs["input_ids"][:, -1] == 29871):
        inputs["input_ids"] = torch.cat((inputs["input_ids"], torch.full_like(inputs["input_ids"][:, :1], 29871)), dim=1)
    model_dtype = next(model.parameters()).dtype
    model_inputs = {}
    for key, value in inputs.items():
        if torch.is_floating_point(value):
            model_inputs[key] = value.to(device=device, dtype=model_dtype)
        else:
            model_inputs[key] = value.to(device=device)
    prompt_len = int(model_inputs["input_ids"].shape[1])
    with torch.inference_mode():
        generated = model.generate(**model_inputs, max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
    tokens = extract_exact_new_tokens(generated.sequences, prompt_len=prompt_len, expected_new_tokens=action_dim)
    token_tensor = np.asarray(tokens, dtype=np.int64)
    vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    discretized = np.clip(vocab_size - token_tensor - 1, 0, int(model.bin_centers.shape[0]) - 1)
    centers = np.asarray(model.bin_centers.detach().cpu() if hasattr(model.bin_centers, "detach") else model.bin_centers)[discretized]
    stats = model.get_action_stats(unnorm_key)
    mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)), dtype=bool)
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    raw_action = np.where(mask, 0.5 * (centers + 1.0) * (high - low) + low, centers).astype(np.float32)
    env_action = np.clip(raw_action, -1.0, 1.0).astype(np.float32)
    env_action[-1] = raw_gripper_to_env_gripper(float(raw_action[-1]))
    env_action = np.clip(env_action, -1.0, 1.0).astype(np.float32)
    return {"prompt": prompt, "processed_size": list(processed.size), "raw_input_hashes": raw_input_hashes, "generation_input_ids_sha256": tensor_sha256(model_inputs["input_ids"]), "tokens": tokens, "raw_action_7d": raw_action.tolist(), "env_action_7d": env_action.tolist(), "raw_gripper": float(raw_action[-1]), "env_gripper": float(env_action[-1])}


def load_student(protocol: Mapping[str, Any], paths: Mapping[str, Path]) -> tuple[Any, np.ndarray, np.ndarray, float, float]:
    import torch

    sys.path.insert(0, str(REPO / "n5/phase3_student"))
    from n5_student_model import N5MultiHeadStudent

    normalization = load_json(paths["normalization"])
    norm = normalization.get("episode_heldout", {}).get("train", {})
    mean = np.asarray(norm.get("mean", []), dtype=np.float32)
    std = np.asarray(norm.get("std", []), dtype=np.float32)
    if mean.shape != (25,) or std.shape != (25,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise RuntimeError("STUDENT_NORMALIZATION_SCHEMA_INVALID")
    thresholds = load_json(paths["thresholds"])
    physical = float(thresholds["physical_criticality"]["threshold"])
    closing = float(thresholds["gripper_closing_state"]["threshold"])
    if (physical, closing) != (0.55, 0.8):
        raise RuntimeError("STUDENT_THRESHOLD_BINDING_MISMATCH")
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu")
    model = N5MultiHeadStudent(input_dim=25, hidden=64, short_rf=32, long_rf=128, dropout=0.0)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, mean, std, physical, closing


def student_trace(model: Any, features: list[list[float]], mean: np.ndarray, std: np.ndarray) -> list[dict[str, float]]:
    import torch

    if not features:
        return []
    raw = np.asarray(features, dtype=np.float32)
    if raw.shape != (len(features), 25) or not np.isfinite(raw).all():
        raise RuntimeError("STUDENT_FEATURE_MATRIX_INVALID")
    x = torch.from_numpy(((raw - mean) / std).astype(np.float32))[None, ...]
    mask = torch.ones((1, len(features)), dtype=torch.bool)
    with torch.no_grad():
        logits = model(x, timestep_mask=mask)
    names = ("physical_criticality", "k10_feasibility", "safe_release", "instability", "gripper_closing_state")
    return [{name: float(torch.sigmoid(logits[name][0, index]).item()) for name in names} for index in range(len(features))]


def schedule(predictions: list[dict[str, float]], candidates: list[bool], horizon: int, physical_threshold: float, closing_threshold: float) -> dict[str, Any]:
    emitted = False
    emit_step: int | None = None
    traces = []
    for step, (prediction, candidate) in enumerate(zip(predictions, candidates)):
        legal = step + 5 + 10 <= horizon
        emit = bool(not emitted and candidate and legal and prediction["physical_criticality"] >= physical_threshold and prediction["gripper_closing_state"] >= closing_threshold)
        if emit:
            emitted = True
            emit_step = step
        traces.append({"step": step, "candidate_close": bool(candidate), "legal_horizon": legal, "emitted_this_step": emit, "physical_criticality": prediction["physical_criticality"], "gripper_closing_state": prediction["gripper_closing_state"]})
    return {"first_emit_step": emit_step, "emitted_count": int(sum(bool(row["emitted_this_step"]) for row in traces)), "traces": traces}


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n" for row in rows), encoding="utf-8")


def run_parent(parent: Mapping[str, Any], protocol: Mapping[str, Any], contract: Mapping[str, Any], model: Any, processor: Any, device: str, action_dim: int, student: tuple[Any, np.ndarray, np.ndarray, float, float], physical_gpu: int, root: Path, attempt: int) -> dict[str, Any]:
    from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
    from gripper_attack.openvla_preprocess import prepare_openvla_image
    from gripper_attack.d8_streaming_features_v3 import D8StreamingFeatureAdapterV3, FEATURE_NAMES

    key = str(parent["canonical_parent_key"])
    suite, task_text, state_text = key.split("/")
    task_idx, state_id = int(task_text.split("_")[1]), int(state_text.split("_")[1])
    horizon = int(HORIZONS[suite])
    out = root / "parents" / f"{int(parent['ordinal']):03d}_{safe_name(key)}" / f"attempt_{int(attempt)}"
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"PARENT_OUTPUT_EXISTS:{out}")
    out.mkdir(parents=True, exist_ok=False)
    suite_cfg = contract["suites"][suite]
    model_start = {"source": source_receipt(), "parent": dict(parent), "canonical_parent_key": key, "condition": "SCREENING_CLEAN", "attempt": int(attempt), "physical_gpu": int(physical_gpu), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    write_json(out / "episode_manifest.json", {"schema": "STAGE_X_X1R_T1D1_SCREENING_CLEAN_EPISODE_MANIFEST_V1", "status": "RUNNING", **model_start, "suite": suite, "task_idx": task_idx, "state_id": state_id, "horizon": horizon, "num_steps_wait": 10, "model_path": suite_cfg["model_path"], "unnorm_key": suite_cfg["unnorm_key"], "clean_only": True, "attack_enabled": False, "physical_intervention": False, "vphys_read": False, "eval160": "UNREAD", "protected_evaluation": "UNREAD"})
    env = None
    writer = None
    rows: list[dict[str, Any]] = []
    counters = {name: 0 for name in COUNTER_NAMES}
    counters["prospective_parent_clean_rollouts"] = 1
    task_success = False
    runtime_error = ""
    first_policy_decision = False
    video_path = out / "clean_rollout.mp4"
    try:
        seed = int(parent["expected_clean_seed"])
        random.seed(seed)
        np.random.seed(seed)
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        from libero.libero import benchmark, get_libero_path
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[suite]()
        task = task_suite.get_task(task_idx)
        initial_states = task_suite.get_task_init_states(task_idx)
        if state_id >= len(initial_states):
            raise RuntimeError(f"STATE_ID_OUT_OF_RANGE:{key}:{len(initial_states)}")
        bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        env, obs = build_v4_exact_env(bddl, physical_gpu, horizon, 10)
        counters["env_reset_for_prospective_parent"] = 1
        obs = env.set_init_state(initial_states[state_id])
        env, obs = apply_dummy_wait(env, obs, 10)
        counters["env_step_calls"] += 10
        instruction = str(task.language)
        import imageio.v2 as imageio

        writer = imageio.get_writer(video_path, fps=20, codec="libx264", macro_block_size=1, quality=7, ffmpeg_log_level="error")
        feature_adapter = D8StreamingFeatureAdapterV3()
        previous_eef: np.ndarray | None = None
        for step in range(horizon):
            raw_image = np.asarray(obs["agentview_image"]).copy()
            if raw_image.dtype != np.uint8:
                raw_image = np.clip(raw_image, 0, 255).astype(np.uint8)
            image_sha = sha256_bytes(raw_image.tobytes())
            writer.append_data(raw_image)
            decoded = decode_clean(model, processor, raw_image, instruction, str(suite_cfg["unnorm_key"]), device, action_dim)
            first_policy_decision = True
            counters["openvla_model_inference_calls"] += 1
            qpos = np.asarray(obs.get("robot0_gripper_qpos", []), dtype=np.float64).reshape(-1)
            eef = np.asarray(obs.get("robot0_eef_pos", []), dtype=np.float64).reshape(-1)
            if qpos.size != 2 or eef.size != 3 or not np.isfinite(qpos).all() or not np.isfinite(eef).all():
                raise RuntimeError(f"TELEMETRY_FIELDS_INVALID:{key}:step={step}")
            velocity = np.zeros(3, dtype=np.float64) if previous_eef is None else eef - previous_eef
            raw_action = np.asarray(decoded["raw_action_7d"], dtype=np.float64)
            env_action = np.asarray(decoded["env_action_7d"], dtype=np.float64)
            feature_result = feature_adapter.update(step_id=step, raw_gripper=float(raw_action[6]), env_gripper=float(env_action[6]), gripper_qpos=float(qpos[0] + qpos[1]), gripper_opening_proxy=float(abs(qpos[0]) + abs(qpos[1])), eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]), eef_vx=float(velocity[0]), eef_vy=float(velocity[1]), eef_vz=float(velocity[2]), action_dx=float(raw_action[0]), action_dy=float(raw_action[1]), action_dz=float(raw_action[2]), action_gripper=float(env_action[6]))
            feature_valid = bool(feature_result.get("valid", False))
            feature_values = [float(feature_result["features"][name]) for name in FEATURE_NAMES] if feature_valid else []
            row = {"step": step, "policy_step": step, "condition": "SCREENING_CLEAN", "canonical_parent_key": key, "raw_agentview_sha256": image_sha, "raw_agentview_shape": list(raw_image.shape), "processor_input_ids_sha256": decoded["raw_input_hashes"].get("input_ids", ""), "processor_pixel_values_sha256": decoded["raw_input_hashes"].get("pixel_values", ""), "processor_attention_mask_sha256": decoded["raw_input_hashes"].get("attention_mask", ""), "generation_input_ids_sha256": decoded["generation_input_ids_sha256"], "prompt": decoded["prompt"], "processed_image_size": decoded["processed_size"], "direct_generated_token_ids": decoded["tokens"], "raw_action_7d": decoded["raw_action_7d"], "action_env_7d": decoded["env_action_7d"], "raw_gripper": decoded["raw_gripper"], "env_gripper": decoded["env_gripper"], "robot0_gripper_qpos": qpos.tolist(), "robot0_eef_pos": eef.tolist(), "robot0_eef_velocity": velocity.tolist(), "feature_valid": feature_valid, "feature_error": str(feature_result.get("error", "")), "features_25d": feature_values, "candidate_close": bool(feature_valid and raw_action[6] < 0.5)}
            obs, reward, done, _info = env.step(env_action.tolist())
            counters["env_step_calls"] += 1
            row["reward"] = float(reward)
            row["done_after_env_step"] = bool(done)
            rows.append(row)
            previous_eef = eef
            if done:
                task_success = True
                break
        writer.close()
        writer = None
        valid_features = all(bool(row["feature_valid"]) for row in rows) and bool(rows)
        student_model, mean, std, physical_threshold, closing_threshold = student
        predictions: list[dict[str, float]] = []
        schedule_result: dict[str, Any]
        if valid_features:
            predictions = student_trace(student_model, [row["features_25d"] for row in rows], mean, std)
            counters["prospective_parent_student_forward_calls"] = 1
            candidates = [bool(row["candidate_close"]) for row in rows]
            schedule_result = schedule(predictions, candidates, horizon, physical_threshold, closing_threshold)
            for row, prediction, trace in zip(rows, predictions, schedule_result["traces"]):
                row["student_probabilities"] = prediction
                row["student_scheduler_trace"] = trace
        else:
            schedule_result = {"first_emit_step": None, "emitted_count": 0, "traces": []}
            for row in rows:
                row["student_probabilities"] = {}
                row["student_scheduler_trace"] = {}
        append_jsonl(out / "step_telemetry.jsonl", rows)
        video_sha = sha256_file(video_path) if video_path.is_file() else ""
        video_size = video_path.stat().st_size if video_path.is_file() else 0
        if video_size > int(protocol["durable_storage"]["per_episode_video_budget_bytes"]):
            raise RuntimeError(f"VIDEO_BUDGET_EXCEEDED:{video_size}")
        receipt = {"schema": "STAGE_X_X1R_T1D1_SCREENING_CLEAN_PARENT_RECEIPT_V1", "status": "PASS_SCREENING_CLEAN_EPISODE", "canonical_parent_key": key, "ordinal": int(parent["ordinal"]), "suite": suite, "task_idx": task_idx, "state_id": state_id, "expected_clean_seed": seed, "condition": "SCREENING_CLEAN", "screening_is_not_clean_eval": True, "policy_horizon": horizon, "policy_steps_executed": len(rows), "clean_success": bool(task_success), "clean_failure": not bool(task_success), "first_emit_step": schedule_result["first_emit_step"], "no_emit_retained": schedule_result["first_emit_step"] is None, "student_status": "PASS_CAUSAL_TRACE" if valid_features else "ABSTAIN_INVALID_FEATURE_STREAM", "attack_eligible_pre_manual_review": bool(task_success and valid_features and schedule_result["first_emit_step"] is not None), "manual_clean_contact_review": "REQUIRED", "video": {"path": str(video_path), "sha256": video_sha, "bytes": video_size, "fps": 20, "overlay": False}, "runtime_source_pre_evidence": source_receipt(), "gpu": gpu_receipt(physical_gpu), "counters": counters, "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "attacked_env_steps": 0, "protected_reads": 0}, "forbidden_actions_executed": [], "timestamp_unix": time.time()}
        write_json(out / "parent_receipt.json", receipt)
        manifest = load_json(out / "episode_manifest.json")
        manifest.update({"status": receipt["status"], "task_name": str(getattr(task, "name", "")), "instruction": instruction, "bddl_file": bddl, "runtime_source_pre_evidence": receipt["runtime_source_pre_evidence"], "counters": counters, "parent_receipt": "parent_receipt.json", "telemetry": "step_telemetry.jsonl", "video": receipt["video"]})
        write_json(out / "episode_manifest.json", manifest)
        return receipt
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}:{exc}"
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        failure = {"schema": "STAGE_X_X1R_T1D1_SCREENING_CLEAN_PARENT_RECEIPT_V1", "status": "RUNTIME_INVALID", "canonical_parent_key": key, "ordinal": int(parent["ordinal"]), "condition": "SCREENING_CLEAN", "first_policy_decision": bool(first_policy_decision), "error": runtime_error, "retry_eligible": not first_policy_decision, "counters": counters, "runtime_source_pre_evidence": source_receipt(), "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "attacked_env_steps": 0, "protected_reads": 0}}
        write_json(out / "parent_receipt.json", failure)
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--ordinal", action="append", type=int, help="Frozen parent ordinal; repeat for one suite shard")
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_json(args.protocol.resolve())
    if protocol.get("schema") != "STAGE_X_X1R_T1D1_SCREENING_CLEAN_PROTOCOL_V1" or protocol.get("status") != "FROZEN_FOR_SCREENING_CLEAN_EXECUTION":
        raise SystemExit("D1_PROTOCOL_NOT_FROZEN")
    source = source_receipt()
    if source["status_porcelain"]:
        raise SystemExit("WORKTREE_NOT_CLEAN")
    if source["branch"] != protocol["implementation"]["branch"]:
        raise SystemExit(f"BRANCH_BINDING_MISMATCH:{source['branch']}")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip().split(",")[0].isdigit() or int(os.environ["CUDA_VISIBLE_DEVICES"].strip().split(",")[0]) != int(args.physical_gpu):
        raise SystemExit("CUDA_VISIBLE_DEVICES_MUST_BIND_SINGLE_REQUESTED_PHYSICAL_GPU")
    parent_rows = load_parent_rows(protocol)
    contract = load_suite_contract(protocol)
    preflight_path = Path(str(protocol["durable_storage"]["root"])) / "preflight" / "D1_DURABLE_STORAGE_PREFLIGHT.json"
    if args.preflight_only:
        preflight = durable_preflight(protocol, verify_models=True)
    elif preflight_path.is_file():
        preflight = load_json(preflight_path)
        if preflight.get("status") != "PASS_DURABLE_STORAGE" or preflight.get("model_contract", {}).get("status") != "PASS":
            raise SystemExit("DURABLE_PREFLIGHT_OR_MODEL_CONTRACT_NOT_PASS")
    else:
        preflight = durable_preflight(protocol, verify_models=True)
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, sort_keys=True))
        return 0
    ordinals = sorted(set(args.ordinal or []))
    if not ordinals:
        raise SystemExit("AT_LEAST_ONE_FROZEN_PARENT_ORDINAL_REQUIRED")
    by_ordinal = {int(row["ordinal"]): row for row in parent_rows}
    missing = [ordinal for ordinal in ordinals if ordinal not in by_ordinal]
    if missing:
        raise SystemExit(f"ORDINAL_NOT_IN_FROZEN_LEDGER:{missing}")
    selected = [by_ordinal[ordinal] for ordinal in ordinals]
    suites = {str(row["canonical_parent_key"]).split("/", 1)[0] for row in selected}
    if len(suites) != 1:
        raise SystemExit("ONE_SUITE_PER_WORKER_REQUIRED")
    gpu = gpu_receipt(int(args.physical_gpu))
    cfg = contract["suites"][next(iter(suites))]
    paths = student_paths(protocol)
    import torch

    torch.set_num_threads(1)
    model, processor, device, action_dim = load_openvla(Path(str(cfg["model_path"])), str(cfg["unnorm_key"]))
    student = load_student(protocol, paths)
    root = Path(str(protocol["durable_storage"]["root"]))
    results = []
    for parent in selected:
        results.append(run_parent(parent, protocol, contract, model, processor, device, action_dim, student, int(args.physical_gpu), root, int(args.attempt)))
    summary = {"schema": "STAGE_X_X1R_T1D1_SCREENING_CLEAN_WORKER_RECEIPT_V1", "status": "PASS", "source": source, "preflight": preflight, "gpu_before_model_load": gpu, "suite": next(iter(suites)), "ordinals": ordinals, "parent_receipts": results, "forbidden_counters": {name: 0 for name in ("pgd_calls", "attack_backward_calls", "adversarial_images", "physical_interventions", "vphys_reads", "attack_outcome_reads", "eval160_reads", "protected_reads", "attacked_env_steps")}, "eval160": "UNREAD", "protected_evaluation": "UNREAD"}
    write_json(root / "workers" / f"worker_{next(iter(suites))}_{os.getpid()}.json", summary)
    print(json.dumps({"status": summary["status"], "suite": summary["suite"], "ordinals": ordinals, "root": str(root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
