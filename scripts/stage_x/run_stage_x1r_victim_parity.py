"""Clean/no-op suite-matched victim parity; never runs PGD or env.step."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_digest(root: Path) -> dict[str, Any]:
    files = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    rows = [{"path": p.relative_to(root).as_posix(), "size": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(files)]

    def digest(predicate) -> str:
        selected = [row for row in rows if predicate(row["path"])]
        raw = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    return {
        "file_count": len(rows),
        "bytes": sum(row["size"] for row in rows),
        "tree_sha256": digest(lambda _: True),
        "weights_sha256": digest(lambda path: path.endswith(".safetensors")),
        "semantic_files_sha256": digest(lambda path: not path.endswith(".safetensors")),
        "key_files": {name: sha256_file(root / name) for name in (
            "config.json", "dataset_statistics.json", "model.safetensors.index.json",
            "processing_prismatic.py", "tokenizer.json", "tokenizer.model"
        )},
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def first_q00(root: Path, suite: str) -> Path:
    candidates = []
    for path in sorted(root.glob("**/CAUSAL_PROBE_SNAPSHOT_V2.json")):
        manifest = load_json(path)
        binding = manifest.get("binding", {})
        if binding.get("probe_id") != "Q00" or str(binding.get("parent_key", "")).split("/", 1)[0] != suite:
            continue
        if manifest.get("status") != "SEALED_PROSPECTIVE_SNAPSHOT":
            raise ValueError(f"SNAPSHOT_NOT_SEALED:{path}")
        candidates.append(path.parent)
    if not candidates:
        raise ValueError(f"NO_Q00_SNAPSHOT:{root}:{suite}")
    return candidates[0]


def register_openvla() -> None:
    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoModelForVision2Seq
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    for register, key, value in (
        (AutoConfig.register, "openvla", OpenVLAConfig),
        (AutoImageProcessor.register, OpenVLAConfig, PrismaticImageProcessor),
        (AutoProcessor.register, OpenVLAConfig, PrismaticProcessor),
        (AutoModelForVision2Seq.register, OpenVLAConfig, OpenVLAForActionPrediction),
    ):
        try:
            register(key, value)
        except ValueError:
            pass


def load_model_and_processor(model_path: Path, device: str):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from scripts.detector_v5.materialize_stage_vii_frozen_embeddings import load_model

    register_openvla()
    model = load_model(model_path, device)
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    model.eval()
    return model, processor


def append_empty_action_token(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch

    result = dict(inputs)
    input_ids = result["input_ids"]
    if not torch.all(input_ids[:, -1] == 29871):
        result["input_ids"] = torch.cat((input_ids, torch.full_like(input_ids[:, :1], 29871)), dim=1)
        if result.get("attention_mask") is not None:
            result["attention_mask"] = torch.cat((result["attention_mask"], torch.ones_like(result["attention_mask"][:, :1])), dim=1)
    return result


def decode_tokens(model: Any, token_ids: Any, unnorm_key: str) -> np.ndarray:
    token_ids = np.asarray(token_ids, dtype=np.int64)
    vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    discretized = np.clip(vocab_size - token_ids - 1, 0, int(model.bin_centers.shape[0]) - 1)
    norm_actions = np.asarray(model.bin_centers.detach().cpu() if hasattr(model.bin_centers, "detach") else model.bin_centers)[discretized]
    stats = model.get_action_stats(unnorm_key)
    mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)), dtype=bool)
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    return np.where(mask, 0.5 * (norm_actions + 1.0) * (high - low) + low, norm_actions).astype(np.float32)


def token_semantics(model: Any, processor: Any, suite: str) -> dict[str, Any]:
    from gripper_attack.attack_adapter import TokenPrefixPGDAttacker

    adapter = TokenPrefixPGDAttacker(model, processor, {}, device="cuda:0")
    region = adapter.get_gripper_region_by_decoded_action(suite, postprocess_gripper=True, open_threshold=0.5)
    return {
        "open_token_ids": sorted(int(x) for x in region["open_token_ids"]),
        "close_token_ids": sorted(int(x) for x in region["close_token_ids"]),
        "boundary_token_ids": sorted(int(x) for x in region["boundary_token_ids"]),
        "open_count": int(region["open_count"]),
        "close_count": int(region["close_count"]),
        "boundary_count": int(len(region["boundary_token_ids"])),
        "canonical_semantics_version": region["canonical_semantics_version"],
    }


def parity_one(model: Any, processor: Any, snapshot_root: Path, suite: str, model_path: Path, tolerance: float) -> dict[str, Any]:
    import torch
    from gripper_attack.openvla_libero_exec_spec import raw_gripper_to_env_gripper
    from gripper_attack.stage_v_causal_observation_snapshot import assert_primary_observation_exact, load_snapshot

    package = load_snapshot(snapshot_root, materialize_torch=True)
    manifest, payload = package["manifest"], package["payload"]
    hashes = assert_primary_observation_exact(payload)
    binding = manifest["binding"]
    decode_config = payload["decode_config"]
    if decode_config.get("base_vla_name") != str(model_path):
        raise ValueError(f"SNAPSHOT_MODEL_PATH_MISMATCH:{snapshot_root}")
    if decode_config.get("unnorm_key") != suite or decode_config.get("center_crop") is not True:
        raise ValueError(f"SNAPSHOT_DECODE_CONTRACT_MISMATCH:{snapshot_root}")
    processed_image = payload["processed_image"]
    if getattr(processed_image, "mode", "") != "RGB" or tuple(getattr(processed_image, "size", ())) != (224, 224):
        raise ValueError(f"PROCESSED_IMAGE_CONTRACT_MISMATCH:{snapshot_root}")

    processed = processor(payload["prompt"], processed_image, return_tensors="pt")
    prepared = append_empty_action_token(processed)
    input_exact = bool(torch.equal(prepared["input_ids"], payload["input_ids"]))
    attention_exact = bool(torch.equal(prepared["attention_mask"], payload["attention_mask"]))
    pixel_cast = prepared["pixel_values"].to(dtype=payload["pixel_values"].dtype)
    pixel_exact = bool(torch.equal(pixel_cast, payload["pixel_values"]))
    if not (input_exact and attention_exact and pixel_exact):
        raise ValueError(f"PROCESSOR_PARITY_FAIL:{snapshot_root}:input={input_exact}:attention={attention_exact}:pixel={pixel_exact}")

    device = "cuda:0"
    model_inputs = {
        "input_ids": prepared["input_ids"].to(device=device),
        "pixel_values": prepared["pixel_values"].to(device=device, dtype=next(model.parameters()).dtype),
    }
    action_dim = int(model.get_action_dim(suite))
    with torch.inference_mode():
        generated = model.generate(**model_inputs, max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
    prompt_len = int(model_inputs["input_ids"].shape[1])
    token_ids = generated.sequences[0, prompt_len:].detach().cpu().numpy().astype(np.int64)
    if len(token_ids) != action_dim:
        raise ValueError(f"CLEAN_GENERATION_LENGTH_FAIL:{snapshot_root}")
    clean_reference = payload["clean_reference_action_window"][0]
    reference_raw = np.asarray(clean_reference["raw_policy_action"], dtype=np.float32)
    reference_env = np.asarray(clean_reference["env_action"], dtype=np.float32)
    decoded = decode_tokens(model, token_ids, suite)
    expected_from_action = TokenPrefixPGDAttacker(model, processor, {}, device=device).action_to_token_ids(reference_raw, suite).detach().cpu().numpy()
    token_exact = bool(np.array_equal(token_ids, expected_from_action))
    action_error = float(np.max(np.abs(decoded - reference_raw)))
    env_decoded = decoded.copy()
    env_decoded[-1] = raw_gripper_to_env_gripper(float(decoded[-1]))
    env_error = float(np.max(np.abs(env_decoded - reference_env)))
    semantic = token_semantics(model, processor, suite)
    return {
        "snapshot_root": str(snapshot_root),
        "snapshot_manifest_sha256": hashes.get("policy_input_sha256"),
        "binding": binding,
        "decode_config": decode_config,
        "processor": {"input_ids_exact": input_exact, "attention_mask_exact": attention_exact, "pixel_values_exact_after_dtype_cast": pixel_exact},
        "clean_generation": {
            "token_ids": [int(x) for x in token_ids],
            "reference_token_ids": [int(x) for x in expected_from_action],
            "token_exact": token_exact,
            "decoded_raw_action": [float(x) for x in decoded],
            "reference_raw_action": [float(x) for x in reference_raw],
            "raw_action_max_abs_error": action_error,
            "decoded_env_action": [float(x) for x in env_decoded],
            "reference_env_action": [float(x) for x in reference_env],
            "env_action_max_abs_error": env_error,
            "action_tolerance": tolerance,
        },
        "semantic_open_target": semantic,
        "pass": bool(token_exact and action_error <= tolerance and env_error <= tolerance),
    }


def nvidia_receipt(physical_gpu: str) -> dict[str, Any]:
    try:
        line = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits", "-i", str(physical_gpu)], text=True).strip()
    except Exception as exc:
        return {"error": repr(exc), "physical_gpu": physical_gpu}
    parts = [part.strip() for part in line.split(",")]
    return {"physical_gpu": int(parts[0]), "gpu_uuid": parts[1], "free_memory_mib": int(parts[2]), "utilization_gpu_percent": int(parts[3])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=REPO / "configs/STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1.json")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    contract = load_json(args.contract)
    if contract.get("schema") != "STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1" or contract.get("scientific_authority") != "X1R_NOT_AUTHORIZED":
        raise ValueError("CONTRACT_SCOPE_INVALID")
    suite_cfg = contract["suites"][args.suite]
    model_path = Path(suite_cfg["model_path"])
    receipt = nvidia_receipt(args.physical_gpu)
    if int(receipt.get("free_memory_mib", 0)) <= 20480:
        raise RuntimeError(f"GPU_RESOURCE_GATE_FAIL:{receipt}")
    observed_digest = model_digest(model_path)
    if observed_digest != suite_cfg["model_identity"]:
        raise RuntimeError(f"MODEL_IDENTITY_MISMATCH:{args.suite}")
    model, processor = load_model_and_processor(model_path, "cuda:0")
    rows = []
    for stage, root_s in contract["snapshot_selection"]["roots"].items():
        snapshot_root = first_q00(Path(root_s), args.suite)
        rows.append(parity_one(model, processor, snapshot_root, args.suite, model_path, args.action_tolerance) | {"stage": stage})
    result = {
        "schema": "STAGE_X_X1R_CLEAN_PARITY_WORKER_V1",
        "status": "PASS_CLEAN_NOOP_PARITY" if all(row["pass"] for row in rows) else "FAIL_CLEAN_NOOP_PARITY",
        "suite": args.suite,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu": receipt,
        "official_environment": contract["runtime"]["official_environment"],
        "runtime_source_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "runtime_source_tree": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD^{tree}"], text=True).strip(),
        "model_path": str(model_path),
        "model_identity": observed_digest,
        "rows": rows,
        "counters": {"pgd_calls": 0, "env_step_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "protected_reads": 0, "eval160_reads": 0},
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS_CLEAN_NOOP_PARITY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
