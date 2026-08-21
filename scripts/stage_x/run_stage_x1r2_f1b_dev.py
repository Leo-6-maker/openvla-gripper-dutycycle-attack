#!/usr/bin/env python3
"""Run the frozen F1-B M0/M1/M2 development comparison on DEV_V3 only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
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

import run_stage_x1r_primary_matrix as primary
import run_stage_x1r2_q3r3_engineering_matrix as engineering
from gripper_attack.failure_evidence import write_failure_receipt

PROTOCOL = ROOT / "configs/STAGE_X_X1R2_F1B_DEV_PROTOCOL_V3.json"
CONTRACT = ROOT / "configs/STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1.json"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
METHODS = ("M0", "M1", "M2")
F1B_FREEZE_DIR = ROOT / "reports/STAGE_X_X1R2_F1B_DEV_METHOD_FREEZE_V3_20260821"
F1B_METHOD_SPEC = F1B_FREEZE_DIR / "F1B_METHOD_SPEC_V3.json"
F1B_PRE_GPU_AUDIT = F1B_FREEZE_DIR / "F1B_PRE_GPU_AUDIT_V3.json"
F1B_ROOT_SEAL = F1B_FREEZE_DIR / "F1B_ROOT_SEAL_V3.json"
F1B_ROOT_SIDECAR = F1B_FREEZE_DIR / "F1B_ROOT_SEAL_V3.sha256"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n"
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


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
        "tree": git("show", "-s", "--format=%T", "HEAD"),
        "status_porcelain": git("status", "--porcelain"),
        "runtime_python": sys.executable,
    }


def gpu_receipt(physical_gpu: int) -> dict[str, Any]:
    fields = [field.strip() for field in subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits", "-i", str(physical_gpu),
    ], text=True).strip().split(",")]
    if len(fields) != 5:
        raise RuntimeError(f"GPU_QUERY_INVALID:{fields}")
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader", "-i", str(physical_gpu),
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
        "foreign_processes_untouched": True,
    }
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != str(physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES_MUST_BIND_SINGLE_PHYSICAL_GPU")
    if receipt["free_memory_mib"] <= 20480:
        raise RuntimeError(f"GPU_RESOURCE_GATE_FAIL:{receipt}")
    return receipt


def durable_preflight(root: Path, minimum_free_bytes: int) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(root)
    free_bytes = int(stat.f_bavail * stat.f_frsize)
    if free_bytes <= int(minimum_free_bytes):
        raise RuntimeError(f"HOLD_DURABLE_STORAGE:{free_bytes}")
    probe = root / f".f1b_durable_probe_{os.getpid()}"
    probe.write_bytes(b"STAGE_X1R2_F1B_DURABLE_PROBE_V3\n")
    probe_sha = sha256_file(probe)
    probe.unlink()
    return {"root": str(root), "free_bytes": free_bytes, "write_probe_sha256": probe_sha}


def validate_f1b_freeze(protocol: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    paths = (F1B_METHOD_SPEC, F1B_PRE_GPU_AUDIT, F1B_ROOT_SEAL, F1B_ROOT_SIDECAR)
    if not all(path.is_file() for path in paths):
        raise SystemExit("F1B_METHOD_FREEZE_ARTIFACT_MISSING")
    root_sha = sha256_file(F1B_ROOT_SEAL)
    if F1B_ROOT_SIDECAR.read_text(encoding="utf-8").split()[0] != root_sha:
        raise SystemExit("F1B_METHOD_FREEZE_ROOT_SIDECAR_MISMATCH")
    seal = load_json(F1B_ROOT_SEAL)
    method = load_json(F1B_METHOD_SPEC)
    audit = load_json(F1B_PRE_GPU_AUDIT)
    if seal.get("status") != "PASS_F1B_PRE_GPU_STATIC_CONTRACT":
        raise SystemExit("F1B_METHOD_FREEZE_NOT_PASS")
    if method.get("status") != "PASS_F1B_METHOD_SPEC_SEALED" or audit.get("status") != "PASS_F1B_PRE_GPU_STATIC_CONTRACT":
        raise SystemExit("F1B_METHOD_FREEZE_STATUS_INVALID")
    protocol_sha = sha256_file(PROTOCOL)
    if seal.get("protocol_sha256") != protocol_sha or method.get("protocol_sha256") != protocol_sha:
        raise SystemExit("F1B_METHOD_FREEZE_PROTOCOL_HASH_MISMATCH")
    if seal.get("method_spec_sha256") != sha256_file(F1B_METHOD_SPEC) or audit.get("method_spec_sha256") != sha256_file(F1B_METHOD_SPEC):
        raise SystemExit("F1B_METHOD_FREEZE_METHOD_SPEC_HASH_MISMATCH")
    if seal.get("pre_gpu_audit_sha256") != sha256_file(F1B_PRE_GPU_AUDIT):
        raise SystemExit("F1B_METHOD_FREEZE_AUDIT_HASH_MISMATCH")
    if seal.get("protected_boundary") != protocol.get("protected_boundary"):
        raise SystemExit("F1B_METHOD_FREEZE_PROTECTED_BOUNDARY_MISMATCH")
    for relative, expected in dict(seal.get("artifact_hashes", {})).items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"F1B_METHOD_FREEZE_ARTIFACT_HASH_MISMATCH:{relative}")
    sealed_commit = str(seal.get("source_commit", ""))
    if not sealed_commit or subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", sealed_commit, str(source["commit"])], check=False).returncode != 0:
        raise SystemExit("F1B_METHOD_FREEZE_SOURCE_NOT_ANCESTOR")


def validate_protocol(protocol: Mapping[str, Any], physical_gpu: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if protocol.get("status") != "FROZEN_F1B_DEV_V3" or protocol.get("scientific_authority") is not False:
        raise SystemExit("F1B_PROTOCOL_NOT_FROZEN")
    official = str(protocol["runtime"]["official_environment"])
    if not str(sys.executable).startswith(official + "/"):
        raise SystemExit(f"F1B_OFFICIAL_ENVIRONMENT_MISMATCH:{sys.executable}")
    source = source_receipt()
    if source["status_porcelain"]:
        raise SystemExit(f"F1B_WORKTREE_NOT_CLEAN:{source['status_porcelain']}")
    validate_f1b_freeze(protocol, source)
    f1a3_root = ROOT / str(protocol["population"]["f1a3_root_seal_path"])
    f1a3_sidecar = f1a3_root.with_suffix(".sha256")
    if not f1a3_root.is_file() or not f1a3_sidecar.is_file():
        raise SystemExit("F1B_F1A3_ROOT_SEAL_MISSING")
    root_sha = sha256_file(f1a3_root)
    if root_sha != str(protocol["population"]["f1a3_root_seal_sha256"]):
        raise SystemExit("F1B_F1A3_ROOT_SEAL_SHA_MISMATCH")
    if not f1a3_sidecar.read_text(encoding="utf-8").split()[0] == root_sha:
        raise SystemExit("F1B_F1A3_ROOT_SIDECAR_MISMATCH")
    seal = load_json(f1a3_root)
    if seal.get("selected_hard_or_unresolved_count") != 0:
        raise SystemExit("F1B_SELECTED_HARD_OR_UNRESOLVED")
    dev_path = ROOT / str(protocol["population"]["path"])
    expected_dev_sha = seal.get("artifact_hashes", {}).get(dev_path.relative_to(ROOT).as_posix())
    if expected_dev_sha != sha256_file(dev_path):
        raise SystemExit("F1B_DEV_LEDGER_ROOT_HASH_MISMATCH")
    ledger = load_json(dev_path)
    rows = list(ledger.get("rows", []))
    if ledger.get("status") != "PASS_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3" or len(rows) != 24:
        raise SystemExit("F1B_DEV_LEDGER_INVALID")
    keys = [str(row.get("canonical_parent_key")) for row in rows]
    if len(set(keys)) != 24 or any(row.get("role") != "DEV_V3" for row in rows):
        raise SystemExit("F1B_DEV_ROLE_OR_UNIQUENESS_INVALID")
    counts = {suite: sum(row.get("suite") == suite for row in rows) for suite in SUITES}
    if counts != {suite: 6 for suite in SUITES}:
        raise SystemExit(f"F1B_DEV_SUITE_COUNTS_INVALID:{counts}")
    frozen = protocol["frozen_attack"]
    if frozen["epsilon_processor_pixel_values"] != 0.03 or frozen["random_start"] is not False or frozen["candidate_policy"] != "STRICT_CANDIDATE_AUDIT_V1":
        raise SystemExit("F1B_ATTACK_BOUNDARY_INVALID")
    if set(protocol["methods"]) != set(METHODS):
        raise SystemExit("F1B_METHOD_FAMILY_INVALID")
    return source, rows


def parent_from_row(row: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    suite, task_text, state_text = str(row["canonical_parent_key"]).split("/")
    return {
        "ordinal": int(ordinal),
        "fixture_id": f"F1B_{suite}_{task_text}_{state_text}",
        "suite": suite,
        "canonical_parent_key": str(row["canonical_parent_key"]),
        "task_idx": int(task_text.split("_")[1]),
        "state_id": int(state_text.split("_")[1]),
        "policy_horizon": int(primary.HORIZONS[suite]),
    }


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def probe_rank(protocol: Mapping[str, Any], key: str, step: int) -> str:
    salt = str(protocol["probe"]["selection_salt"])
    return hashlib.sha256(f"{salt}|{key}|{int(step)}".encode()).hexdigest()


def attack_seed(protocol: Mapping[str, Any], method: str, steps: int, key: str, step: int) -> int:
    salt = "STAGE_X1R2_F1B_ATTACK_V3_20260821"
    return int(hashlib.sha256(f"{salt}|{method}|{int(steps)}|{key}|{int(step)}".encode()).hexdigest()[:8], 16)


def normalize_image(value: Any) -> np.ndarray:
    image = np.asarray(value).copy()
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape != (256, 256, 3):
        raise RuntimeError(f"F1B_AGENTVIEW_SHAPE_INVALID:{list(image.shape)}")
    return image


def protected_boundary() -> dict[str, Any]:
    return {
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
        "protected_reads": 0,
        "vphys_reads": 0,
        "physical_interventions": 0,
        "attack_outcome_reads": 0,
        "attacked_env_steps": 0,
    }


def clean_rollout_multi(parent: Mapping[str, Any], suite_cfg: Mapping[str, Any], model: Any, processor: Any, device: str, output: Path, protocol: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    key = str(parent["canonical_parent_key"])
    suite = str(parent["suite"])
    seed = int(hashlib.sha256(f"STAGE_X1R2_F1B_CLEAN_V3|{key}".encode()).hexdigest()[:8], 16)
    primary.set_seed(seed)
    counters = {"env_reset_calls": 0, "env_step_calls": 0, "model_inference_calls": 0}
    rows: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    runtime_valid = True
    stop_reason = "HORIZON_EXHAUSTED"
    env = None
    try:
        env, obs, instruction, _task_idx, _state_id, _bddl = primary.build_parent_env(parent, suite_cfg, int(os.environ["CUDA_VISIBLE_DEVICES"]))
        counters["env_reset_calls"] = 1
        horizon = int(parent["policy_horizon"])
        for step in range(horizon):
            image = normalize_image(obs["agentview_image"])
            prepared = primary.prepare_generation(model, processor, image, instruction, suite, device)
            decoded = primary.decode_tokens(model, prepared["tokens"], suite)
            decoded.update({"generated": prepared["generated"], "inputs": prepared["inputs"], "prompt_len": int(prepared["inputs"]["input_ids"].shape[1]), "raw_hashes": prepared["raw_hashes"]})
            counters["model_inference_calls"] += 1
            tokens = [int(value) for value in decoded["tokens"]]
            if len(tokens) != 7:
                runtime_valid = False
                stop_reason = f"CLEAN_TOKEN_COUNT_INVALID:{len(tokens)}"
                break
            semantics = primary.classify_gripper(model, suite, tokens[6])
            is_eligible = semantics.get("execution_class") != "NATIVE_OPEN" and step + 14 < horizon
            row = {
                "step": int(step),
                "observation_sha256": sha256_bytes(image.tobytes()),
                "direct_generated_token_ids": tokens,
                "gripper": semantics,
                "eligible": bool(is_eligible),
                "probe_rank": probe_rank(protocol, key, step) if is_eligible else None,
            }
            rows.append(row)
            if is_eligible:
                eligible.append({
                    "step": int(step),
                    "rank": row["probe_rank"],
                    "image_bytes": image.tobytes(),
                    "instruction": instruction,
                    "clean_tokens": tokens,
                    "clean_gripper": semantics,
                    "clean_raw_hashes": dict(prepared["raw_hashes"]),
                    "clean_action_7d": decoded["raw_action_7d"],
                    "prompt_len": int(prepared["inputs"]["input_ids"].shape[1]),
                })
            if step + 1 >= horizon:
                break
            obs, _reward, done, _info = env.step(list(decoded["env_action_7d"]))
            counters["env_step_calls"] += 1
            if done:
                stop_reason = "CLEAN_RUNTIME_TERMINATED"
                break
        selected = sorted(eligible, key=lambda item: str(item["rank"]))[: int(protocol["probe"]["max_per_parent"])]
        selected_rows = []
        for probe_index, item in enumerate(selected):
            probe_path = output / f"probe_{probe_index:02d}_step_{int(item['step']):04d}.bin"
            probe_path.write_bytes(item["image_bytes"])
            selected_rows.append({key: value for key, value in item.items() if key != "image_bytes"} | {
                "probe_index": int(probe_index),
                "observation_path": str(probe_path),
                "observation_sha256": sha256_file(probe_path),
            })
        receipt = {
            "schema": "STAGE_X1R2_F1B_CLEAN_PROBE_RECEIPT_V3",
            "status": "PASS_F1B_CLEAN_RUNTIME" if runtime_valid else "HOLD_F1B_CLEAN_RUNTIME",
            "suite": suite,
            "fixture_id": parent["fixture_id"],
            "canonical_parent_key": key,
            "clean_rollout_runtime_valid": runtime_valid,
            "clean_rollout_stop_reason": stop_reason,
            "observed_rows": len(rows),
            "eligible_rows": sum(bool(row["eligible"]) for row in rows),
            "selected_probe_count": len(selected_rows),
            "rows": rows,
            "selected_probes": selected_rows,
            "counters": counters,
            "student_used": False,
            "student_emit_used": False,
            "protected_boundary": protected_boundary(),
        }
        write_json(output / "clean_probe_receipt.json", receipt)
        return receipt, selected_rows
    except Exception as exc:
        receipt = {
            "schema": "STAGE_X1R2_F1B_CLEAN_PROBE_RECEIPT_V3",
            "status": "HOLD_F1B_CLEAN_RUNTIME",
            "suite": suite,
            "fixture_id": parent["fixture_id"],
            "canonical_parent_key": key,
            "clean_rollout_runtime_valid": False,
            "error": f"{type(exc).__name__}:{exc}",
            "rows": rows,
            "selected_probes": [],
            "counters": counters,
            "student_used": False,
            "student_emit_used": False,
            "protected_boundary": protected_boundary(),
        }
        write_json(output / "clean_probe_receipt.json", receipt)
        return receipt, []
    finally:
        if env is not None:
            env.close()


def build_attack(method: str, steps: int, seed: int, model: Any, processor: Any, device: str, protocol: Mapping[str, Any], temporal_init: str = "none") -> Any:
    from gripper_attack.attack_adapter import OpenVLAVisualAttacker

    objective = str(protocol["methods"][method]["objective"])
    epsilon = float(protocol["frozen_attack"]["epsilon_processor_pixel_values"])
    step_size = float(protocol["frozen_attack"]["step_size_by_iterations"][str(steps)])
    config = {"attack_optimizer": {
        "method": "token_prefix_pgd",
        "strict_route": True,
        "allow_fallback": False,
        "objective": objective,
        "target_token_id": int(protocol["frozen_attack"]["target_token_id_secondary"]),
        "target_execution_class": str(protocol["frozen_attack"]["target_execution_class"]),
        "epsilon": epsilon,
        "step_size": step_size,
        "num_steps": int(steps),
        "cw_margin": 5.0,
        "gripper_margin": 5.0,
        "random_start": False,
        "temporal_init": str(temporal_init),
        "temporal_smooth_lambda": 0.0,
        "prefix_refresh_interval": 1,
        "surrogate_score_path": "cached_autoregressive_generate_v1",
        "gradient_transform": "none",
        "gradient_transform_seed": int(seed),
        "arm_preserve_weight": 0.1,
        "arm_isolation_candidate_policy": "STRICT_CANDIDATE_AUDIT_V1",
    }}
    return OpenVLAVisualAttacker(
        model,
        processor,
        config,
        seed=seed,
        preprocess_kwargs={"center_crop": True, "resize_size": 224, "libero_preprocess_backend": "official_pil_lanczos", "postprocess_gripper": True},
        device=device,
    )


def complete_audit(audit: Any, expected_count: int) -> bool:
    if not isinstance(audit, list) or len(audit) != int(expected_count):
        return False
    sources = ["delta0", *(f"pgd_iteration_{i}" for i in range(1, int(expected_count)))]
    required = ("candidate_index", "candidate_source", "direct_generated_token_ids", "arm_token_ids_equal", "direct_generated_gripper_is_native_open")
    return [row.get("candidate_index") for row in audit] == list(range(int(expected_count))) and [row.get("candidate_source") for row in audit] == sources and all(all(row.get(k) is not None for k in required) for row in audit)


def run_attack(parent: Mapping[str, Any], probe: Mapping[str, Any], method: str, steps: int, model: Any, processor: Any, device: str, protocol: Mapping[str, Any], output: Path) -> dict[str, Any]:
    suite = str(parent["suite"])
    key = str(parent["canonical_parent_key"])
    image = np.frombuffer(Path(str(probe["observation_path"])).read_bytes(), dtype=np.uint8).reshape((256, 256, 3)).copy()
    expected_count = int(steps) + 1
    seed = attack_seed(protocol, method, steps, key, int(probe["step"]))
    counters = {
        "model_inference_calls": 0,
        "attack_invocation_count": 0,
        "true_invocation_reached": 0,
        "pgd_calls": 0,
        "attack_backward_calls": 0,
        "loss_forward_count": 0,
        "attacked_env_steps": 0,
        "physical_interventions": 0,
        "vphys_reads": 0,
        "protected_reads": 0,
        "eval160_reads": 0,
        "attack_outcome_reads": 0,
    }
    trace: dict[str, Any] = {}
    output.mkdir(parents=True, exist_ok=False)
    try:
        clean_prepared = primary.prepare_generation(model, processor, image, str(probe["instruction"]), suite, device)
        clean_decoded = primary.decode_tokens(model, clean_prepared["tokens"], suite)
        clean_decoded.update({"generated": clean_prepared["generated"], "inputs": clean_prepared["inputs"], "prompt_len": int(clean_prepared["inputs"]["input_ids"].shape[1]), "raw_hashes": clean_prepared["raw_hashes"]})
        counters["model_inference_calls"] += 1
        clean_tokens = [int(value) for value in clean_decoded["tokens"]]
        if clean_tokens != [int(value) for value in probe["clean_tokens"]]:
            raise RuntimeError("F1B_CLEAN_PROBE_TOKEN_MISMATCH")
        if primary.classify_gripper(model, suite, clean_tokens[6]).get("execution_class") == "NATIVE_OPEN":
            raise RuntimeError("F1B_CLEAN_PROBE_OPEN_INVALID")
        attacker = build_attack(method, steps, seed, model, processor, device, protocol)
        attacker.reset_temporal_state()
        counters["attack_invocation_count"] = 1
        counters["true_invocation_reached"] = 1
        try:
            result = attacker.attack(
                image,
                str(probe["instruction"]),
                clean_action=np.asarray(clean_decoded["raw_action_7d"], dtype=np.float32),
                target_action=np.asarray(clean_decoded["raw_action_7d"], dtype=np.float32),
                clean_model_output=clean_decoded["generated"],
                unnorm_key=suite,
                execution_trace=trace,
            )
        except Exception as exc:
            diagnostics = getattr(getattr(attacker, "adapter", None), "last_attack_diagnostics", {}) or {}
            audit = diagnostics.get("candidate_audit") if isinstance(diagnostics, Mapping) else None
            counters["pgd_calls"] = 1 if complete_audit(audit, expected_count) else 0
            failure = {
                "schema": "STAGE_X1R2_F1B_ATTACK_RECEIPT_V3",
                "status": "HOLD_F1B_EXECUTABLE_EVIDENCE_INSUFFICIENT",
                "suite": suite,
                "fixture_id": parent["fixture_id"],
                "canonical_parent_key": key,
                "method": method,
                "iterations": int(steps),
                "probe": dict(probe),
                "clean_direct_token_ids": clean_tokens,
                "counters": counters,
                "protected_boundary": protected_boundary(),
                "execution_trace": dict(trace),
                "student_used": False,
                "student_emit_used": False,
            }
            receipt = write_failure_receipt(output / "attack_receipt.json", failure, exc, attacker, expected_count=expected_count)
            receipt["candidate_audit_complete"] = bool(receipt.get("candidate_audit_complete"))
            if str(exc) == "STRUCTURAL_INVALID_NO_SELECTIVE_CANDIDATE" and receipt["candidate_audit_complete"]:
                receipt["status"] = "F1B_NO_STRICT_CANDIDATE"
            write_json(output / "attack_receipt.json", receipt)
            return receipt
        route = primary.summarize_attack(result)
        counters["pgd_calls"] = 1
        counters["attack_backward_calls"] = int(route.get("num_backwards") or 0)
        counters["loss_forward_count"] = int(route.get("num_loss_forwards") or 0)
        adv_inputs = result.debug["adv_inputs"]
        executed = primary.decode_from_inputs(model, adv_inputs, int(clean_decoded["prompt_len"]), suite)
        executed_tokens = [int(value) for value in executed["tokens"]]
        direct_audit = primary.audit_direct_action_tokens(clean_tokens, executed_tokens)
        audit = route.get("arm_isolation_candidate_audit")
        selected_index = route.get("selected_candidate_index")
        selected = next((row for row in audit or [] if int(row.get("candidate_index", -1)) == int(selected_index)), None) if selected_index is not None else None
        valid = bool(
            route.get("strict_route") is True
            and route.get("allow_fallback") is False
            and route.get("fallback_used") is False
            and complete_audit(audit, expected_count)
            and selected is not None
            and direct_audit.get("arm_token_ids_equal") is True
            and selected.get("clean_gripper_is_native_open") is False
            and selected.get("gripper_token_changed") is True
            and selected.get("direct_generated_gripper_is_native_open") is True
            and selected.get("direct_generated_token_ids") == executed_tokens
        )
        receipt = {
            "schema": "STAGE_X1R2_F1B_ATTACK_RECEIPT_V3",
            "status": "PASS_F1B_VALID_CANDIDATE" if valid else "HOLD_F1B_EXECUTABLE_EVIDENCE_INSUFFICIENT",
            "suite": suite,
            "fixture_id": parent["fixture_id"],
            "canonical_parent_key": key,
            "method": method,
            "iterations": int(steps),
            "probe": dict(probe),
            "clean_direct_token_ids": clean_tokens,
            "candidate_direct_token_ids": executed_tokens,
            "direct_action_audit": direct_audit,
            "candidate_audit": audit,
            "candidate_audit_complete": complete_audit(audit, expected_count),
            "selected_candidate_index": selected_index,
            "selected_candidate_source": route.get("selected_candidate_source"),
            "attack_route": route,
            "counters": counters,
            "protected_boundary": protected_boundary(),
            "execution_trace": dict(trace),
            "student_used": False,
            "student_emit_used": False,
        }
        write_json(output / "attack_receipt.json", receipt)
        return receipt
    except Exception as exc:
        receipt = {
            "schema": "STAGE_X1R2_F1B_ATTACK_RECEIPT_V3",
            "status": "HOLD_F1B_EXECUTABLE_EVIDENCE_INSUFFICIENT",
            "suite": suite,
            "fixture_id": parent["fixture_id"],
            "canonical_parent_key": key,
            "method": method,
            "iterations": int(steps),
            "probe": dict(probe),
            "error": f"{type(exc).__name__}:{exc}",
            "counters": counters,
            "protected_boundary": protected_boundary(),
            "execution_trace": dict(trace),
            "student_used": False,
            "student_emit_used": False,
        }
        write_json(output / "attack_receipt.json", receipt)
        return receipt


def run_parent(parent: Mapping[str, Any], suite_cfg: Mapping[str, Any], model: Any, processor: Any, device: str, protocol: Mapping[str, Any], root: Path) -> dict[str, Any]:
    output = root / str(parent["suite"]) / safe_name(str(parent["canonical_parent_key"]))
    output.mkdir(parents=True, exist_ok=False)
    clean_receipt, probes = clean_rollout_multi(parent, suite_cfg, model, processor, device, output, protocol)
    attack_receipts: list[dict[str, Any]] = []
    if clean_receipt.get("status") == "PASS_F1B_CLEAN_RUNTIME":
        for probe in probes:
            for method in METHODS:
                for steps in (5, 10):
                    attack_output = output / f"probe_{int(probe['probe_index']):02d}_step_{int(probe['step']):04d}" / method / f"steps_{steps}"
                    attack_receipts.append(run_attack(parent, probe, method, steps, model, processor, device, protocol, attack_output))
    status = "PASS_F1B_PARENT_EXECUTED" if clean_receipt.get("status") == "PASS_F1B_CLEAN_RUNTIME" else "HOLD_F1B_CLEAN_RUNTIME"
    receipt = {
        "schema": "STAGE_X1R2_F1B_PARENT_RECEIPT_V3",
        "status": status,
        "suite": parent["suite"],
        "fixture_id": parent["fixture_id"],
        "canonical_parent_key": parent["canonical_parent_key"],
        "clean_probe": clean_receipt,
        "attack_count": len(attack_receipts),
        "attack_status_counts": {status: sum(row.get("status") == status for row in attack_receipts) for status in sorted({row.get("status") for row in attack_receipts})},
        "attack_receipts": [{"method": row.get("method"), "iterations": row.get("iterations"), "status": row.get("status"), "path": str(output / f"probe_{int(row['probe']['probe_index']):02d}_step_{int(row['probe']['step']):04d}" / str(row.get("method")) / f"steps_{int(row.get('iterations'))}" / "attack_receipt.json")} for row in attack_receipts],
        "student_used": False,
        "student_emit_used": False,
        "protected_boundary": protected_boundary(),
    }
    write_json(output / "parent_receipt.json", receipt)
    return receipt


def run_worker(protocol_path: Path, physical_gpu: int, worker_index: int, worker_count: int) -> int:
    protocol = load_json(protocol_path)
    source, rows = validate_protocol(protocol, physical_gpu)
    if worker_count < 1 or worker_count > int(protocol["resource"]["max_project_workers"]):
        raise SystemExit("F1B_WORKER_COUNT_INVALID")
    root = Path(str(protocol["runtime"]["durable_output_root"]))
    durable = durable_preflight(root, int(protocol["resource"]["minimum_free_bytes"]))
    gpu = gpu_receipt(physical_gpu)
    assigned = [row for index, row in enumerate(rows) if index % int(worker_count) == int(worker_index)]
    worker_root = root / "workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    worker_receipt: dict[str, Any] = {
        "schema": "STAGE_X1R2_F1B_WORKER_RECEIPT_V3",
        "status": "RUNNING",
        "worker_index": int(worker_index),
        "worker_count": int(worker_count),
        "physical_gpu": int(physical_gpu),
        "assigned_keys": [row["canonical_parent_key"] for row in assigned],
        "source": source,
        "durable_storage": durable,
        "gpu_before_model_load": gpu,
        "protected_boundary": protected_boundary(),
    }
    write_json(worker_root / f"worker_{worker_index:02d}_receipt.json", worker_receipt)
    contract = load_json(CONTRACT)
    parent_receipts: list[dict[str, Any]] = []
    start = time.time()
    try:
        for suite in SUITES:
            suite_rows = [row for row in assigned if row["suite"] == suite]
            if not suite_rows:
                continue
            suite_cfg = contract["suites"][suite]
            primary.verify_model_identity(contract, suite)
            model, processor, device, action_dim = engineering.load_openvla(Path(str(suite_cfg["model_path"])), suite)
            if int(action_dim) != 7:
                raise RuntimeError(f"F1B_ACTION_DIM_INVALID:{suite}:{action_dim}")
            for row in suite_rows:
                parent = parent_from_row(row, rows.index(row))
                try:
                    parent_receipts.append(run_parent(parent, suite_cfg, model, processor, device, protocol, root))
                except Exception as exc:
                    output = root / str(suite) / safe_name(str(row["canonical_parent_key"]))
                    output.mkdir(parents=True, exist_ok=True)
                    failure = {
                        "schema": "STAGE_X1R2_F1B_PARENT_RECEIPT_V3",
                        "status": "HOLD_F1B_RUNTIME",
                        "suite": suite,
                        "canonical_parent_key": row["canonical_parent_key"],
                        "error": f"{type(exc).__name__}:{exc}",
                        "source": source,
                        "protected_boundary": protected_boundary(),
                    }
                    write_json(output / "parent_receipt.json", failure)
                    parent_receipts.append(failure)
            del model, processor
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        worker_receipt.update({
            "status": "PASS_F1B_WORKER_COMPLETED",
            "parent_receipts": [{"suite": row.get("suite"), "canonical_parent_key": row.get("canonical_parent_key"), "status": row.get("status")} for row in parent_receipts],
            "elapsed_seconds": time.time() - start,
            "gpu_after": gpu_receipt(physical_gpu),
        })
        write_json(worker_root / f"worker_{worker_index:02d}_receipt.json", worker_receipt)
        print(json.dumps({"status": worker_receipt["status"], "worker_index": worker_index, "parents": len(parent_receipts)}, sort_keys=True))
        return 0
    except Exception as exc:
        worker_receipt.update({"status": "HOLD_F1B_RUNTIME", "error": f"{type(exc).__name__}:{exc}", "elapsed_seconds": time.time() - start})
        write_json(worker_root / f"worker_{worker_index:02d}_receipt.json", worker_receipt)
        print(json.dumps({"status": worker_receipt["status"], "worker_index": worker_index, "error": worker_receipt["error"]}, sort_keys=True))
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    args = parser.parse_args()
    if args.worker_index < 0 or args.worker_index >= args.worker_count:
        raise SystemExit("F1B_WORKER_INDEX_INVALID")
    return run_worker(args.protocol.resolve(), args.physical_gpu, args.worker_index, args.worker_count)


if __name__ == "__main__":
    raise SystemExit(main())
