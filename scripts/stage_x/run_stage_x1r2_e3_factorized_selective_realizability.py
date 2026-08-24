#!/usr/bin/env python3
"""Run one E3 parent: clean timing-decoupled probe, then TRUE candidate-only PGD."""

from __future__ import annotations

import argparse
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

import run_stage_x1r2_q3r3_engineering_matrix as engineering
import run_stage_x1r_primary_matrix as primary
from gripper_attack.failure_evidence import write_failure_receipt

PROTOCOL = ROOT / "configs/STAGE_X_X1R2_E3_FACTORIZED_SELECTIVE_REALIZABILITY_PROTOCOL_V1.json"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
AUDIT_KEYS = (
    "candidate_index", "candidate_source", "processor_input_sha256", "delta_sha256",
    "pixel_budget_adv_inputs_linf", "direct_generated_token_ids", "clean_arm_token_ids",
    "direct_generated_arm_token_ids", "arm_token_ids_equal", "arm_mismatch_dimensions",
    "clean_gripper_token_id", "clean_gripper_is_native_open", "direct_generated_gripper_token_id",
    "direct_generated_gripper_is_native_open", "gripper_token_changed",
)


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
    raise TypeError(type(value).__name__)


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


def source_receipt(protocol_path: Path) -> dict[str, Any]:
    paths = (
        protocol_path.relative_to(ROOT).as_posix(),
        "reports/STAGE_X_X1R2_E3_SELECTIVE_REALIZABILITY_POOL_V1.json",
        "scripts/stage_x/run_stage_x1r2_e3_factorized_selective_realizability.py",
        "scripts/stage_x/run_stage_x1r2_q3r3_engineering_matrix.py",
        "scripts/stage_x/run_stage_x1r_primary_matrix.py",
        "src/gripper_attack/attack_adapter.py",
        "src/gripper_attack/failure_evidence.py",
        "configs/STAGE_X_X1R2_GRIPPER_SELECTIVE_ATTACK_CONTRACT_V1.json",
    )
    return {
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "status_porcelain": git("status", "--porcelain"),
        "runtime_file_blobs": {path: git("rev-parse", f"HEAD:{path}") for path in paths},
    }


def normalize_image(value: Any) -> np.ndarray:
    image = np.asarray(value).copy()
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape != (256, 256, 3):
        raise RuntimeError(f"E3_AGENTVIEW_SHAPE_INVALID:{list(image.shape)}")
    return image


def validate_protocol(protocol: Mapping[str, Any], protocol_path: Path, physical_gpu: int) -> dict[str, Any]:
    if protocol.get("status") != "FROZEN_E3_FACTORIZED_SELECTIVE_REALIZABILITY" or protocol.get("scientific_authority") is not False:
        raise SystemExit("E3_PROTOCOL_NOT_FROZEN")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != str(physical_gpu):
        raise SystemExit("CUDA_VISIBLE_DEVICES_MUST_BIND_SINGLE_PHYSICAL_GPU")
    source = source_receipt(protocol_path)
    if source["status_porcelain"]:
        raise SystemExit("WORKTREE_NOT_CLEAN")
    population_path = ROOT / str(protocol["population"]["path"])
    if sha256_file(population_path) != str(protocol["population"]["raw_sha256"]):
        raise SystemExit("E3_POOL_RAW_SHA_MISMATCH")
    if git("rev-parse", f"HEAD:{population_path.relative_to(ROOT).as_posix()}") != str(protocol["population"]["git_blob_sha256"]):
        raise SystemExit("E3_POOL_GIT_BLOB_MISMATCH")
    contract_path = ROOT / str(protocol["attack_contract"]["path"])
    if sha256_file(contract_path) != str(protocol["attack_contract"]["raw_sha256"]):
        raise SystemExit("E3_ATTACK_CONTRACT_RAW_SHA_MISMATCH")
    if git("rev-parse", f"HEAD:{contract_path.relative_to(ROOT).as_posix()}") != str(protocol["attack_contract"]["git_blob_sha256"]):
        raise SystemExit("E3_ATTACK_CONTRACT_GIT_BLOB_MISMATCH")
    return source


def parent_from_row(row: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    suite = str(row["suite"])
    return {
        "ordinal": int(ordinal),
        "fixture_id": str(row["fixture_id"]),
        "suite": suite,
        "canonical_parent_key": str(row["canonical_parent_key"]),
        "task_idx": int(row["task_idx"]),
        "state_id": int(row["state_id"]),
        "policy_horizon": int(primary.HORIZONS[suite]),
    }


def probe_rank(protocol: Mapping[str, Any], key: str, step: int) -> str:
    salt = str(protocol["clean_probe"]["step_selection_salt"])
    return hashlib.sha256(f"{salt}|{key}|{int(step)}".encode()).hexdigest()


def attack_seed(protocol: Mapping[str, Any], key: str, step: int) -> int:
    salt = str(protocol["true_execution"]["attack_seed_salt"])
    return int(hashlib.sha256(f"{salt}|{key}|{int(step)}".encode()).hexdigest()[:8], 16)


def protected_boundary() -> dict[str, Any]:
    return {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "protected_reads": 0, "vphys_reads": 0, "physical_interventions": 0, "attack_outcome_reads": 0, "attacked_env_steps": 0}


def fresh_counters() -> dict[str, int]:
    return {"env_reset_calls": 0, "env_step_calls": 0, "model_inference_calls": 0, "pgd_calls": 0, "attack_backward_calls": 0, "loss_forward_count": 0, "true_invocation_reached": 0, "attacked_env_steps": 0, "physical_interventions": 0, "vphys_reads": 0, "protected_reads": 0, "eval160_reads": 0, "attack_outcome_reads": 0}


def complete_audit(audit: Any) -> bool:
    if not isinstance(audit, list) or len(audit) != 6:
        return False
    sources = ["delta0", *(f"pgd_iteration_{i}" for i in range(1, 6))]
    return all(isinstance(row, Mapping) and all(row.get(key) is not None for key in AUDIT_KEYS) for row in audit) and [row["candidate_index"] for row in audit] == list(range(6)) and [row["candidate_source"] for row in audit] == sources


def clean_rollout(parent: Mapping[str, Any], suite_cfg: Mapping[str, Any], model: Any, processor: Any, device: str, physical_gpu: int, output: Path, protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    key = str(parent["canonical_parent_key"])
    suite = str(parent["suite"])
    counters = fresh_counters()
    seed = int(hashlib.sha256(f"STAGE_X1R2_E3_CLEAN|{key}".encode()).hexdigest()[:8], 16)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    runtime_valid = True
    stop_reason = "HORIZON_EXHAUSTED"
    env = None
    try:
        primary.set_seed(seed)
        env, obs, instruction, _task_idx, _state_id, _bddl = primary.build_parent_env(parent, suite_cfg, physical_gpu)
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
            eligible = semantics.get("execution_class") != "NATIVE_OPEN" and step + 14 < horizon
            row = {"step": int(step), "observation_sha256": sha256_bytes(image.tobytes()), "direct_generated_token_ids": tokens, "gripper": semantics, "eligible": bool(eligible), "probe_rank": probe_rank(protocol, key, step) if eligible else None}
            rows.append(row)
            if eligible and (best is None or str(row["probe_rank"]) < str(best["probe_rank"])):
                best = {"step": int(step), "probe_rank": row["probe_rank"], "image_bytes": image.tobytes(), "instruction": instruction, "clean_tokens": tokens, "clean_gripper": semantics, "clean_raw_hashes": dict(prepared["raw_hashes"]), "clean_action_7d": decoded["env_action_7d"]}
            if step + 1 >= horizon:
                break
            obs, _reward, done, _info = env.step(list(decoded["env_action_7d"]))
            counters["env_step_calls"] += 1
            if done:
                stop_reason = "CLEAN_RUNTIME_TERMINATED"
                break
        probe = None
        if best is not None:
            probe_path = output / "probe_observation.bin"
            probe_path.write_bytes(best["image_bytes"])
            probe = {key: value for key, value in best.items() if key != "image_bytes"}
            probe.update({"observation_path": str(probe_path), "observation_sha256": sha256_file(probe_path), "step": int(best["step"])})
        receipt = {"schema": "STAGE_X1R2_E3_CLEAN_PROBE_RECEIPT_V1", "status": "PASS_E3_CLEAN_RUNTIME" if runtime_valid else "HOLD_E3_CLEAN_RUNTIME", "suite": suite, "fixture_id": parent["fixture_id"], "canonical_parent_key": key, "clean_rollout_runtime_valid": runtime_valid, "clean_rollout_stop_reason": stop_reason, "observed_rows": len(rows), "eligible_rows": sum(bool(row["eligible"]) for row in rows), "probe": probe, "rows": rows, "counters": counters, "protected_boundary": protected_boundary()}
        write_json(output / "clean_probe_receipt.json", receipt)
        return receipt, probe
    except Exception as exc:
        receipt = {"schema": "STAGE_X1R2_E3_CLEAN_PROBE_RECEIPT_V1", "status": "HOLD_E3_CLEAN_RUNTIME", "suite": suite, "fixture_id": parent["fixture_id"], "canonical_parent_key": key, "clean_rollout_runtime_valid": False, "error": f"{type(exc).__name__}:{exc}", "observed_rows": len(rows), "rows": rows, "counters": counters, "protected_boundary": protected_boundary()}
        write_json(output / "clean_probe_receipt.json", receipt)
        return receipt, None
    finally:
        if env is not None:
            env.close()


def run_true(parent: Mapping[str, Any], suite_cfg: Mapping[str, Any], model: Any, processor: Any, device: str, physical_gpu: int, output: Path, protocol: Mapping[str, Any], clean_receipt: Mapping[str, Any], probe: Mapping[str, Any], counters: dict[str, int], source: Mapping[str, Any]) -> dict[str, Any]:
    suite = str(parent["suite"])
    key = str(parent["canonical_parent_key"])
    image = np.frombuffer((ROOT / str(probe["observation_path"])).read_bytes(), dtype=np.uint8).reshape((256, 256, 3)).copy()
    clean_prepared = primary.prepare_generation(model, processor, image, str(probe["instruction"]), suite, device)
    clean_decoded = primary.decode_tokens(model, clean_prepared["tokens"], suite)
    clean_decoded.update({"generated": clean_prepared["generated"], "inputs": clean_prepared["inputs"], "prompt_len": int(clean_prepared["inputs"]["input_ids"].shape[1]), "raw_hashes": clean_prepared["raw_hashes"]})
    counters["model_inference_calls"] += 1
    clean_tokens = [int(value) for value in clean_decoded["tokens"]]
    if clean_tokens != [int(value) for value in probe["clean_tokens"]]:
        raise RuntimeError("E3_CLEAN_PROBE_TOKEN_MISMATCH")
    if len(clean_tokens) != 7:
        raise RuntimeError("E3_CLEAN_TOKEN_COUNT_INVALID")
    clean_semantics = primary.classify_gripper(model, suite, clean_tokens[6])
    if clean_semantics.get("execution_class") == "NATIVE_OPEN":
        raise RuntimeError("E3_CLEAN_PROBE_OPEN_INVALID")
    seed = attack_seed(protocol, key, int(probe["step"]))
    attacker = engineering.build_attack("TRUE_PGD_T5", seed, model, processor, device)
    attacker.reset_temporal_state()
    trace: dict[str, Any] = {}
    counters["true_invocation_reached"] = 1
    try:
        result = attacker.attack(image, str(probe["instruction"]), clean_action=np.asarray(clean_decoded["raw_action_7d"], dtype=np.float32), target_action=np.asarray(clean_decoded["raw_action_7d"], dtype=np.float32), clean_model_output=clean_decoded["generated"], unnorm_key=suite, execution_trace=trace)
    except Exception as exc:
        failure = {"schema": "STAGE_X1R2_E3_TRUE_RECEIPT_V1", "status": "HOLD_E3_TRUE_EVIDENCE_INSUFFICIENT", "suite": suite, "fixture_id": parent["fixture_id"], "canonical_parent_key": key, "probe": dict(probe), "clean_probe_receipt": dict(clean_receipt), "counters": counters, "protected_boundary": protected_boundary(), "source": source, "execution_trace": dict(trace)}
        receipt = write_failure_receipt(output / "e3_true_receipt.json", failure, exc, attacker)
        receipt.update({"probe": dict(probe), "clean_probe_receipt": dict(clean_receipt), "counters": counters, "protected_boundary": protected_boundary(), "source": source, "execution_trace": dict(trace)})
        if str(exc) == "STRUCTURAL_INVALID_NO_SELECTIVE_CANDIDATE" and receipt.get("candidate_audit_complete") is True:
            receipt["status"] = "COMPLETE_E3_TRUE_NO_VALID_CANDIDATE"
        write_json(output / "e3_true_receipt.json", receipt)
        return receipt
    route = primary.summarize_attack(result)
    counters["pgd_calls"] = 1
    counters["attack_backward_calls"] = int(route.get("num_backwards") or 0)
    counters["loss_forward_count"] = int(route.get("num_loss_forwards") or 0)
    audit = route.get("arm_isolation_candidate_audit")
    adapter_diag = getattr(getattr(attacker, "adapter", None), "last_attack_diagnostics", None)
    selected_index = route.get("selected_candidate_index")
    selected_row = None
    if selected_index is not None:
        selected_row = next((row for row in audit or [] if int(row.get("candidate_index", -1)) == int(selected_index)), None)
    selected_source = selected_row.get("candidate_source") if isinstance(selected_row, Mapping) else None
    route["selected_candidate_source"] = selected_source
    expected_diag = {"candidate_policy": "STRICT_CANDIDATE_AUDIT_V1", "candidate_audit": audit, "selected_candidate_index": selected_index, "selected_candidate_source": selected_source}
    diagnostics_equal = adapter_diag == expected_diag
    candidate_complete = complete_audit(audit)
    if selected_index is None or not candidate_complete or not diagnostics_equal:
        receipt = {"schema": "STAGE_X1R2_E3_TRUE_RECEIPT_V1", "status": "HOLD_E3_TRUE_EVIDENCE_INSUFFICIENT", "suite": suite, "fixture_id": parent["fixture_id"], "canonical_parent_key": key, "probe": dict(probe), "clean_direct_token_ids": clean_tokens, "candidate_audit": audit, "candidate_audit_complete": candidate_complete, "diagnostics_sources_equal": diagnostics_equal, "selected_candidate_index": selected_index, "attack_route": route, "counters": counters, "protected_boundary": protected_boundary(), "source": source, "execution_trace": dict(trace)}
        write_json(output / "e3_true_receipt.json", receipt)
        return receipt
    adv_inputs = result.debug["adv_inputs"]
    executed = primary.decode_from_inputs(model, adv_inputs, int(clean_decoded["prompt_len"]), suite)
    executed_tokens = [int(value) for value in executed["tokens"]]
    direct_audit = primary.audit_direct_action_tokens(clean_tokens, executed_tokens)
    selected = selected_row
    valid = isinstance(selected, Mapping) and selected.get("arm_token_ids_equal") is True and selected.get("clean_gripper_is_native_open") is False and selected.get("gripper_token_changed") is True and selected.get("direct_generated_gripper_is_native_open") is True and selected.get("direct_generated_token_ids") == executed_tokens and direct_audit.get("arm_token_ids_equal") is True
    receipt = {"schema": "STAGE_X1R2_E3_TRUE_RECEIPT_V1", "status": "PASS_E3_VALID_CANDIDATE" if valid else "HOLD_E3_TRUE_EVIDENCE_INSUFFICIENT", "suite": suite, "fixture_id": parent["fixture_id"], "canonical_parent_key": key, "probe": dict(probe), "clean_direct_token_ids": clean_tokens, "candidate_direct_token_ids": executed_tokens, "clean_gripper": clean_semantics, "candidate_gripper": primary.classify_gripper(model, suite, executed_tokens[6]), "direct_action_audit": direct_audit, "candidate_audit": audit, "candidate_audit_complete": candidate_complete, "diagnostics_sources_equal": diagnostics_equal, "selected_candidate_index": selected_index, "selected_candidate_source": route.get("selected_candidate_source"), "attack_route": route, "counters": counters, "protected_boundary": protected_boundary(), "source": source, "execution_trace": dict(trace)}
    write_json(output / "e3_true_receipt.json", receipt)
    return receipt


def run_parent(protocol_path: Path, parent: Mapping[str, Any], physical_gpu: int) -> int:
    protocol = load_json(protocol_path)
    source = validate_protocol(protocol, protocol_path, physical_gpu)
    root = Path(str(protocol["runtime"]["durable_output_root"]))
    root.mkdir(parents=True, exist_ok=True)
    durable = engineering.durable_preflight(root, int(protocol["resource"]["minimum_free_bytes"]))
    gpu = engineering.gpu_receipt(physical_gpu, require_free=True)
    output = root / str(parent["suite"]) / str(parent["fixture_id"])
    if output.exists():
        attempts = sorted(path for path in output.glob("attempt_*") if path.is_dir())
        output = output / f"attempt_{len(attempts) + 1:02d}"
    output.mkdir(parents=True)
    counters = fresh_counters()
    start = time.time()
    try:
        contract = load_json(ROOT / "configs/STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1.json")
        suite_cfg = contract["suites"][str(parent["suite"])]
        primary.verify_model_identity(contract, str(parent["suite"]))
        model, processor, device, _action_dim = engineering.load_openvla(Path(str(suite_cfg["model_path"])), str(parent["suite"]))
        clean_receipt, probe = clean_rollout(parent, suite_cfg, model, processor, device, physical_gpu, output, protocol)
        counters.update(clean_receipt.get("counters", {}))
        if probe is None:
            status = "NO_E3_PROBE_STEP" if clean_receipt.get("status") == "PASS_E3_CLEAN_RUNTIME" else "HOLD_E3_CLEAN_RUNTIME"
            receipt = {"schema": "STAGE_X1R2_E3_PARENT_RECEIPT_V1", "status": status, "suite": parent["suite"], "fixture_id": parent["fixture_id"], "canonical_parent_key": parent["canonical_parent_key"], "clean_probe": clean_receipt, "counters": counters, "durable_storage": durable, "gpu_before_model_load": gpu, "protected_boundary": protected_boundary(), "source": source, "elapsed_seconds": time.time() - start}
        else:
            true_receipt = run_true(parent, suite_cfg, model, processor, device, physical_gpu, output, protocol, clean_receipt, probe, counters, source)
            receipt = {"schema": "STAGE_X1R2_E3_PARENT_RECEIPT_V1", "status": true_receipt.get("status"), "suite": parent["suite"], "fixture_id": parent["fixture_id"], "canonical_parent_key": parent["canonical_parent_key"], "clean_probe": clean_receipt, "true_receipt": true_receipt, "counters": true_receipt.get("counters", counters), "durable_storage": durable, "gpu_before_model_load": gpu, "protected_boundary": protected_boundary(), "source": source, "elapsed_seconds": time.time() - start}
        write_json(output / "parent_receipt.json", receipt)
        return 0
    except Exception as exc:
        receipt = {"schema": "STAGE_X1R2_E3_PARENT_RECEIPT_V1", "status": "HOLD_E3_RUNTIME", "suite": parent["suite"], "fixture_id": parent["fixture_id"], "canonical_parent_key": parent["canonical_parent_key"], "error": f"{type(exc).__name__}:{exc}", "counters": counters, "protected_boundary": protected_boundary(), "source": source, "elapsed_seconds": time.time() - start}
        write_json(output / "parent_receipt.json", receipt)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    population = load_json(ROOT / str(protocol["population"]["path"]))
    rows = [row for row in population["selected"] if row["suite"] == args.suite and row["fixture_id"] == args.fixture_id]
    if len(rows) != 1:
        raise SystemExit("E3_PARENT_NOT_BOUND")
    ordinal = next(index for index, row in enumerate(population["selected"]) if row["fixture_id"] == args.fixture_id)
    return run_parent(args.protocol.resolve(), parent_from_row(rows[0], ordinal), args.physical_gpu)


if __name__ == "__main__":
    raise SystemExit(main())
