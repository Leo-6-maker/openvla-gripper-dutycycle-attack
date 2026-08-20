#!/usr/bin/env python3
"""Run one E2 TRUE strict-feasibility probe and stop before attacked env.step."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "stage_x"))

import run_stage_x1r2_q3r3_branch_replay as branch
import run_stage_x1r2_q3r3_engineering_matrix as engineering
import run_stage_x1r_primary_matrix as primary
from gripper_attack.failure_evidence import write_failure_receipt


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


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


def source_receipt(protocol_path: Path) -> dict[str, Any]:
    commit = engineering.git("rev-parse", "HEAD")
    tree = engineering.git("rev-parse", "HEAD^{tree}")
    protocol_rel = protocol_path.resolve().relative_to(ROOT).as_posix()
    source_paths = (
        protocol_rel,
        "scripts/stage_x/run_stage_x1r2_q3r3_e2_feasibility.py",
        "scripts/stage_x/run_stage_x1r2_q3r3_engineering_matrix.py",
        "scripts/stage_x/run_stage_x1r2_q3r3_branch_replay.py",
        "src/gripper_attack/attack_adapter.py",
        "src/gripper_attack/failure_evidence.py",
        "configs/STAGE_X_X1R2_GRIPPER_SELECTIVE_ATTACK_CONTRACT_V1.json",
    )
    return {
        "branch": engineering.git("branch", "--show-current"),
        "commit": commit,
        "tree": tree,
        "status_porcelain": engineering.git("status", "--porcelain"),
        "runtime_file_blobs": {path: engineering.git("rev-parse", f"HEAD:{path}") for path in source_paths},
    }


def run_suite(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_json(protocol_path)
    contract = load_json(ROOT / "configs/STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1.json")
    if protocol.get("status") != "FROZEN_E2_TRUE_FEASIBILITY_ONLY" or protocol.get("scientific_authority") is not False:
        raise SystemExit("E2_PROTOCOL_NOT_FROZEN")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != str(args.physical_gpu):
        raise SystemExit("CUDA_VISIBLE_DEVICES_MUST_BIND_SINGLE_PHYSICAL_GPU")
    source = source_receipt(protocol_path)
    if source["status_porcelain"]:
        raise SystemExit("WORKTREE_NOT_CLEAN")
    engineering.validate_c_root(protocol)
    contract_path = ROOT / str(protocol["attack_contract"]["path"])
    if engineering.sha256_file(contract_path).lower() != str(protocol["attack_contract"]["file_sha256"]).lower():
        raise SystemExit("ATTACK_CONTRACT_RAW_SHA_MISMATCH")
    if engineering.git("rev-parse", f"HEAD:{contract_path.relative_to(ROOT).as_posix()}") != str(protocol["attack_contract"]["git_blob_sha"]):
        raise SystemExit("ATTACK_CONTRACT_GIT_BLOB_MISMATCH")
    suite = str(args.suite)
    fixture_rows = [row for row in protocol["fixtures"] if row["suite"] == suite]
    if args.fixture_id:
        fixture_rows = [row for row in fixture_rows if row["fixture_id"] == args.fixture_id]
    if len(fixture_rows) != 1:
        raise SystemExit("E2_SUITE_NOT_BOUND")
    fixture = fixture_rows[0]
    data = engineering.load_fixture(protocol, fixture, contract)
    parent = data["parent"]
    suite_cfg = contract["suites"][suite]
    root = Path(str(protocol["resource"]["durable_output_root"]))
    durable = engineering.durable_preflight(root, int(protocol["resource"]["minimum_free_bytes"]))
    gpu = engineering.gpu_receipt(args.physical_gpu, require_free=True)
    primary.verify_model_identity(contract, suite)
    model, processor, device, _action_dim = engineering.load_openvla(Path(str(suite_cfg["model_path"])), suite)
    emit = int(parent["first_emit_step"])
    key = str(parent["canonical_parent_key"])
    seed = int(hashlib.sha256(f"STAGE_X1R2_Q3R3_E2_TRUE|{key}".encode()).hexdigest()[:8], 16)
    output = root / suite / str(parent["fixture_id"]) / "E2_TRUE_FEASIBILITY"
    if output.exists():
        raise SystemExit(f"E2_OUTPUT_EXISTS:{output}")
    output.mkdir(parents=True)
    env = None
    attacker = None
    counters = {"env_reset_calls": 0, "env_step_calls": 0, "model_inference_calls": 0, "pgd_calls": 0, "attack_backward_calls": 0, "loss_forward_count": 0, "attacked_env_steps": 0, "physical_interventions": 0, "vphys_reads": 0, "protected_reads": 0, "eval160_reads": 0, "attack_outcome_reads": 0}
    start = time.time()
    try:
        primary.set_seed(seed)
        env, obs, instruction, _task_idx, _state_id, _bddl = primary.build_parent_env(parent, suite_cfg, args.physical_gpu)
        counters["env_reset_calls"] = 1
        for row in data["rows"][:emit]:
            obs, _reward, done, _info = env.step(list(row["action_env_7d"]))
            counters["env_step_calls"] += 1
            if done:
                raise RuntimeError(f"E2_REFERENCE_PREFIX_TERMINATED:{emit}:{row['step']}")
        branch_state = branch.capture_branch_state(env, parent, {}, suite_cfg, seed, emit)
        state_audit = branch.compare_branch_state(data["branch_state"], branch_state)
        if not state_audit.get("equal"):
            raise RuntimeError(f"E2_BRANCH_STATE_MISMATCH:{state_audit}")
        common_image = np.frombuffer(data["reference_image"], dtype=np.uint8).reshape((256, 256, 3)).copy()
        clean_prepared = primary.prepare_generation(model, processor, common_image, instruction, suite, device)
        clean_decoded = primary.decode_tokens(model, clean_prepared["tokens"], suite)
        clean_decoded.update({"generated": clean_prepared["generated"], "inputs": clean_prepared["inputs"], "prompt_len": int(clean_prepared["inputs"]["input_ids"].shape[1]), "raw_hashes": clean_prepared["raw_hashes"]})
        counters["model_inference_calls"] += 1
        clean_tokens = [int(value) for value in clean_decoded["tokens"]]
        expected_tokens = [int(value) for value in data["rows"][emit]["direct_generated_token_ids"]]
        if clean_tokens != expected_tokens:
            raise RuntimeError("E2_CLEAN_REFERENCE_TOKEN_MISMATCH")
        if len(clean_tokens) != 7:
            raise RuntimeError("E2_CLEAN_TOKEN_COUNT_INVALID")
        clean_semantics = primary.classify_gripper(model, suite, clean_tokens[6])
        attacker = engineering.build_attack("TRUE_PGD_T5", seed, model, processor, device)
        attacker.reset_temporal_state()
        trace: dict[str, Any] = {}
        try:
            result = attacker.attack(common_image, instruction, clean_action=np.asarray(clean_decoded["raw_action_7d"], dtype=np.float32), target_action=np.asarray(clean_decoded["raw_action_7d"], dtype=np.float32), clean_model_output=clean_decoded["generated"], unnorm_key=suite, execution_trace=trace)
        except Exception as exc:
            failure = {"schema": "STAGE_X1R2_Q3R3_E2_TRUE_FEASIBILITY_RECEIPT_V1", "status": "HOLD_E2_TRUE_SELECTOR_FAILURE", "suite": suite, "fixture_id": parent["fixture_id"], "canonical_parent_key": key, "branch_state_audit": state_audit, "reference_observation_sha256": sha256_bytes(data["reference_image"]), "counters": counters, "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "attacked_env_steps": 0, "physical_interventions": 0, "vphys_reads": 0, "protected_reads": 0, "attack_outcome_reads": 0}, "source": source}
            receipt = write_failure_receipt(output / "e2_true_receipt.json", failure, exc, attacker)
            receipt.update({"reference_observation_sha256": sha256_bytes(data["reference_image"]), "execution_trace": dict(trace), "elapsed_seconds": time.time() - start})
            write_json(output / "e2_true_receipt.json", receipt)
            return 0
        route = primary.summarize_attack(result)
        counters["pgd_calls"] = 1
        counters["attack_backward_calls"] = int(route.get("num_backwards") or 0)
        counters["loss_forward_count"] = int(route.get("num_loss_forwards") or 0)
        adv_inputs = result.debug["adv_inputs"]
        executed = primary.decode_from_inputs(model, adv_inputs, clean_decoded["prompt_len"], suite)
        executed_tokens = [int(value) for value in executed["tokens"]]
        direct_audit = primary.audit_direct_action_tokens(clean_tokens, executed_tokens)
        executed_semantics = primary.classify_gripper(model, suite, executed_tokens[6])
        audit = route.get("arm_isolation_candidate_audit")
        selected_index = route.get("selected_candidate_index")
        status = "PASS_E2_TRUE_VALID_CANDIDATE" if selected_index is not None else "COMPLETE_E2_TRUE_NO_VALID_CANDIDATE"
        receipt = {"schema": "STAGE_X1R2_Q3R3_E2_TRUE_FEASIBILITY_RECEIPT_V1", "status": status, "selector_completed": True, "suite": suite, "fixture_id": parent["fixture_id"], "canonical_parent_key": key, "branch_state_audit": state_audit, "reference_observation_sha256": sha256_bytes(data["reference_image"]), "clean_direct_token_ids": clean_tokens, "candidate_direct_token_ids": executed_tokens, "clean_gripper": clean_semantics, "candidate_gripper": executed_semantics, "direct_action_audit": direct_audit, "candidate_audit": audit, "selected_candidate_index": selected_index, "selected_candidate_source": (audit[selected_index].get("candidate_source") if isinstance(audit, list) and selected_index is not None else None), "attack_route": route, "execution_trace": dict(trace), "counters": counters, "durable_storage": durable, "gpu_before_model_load": gpu, "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "attacked_env_steps": 0, "physical_interventions": 0, "vphys_reads": 0, "protected_reads": 0, "attack_outcome_reads": 0}, "source": source, "elapsed_seconds": time.time() - start}
        write_json(output / "e2_true_receipt.json", receipt)
        return 0
    finally:
        if env is not None:
            env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--fixture-id")
    return run_suite(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
