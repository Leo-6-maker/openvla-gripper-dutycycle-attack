#!/usr/bin/env python3
"""Run the sealed F1-C M1-10 temporal comparison on C_CANARY_V3 only."""

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
import run_stage_x1r2_f1b_dev as dev
import run_stage_x1r2_q3r3_engineering_matrix as engineering
from gripper_attack.failure_evidence import write_failure_receipt

PROTOCOL = ROOT / "configs/STAGE_X_X1R2_F1C_METHOD_FREEZE_T5_CANARY_PROTOCOL_V3.json"
CONTRACT = ROOT / "configs/STAGE_X_X1R2_F1B_DEV_PROTOCOL_V3.json"
SUITE_CONTRACT = ROOT / "configs/STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1.json"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
ARMS = ("none", "prev_delta")
FREEZE_DIR = ROOT / "reports/STAGE_X_X1R2_F1C_METHOD_FREEZE_T5_CANARY_V3_20260821"
METHOD_SPEC = FREEZE_DIR / "F1C_METHOD_SPEC_V3.json"
PRE_GPU_AUDIT = FREEZE_DIR / "F1C_PRE_GPU_AUDIT_V3.json"
ROOT_SEAL = FREEZE_DIR / "F1C_ROOT_SEAL_V3.json"
ROOT_SIDECAR = FREEZE_DIR / "F1C_ROOT_SEAL_V3.sha256"

F1A3_ROOT = ROOT / "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_ROOT_SEAL_V3.json"
CANARY_LEDGER = ROOT / "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_C_CANARY_V3_LEDGER_V3.json"
F1B_DECISION = ROOT / "reports/STAGE_X_X1R2_F1B_DEV_RESULT_AGGREGATION_V3_20260821/F1B_DEV_DECISION_V3.json"
F1B_RESULT_ROOT = ROOT / "reports/STAGE_X_X1R2_F1B_DEV_RESULT_AGGREGATION_V3_20260821/F1B_DEV_ROOT_SEAL_V3.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    dev.write_json(path, value)


def sha256_file(path: Path) -> str:
    return dev.sha256_file(path)


def source_receipt() -> dict[str, Any]:
    return dev.source_receipt()


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


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def parent_from_row(row: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    suite, task_text, state_text = str(row["canonical_parent_key"]).split("/")
    return {
        "ordinal": int(ordinal),
        "fixture_id": f"F1C_{suite}_{task_text}_{state_text}",
        "suite": suite,
        "canonical_parent_key": str(row["canonical_parent_key"]),
        "task_idx": int(task_text.split("_")[1]),
        "state_id": int(state_text.split("_")[1]),
        "policy_horizon": int(primary.HORIZONS[suite]),
    }


def probe_rank(protocol: Mapping[str, Any], key: str, step: int) -> str:
    salt = str(protocol["probe"]["selection_salt"])
    return hashlib.sha256(f"{salt}|{key}|{int(step)}".encode()).hexdigest()


def clean_seed(protocol: Mapping[str, Any], key: str) -> int:
    salt = str(protocol["runtime"]["attack_seed_salt"])
    return int(hashlib.sha256(f"{salt}|CLEAN|{key}".encode()).hexdigest()[:8], 16)


def attack_seed(protocol: Mapping[str, Any], key: str, step: int) -> int:
    salt = str(protocol["runtime"]["attack_seed_salt"])
    return int(hashlib.sha256(f"{salt}|M1|10|{key}|{int(step)}".encode()).hexdigest()[:8], 16)


def complete_audit(audit: Any, expected_count: int) -> bool:
    if not isinstance(audit, list) or len(audit) != int(expected_count):
        return False
    sources = ["delta0", *(f"pgd_iteration_{i}" for i in range(1, int(expected_count)))]
    required = ("candidate_index", "candidate_source", "direct_generated_token_ids", "arm_token_ids_equal", "direct_generated_gripper_is_native_open")
    return (
        [row.get("candidate_index") for row in audit] == list(range(int(expected_count)))
        and [row.get("candidate_source") for row in audit] == sources
        and all(all(row.get(key) is not None for key in required) for row in audit)
    )


def validate_f1c_freeze(protocol: Mapping[str, Any], physical_gpu: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if protocol.get("status") != "FROZEN_F1C_T5_CANARY_V3" or protocol.get("scientific_authority") is not False:
        raise SystemExit("F1C_PROTOCOL_NOT_FROZEN")
    official = str(protocol["runtime"]["official_environment"])
    if not str(sys.executable).startswith(official + "/"):
        raise SystemExit(f"F1C_OFFICIAL_ENVIRONMENT_MISMATCH:{sys.executable}")
    source = source_receipt()
    if source["status_porcelain"]:
        raise SystemExit(f"F1C_WORKTREE_NOT_CLEAN:{source['status_porcelain']}")
    for path in (METHOD_SPEC, PRE_GPU_AUDIT, ROOT_SEAL, ROOT_SIDECAR):
        if not path.is_file():
            raise SystemExit(f"F1C_METHOD_FREEZE_ARTIFACT_MISSING:{path}")
    root_sha = sha256_file(ROOT_SEAL)
    if ROOT_SIDECAR.read_text(encoding="utf-8").split()[0] != root_sha:
        raise SystemExit("F1C_METHOD_FREEZE_ROOT_SIDECAR_MISMATCH")
    seal, method, audit = load_json(ROOT_SEAL), load_json(METHOD_SPEC), load_json(PRE_GPU_AUDIT)
    if seal.get("status") != "PASS_F1C_PRE_GPU_STATIC_CONTRACT" or method.get("status") != "PASS_F1C_METHOD_SPEC_SEALED" or audit.get("status") != "PASS_F1C_PRE_GPU_STATIC_CONTRACT":
        raise SystemExit("F1C_METHOD_FREEZE_NOT_PASS")
    protocol_sha = sha256_file(PROTOCOL)
    if seal.get("protocol_sha256") != protocol_sha or method.get("protocol_sha256") != protocol_sha:
        raise SystemExit("F1C_METHOD_FREEZE_PROTOCOL_HASH_MISMATCH")
    if seal.get("method_spec_sha256") != sha256_file(METHOD_SPEC) or seal.get("pre_gpu_audit_sha256") != sha256_file(PRE_GPU_AUDIT):
        raise SystemExit("F1C_METHOD_FREEZE_ARTIFACT_HASH_MISMATCH")
    if seal.get("protected_boundary") != protocol.get("protected_boundary"):
        raise SystemExit("F1C_METHOD_FREEZE_PROTECTED_BOUNDARY_MISMATCH")
    for relative, expected in dict(seal.get("artifact_hashes", {})).items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"F1C_METHOD_FREEZE_SOURCE_HASH_MISMATCH:{relative}")
    sealed_commit = str(seal.get("source_commit", ""))
    if not sealed_commit or subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", sealed_commit, str(source["commit"])], check=False).returncode != 0:
        raise SystemExit("F1C_METHOD_FREEZE_SOURCE_NOT_ANCESTOR")
    f1a3_root = ROOT / str(protocol["population"]["f1a3_root_seal_path"])
    if sha256_file(f1a3_root) != str(protocol["population"]["f1a3_root_seal_sha256"]):
        raise SystemExit("F1C_F1A3_ROOT_HASH_MISMATCH")
    f1a3_sidecar = f1a3_root.with_suffix(".sha256")
    if f1a3_sidecar.read_text(encoding="utf-8").split()[0] != sha256_file(f1a3_root):
        raise SystemExit("F1C_F1A3_ROOT_SIDECAR_MISMATCH")
    if sha256_file(CANARY_LEDGER) != str(protocol["population"]["canary_ledger_sha256"]):
        raise SystemExit("F1C_CANARY_LEDGER_HASH_MISMATCH")
    f1b_decision = load_json(F1B_DECISION)
    f1b_root = load_json(F1B_RESULT_ROOT)
    if f1b_decision.get("status") != "F1B_NEW_METHOD_SELECTED_FOR_F1C" or f1b_decision.get("selected_method", {}).get("method") != "M1" or int(f1b_decision.get("selected_method", {}).get("iterations", -1)) != 10:
        raise SystemExit("F1C_F1B_METHOD_SELECTION_INVALID")
    if sha256_file(F1B_DECISION) != str(protocol["upstream"]["f1b_decision_sha256"]) or sha256_file(F1B_RESULT_ROOT) != str(protocol["upstream"]["f1b_result_root_sha256"]):
        raise SystemExit("F1C_F1B_UPSTREAM_HASH_MISMATCH")
    if f1b_root.get("status") != "PASS_F1B_DEV_RESULT_AGGREGATION":
        raise SystemExit("F1C_F1B_RESULT_ROOT_NOT_PASS")
    ledger = load_json(CANARY_LEDGER)
    rows = list(ledger.get("rows", []))
    if ledger.get("status") != "PASS_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3" or len(rows) != 8:
        raise SystemExit("F1C_CANARY_LEDGER_INVALID")
    if any(row.get("role") != "C_CANARY_V3" or row.get("permanent_exclusion") is not True or row.get("outcome_read") is not False for row in rows):
        raise SystemExit("F1C_CANARY_ROLE_FIREWALL_INVALID")
    counts = {suite: sum(row.get("suite") == suite for row in rows) for suite in SUITES}
    if counts != {suite: 2 for suite in SUITES} or len({row.get("canonical_parent_key") for row in rows}) != 8:
        raise SystemExit(f"F1C_CANARY_COUNTS_INVALID:{counts}")
    if int(protocol["execution"]["attempted_steps"]) != 5 or tuple(protocol["temporal_arms"]) != ARMS:
        raise SystemExit("F1C_EXECUTION_BOUNDARY_INVALID")
    if int(protocol["resource"]["max_project_workers"]) > 8 or protocol["resource"]["one_project_worker_per_physical_gpu"] is not True:
        raise SystemExit("F1C_RESOURCE_BOUNDARY_INVALID")
    return source, rows


def clean_probe_selection(parent: Mapping[str, Any], suite_cfg: Mapping[str, Any], model: Any, processor: Any, device: str, protocol: Mapping[str, Any], output: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    key, suite = str(parent["canonical_parent_key"]), str(parent["suite"])
    counters = {"env_reset_calls": 0, "env_step_calls": 0, "model_inference_calls": 0}
    rows: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    runtime_valid, stop_reason = True, "HORIZON_EXHAUSTED"
    env = None
    try:
        primary.set_seed(clean_seed(protocol, key))
        env, obs, instruction, _task_idx, _state_id, _bddl = primary.build_parent_env(parent, suite_cfg, int(os.environ["CUDA_VISIBLE_DEVICES"]))
        counters["env_reset_calls"] = 1
        for step in range(int(parent["policy_horizon"])):
            image = dev.normalize_image(obs["agentview_image"])
            prepared = primary.prepare_generation(model, processor, image, instruction, suite, device)
            decoded = primary.decode_tokens(model, prepared["tokens"], suite)
            decoded.update({"generated": prepared["generated"], "inputs": prepared["inputs"], "prompt_len": int(prepared["inputs"]["input_ids"].shape[1]), "raw_hashes": prepared["raw_hashes"]})
            counters["model_inference_calls"] += 1
            tokens = [int(value) for value in decoded["tokens"]]
            if len(tokens) != 7:
                runtime_valid, stop_reason = False, f"CLEAN_TOKEN_COUNT_INVALID:{len(tokens)}"
                break
            gripper = primary.classify_gripper(model, suite, tokens[6])
            eligible_now = gripper.get("execution_class") != "NATIVE_OPEN" and step + 14 < int(parent["policy_horizon"])
            row = {
                "step": int(step),
                "observation_sha256": dev.sha256_bytes(image.tobytes()),
                "direct_generated_token_ids": tokens,
                "clean_env_action_7d": [float(value) for value in decoded["env_action_7d"]],
                "gripper": gripper,
                "eligible": bool(eligible_now),
                "probe_rank": probe_rank(protocol, key, step) if eligible_now else None,
            }
            rows.append(row)
            if eligible_now:
                eligible.append({
                    "step": int(step),
                    "rank": row["probe_rank"],
                    "image_bytes": image.tobytes(),
                    "observation_sha256": row["observation_sha256"],
                    "instruction": instruction,
                    "clean_tokens": tokens,
                    "clean_gripper": gripper,
                    "clean_raw_hashes": dict(prepared["raw_hashes"]),
                    "clean_action_7d": decoded["raw_action_7d"],
                    "prefix_clean_env_actions_7d": [list(row["clean_env_action_7d"]) for row in rows[:step]],
                    "prompt_len": int(prepared["inputs"]["input_ids"].shape[1]),
                })
            if step + 1 >= int(parent["policy_horizon"]):
                break
            obs, _reward, done, _info = env.step(list(decoded["env_action_7d"]))
            counters["env_step_calls"] += 1
            if done:
                stop_reason = "CLEAN_RUNTIME_TERMINATED"
                break
        selected = sorted(eligible, key=lambda item: str(item["rank"]))[:1]
        selected_row = None
        if selected:
            item = selected[0]
            probe_path = output / "selected_probe.bin"
            probe_path.write_bytes(item["image_bytes"])
            selected_row = {key: value for key, value in item.items() if key != "image_bytes"} | {
                "probe_index": 0,
                "observation_path": str(probe_path),
                "observation_sha256": dev.sha256_file(probe_path),
            }
        receipt = {
            "schema": "STAGE_X1R2_F1C_CLEAN_PROBE_RECEIPT_V3",
            "status": "PASS_F1C_CLEAN_PROBE" if runtime_valid and selected_row is not None else "HOLD_F1C_NO_PROBE" if runtime_valid else "HOLD_F1C_CLEAN_RUNTIME",
            "suite": suite,
            "fixture_id": parent["fixture_id"],
            "canonical_parent_key": key,
            "clean_rollout_runtime_valid": runtime_valid,
            "clean_rollout_stop_reason": stop_reason,
            "observed_rows": len(rows),
            "eligible_rows": sum(bool(row["eligible"]) for row in rows),
            "selected_probe_count": 1 if selected_row is not None else 0,
            "rows": rows,
            "selected_probe": selected_row,
            "counters": counters,
            "student_used": False,
            "student_emit_used": False,
            "protected_boundary": protected_boundary(),
        }
        write_json(output / "clean_probe_receipt.json", receipt)
        return receipt, selected_row
    except Exception as exc:
        receipt = {
            "schema": "STAGE_X1R2_F1C_CLEAN_PROBE_RECEIPT_V3",
            "status": "HOLD_F1C_CLEAN_RUNTIME",
            "suite": suite,
            "fixture_id": parent["fixture_id"],
            "canonical_parent_key": key,
            "clean_rollout_runtime_valid": False,
            "error": f"{type(exc).__name__}:{exc}",
            "rows": rows,
            "selected_probe": None,
            "counters": counters,
            "student_used": False,
            "student_emit_used": False,
            "protected_boundary": protected_boundary(),
        }
        write_json(output / "clean_probe_receipt.json", receipt)
        return receipt, None
    finally:
        if env is not None:
            env.close()


def replay_clean_prefix(env: Any, obs: Any, prefix: Any) -> tuple[Any, int]:
    if not isinstance(prefix, list):
        raise RuntimeError("F1C_REPLAY_PREFIX_ACTIONS_MISSING")
    for step, action in enumerate(prefix):
        if not isinstance(action, list) or len(action) != 7:
            raise RuntimeError(f"F1C_REPLAY_PREFIX_ACTION_INVALID:{step}")
        obs, _reward, done, _info = env.step([float(value) for value in action])
        if done:
            raise RuntimeError("F1C_REPLAY_TERMINATED_BEFORE_PROBE")
    return obs, len(prefix)


def reconstruct_to_probe(parent: Mapping[str, Any], suite_cfg: Mapping[str, Any], model: Any, processor: Any, device: str, protocol: Mapping[str, Any], probe: Mapping[str, Any]) -> tuple[Any, Any, str, dict[str, Any], dict[str, int]]:
    key, suite, target = str(parent["canonical_parent_key"]), str(parent["suite"]), int(probe["step"])
    counters = {"env_reset_calls": 0, "env_step_calls": 0, "model_inference_calls": 0}
    primary.set_seed(clean_seed(protocol, key))
    env, obs, instruction, _task_idx, _state_id, _bddl = primary.build_parent_env(parent, suite_cfg, int(os.environ["CUDA_VISIBLE_DEVICES"]))
    counters["env_reset_calls"] = 1
    try:
        prefix = probe.get("prefix_clean_env_actions_7d")
        if not isinstance(prefix, list) or len(prefix) != target:
            raise RuntimeError("F1C_REPLAY_PREFIX_ACTIONS_MISSING")
        obs, replay_steps = replay_clean_prefix(env, obs, prefix)
        counters["env_step_calls"] += replay_steps
        image = dev.normalize_image(obs["agentview_image"])
        prepared = primary.prepare_generation(model, processor, image, instruction, suite, device)
        decoded = primary.decode_tokens(model, prepared["tokens"], suite)
        decoded.update({"generated": prepared["generated"], "inputs": prepared["inputs"], "prompt_len": int(prepared["inputs"]["input_ids"].shape[1]), "raw_hashes": prepared["raw_hashes"]})
        counters["model_inference_calls"] += 1
        tokens = [int(value) for value in decoded["tokens"]]
        if len(tokens) != 7:
            raise RuntimeError(f"F1C_REPLAY_TOKEN_COUNT_INVALID:{len(tokens)}")
        if dev.sha256_bytes(image.tobytes()) != str(probe["observation_sha256"]):
            raise RuntimeError("F1C_REPLAY_OBSERVATION_HASH_MISMATCH")
        if tokens != [int(value) for value in probe["clean_tokens"]]:
            raise RuntimeError("F1C_REPLAY_DIRECT_TOKEN_MISMATCH")
        return env, obs, instruction, decoded, counters
    except Exception:
        env.close()
        raise


def attack_step(parent: Mapping[str, Any], probe: Mapping[str, Any], arm: str, attempt: int, clean_decoded: Mapping[str, Any], attacker: Any, model: Any, device: str, protocol: Mapping[str, Any], output: Path) -> tuple[dict[str, Any], list[float], bool]:
    suite, key = str(parent["suite"]), str(parent["canonical_parent_key"])
    clean_tokens = [int(value) for value in clean_decoded["tokens"]]
    clean_gripper = primary.classify_gripper(model, suite, clean_tokens[6])
    counters = {"attack_invocation_count": 0, "true_invocation_reached": 0, "pgd_calls": 0, "attack_backward_calls": 0, "loss_forward_count": 0, "attacked_env_steps": 0, "physical_interventions": 0, "vphys_reads": 0, "protected_reads": 0, "eval160_reads": 0, "attack_outcome_reads": 0}
    trace: dict[str, Any] = {}
    step_dir = output / f"attempt_{int(attempt):02d}"
    step_dir.mkdir(parents=True, exist_ok=False)
    clean_action = [float(value) for value in clean_decoded["env_action_7d"]]
    if clean_gripper.get("execution_class") == "NATIVE_OPEN":
        receipt = {
            "schema": "STAGE_X1R2_F1C_ATTACK_RECEIPT_V3",
            "status": "F1C_CLEAN_NATIVE_OPEN_NO_ATTACK",
            "suite": suite,
            "canonical_parent_key": key,
            "temporal_init": arm,
            "attempt": int(attempt),
            "clean_direct_token_ids": clean_tokens,
            "clean_gripper": clean_gripper,
            "candidate_audit_complete": False,
            "candidate_audit": [],
            "counters": counters,
            "protected_boundary": protected_boundary(),
            "student_used": False,
            "student_emit_used": False,
        }
        write_json(step_dir / "attack_receipt.json", receipt)
        return receipt, clean_action, False
    counters["attack_invocation_count"] = 1
    counters["true_invocation_reached"] = 1
    try:
        result = attacker.attack(
            np.asarray(clean_decoded["image"]),
            str(probe["instruction"]),
            clean_action=np.asarray(clean_decoded["raw_action_7d"], dtype=np.float32),
            target_action=np.asarray(clean_decoded["raw_action_7d"], dtype=np.float32),
            clean_model_output=clean_decoded["generated"],
            unnorm_key=suite,
            execution_trace=trace,
        )
        route = primary.summarize_attack(result)
        counters["pgd_calls"] = 1
        counters["attack_backward_calls"] = int(route.get("num_backwards") or 0)
        counters["loss_forward_count"] = int(route.get("num_loss_forwards") or 0)
        adv_inputs = result.debug.get("adv_inputs") if getattr(result, "debug", None) else None
        audit = route.get("arm_isolation_candidate_audit")
        selected_index = route.get("selected_candidate_index")
        selected = next((row for row in audit or [] if int(row.get("candidate_index", -1)) == int(selected_index)), None) if selected_index is not None else None
        executed = None
        direct_audit = None
        if isinstance(adv_inputs, Mapping) and adv_inputs.get("pixel_values") is not None:
            executed = primary.decode_from_inputs(model, adv_inputs, int(clean_decoded["prompt_len"]), suite)
            executed_tokens = [int(value) for value in executed["tokens"]]
            direct_audit = primary.audit_direct_action_tokens(clean_tokens, executed_tokens)
        valid = bool(
            route.get("strict_route") is True
            and route.get("allow_fallback") is False
            and route.get("fallback_used") is False
            and complete_audit(audit, 11)
            and selected is not None
            and direct_audit is not None
            and direct_audit.get("arm_token_ids_equal") is True
            and selected.get("clean_gripper_is_native_open") is False
            and selected.get("gripper_token_changed") is True
            and selected.get("direct_generated_gripper_is_native_open") is True
            and selected.get("direct_generated_token_ids") == direct_audit.get("executed_token_ids")
        )
        receipt = {
            "schema": "STAGE_X1R2_F1C_ATTACK_RECEIPT_V3",
            "status": "PASS_F1C_STRICT_CANDIDATE" if valid else "F1C_NO_STRICT_CANDIDATE" if complete_audit(audit, 11) else "HOLD_F1C_EXECUTABLE_EVIDENCE_INSUFFICIENT",
            "suite": suite,
            "canonical_parent_key": key,
            "temporal_init": arm,
            "attempt": int(attempt),
            "probe": dict(probe),
            "clean_direct_token_ids": clean_tokens,
            "candidate_direct_token_ids": direct_audit.get("executed_token_ids") if direct_audit else None,
            "direct_action_audit": direct_audit,
            "candidate_audit": audit,
            "candidate_audit_complete": complete_audit(audit, 11),
            "selected_candidate_index": selected_index,
            "selected_candidate_source": route.get("selected_candidate_source"),
            "attack_route": route,
            "counters": counters,
            "protected_boundary": protected_boundary(),
            "execution_trace": dict(trace),
            "student_used": False,
            "student_emit_used": False,
        }
        write_json(step_dir / "attack_receipt.json", receipt)
        return receipt, [float(value) for value in (executed["env_action_7d"] if valid else clean_decoded["env_action_7d"])], valid
    except Exception as exc:
        diagnostics = getattr(getattr(attacker, "adapter", None), "last_attack_diagnostics", {}) or {}
        audit = diagnostics.get("candidate_audit") if isinstance(diagnostics, Mapping) else None
        failure = {
            "schema": "STAGE_X1R2_F1C_ATTACK_RECEIPT_V3",
            "status": "HOLD_F1C_EXECUTABLE_EVIDENCE_INSUFFICIENT",
            "suite": suite,
            "canonical_parent_key": key,
            "temporal_init": arm,
            "attempt": int(attempt),
            "probe": dict(probe),
            "clean_direct_token_ids": clean_tokens,
            "counters": counters,
            "protected_boundary": protected_boundary(),
            "execution_trace": dict(trace),
            "student_used": False,
            "student_emit_used": False,
        }
        receipt = write_failure_receipt(step_dir / "attack_receipt.json", failure, exc, attacker, expected_count=11)
        receipt["schema"] = "STAGE_X1R2_F1C_ATTACK_RECEIPT_V3"
        receipt["candidate_audit_complete"] = bool(receipt.get("candidate_audit_complete"))
        if str(exc) == "STRUCTURAL_INVALID_NO_SELECTIVE_CANDIDATE" and receipt["candidate_audit_complete"]:
            receipt["status"] = "F1C_NO_STRICT_CANDIDATE"
        write_json(step_dir / "attack_receipt.json", receipt)
        return receipt, clean_action, False


def run_arm(parent: Mapping[str, Any], suite_cfg: Mapping[str, Any], probe: Mapping[str, Any], arm: str, model: Any, processor: Any, device: str, protocol: Mapping[str, Any], output: Path) -> dict[str, Any]:
    suite, key = str(parent["suite"]), str(parent["canonical_parent_key"])
    output.mkdir(parents=True, exist_ok=False)
    counters = {"env_reset_calls": 0, "env_step_calls": 0, "model_inference_calls": 0, "attack_invocation_count": 0, "pgd_calls": 0, "attacked_env_steps": 0, "physical_interventions": 0, "vphys_reads": 0, "protected_reads": 0, "eval160_reads": 0, "attack_outcome_reads": 0}
    step_rows: list[dict[str, Any]] = []
    env = None
    try:
        env, obs, instruction, _initial_decoded, replay_counts = reconstruct_to_probe(parent, suite_cfg, model, processor, device, protocol, probe)
        for name, value in replay_counts.items():
            counters[name] += int(value)
        attacker = dev.build_attack("M1", 10, attack_seed(protocol, key, int(probe["step"])), model, processor, device, protocol, temporal_init=arm)
        attacker.reset_temporal_state()
        for attempt in range(int(protocol["execution"]["attempted_steps"])):
            image = dev.normalize_image(obs["agentview_image"])
            prepared = primary.prepare_generation(model, processor, image, instruction, suite, device)
            clean_decoded = primary.decode_tokens(model, prepared["tokens"], suite)
            clean_decoded.update({"generated": prepared["generated"], "inputs": prepared["inputs"], "prompt_len": int(prepared["inputs"]["input_ids"].shape[1]), "raw_hashes": prepared["raw_hashes"], "image": image})
            counters["model_inference_calls"] += 1
            clean_tokens = [int(value) for value in clean_decoded["tokens"]]
            if len(clean_tokens) != 7:
                raise RuntimeError(f"F1C_STEP_TOKEN_COUNT_INVALID:{attempt}:{len(clean_tokens)}")
            attack_receipt, action, attacked = attack_step(parent, probe, arm, attempt, clean_decoded, attacker, model, device, protocol, output)
            counters["attack_invocation_count"] += int(attack_receipt.get("counters", {}).get("attack_invocation_count", 0))
            counters["pgd_calls"] += int(attack_receipt.get("counters", {}).get("pgd_calls", 0))
            counters["attacked_env_steps"] += int(attacked)
            if attacked and attack_receipt.get("status") != "PASS_F1C_STRICT_CANDIDATE":
                raise RuntimeError("F1C_ATTACKED_STEP_STATUS_MISMATCH")
            before_sha = dev.sha256_bytes(image.tobytes())
            obs, _reward, done, _info = env.step(action)
            counters["env_step_calls"] += 1
            step_rows.append({
                "attempt": int(attempt),
                "observation_sha256": before_sha,
                "clean_direct_token_ids": clean_tokens,
                "attack_status": attack_receipt.get("status"),
                "attacked_action_executed": bool(attacked),
                "executed_action_class": "STRICT_VISUAL_OPEN" if attacked else "CLEAN_ACTION",
                "attack_receipt_path": str(output / f"attempt_{int(attempt):02d}" / "attack_receipt.json"),
                "env_step_completed": True,
                "done": bool(done),
            })
            if done:
                break
        status = "PASS_F1C_ARM_COMPLETED"
        receipt = {
            "schema": "STAGE_X1R2_F1C_TEMPORAL_ARM_RECEIPT_V3",
            "status": status,
            "suite": suite,
            "fixture_id": parent["fixture_id"],
            "canonical_parent_key": key,
            "temporal_init": arm,
            "probe": dict(probe),
            "attempted_step_budget": int(protocol["execution"]["attempted_steps"]),
            "attempted_step_count": len(step_rows),
            "step_rows": step_rows,
            "counters": counters,
            "student_used": False,
            "student_emit_used": False,
            "protected_boundary": protected_boundary(),
        }
        write_json(output / "arm_receipt.json", receipt)
        return receipt
    except Exception as exc:
        receipt = {
            "schema": "STAGE_X1R2_F1C_TEMPORAL_ARM_RECEIPT_V3",
            "status": "HOLD_F1C_RUNTIME",
            "suite": suite,
            "fixture_id": parent["fixture_id"],
            "canonical_parent_key": key,
            "temporal_init": arm,
            "probe": dict(probe),
            "attempted_step_budget": int(protocol["execution"]["attempted_steps"]),
            "attempted_step_count": len(step_rows),
            "step_rows": step_rows,
            "error": f"{type(exc).__name__}:{exc}",
            "counters": counters,
            "student_used": False,
            "student_emit_used": False,
            "protected_boundary": protected_boundary(),
        }
        write_json(output / "arm_receipt.json", receipt)
        return receipt
    finally:
        if env is not None:
            env.close()


def run_parent(parent: Mapping[str, Any], suite_cfg: Mapping[str, Any], model: Any, processor: Any, device: str, protocol: Mapping[str, Any], root: Path) -> dict[str, Any]:
    output = root / str(parent["suite"]) / safe_name(str(parent["canonical_parent_key"]))
    output.mkdir(parents=True, exist_ok=False)
    clean_receipt, probe = clean_probe_selection(parent, suite_cfg, model, processor, device, protocol, output)
    arms: list[dict[str, Any]] = []
    if probe is not None and clean_receipt.get("status") == "PASS_F1C_CLEAN_PROBE":
        for arm in ARMS:
            arms.append(run_arm(parent, suite_cfg, probe, arm, model, processor, device, protocol, output / f"temporal_{arm}"))
    status = "PASS_F1C_PARENT_COMPLETED" if clean_receipt.get("status") == "PASS_F1C_CLEAN_PROBE" and len(arms) == 2 and all(row.get("status") == "PASS_F1C_ARM_COMPLETED" for row in arms) else "HOLD_F1C_PARENT"
    receipt = {
        "schema": "STAGE_X1R2_F1C_PARENT_RECEIPT_V3",
        "status": status,
        "suite": parent["suite"],
        "fixture_id": parent["fixture_id"],
        "canonical_parent_key": parent["canonical_parent_key"],
        "clean_probe": clean_receipt,
        "temporal_arm_count": len(arms),
        "temporal_arms": [{"temporal_init": row.get("temporal_init"), "status": row.get("status"), "path": str(output / f"temporal_{row.get('temporal_init')}" / "arm_receipt.json")} for row in arms],
        "student_used": False,
        "student_emit_used": False,
        "protected_boundary": protected_boundary(),
    }
    write_json(output / "parent_receipt.json", receipt)
    return receipt


def run_worker(protocol_path: Path, physical_gpu: int, worker_index: int, worker_count: int) -> int:
    protocol = load_json(protocol_path)
    source, rows = validate_f1c_freeze(protocol, physical_gpu)
    if worker_count < 1 or worker_count > int(protocol["resource"]["max_project_workers"]) or worker_index < 0 or worker_index >= worker_count:
        raise SystemExit("F1C_WORKER_ASSIGNMENT_INVALID")
    root = Path(str(protocol["runtime"]["durable_output_root"]))
    root.mkdir(parents=True, exist_ok=True)
    worker_root = root / "workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    assigned = [row for index, row in enumerate(rows) if index % worker_count == worker_index]
    worker_receipt_path = worker_root / f"worker_{worker_index:02d}_receipt.json"
    existing_parents = [
        root / str(row["suite"]) / safe_name(str(row["canonical_parent_key"]))
        for row in assigned
        if (root / str(row["suite"]) / safe_name(str(row["canonical_parent_key"]))).exists()
    ]
    if worker_receipt_path.exists() or existing_parents:
        raise SystemExit(f"F1C_OUTPUT_ROOT_NOT_FRESH:{worker_receipt_path}:{existing_parents}")
    worker_receipt: dict[str, Any] = {
        "schema": "STAGE_X1R2_F1C_WORKER_RECEIPT_V3",
        "status": "RUNNING",
        "worker_index": int(worker_index),
        "worker_count": int(worker_count),
        "physical_gpu": int(physical_gpu),
        "assigned_keys": [row["canonical_parent_key"] for row in assigned],
        "source": source,
        "gpu_before_model_load": dev.gpu_receipt(physical_gpu),
        "protected_boundary": protected_boundary(),
    }
    write_json(worker_receipt_path, worker_receipt)
    contract = load_json(SUITE_CONTRACT)
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
                raise RuntimeError(f"F1C_ACTION_DIM_INVALID:{suite}:{action_dim}")
            for row in suite_rows:
                parent = parent_from_row(row, rows.index(row))
                try:
                    parent_receipts.append(run_parent(parent, suite_cfg, model, processor, device, protocol, root))
                except Exception as exc:
                    output = root / str(suite) / safe_name(str(row["canonical_parent_key"]))
                    output.mkdir(parents=True, exist_ok=True)
                    failure = {"schema": "STAGE_X1R2_F1C_PARENT_RECEIPT_V3", "status": "HOLD_F1C_RUNTIME", "suite": suite, "canonical_parent_key": row["canonical_parent_key"], "error": f"{type(exc).__name__}:{exc}", "source": source, "protected_boundary": protected_boundary()}
                    write_json(output / "parent_receipt.json", failure)
                    parent_receipts.append(failure)
            del model, processor
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        worker_receipt.update({"status": "PASS_F1C_WORKER_COMPLETED", "parent_receipts": [{"suite": row.get("suite"), "canonical_parent_key": row.get("canonical_parent_key"), "status": row.get("status")} for row in parent_receipts], "elapsed_seconds": time.time() - start, "gpu_after": dev.gpu_receipt(physical_gpu)})
        write_json(worker_receipt_path, worker_receipt)
        print(json.dumps({"status": worker_receipt["status"], "worker_index": worker_index, "parents": len(parent_receipts)}, sort_keys=True))
        return 0
    except Exception as exc:
        worker_receipt.update({"status": "HOLD_F1C_RUNTIME", "error": f"{type(exc).__name__}:{exc}", "elapsed_seconds": time.time() - start})
        write_json(worker_receipt_path, worker_receipt)
        print(json.dumps({"status": worker_receipt["status"], "worker_index": worker_index, "error": worker_receipt["error"]}, sort_keys=True))
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    args = parser.parse_args()
    return run_worker(args.protocol.resolve(), args.physical_gpu, args.worker_index, args.worker_count)


if __name__ == "__main__":
    raise SystemExit(main())
