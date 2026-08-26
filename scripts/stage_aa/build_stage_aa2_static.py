#!/usr/bin/env python3
"""Freeze the AA2 clean-only protocol and its complete 324-cell launch list."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
SUITES = ("libero_10", "libero_object", "libero_spatial")
CRITICAL_SALT = "STAGE_AA_AA2_CRITICAL_ANCHOR_V1_20260826"
NONCRITICAL_SALT = "STAGE_AA_AA2_NONCRITICAL_ANCHOR_V1_20260826"
COMMON_SALT = "STAGE_AA_AA2_COMMON_PARENT_SELECTION_V1_20260826"
CLEAN_SEED_SALT = "STAGE_AA_AA2_CLEAN_SCREEN_SEED_V1_20260826"
PI_COMMENT_ID = 5419855172


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def binding(path: str) -> dict[str, Any]:
    file = ROOT / path
    return {"path": path, "bytes": file.stat().st_size, "sha256": digest_file(file)}


def seed_for(family: str, parent_key: str) -> tuple[int, str]:
    digest = digest_bytes(f"{CLEAN_SEED_SALT}|{family}|{parent_key}".encode())
    return int(digest[:15], 16) % (2**31 - 1), digest


def build(*, write: bool) -> dict[str, Any]:
    aa0_path = ROOT / "configs/STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_V1.json"
    capacity_path = ROOT / "reports/STAGE_AA_AA0_FRESH_CAPACITY_INVENTORY_V1.json"
    aa1r1_root_path = ROOT / "reports/STAGE_AA_AA1R1_ROOT_SEAL_V1.json"
    z1_config_path = ROOT / "configs/STAGE_Z_Z1_RUNTIME_PROTOCOL_V11.json"
    runner_path = ROOT / "scripts/stage_aa/run_stage_aa2_clean_screen.py"
    telemetry_path = ROOT / "src/gripper_attack/stage_v_m3_5_physical_taxonomy.py"
    aa0 = load_json(aa0_path)
    capacity = load_json(capacity_path)
    z1_config = load_json(z1_config_path)
    if aa0.get("status") != "STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_FROZEN_STOP_FOR_PI":
        raise RuntimeError("AA0_PROTOCOL_NOT_FROZEN")
    pool = list(capacity["analysis_pool_after_aa1_reservation"]["keys"])
    if len(pool) != 108 or len(set(pool)) != 108:
        raise RuntimeError("AA2_POOL_NOT_EXACTLY_108")
    canaries = {row["canonical_parent_key"] for row in capacity["aa1_engineering_canary_reservation"]["reserved_rows"]}
    if canaries.intersection(pool):
        raise RuntimeError("AA2_CANARY_OVERLAP")
    inventory = {row["canonical_parent_key"]: row for row in capacity["full_fresh_inventory"]}
    if set(pool) - set(inventory):
        raise RuntimeError("AA2_INVENTORY_ROWS_MISSING")
    code_commit = git("rev-parse", "HEAD")
    code_tree = git("rev-parse", "HEAD^{tree}")
    runner_binding = binding("scripts/stage_aa/run_stage_aa2_clean_screen.py")
    runtime_files = [
        binding("scripts/stage_aa/run_stage_aa2_clean_screen.py"),
        binding("scripts/stage_aa/run_stage_aa1_engineering_canary.py"),
        binding("scripts/stage_z/run_stage_z_z1_runtime_canary.py"),
        binding("src/gripper_attack/stage_v_m3_5_physical_taxonomy.py"),
        binding("src/stage_z_preparation/action_semantics.py"),
        binding("configs/STAGE_Z_Z1_RUNTIME_PROTOCOL_V11.json"),
    ]
    protocol = {
        "schema": "STAGE_AA_AA2_CLEAN_SCREEN_PROTOCOL_V1",
        "status": "STAGE_AA_AA2_CLEAN_SCREEN_PROTOCOL_AUTHORIZED_PRE_EXPOSURE",
        "gate": "STAGE_AA_AA2_CLEAN_ONLY_FULL_CENSUS_AND_COMMON_DENOMINATOR_FREEZE_V1",
        "authorization_pi_comment_id": PI_COMMENT_ID,
        "clean_only": True,
        "model_inference_allowed": True,
        "env_step_allowed": True,
        "open_intervention_allowed": False,
        "attack_or_pgd_allowed": False,
        "protected_or_eval160_allowed": False,
        "stage_z_identity_reuse_allowed": False,
        "population": {
            "analysis_pool_count": 108,
            "model_families": list(MODELS),
            "suites": list(SUITES),
            "model_parent_cell_count": 324,
            "full_census_required": True,
            "aa1_canaries_excluded": sorted(canaries),
            "replacement_or_top_up": False,
        },
        "source_bindings": {
            "aa0_protocol": binding("configs/STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_V1.json"),
            "aa0_capacity": binding("reports/STAGE_AA_AA0_FRESH_CAPACITY_INVENTORY_V1.json"),
            "aa1r1_root": binding("reports/STAGE_AA_AA1R1_ROOT_SEAL_V1.json"),
            "runtime_source_authority": {"path": "reports/STAGE_AA_AA2_RUNTIME_SOURCE_AUTHORITY_V1.json"},
            "z1_runtime_protocol": binding("configs/STAGE_Z_Z1_RUNTIME_PROTOCOL_V11.json"),
        },
        "literal_salts": {
            "critical_anchor": CRITICAL_SALT,
            "noncritical_anchor": NONCRITICAL_SALT,
            "common_parent_selection": COMMON_SALT,
            "clean_seed": CLEAN_SEED_SALT,
        },
        "clean_eligibility": {
            "stable_grasp_window_steps": 3,
            "clean_stable_continuation_steps": 20,
            "minimum_remaining_horizon_steps": 20,
            "lift_delta_min_m": 0.015,
            "object_eef_distance_max_m": 0.12,
            "carry_relative_distance_max_m": 0.04,
            "gripper_contact_true_for_window": True,
            "object_support_contact_false_for_window": True,
            "object_identity_and_baseline_valid": True,
            "telemetry_valid_for_eligibility_window": True,
            "complete_trajectory": True,
            "natural_release_in_continuation": "INELIGIBLE_NOT_PHYSICAL_FAILURE",
        },
        "anchor_selection": {
            "critical": "minimum hash rank among eligible clean established-grasp candidates",
            "noncritical": "minimum hash rank among separate pre-contact candidates",
            "outcome_blind": True,
            "noncritical_affects_primary_denominator": False,
            "rank_inputs": "fixed salt|model_family|canonical_parent_key|step",
        },
        "denominator_freeze": {
            "model_specific_sets": ["E_M0", "E_M1", "E_M2"],
            "common_set": "E_M0 intersection E_M1 intersection E_M2",
            "if_common_n_at_least_32": "minimum sha256(COMMON_PARENT_SALT|canonical_parent_key), select exactly 32",
            "if_common_n_24_to_31": "use all common eligible parents",
            "if_common_n_below_24": "STAGE_AA_AA2_CAPACITY_LIMIT_STOP_FOR_PI",
            "selection_inputs": ["canonical_parent_key", "clean eligibility only", COMMON_SALT],
        },
        "technical_retry": {
            "allowed_causes": ["OOM", "allocator", "process crash", "simulator exception", "telemetry corruption", "checkpoint materialization failure"],
            "maximum_retries_per_model_parent_seed": 1,
            "same_model_parent_seed_required": True,
            "scientific_ineligibility_may_not_trigger_retry": True,
        },
        "scientific_firewall": {
            "open_intervention_steps": 0,
            "attacked_env_steps": 0,
            "pgd_calls": 0,
            "aa_v_phys_reads": 0,
            "stage_z_v_phys_reads": 0,
            "task_success_reads": 0,
            "eval160_reads": 0,
            "protected_reads": 0,
            "bridge_f1_reads": 0,
            "paper_promotion": False,
        },
        "next_legal_action": "STOP_FOR_PI_AFTER_FULL_CENSUS",
    }
    protocol_path = ROOT / "configs/STAGE_AA_AA2_CLEAN_SCREEN_PROTOCOL_V1.json"
    if write:
        write_json(protocol_path, protocol)
    protocol_binding = binding("configs/STAGE_AA_AA2_CLEAN_SCREEN_PROTOCOL_V1.json") if write else {"path": protocol_path.as_posix()}
    source_authority = {
        "schema": "STAGE_AA_AA2_RUNTIME_SOURCE_AUTHORITY_V1",
        "status": "STAGE_AA_AA2_RUNTIME_SOURCE_AUTHORITY_FROZEN",
        "gate": protocol["gate"],
        "authorization_pi_comment_id": PI_COMMENT_ID,
        "claim_boundary": "AA2 clean-only eligibility census and denominator diagnostics; no treatment or endpoint result",
        "git_binding": {"repository": "Leo-6-maker/openvla-gripper-dutycycle-attack", "branch": git("branch", "--show-current"), "commit": code_commit, "tree": code_tree},
        "runtime_files": runtime_files,
        "parent_authorities": {
            "aa0_protocol": binding("configs/STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_V1.json"),
            "aa0_capacity": binding("reports/STAGE_AA_AA0_FRESH_CAPACITY_INVENTORY_V1.json"),
            "aa1r1_root": binding("reports/STAGE_AA_AA1R1_ROOT_SEAL_V1.json"),
            "z1_runtime_protocol": binding("configs/STAGE_Z_Z1_RUNTIME_PROTOCOL_V11.json"),
            "aa2_protocol": protocol_binding,
        },
        "server_runtime": {
            "environment": "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800",
            "python": "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python",
            "common_libero_authority": z1_config["environment"]["common_libero_checkout"],
            "gpu_admission": "free_memory_mib > 20480 immediately before every launch",
            "one_worker_per_gpu": True,
            "foreign_process_policy": "record and preserve; never terminate or alter",
        },
        "scientific_firewall": protocol["scientific_firewall"],
        "next_legal_action": "Execute only the 324-cell AA2 clean-only launch manifest",
    }
    source_path = ROOT / "reports/STAGE_AA_AA2_RUNTIME_SOURCE_AUTHORITY_V1.json"
    if write:
        write_json(source_path, source_authority)
    source_binding = binding("reports/STAGE_AA_AA2_RUNTIME_SOURCE_AUTHORITY_V1.json") if write else {"path": source_path.as_posix()}
    cells: list[dict[str, Any]] = []
    for family in MODELS:
        for parent_key in pool:
            row = inventory[parent_key]
            suite, task, state = parent_key.split("/")
            if suite not in SUITES or row["suite"] != suite or row["task"] != task or row["state"] != state:
                raise RuntimeError(f"AA2_PARENT_KEY_ROW_MISMATCH:{parent_key}")
            seed, seed_digest = seed_for(family, parent_key)
            checkpoint = z1_config["model_families"][family]["paths"][suite] if family == "M0_OPENVLA" else (str(Path(z1_config["model_families"][family]["checkpoint_root"]) / suite) if family == "M1_OPENVLA_OFT" else z1_config["model_families"][family]["checkpoint"])
            cell_number = len(cells) + 1
            cells.append(
                {
                    "cell_id": f"AA2-{cell_number:04d}",
                    "cell_index": cell_number,
                    "model_family": family,
                    "canonical_parent_key": parent_key,
                    "suite": suite,
                    "task": task,
                    "source_task_idx": int(row["source_task_idx"]),
                    "state": state,
                    "checkpoint": checkpoint,
                    "checkpoint_manifest_sha256": z1_config["model_families"][family].get("checkpoint_manifests_sha256") if family == "M1_OPENVLA_OFT" else z1_config["model_families"][family].get("checkpoint_manifest_sha256"),
                    "env_seed": 0,
                    "seed": seed,
                    "seed_digest": seed_digest,
                    "eligibility_implementation_sha256": runner_binding["sha256"],
                    "telemetry_schema": "stage_v_m3_5_physical_taxonomy.telemetry_from_env.v1",
                    "telemetry_implementation_sha256": digest_file(telemetry_path),
                    "clean_only_authorization": "AA2_CLEAN_ONLY_NO_OPEN_NO_ATTACK_NO_PROTECTED",
                }
            )
    manifest = {
        "schema": "STAGE_AA_AA2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1",
        "status": "STAGE_AA_AA2_CLEAN_SCREEN_LAUNCH_MANIFEST_FROZEN_PRE_EXPOSURE",
        "gate": protocol["gate"],
        "authorization_pi_comment_id": PI_COMMENT_ID,
        "cell_count": len(cells),
        "full_census_required": True,
        "population": {"parent_count": len(pool), "model_count": len(MODELS), "parent_pool_sha256": digest_bytes(json.dumps(pool, separators=(",", ":")).encode())},
        "git_binding": {"repository": "Leo-6-maker/openvla-gripper-dutycycle-attack", "branch": git("branch", "--show-current"), "commit": code_commit, "tree": code_tree},
        "source_bindings": {
            "aa0_protocol": binding("configs/STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_V1.json"),
            "aa0_capacity": binding("reports/STAGE_AA_AA0_FRESH_CAPACITY_INVENTORY_V1.json"),
            "runtime_source_authority": source_binding,
            "protocol": protocol_binding,
        },
        "cells": cells,
        "pre_exposure_counters": {"model_inference_calls": 0, "env_step_calls": 0, "open_intervention_steps": 0, "pgd_calls": 0, "protected_reads": 0, "aa2_exposure": 0},
        "next_legal_action": "Execute all cells exactly once unless a pre-registered technical retry is required",
    }
    manifest_path = ROOT / "reports/STAGE_AA_AA2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1.json"
    if write:
        write_json(manifest_path, manifest)
    if len(cells) != 324 or len({row["cell_id"] for row in cells}) != 324:
        raise RuntimeError("AA2_CELL_MANIFEST_INVALID")
    return {"protocol": protocol, "source_authority": source_authority, "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = build(write=args.write)
    print(json.dumps({"status": "AA2_STATIC_FREEZE_PASS", "cell_count": result["manifest"]["cell_count"], "write": args.write}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
