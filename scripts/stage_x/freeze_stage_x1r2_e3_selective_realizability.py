#!/usr/bin/env python3
"""Freeze the fresh, outcome-blind E3 selective-realizability population."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SALT = "STAGE_X_X1R2_E3_SELECTIVE_REALIZABILITY_POOL_V1_20260821"
PROBE_SALT = "STAGE_X_X1R2_E3_OUTCOME_BLIND_PROBE_STEP_V1_20260821"
ATTACK_SALT = "STAGE_X_X1R2_E3_TRUE_ATTACK_V1_20260821"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
PER_SUITE = 3

G10 = REPO / "reports/STAGE_X_X1R_T1D0R_G10_IDENTITY_EXCLUSION_LEDGER_V1.json"
Q3R2 = REPO / "reports/STAGE_X_X1R2_Q3R2_ENGINEERING_FIXTURE_POOL_V1.json"
E2 = REPO / "reports/STAGE_X_X1R2_Q3R3_E2_SUCCESSOR_ENGINEERING_POOL_V1.json"
POOL = REPO / "reports/STAGE_X_X1R2_E3_SELECTIVE_REALIZABILITY_POOL_V1.json"
PROTOCOL = REPO / "configs/STAGE_X_X1R2_E3_FACTORIZED_SELECTIVE_REALIZABILITY_PROTOCOL_V1.json"
CONTRACT = REPO / "configs/STAGE_X_X1R2_GRIPPER_SELECTIVE_ATTACK_CONTRACT_V1.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_blob(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "hash-object", str(path)], text=True).strip()


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict) and isinstance(value.get("rows"), list):
        return list(value["rows"])
    if isinstance(value, dict) and isinstance(value.get("selected"), list):
        return list(value["selected"])
    raise ValueError(f"unsupported identity source: {path}")


def rank(key: str) -> str:
    return hashlib.sha256(f"{SALT}|{key}".encode()).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> None:
    g10_rows = load_rows(G10)
    q3r2_rows = load_rows(Q3R2)
    e2_rows = load_rows(E2)
    prior_pool_keys = {
        str(row["canonical_parent_key"])
        for row in [*q3r2_rows, *e2_rows]
    }
    historical_keys = {
        str(row["canonical_parent_key"])
        for row in g10_rows
        if bool(row.get("excluded_union"))
    }
    excluded_keys = historical_keys | prior_pool_keys
    fresh = [row for row in g10_rows if str(row["canonical_parent_key"]) not in excluded_keys]
    selected: list[dict[str, Any]] = []
    for suite in SUITES:
        ordered = sorted(
            (row for row in fresh if str(row["suite"]) == suite),
            key=lambda row: (rank(str(row["canonical_parent_key"])), str(row["canonical_parent_key"])),
        )
        if len(ordered) < PER_SUITE:
            raise SystemExit(f"INSUFFICIENT_E3_IDENTITIES:{suite}:{len(ordered)}")
        for ordinal, row in enumerate(ordered[:PER_SUITE], start=1):
            key = str(row["canonical_parent_key"])
            selected.append({
                "fixture_id": f"E3-{suite.upper()}-{ordinal:02d}",
                "suite": suite,
                "task_idx": int(row["task_idx"]),
                "state_id": int(row["state_id"]),
                "canonical_parent_key": key,
                "rank_sha256": rank(key),
                "source_row": {
                    "excluded_union": bool(row.get("excluded_union")),
                    "fresh_after_exclusion": bool(row.get("fresh_after_exclusion")),
                    "prior_clean_attempt": bool(row.get("prior_clean_attempt")),
                    "prior_exposure": bool(row.get("prior_exposure")),
                    "prior_physical_intervention_named_roots": bool(row.get("prior_physical_intervention_named_roots")),
                },
                "permanent_exclusion": True,
                "scientific_use": False,
                "outcome_read": False,
            })

    if len({row["canonical_parent_key"] for row in selected}) != len(selected):
        raise SystemExit("DUPLICATE_E3_IDENTITY")
    pool = {
        "schema": "STAGE_X1R2_E3_SELECTIVE_REALIZABILITY_POOL_V1",
        "status": "FROZEN_E3_ENGINEERING_ONLY",
        "purpose": "Timing-decoupled, model-side selective-realizability feasibility only; every identity is permanently excluded from future science and protected populations.",
        "selection": {
            "salt": SALT,
            "candidate_order": "sha256(salt|canonical_parent_key), then canonical_parent_key",
            "per_suite_count": PER_SUITE,
            "selected_count": len(selected),
        },
        "candidate_universe": {
            "path": G10.relative_to(REPO).as_posix(),
            "raw_sha256": sha256_file(G10),
            "git_blob_sha256": git_blob(G10),
            "rows": len(g10_rows),
            "historical_excluded_rows": len(historical_keys),
            "prior_q3r2_e2_excluded_keys": len(prior_pool_keys),
            "excluded_union_keys": len(excluded_keys),
            "fresh_after_full_exclusion": len(fresh),
        },
        "exclusion_union": {
            "canonical_parent_keys_sha256": sha256_bytes(canonical_json(sorted(excluded_keys))),
            "canonical_parent_keys": sorted(excluded_keys),
            "sources": [
                {"path": G10.relative_to(REPO).as_posix(), "raw_sha256": sha256_file(G10), "git_blob_sha256": git_blob(G10)},
                {"path": Q3R2.relative_to(REPO).as_posix(), "raw_sha256": sha256_file(Q3R2), "git_blob_sha256": git_blob(Q3R2)},
                {"path": E2.relative_to(REPO).as_posix(), "raw_sha256": sha256_file(E2), "git_blob_sha256": git_blob(E2)},
            ],
            "selection_inputs_forbidden": ["Student", "clean outcome", "emit", "V_phys", "attack result", "manual outcome", "protected data"],
        },
        "selected": selected,
        "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "protected_reads": 0, "vphys_reads": 0, "physical_interventions": 0, "attacked_env_steps": 0},
        "next_gate": "STAGE_X_X1R2_E3_FACTORIZED_SELECTIVE_REALIZABILITY",
    }
    POOL.write_text(json.dumps(pool, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol = {
        "schema": "STAGE_X_X1R2_E3_FACTORIZED_SELECTIVE_REALIZABILITY_PROTOCOL_V1",
        "status": "FROZEN_E3_FACTORIZED_SELECTIVE_REALIZABILITY",
        "date": "2026-08-21",
        "scientific_authority": False,
        "authority_scope": "One bounded timing-decoupled model-side feasibility gate; no physical efficacy, detector, Student, protected, or method tuning authority.",
        "population": {"path": POOL.relative_to(REPO).as_posix(), "raw_sha256": sha256_file(POOL), "git_blob_sha256": git_blob(POOL), "per_suite_count": PER_SUITE, "selected_count": len(selected), "permanent_exclusion": True},
        "exclusion_union": pool["exclusion_union"],
        "runtime": {"official_environment": "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800", "source_receipt": "bind_current_exact_commit_tree_and_runtime_file_blobs_before_first_model_load", "durable_output_root": "/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x1r2_e3_factorized_selective_realizability_20260821"},
        "resource": {"free_memory_mib_strictly_greater_than": 20480, "one_project_worker_per_physical_gpu": True, "max_project_workers": 8, "foreign_processes_untouched": True, "minimum_free_bytes": 4294967296},
        "clean_probe": {"student_emit_used": False, "student_used": False, "outcome_blind": True, "eligible": ["runtime-valid through step", "exactly seven direct generated tokens", "clean gripper is not NATIVE_OPEN", "step+14 < official horizon"], "step_selection_salt": PROBE_SALT, "step_selection_rule": "minimum sha256(step_salt|canonical_parent_key|step) among eligible clean steps", "no_replacement": True},
        "true_execution": {"one_call_per_parent_with_probe": True, "candidate_order": ["delta0", "pgd_iteration_1", "pgd_iteration_2", "pgd_iteration_3", "pgd_iteration_4", "pgd_iteration_5"], "required_candidate_count": 6, "stop_before_attacked_env_step": True, "attack_seed_salt": ATTACK_SALT, "rand": False, "shuffled": False},
        "attack_contract": {"path": CONTRACT.relative_to(REPO).as_posix(), "raw_sha256": sha256_file(CONTRACT), "git_blob_sha256": git_blob(CONTRACT)},
        "frozen_method": {"epsilon": 0.03, "step_size": 0.006, "num_steps": 5, "random_start": False, "objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3", "candidate_policy": "STRICT_CANDIDATE_AUDIT_V1", "target_execution_class": "NATIVE_OPEN", "secondary_target_token_id": 31745, "exact_arm_dimensions": [0, 1, 2, 3, 4, 5], "direct_action_token_count": 7, "strict_route": True, "allow_fallback": False, "no_actuator_overwrite": True, "no_decode_reencode": True},
        "required_candidate_fields": ["candidate_index", "candidate_source", "processor_input_sha256", "delta_sha256", "pixel_budget_adv_inputs_linf", "direct_generated_token_ids", "clean_arm_token_ids", "direct_generated_arm_token_ids", "arm_token_ids_equal", "arm_mismatch_dimensions", "clean_gripper_token_id", "clean_gripper_is_native_open", "direct_generated_gripper_token_id", "direct_generated_gripper_is_native_open", "gripper_token_changed"],
        "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "protected_reads": 0, "vphys_reads": 0, "physical_interventions": 0, "attack_outcome_reads": 0, "attacked_env_steps": 0},
        "next_gate": "E3_AGGREGATE_DECISION_THEN_OWNER_PI_REVIEW_STOP",
    }
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pool": str(POOL), "protocol": str(PROTOCOL), "selected": len(selected), "remaining": len(fresh), "by_suite": {suite: sum(row["suite"] == suite for row in selected) for suite in SUITES}}, sort_keys=True))


if __name__ == "__main__":
    main()
