"""Audit T1-D0 without importing a model, touching a GPU, or stepping an env."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BASE_COMMIT = "c6a4c5a9e7d63121a75814b3071c9047e1d9e0d0"
EXPECTED_BASE_TREE = "aa22ae95ed760a32cde01729c47b40b3331f668a"
EXPECTED_SALT = "STAGE_X_X1R_T1D0_PARENT_AUTHORITY_V1_20260818"
EXPECTED_COUNTS = {
    "libero_10": [6, 2, 5, 3, 7, 7, 6, 5, 3, 5],
    "libero_goal": [2, 0, 2, 2, 3, 2, 1, 5, 2, 2],
    "libero_object": [9, 4, 10, 5, 5, 8, 8, 5, 6, 10],
    "libero_spatial": [3, 4, 12, 6, 8, 8, 5, 8, 7, 9],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(salt: str, key: str) -> str:
    return hashlib.sha256(f"{salt}::{key}".encode()).hexdigest()


def audit(protocol: dict[str, Any], stage_ix: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    load = protocol["attack_load"]
    stage_load = stage_ix["optimization"]
    expected_load = {
        "epsilon": 0.1,
        "step_size": 0.02,
        "num_steps": 20,
        "cw_margin": 5.0,
        "random_start": False,
        "temporal_init": "prev_delta",
        "temporal_smoothing_lambda": 0.0,
    }
    if stage_ix.get("status") != "FROZEN_BEFORE_F0_EXECUTION":
        errors.append("STAGE_IX_SOURCE_NOT_FROZEN")
    for key, expected in expected_load.items():
        if stage_load.get(key) != expected or load.get(key) != expected:
            errors.append(f"LOAD_MISMATCH:{key}")
    if load.get("master_dtype") != "fp32" or load.get("iterate_selection") != "FINAL_ONLY":
        errors.append("NUMERICAL_DTYPE_OR_ITERATE_RULE_MISMATCH")
    if protocol["target_contract"].get("global_open_token_id_assumption") is not False:
        errors.append("GLOBAL_OPEN_TOKEN_ASSUMPTION_NOT_FALSE")
    if protocol["target_contract"].get("native_token_authority") != "STAGE_X_X1R_T1_NATIVE_ACTION_TOKEN_AUTHORITY_V2":
        errors.append("NATIVE_TOKEN_AUTHORITY_NOT_BOUND")

    parent = protocol["parent_authority"]
    selected = parent["selected_parent_keys"]
    if len(selected) != 39 or len(set(selected)) != len(selected):
        errors.append("EXPECTED_39_UNIQUE_STATIC_CANDIDATES_NOT_PRESENT")
    if parent.get("selection_salt") != EXPECTED_SALT:
        errors.append("PARENT_SELECTION_SALT_MISMATCH")
    if parent.get("missing_task_slots") != ["libero_goal/task_01"]:
        errors.append("MISSING_TASK_SLOT_MISMATCH")
    if parent.get("candidate_counts_by_task") != EXPECTED_COUNTS:
        errors.append("CANDIDATE_COUNT_MATRIX_MISMATCH")
    for key in selected:
        if rank(EXPECTED_SALT, key) != rank(parent["selection_salt"], key):
            errors.append("RANK_RECOMPUTATION_FAILED")
            break
    if parent.get("parent_manifest_ready") is not False:
        errors.append("INCOMPLETE_PARENT_MANIFEST_MARKED_READY")
    for flag in ("selection_outcomes_read", "clean_success_read", "emit_read", "vphys_read", "attack_outcome_read"):
        if parent.get(flag) is not False:
            errors.append(f"OUTCOME_FIREWALL_VIOLATION:{flag}")

    timing = protocol["timing_anchor"]
    if timing.get("status") != "STAGE_X_X1R_T1D0_HOLD_TIMING_ANCHOR_AUTHORITY":
        errors.append("TIMING_ANCHOR_HOLD_NOT_DECLARED")
    if timing.get("observed_fields", {}).get("explicit_attack_start_field_present") is not False:
        errors.append("TIMING_ANCHOR_FALSELY_DECLARED_EXPLICIT")
    if protocol["authorization"].get("pgd_authorized") is not False:
        errors.append("PGD_AUTHORIZATION_NOT_FALSE")
    if protocol["authorization"].get("env_step_authorized") is not False:
        errors.append("ENV_STEP_AUTHORIZATION_NOT_FALSE")
    if protocol["authorization"].get("physical_intervention_authorized") is not False:
        errors.append("PHYSICAL_AUTHORIZATION_NOT_FALSE")
    if protocol["authorization"].get("attack_outcome_authorized") is not False:
        errors.append("OUTCOME_AUTHORIZATION_NOT_FALSE")
    if protocol["source_binding"].get("base_commit") != EXPECTED_BASE_COMMIT:
        errors.append("BASE_COMMIT_MISMATCH")
    if protocol["source_binding"].get("base_tree") != EXPECTED_BASE_TREE:
        errors.append("BASE_TREE_MISMATCH")
    counters = protocol["protected_boundary"]["counters"]
    if any(value != 0 for value in counters.values()):
        errors.append("PROTECTED_COUNTER_NONZERO")

    return {
        "schema": "STAGE_X_X1R_T1D0_STATIC_AUTHORITY_AUDIT_V1",
        "status": protocol["status"] if not errors else "HOLD_T1D0_STATIC_AUDIT_INVALID",
        "static_load_gate": "PASS",
        "native_token_gate": "PASS",
        "timing_anchor_gate": timing["status"],
        "parent_authority_gate": parent["status"],
        "selected_static_candidate_count": len(selected),
        "required_static_candidate_count": 40,
        "missing_task_slots": parent["missing_task_slots"],
        "source_binding": protocol["source_binding"],
        "protected_boundary": protocol["protected_boundary"],
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
        "x1r_pgd_executed": False,
        "env_step_executed": False,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/STAGE_X_X1R_T1D0_ATTACK_PARENT_AUTHORITY_V1.json")
    parser.add_argument("--stage-ix", type=Path, default=ROOT / "configs/STAGE_IX_CANONICAL_PGD_CONTRACT_V1.json")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    stage_ix = json.loads(args.stage_ix.read_text(encoding="utf-8"))
    receipt = audit(protocol, stage_ix)
    receipt["protocol_sha256"] = sha256(args.protocol)
    receipt["stage_ix_sha256"] = sha256(args.stage_ix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "errors": receipt["errors"]}, sort_keys=True))
    return 0 if not receipt["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
