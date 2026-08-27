#!/usr/bin/env python3
"""Rebind the frozen Z3 execution rows to protocol V2 without reranking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V1_MANIFEST = ROOT / "reports/STAGE_Z_Z3_EXECUTION_MANIFEST_V1.json"
V1_PROTOCOL = ROOT / "configs/STAGE_Z_Z3_CROSS_MODEL_COMMAND_OPEN_PHYSICAL_MATRIX_PROTOCOL_V1.json"
V2_PROTOCOL = ROOT / "configs/STAGE_Z_Z3_CROSS_MODEL_COMMAND_OPEN_PHYSICAL_MATRIX_PROTOCOL_V2.json"
Z3R1_TERMINAL = ROOT / "reports/STAGE_Z_Z3R1_SENTINEL_RECOVERY_TERMINAL_V1.json"
V2_MANIFEST = ROOT / "reports/STAGE_Z_Z3_EXECUTION_MANIFEST_V2.json"
RECONCILIATION = ROOT / "reports/STAGE_Z_Z3_EXECUTION_MANIFEST_RECONCILIATION_V1.json"

EXECUTION_KEYS = (
    "action_contract",
    "branch_contract",
    "five_arms",
    "forbidden_scope",
    "historical_roots",
    "manual_audit",
    "model_boundaries",
    "physical_contract",
    "population",
    "resource_contract",
    "storage",
    "z1_runtime_authority",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-manifest", type=Path, default=V1_MANIFEST)
    parser.add_argument("--v1-protocol", type=Path, default=V1_PROTOCOL)
    parser.add_argument("--v2-protocol", type=Path, default=V2_PROTOCOL)
    parser.add_argument("--z3r1-terminal", type=Path, default=Z3R1_TERMINAL)
    parser.add_argument("--output-manifest", type=Path, default=V2_MANIFEST)
    parser.add_argument("--output-reconciliation", type=Path, default=RECONCILIATION)
    args = parser.parse_args()

    source = load(args.v1_manifest)
    old_protocol = load(args.v1_protocol)
    new_protocol = load(args.v2_protocol)
    terminal = load(args.z3r1_terminal)
    source_hash = sha(args.v1_manifest)
    old_protocol_hash = sha(args.v1_protocol)
    new_protocol_hash = sha(args.v2_protocol)
    terminal_hash = sha(args.z3r1_terminal)

    require(source["schema"] == "STAGE_Z_Z3_EXECUTION_MANIFEST_V1", "V1_SCHEMA")
    require(source["status"] == "STAGE_Z_Z3_EXECUTION_MANIFEST_FROZEN_NOT_EXECUTED", "V1_STATUS")
    require(source["protocol_sha256"] == old_protocol_hash, "V1_PROTOCOL_BINDING")
    require(old_protocol["status"] == "STAGE_Z_Z3_SOURCE_AUTHORITY_FROZEN", "V1_PROTOCOL_STATUS")
    require(new_protocol["status"] == "STAGE_Z_Z3_SOURCE_AUTHORITY_FROZEN", "V2_PROTOCOL_STATUS")
    require(terminal["status"] == "STAGE_Z_Z3R1_SENTINEL_RECOVERY_PASS_STOP_FOR_PI", "Z3R1_STATUS")
    require(terminal["source_authority"]["protocol_sha256"] == new_protocol_hash, "Z3R1_PROTOCOL_BINDING")
    require(terminal["scientific_matrix_started"] is False, "Z3R1_MATRIX_STARTED")
    require(terminal["scientific_branch_receipts"] == 0, "Z3R1_BRANCH_RECEIPTS")

    for key in EXECUTION_KEYS:
        require(old_protocol.get(key) == new_protocol.get(key), f"PROTOCOL_EXECUTION_DIFF:{key}")

    jobs = source["jobs"]
    require(len(jobs) == 460, "JOB_COUNT")
    require(len({job["branch_id"] for job in jobs}) == 460, "BRANCH_ID_UNIQUENESS")
    parent_keys = [(job["model_family"], job["suite"], job["canonical_parent_key"]) for job in jobs]
    require(len(set(parent_keys)) == 92, "PARENT_COUNT")
    for parent in sorted(set(parent_keys)):
        rows = [job for job in jobs if (job["model_family"], job["suite"], job["canonical_parent_key"]) == parent]
        require(len(rows) == 5, f"ARM_COUNT:{parent}")
        require([row["arm"] for row in rows] == [arm["name"] for arm in new_protocol["five_arms"]], f"ARM_ORDER:{parent}")

    ordered_branch_ids_sha = digest([job["branch_id"] for job in jobs])
    ordered_jobs_sha = digest(jobs)
    selected_fields = ("branch_id", "canonical_parent_key", "anchor_step", "anchor_state_sha256", "anchor_rank_digest", "arm", "duration", "manual_audit_id", "blinded_video_id", "receipt_path", "receipt_sha256")
    binding_sha = digest([{field: job.get(field) for field in selected_fields} for job in jobs])

    target = dict(source)
    target["schema"] = "STAGE_Z_Z3_EXECUTION_MANIFEST_V2"
    target["protocol_sha256"] = new_protocol_hash
    target["protocol_path"] = str(args.v2_protocol.relative_to(ROOT)).replace("\\", "/")
    target["next_legal_action"] = "Z3_C_FIXED_SCIENTIFIC_MATRIX"
    target["reconciliation"] = {
        "status": "STAGE_Z_Z3_EXECUTION_MANIFEST_RECONCILED_TO_V2_PASS",
        "source_manifest_path": str(args.v1_manifest.relative_to(ROOT)).replace("\\", "/"),
        "source_manifest_sha256": source_hash,
        "source_protocol_sha256": old_protocol_hash,
        "target_protocol_sha256": new_protocol_hash,
        "accepted_z3r1_terminal_path": str(args.z3r1_terminal.relative_to(ROOT)).replace("\\", "/"),
        "accepted_z3r1_terminal_sha256": terminal_hash,
        "ordered_branch_ids_sha256": ordered_branch_ids_sha,
        "ordered_job_rows_sha256": ordered_jobs_sha,
        "anchor_and_branch_binding_sha256": binding_sha,
        "exact_job_rows_inherited": True,
        "scientific_rows_changed": False,
        "new_identity": False,
        "rerank": False,
        "top_up": False,
        "anchor_state_or_rank_changed": False,
        "five_arm_schedule_changed": False,
        "manual_selection_changed": False,
        "model_parent_denominator_changed": False,
    }
    target["jobs"] = source["jobs"]

    target_json = json.dumps(target, indent=2, sort_keys=True) + "\n"
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(target_json, encoding="utf-8")
    target_hash = sha(args.output_manifest)

    report = {
        "schema": "STAGE_Z_Z3_EXECUTION_MANIFEST_RECONCILIATION_V1",
        "status": "STAGE_Z_Z3_EXECUTION_MANIFEST_RECONCILED_TO_V2_PASS",
        "source_manifest": {"path": str(args.v1_manifest.relative_to(ROOT)).replace("\\", "/"), "sha256": source_hash, "schema": source["schema"], "protocol_sha256": old_protocol_hash},
        "target_manifest": {"path": str(args.output_manifest.relative_to(ROOT)).replace("\\", "/"), "sha256": target_hash, "schema": target["schema"], "protocol_sha256": new_protocol_hash},
        "source_protocol": {"path": str(args.v1_protocol.relative_to(ROOT)).replace("\\", "/"), "sha256": old_protocol_hash, "schema": old_protocol["schema"]},
        "target_protocol": {"path": str(args.v2_protocol.relative_to(ROOT)).replace("\\", "/"), "sha256": new_protocol_hash, "schema": new_protocol["schema"]},
        "accepted_z3r1_terminal": {"path": str(args.z3r1_terminal.relative_to(ROOT)).replace("\\", "/"), "sha256": terminal_hash, "schema": terminal["schema"], "status": terminal["status"]},
        "equivalence": {
            "eligible_model_parent_pairs": 92,
            "fixed_branch_count": 460,
            "arms_per_pair": 5,
            "ordered_branch_ids_sha256_before": ordered_branch_ids_sha,
            "ordered_branch_ids_sha256_after": digest([job["branch_id"] for job in target["jobs"]]),
            "ordered_job_rows_sha256_before": ordered_jobs_sha,
            "ordered_job_rows_sha256_after": digest(target["jobs"]),
            "anchor_and_branch_binding_sha256_before": binding_sha,
            "anchor_and_branch_binding_sha256_after": digest([{field: job.get(field) for field in selected_fields} for job in target["jobs"]]),
            "job_rows_exactly_equal": target["jobs"] == source["jobs"],
            "eligibility_sha256_equal": target["eligibility_sha256"] == source["eligibility_sha256"],
            "fixed_incomplete_model_parent_pairs_equal": target["fixed_incomplete_model_parent_pairs"] == source["fixed_incomplete_model_parent_pairs"],
            "manual_audit_equal": target["manual_audit"] == source["manual_audit"],
            "execution_protocol_fields_equal": all(old_protocol.get(key) == new_protocol.get(key) for key in EXECUTION_KEYS),
        },
        "allowed_authority_only_changes": ["protocol schema/gate binding", "protocol SHA binding", "approved V2 source revision/source-file binding", "append-only reconciliation metadata", "next legal action"],
        "scientific_execution_started_before_reconciliation": False,
        "next_legal_action": "Z3_C_FIXED_SCIENTIFIC_MATRIX",
    }
    args.output_reconciliation.parent.mkdir(parents=True, exist_ok=True)
    args.output_reconciliation.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    require(report["equivalence"]["job_rows_exactly_equal"], "TARGET_ROWS_CHANGED")
    require(report["equivalence"]["ordered_branch_ids_sha256_before"] == report["equivalence"]["ordered_branch_ids_sha256_after"], "BRANCH_ORDER_CHANGED")
    require(report["equivalence"]["anchor_and_branch_binding_sha256_before"] == report["equivalence"]["anchor_and_branch_binding_sha256_after"], "BINDINGS_CHANGED")
    print(json.dumps({"status": report["status"], "manifest_sha256": target_hash, "reconciliation_sha256": sha(args.output_reconciliation), "jobs": 460}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
