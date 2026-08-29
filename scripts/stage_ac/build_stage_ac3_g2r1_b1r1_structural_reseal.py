#!/usr/bin/env python3
"""Append-only structural reseal after the seed-bound B1R1 action audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TARGET = "AC3-65bcfd948a45dd0be9ac"
UNKNOWN = "UNKNOWN_ACTION_SEMANTICS_AFTER_B1R1"
HORIZON = "TRUE_SIMULATOR_TERMINAL_HORIZON_CENSOR"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def record(path: Path) -> tuple[Any, dict[str, Any]]:
    data = path.read_bytes()
    return json.loads(data.decode("utf-8")), {"path": str(path), "bytes": len(data), "sha256": digest(data)}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def write_new(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    require(not path.exists(), f"AC3_G2R1_B1R1_RESEAL_OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return {"path": str(path), "bytes": len(data), "sha256": digest(data)}


def verify_local_record(path: Path, expected: dict[str, Any], label: str) -> dict[str, Any]:
    actual = record(path)[1]
    for key in ("bytes", "sha256"):
        require(str(actual[key]) == str(expected[key]), f"AC3_G2R1_B1R1_RESEAL:{label}:{key}")
    return actual


def build(args: argparse.Namespace) -> dict[str, Any]:
    g2, g2_record = record(args.g2_index)
    g2_root, g2_root_record = record(args.g2_root)
    adjudication, adjudication_record = record(args.adjudication)
    b0, b0_record = record(args.b0_report)
    b0_root, b0_root_record = record(args.b0_root)
    b1, b1_record = record(args.b1_report)
    b1_root, b1_root_record = record(args.b1_root)

    require(g2["schema"] == "STAGE_AC_AC3_G2_BRANCH_RECEIPT_INDEX_V1", "AC3_G2R1_B1R1_RESEAL_G2_SCHEMA")
    require(g2["status"] == "HOLD_AC3_G2_ENGINEERING_OR_HORIZON", "AC3_G2R1_B1R1_RESEAL_G2_STATUS")
    require(g2["counts"]["manifest_branches"] == 384, "AC3_G2R1_B1R1_RESEAL_MANIFEST_COUNT")
    require(g2["counts"]["pass_branches"] == 372, "AC3_G2R1_B1R1_RESEAL_PASS_COUNT")
    require(g2["counts"]["invalid_or_horizon_censored_branches"] == 12, "AC3_G2R1_B1R1_RESEAL_INVALID_COUNT")
    require(g2_root["status"] == "STAGE_AC_AC3_G2_ENGINEERING_OR_HORIZON_HOLD_STOP_FOR_PI", "AC3_G2R1_B1R1_RESEAL_G2_ROOT_STATUS")

    require(b0["status"] == "STAGE_AC_AC3_G2R1_B0R1_ACTION_ONLY_AUDIT_PASS_CONTINUE", "AC3_G2R1_B1R1_RESEAL_B0_STATUS")
    require(b0["remote_action_only_audit"]["receipt_summary"]["v2_rejected"] == 0, "AC3_G2R1_B1R1_RESEAL_B0_V2_REJECTED")
    require(digest(canonical(b0_root["root_payload"])) == b0_root["root_payload_sha256"], "AC3_G2R1_B1R1_RESEAL_B0_ROOT")

    require(b1["schema"] == "STAGE_AC_AC3_G2R1_B1R1_TARGET_INFERENCE_ONLY_RECEIPT_V1", "AC3_G2R1_B1R1_RESEAL_B1_SCHEMA")
    require(b1["branch_id"] == TARGET, "AC3_G2R1_B1R1_RESEAL_TARGET_ID")
    require(b1["status"] == "STAGE_AC_AC3_G2R1_B1R1_UNKNOWN_ACTION_SEMANTICS_CONTINUE_TO_CENSORING_ANALYSIS", "AC3_G2R1_B1R1_RESEAL_B1_STATUS")
    require(b1["seed_bound_before_inference"] is True, "AC3_G2R1_B1R1_RESEAL_SEED_BINDING")
    require(b1["inference"]["model_inference_calls"] == 1, "AC3_G2R1_B1R1_RESEAL_INFERENCE_COUNT")
    require(b1["scientific_firewall"]["new_env_step_calls"] == 0, "AC3_G2R1_B1R1_RESEAL_ENV_STEPS")
    require(b1["scientific_firewall"]["new_open_intervention_steps"] == 0, "AC3_G2R1_B1R1_RESEAL_OPEN_STEPS")
    require(b1["scientific_firewall"]["new_protected_reads"] == 0, "AC3_G2R1_B1R1_RESEAL_PROTECTED")
    require(b1["reconciliation"]["v1_accepted"] == 8 and b1["reconciliation"]["v2_accepted"] == 8, "AC3_G2R1_B1R1_RESEAL_QUEUE_AUDIT")
    require(b1["reconciliation"]["v1_rejected"] == 0 and b1["reconciliation"]["v2_rejected"] == 0, "AC3_G2R1_B1R1_RESEAL_QUEUE_REJECTED")
    require(digest(canonical(b1_root["root_payload"])) == b1_root["root_payload_sha256"], "AC3_G2R1_B1R1_RESEAL_B1_ROOT")

    branches = adjudication["branches"]
    require(len(branches) == 12, "AC3_G2R1_B1R1_RESEAL_ADJUDICATION_COUNT")
    target_rows = [row for row in branches if row["branch_id"] == TARGET]
    require(len(target_rows) == 1, "AC3_G2R1_B1R1_RESEAL_TARGET_ADJUDICATION")
    require(target_rows[0]["detail"]["classification"] == "ACTION_SEMANTICS_VALIDATOR_FAILURE_UNRESOLVED", "AC3_G2R1_B1R1_RESEAL_TARGET_CLASS")
    horizon_rows = [row for row in branches if row["branch_id"] != TARGET]
    require(len(horizon_rows) == 11, "AC3_G2R1_B1R1_RESEAL_HORIZON_COUNT")
    require(all(row["detail"]["classification"] == HORIZON for row in horizon_rows), "AC3_G2R1_B1R1_RESEAL_HORIZON_CLASS")
    require(all(row["detail"]["physical_outcome_read"] is False for row in branches), "AC3_G2R1_B1R1_RESEAL_OUTCOME_FIREWALL")

    target = target_rows[0]
    target_update = {
        "branch_id": TARGET,
        "prior_classification": target["detail"]["classification"],
        "post_b1r1_classification": UNKNOWN,
        "b1r1_status": b1["status"],
        "physical_recovery_authorized": False,
        "physical_outcome_read": False,
        "reason": "B1R1 V1/V2 action-only replay accepted all 8 queue pairs and did not reproduce the historical validator failure; no physical recovery is authorized.",
    }
    preserved_horizon = [
        {
            "branch_id": row["branch_id"],
            "model_family": row["model_family"],
            "suite": row["suite"],
            "canonical_parent_key": row["canonical_parent_key"],
            "condition": row["condition"],
            "dose": row["dose"],
            "classification": row["detail"]["classification"],
            "recovery_authorized": row["detail"]["recovery_authorized"],
            "physical_outcome_read": row["detail"]["physical_outcome_read"],
        }
        for row in sorted(horizon_rows, key=lambda item: item["branch_id"])
    ]
    status = "STAGE_AC_AC3_G2R1_B1R1_STRUCTURAL_RESEAL_PASS_CONTINUE_TO_G2R1_C"
    next_action = "G2R1_C_CENSORING_AWARE_ANALYSIS"
    payload = {
        "status": status,
        "gate": "STAGE_AC_AC3_G2R1_B1R1_STRUCTURAL_RESEAL_V1",
        "source_authority": {"g2": {"index": g2_record, "root": g2_root_record}, "b0r1": {"report": b0_record, "root": b0_root_record}, "g2r1_a": {"adjudication": adjudication_record}, "b1r1": {"report": b1_record, "root": b1_root_record}},
        "g2": {"index": g2_record, "root": g2_root_record},
        "b0r1": {"report": b0_record, "root": b0_root_record},
        "g2r1_a": {"adjudication": adjudication_record},
        "b1r1": {"report": b1_record, "root": b1_root_record},
        "counts": {"authoritative_branches": 384, "pass_branches": 372, "true_horizon_censors": 11, "target_unknown_action_semantics": 1},
        "target_update": target_update,
        "preserved_true_horizon_censors": preserved_horizon,
        "coverage_preserved": adjudication["model_coverage"],
        "outcome_firewall": {
            "new_inference": 0,
            "new_env_steps": 0,
            "new_open_interventions": 0,
            "physical_outcome_read": False,
            "protected_reads": 0,
        },
        "claim_boundary": "B1R1 structural reclassification only; no physical outcome interpretation and no censoring branch recovery.",
        "next_legal_action": next_action,
    }
    report = {
        "schema": "STAGE_AC_AC3_G2R1_B1R1_STRUCTURAL_RESEAL_V1",
        "status": status,
        "gate": payload["gate"],
        "claim_boundary": payload["claim_boundary"],
        "source_authority": {"g2": payload["g2"], "b0r1": payload["b0r1"], "g2r1_a": payload["g2r1_a"], "b1r1": payload["b1r1"]},
        "counts": payload["counts"],
        "target_update": target_update,
        "preserved_true_horizon_censors": preserved_horizon,
        "coverage_preserved": adjudication["model_coverage"],
        "outcome_firewall": payload["outcome_firewall"],
        "next_legal_action": next_action,
    }
    report_artifact = write_new(args.output_dir / "STAGE_AC_AC3_G2R1_B1R1_STRUCTURAL_RESEAL_V1.json", report)
    root_payload = {"gate": payload["gate"], "status": status, "report": report_artifact, "source_authority": payload["source_authority"], "counts": payload["counts"], "target_update": target_update, "outcome_firewall": payload["outcome_firewall"], "next_legal_action": next_action}
    root = {
        "schema": "STAGE_AC_AC3_G2R1_B1R1_STRUCTURAL_RESEAL_ROOT_V1",
        "status": status,
        "root_payload": root_payload,
        "root_payload_sha256": digest(canonical(root_payload)),
        "artifacts": {"report": report_artifact},
        "claim_boundary": payload["claim_boundary"],
        "next_legal_action": next_action,
    }
    root_artifact = write_new(args.output_dir / "STAGE_AC_AC3_G2R1_B1R1_STRUCTURAL_RESEAL_ROOT_V1.json", root)
    return {"status": status, "counts": payload["counts"], "artifacts": {"report": report_artifact, "root": root_artifact}, "root_payload_sha256": root["root_payload_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--g2-index", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2_BRANCH_RECEIPT_INDEX_V1.json")
    parser.add_argument("--g2-root", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2_ROOT_SEAL_V1.json")
    parser.add_argument("--adjudication", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2R1_A_CENSOR_ADJUDICATION_V1.json")
    parser.add_argument("--b0-report", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2R1_B0R1_ACTION_ONLY_AUDIT_V1.json")
    parser.add_argument("--b0-root", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2R1_B0R1_ROOT_SEAL_V1.json")
    parser.add_argument("--b1-report", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2R1_B1R1_TARGET_INFERENCE_ONLY_V1/AC3-65bcfd948a45dd0be9ac_INFERENCE_ONLY.json")
    parser.add_argument("--b1-root", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2R1_B1R1_TARGET_INFERENCE_ONLY_V1/STAGE_AC_AC3_G2R1_B1R1_ROOT_SEAL_V1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2R1_B1R1_STRUCTURAL_RESEAL_V1")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
