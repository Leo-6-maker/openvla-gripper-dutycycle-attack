#!/usr/bin/env python3
"""Seal the AA2R2 Phase-A engineering-only canary requalification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/STAGE_AA_AA2R2_ACTION_SEMANTICS_AMENDMENT_V1.json"
SOURCE = ROOT / "reports/STAGE_AA_AA2R2_RUNTIME_SOURCE_AUTHORITY_V2.json"
PLAN = ROOT / "reports/STAGE_AA_AA2R2_ENGINEERING_CANARY_PLAN_V1.json"
STATIC_REPORT = ROOT / "reports/STAGE_AA_AA2R2_STATIC_SEMANTICS_RECONCILIATION_V2.json"
RECEIPTS = ROOT / "reports/server_evidence/STAGE_AA_AA2R2/phase_a_retry_v2/receipts"
LOGS = ROOT / "reports/server_evidence/STAGE_AA_AA2R2/phase_a_retry_v2/logs"

HORIZONS = {"libero_10": 520, "libero_object": 280, "libero_spatial": 220}
QUEUE_LENGTHS = {"M0_OPENVLA": 1, "M1_OPENVLA_OFT": 8, "M2_PI05_LIBERO": 5}
FORBIDDEN = (
    "open_intervention_steps",
    "attacked_env_steps",
    "pgd_calls",
    "aa_v_phys_reads",
    "stage_z_v_phys_reads",
    "v_phys_reads",
    "attack_outcome_reads",
    "task_success_reads",
    "eval160_reads",
    "protected_reads",
    "scientific_parent_exposure",
    "aa2_exposure",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def receipt_name(cell_id: str) -> str:
    return f"{cell_id.replace('-A-', '-R2-')}.json"


def log_name(cell_id: str) -> str:
    return receipt_name(cell_id).removesuffix(".json") + ".log"


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def audit_cell(cell: dict[str, Any]) -> dict[str, Any]:
    path = RECEIPTS / receipt_name(str(cell["cell_id"]))
    require(path.is_file(), f"AA2R2_RECEIPT_MISSING:{cell['cell_id']}")
    receipt = read_json(path)
    require(receipt.get("status") == "PASS_AA2R2_ENGINEERING_CANARY_CELL", f"AA2R2_RECEIPT_STATUS:{cell['cell_id']}")
    for key in ("model_family", "canonical_parent_key", "suite", "task_idx", "state_id", "seed"):
        require(receipt.get(key) == cell.get(key), f"AA2R2_RECEIPT_BINDING:{cell['cell_id']}:{key}")
    require(receipt.get("canary_permanent_exclusion") is True, f"AA2R2_EXCLUSION:{cell['cell_id']}")
    require(receipt.get("scientific_use") is False, f"AA2R2_SCIENTIFIC_USE:{cell['cell_id']}")

    suite = str(cell["suite"])
    family = str(cell["model_family"])
    horizon = HORIZONS[suite]
    queue_length = QUEUE_LENGTHS[family]
    expected_inference = math.ceil(horizon / queue_length)
    counters = receipt.get("runtime_counters", {})
    require(counters.get("env_step_calls") == horizon, f"AA2R2_ENV_STEPS:{cell['cell_id']}")
    require(counters.get("model_inference_calls") == expected_inference, f"AA2R2_INFERENCE_CALLS:{cell['cell_id']}")
    require(counters.get("dummy_wait_env_step_calls") == 10, f"AA2R2_DUMMY_STEPS:{cell['cell_id']}")
    require(counters.get("physical_telemetry_reads") == horizon, f"AA2R2_TELEMETRY:{cell['cell_id']}")
    require(all(counters.get(key, 0) == 0 for key in FORBIDDEN), f"AA2R2_FIREWALL:{cell['cell_id']}")

    clean = receipt.get("clean_runtime", {})
    require(clean.get("status") == "PASS_AA2R2_ENGINEERING_CLEAN_TRAJECTORY", f"AA2R2_CLEAN_STATUS:{cell['cell_id']}")
    require(clean.get("horizon") == horizon and clean.get("steps_captured") == horizon, f"AA2R2_CLEAN_HORIZON:{cell['cell_id']}")
    require(clean.get("complete_trajectory") is True, f"AA2R2_CLEAN_INCOMPLETE:{cell['cell_id']}")
    require(clean.get("boundary_count") == expected_inference, f"AA2R2_BOUNDARY_COUNT:{cell['cell_id']}")

    rows = receipt.get("clean_rows", [])
    audits = receipt.get("action_pair_audit", [])
    require(len(rows) == horizon, f"AA2R2_ROW_COUNT:{cell['cell_id']}")
    require(len(audits) == expected_inference * queue_length, f"AA2R2_PAIR_COUNT:{cell['cell_id']}")
    for row in rows:
        require(len(row.get("raw_action_7d", [])) == 7 and len(row.get("env_action_7d", [])) == 7, f"AA2R2_ACTION_DIM:{cell['cell_id']}")
    for audit in audits:
        semantics = audit.get("semantics", {})
        require(semantics.get("accepted") is True, f"AA2R2_ACTION_REJECTED:{cell['cell_id']}")
        require(semantics.get("validator_version") == "STAGE_AA_AA2R2_ACTION_SEMANTICS_V2", f"AA2R2_VALIDATOR_VERSION:{cell['cell_id']}")

    log = LOGS / log_name(str(cell["cell_id"]))
    record = {
        "cell_id": cell["cell_id"],
        "model_family": family,
        "canonical_parent_key": cell["canonical_parent_key"],
        "suite": suite,
        "seed": cell["seed"],
        "status": receipt["status"],
        "receipt": artifact(path),
        "log": artifact(log) if log.is_file() else None,
        "horizon": horizon,
        "env_step_calls": counters["env_step_calls"],
        "model_inference_calls": counters["model_inference_calls"],
        "dummy_wait_env_step_calls": counters["dummy_wait_env_step_calls"],
        "physical_telemetry_reads": counters["physical_telemetry_reads"],
        "action_pair_audit_count": len(audits),
        "boundary_count": clean["boundary_count"],
        "terminal_before_count": sum(bool(row.get("terminal_before")) for row in rows),
        "terminal_after_count": sum(bool(row.get("terminal_after")) for row in rows),
        "clean_trajectory_digest": clean.get("clean_trajectory_digest"),
        "action_pair_audit_sha256": receipt.get("action_pair_audit_sha256"),
        "scientific_claim": receipt.get("scientific_claim"),
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    protocol = read_json(PROTOCOL)
    source = read_json(SOURCE)
    plan = read_json(PLAN)
    require(protocol.get("status") == "STAGE_AA_AA2R2_ACTION_SEMANTICS_AMENDMENT_AUTHORIZED", "AA2R2_PROTOCOL")
    require(source.get("status") == "STAGE_AA_AA2R2_RUNTIME_SOURCE_AUTHORITY_FROZEN", "AA2R2_SOURCE")
    require(plan.get("cell_count") == 9 and len(plan.get("canaries", [])) == 9, "AA2R2_PLAN_COUNT")
    require(STATIC_REPORT.is_file(), "AA2R2_STATIC_REPORT_MISSING")

    records = [audit_cell(cell) for cell in plan["canaries"]]
    aggregate = {key: sum(int(record.get(key, 0)) for record in records) for key in ("env_step_calls", "model_inference_calls", "dummy_wait_env_step_calls", "physical_telemetry_reads", "action_pair_audit_count")}
    aggregate.update({key: 0 for key in FORBIDDEN})
    terminal = {
        "schema": "STAGE_AA_AA2R2_PHASE_A_TERMINAL_V2",
        "status": "STAGE_AA_AA2R2_THREE_STATE_SEMANTICS_ENGINEERING_REQUALIFIED_STOP_FOR_PI",
        "gate": protocol["gate"],
        "authorization_pi_comment_id": protocol["authorization"]["pi_comment_id"],
        "claim_boundary": "Phase-A engineering-only action-semantics and clean-runtime qualification; no AA2 scientific eligibility, treatment, endpoint, denominator, or promotion claim.",
        "source_authority": artifact(SOURCE),
        "protocol": artifact(PROTOCOL),
        "canary_plan": artifact(PLAN),
        "static_semantics_report": artifact(STATIC_REPORT),
        "requalification": {"expected_cells": 9, "pass_cells": len(records), "by_model": {family: sum(record["model_family"] == family for record in records) for family in QUEUE_LENGTHS}, "records": records},
        "aggregate_runtime_counters": dict(sorted(aggregate.items())),
        "scientific_firewall": {key: aggregate[key] for key in FORBIDDEN},
        "phase_b_authorized": False,
        "aa3_authorized": False,
        "next_legal_action": "STOP_FOR_PI",
    }
    index = {
        "schema": "STAGE_AA_AA2R2_ENGINEERING_CANARY_RECEIPT_INDEX_V2",
        "status": terminal["status"],
        "gate": terminal["gate"],
        "source_authority": artifact(SOURCE),
        "receipt_count": len(records),
        "receipts": records,
        "terminal_sha256_after_write": None,
    }
    terminal_path = ROOT / "reports/STAGE_AA_AA2R2_PHASE_A_TERMINAL_V2.json"
    index_path = ROOT / "reports/STAGE_AA_AA2R2_ENGINEERING_CANARY_RECEIPT_INDEX_V2.json"
    root_path = ROOT / "reports/STAGE_AA_AA2R2_PHASE_A_ROOT_SEAL_V2.json"
    if args.write:
        write_json(terminal_path, terminal)
        index["terminal_sha256_after_write"] = sha256_file(terminal_path)
        write_json(index_path, index)
        root = {
            "schema": "STAGE_AA_AA2R2_PHASE_A_ROOT_SEAL_V2",
            "status": terminal["status"],
            "gate": terminal["gate"],
            "authorization_pi_comment_id": terminal["authorization_pi_comment_id"],
            "terminal": artifact(terminal_path),
            "receipt_index": artifact(index_path),
            "source_authority": artifact(SOURCE),
            "protocol": artifact(PROTOCOL),
            "canary_plan": artifact(PLAN),
            "static_semantics_report": artifact(STATIC_REPORT),
            "receipt_count": len(records),
            "receipts": records,
            "aggregate_runtime_counters": terminal["aggregate_runtime_counters"],
            "scientific_firewall": terminal["scientific_firewall"],
            "phase_b_authorized": False,
            "aa3_authorized": False,
            "next_legal_action": "STOP_FOR_PI",
        }
        root["root_payload_sha256"] = hashlib.sha256(canonical(root)).hexdigest()
        write_json(root_path, root)
        root_path.with_suffix(".sha256").write_text(f"{sha256_file(root_path)}  {root_path.name}\n", encoding="utf-8")
    print(json.dumps({"status": terminal["status"], "receipt_count": len(records), "write": args.write}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
