#!/usr/bin/env python3
"""Seal the three AC2R1 M1 clean-only canary receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "reports/STAGE_AA_AA2R2_ENGINEERING_CANARY_PLAN_V1.json"
PROTOCOL = ROOT / "configs/STAGE_AC_AC2R1_M1_MANIFEST_REQUALIFICATION_PROTOCOL_V1.json"
SOURCE = ROOT / "reports/STAGE_AC_AC2R1_M1_RUNTIME_SOURCE_AUTHORITY_V1.json"
PRE_ROOT = ROOT / "reports/STAGE_AC_AC2R1_PRE_GPU_ROOT_SEAL_V1.json"
M1 = "M1_OPENVLA_OFT"
EXPECTED_KEYS = {
    "libero_10/task_04/state_20",
    "libero_object/task_02/state_42",
    "libero_spatial/task_05/state_34",
}
HORIZONS = {"libero_10": 520, "libero_object": 280, "libero_spatial": 220}
FORBIDDEN = (
    "open_intervention_steps",
    "pgd_calls",
    "attacked_env_steps",
    "aa_v_phys_reads",
    "v_phys_reads",
    "task_success_reads",
    "attack_outcome_reads",
    "eval160_reads",
    "protected_reads",
    "scientific_parent_exposure",
    "aa2_exposure",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: dict[str, Any]) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError(f"AC2R1_APPEND_ONLY_CONFLICT:{path}")
    if not path.exists():
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        temporary.write_bytes(data)
        os.replace(temporary, path)


def repo_artifact(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def external_artifact(path: Path, evidence_root: Path) -> dict[str, Any]:
    return {"path": str(path), "relative_path": path.relative_to(evidence_root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def audit_receipt(path: Path, expected: dict[str, Any], evidence_root: Path) -> dict[str, Any]:
    receipt = load_json(path)
    cell_id = str(expected["cell_id"])
    require(receipt.get("status") == "STAGE_AC_AC2R1_M1_CANARY_PASS", f"AC2R1_RECEIPT_STATUS:{cell_id}")
    for key in ("model_family", "canonical_parent_key", "suite", "task_idx", "state_id", "seed"):
        require(receipt.get(key) == expected.get(key), f"AC2R1_RECEIPT_BINDING:{cell_id}:{key}")
    require(receipt.get("permanent_exclusion") is True and receipt.get("scientific_use") is False, f"AC2R1_EXCLUSION:{cell_id}")
    suite = str(expected["suite"])
    horizon = HORIZONS[suite]
    counters = receipt.get("runtime_counters", {})
    require(counters.get("env_step_calls") == horizon, f"AC2R1_ENV_STEPS:{cell_id}")
    require(counters.get("model_inference_calls") == math.ceil(horizon / 8), f"AC2R1_INFERENCE_CALLS:{cell_id}")
    require(counters.get("dummy_wait_env_step_calls") == 10, f"AC2R1_DUMMY_STEPS:{cell_id}")
    require(counters.get("physical_telemetry_reads") == horizon, f"AC2R1_TELEMETRY:{cell_id}")
    require(all(counters.get(key, 0) == 0 for key in FORBIDDEN), f"AC2R1_FIREWALL:{cell_id}")
    clean = receipt.get("clean_runtime", {})
    require(clean.get("status") == "PASS_AA2R2_ENGINEERING_CLEAN_TRAJECTORY", f"AC2R1_CLEAN_STATUS:{cell_id}")
    require(clean.get("horizon") == horizon and clean.get("steps_captured") == horizon and clean.get("complete_trajectory") is True, f"AC2R1_CLEAN_HORIZON:{cell_id}")
    rows = receipt.get("clean_rows", [])
    audits = receipt.get("action_pair_audit", [])
    require(len(rows) == horizon, f"AC2R1_ROW_COUNT:{cell_id}")
    require(len(audits) == counters["model_inference_calls"] * 8, f"AC2R1_PAIR_COUNT:{cell_id}")
    for row in rows:
        require(len(row.get("raw_action_7d", [])) == 7 and len(row.get("env_action_7d", [])) == 7, f"AC2R1_ACTION_DIM:{cell_id}")
    for audit in audits:
        semantics = audit.get("semantics", {})
        require(semantics.get("accepted") is True, f"AC2R1_ACTION_REJECTED:{cell_id}")
        require(semantics.get("validator_version") == "STAGE_AA_AA2R2_ACTION_SEMANTICS_V2", f"AC2R1_VALIDATOR_VERSION:{cell_id}")
    return {
        "cell_id": cell_id,
        "model_family": M1,
        "canonical_parent_key": expected["canonical_parent_key"],
        "suite": suite,
        "seed": expected["seed"],
        "status": receipt["status"],
        "receipt": external_artifact(path, evidence_root),
        "horizon": horizon,
        "env_step_calls": counters["env_step_calls"],
        "model_inference_calls": counters["model_inference_calls"],
        "dummy_wait_env_step_calls": counters["dummy_wait_env_step_calls"],
        "physical_telemetry_reads": counters["physical_telemetry_reads"],
        "action_pair_audit_count": len(audits),
        "clean_trajectory_digest": clean.get("clean_trajectory_digest"),
        "action_pair_audit_sha256": receipt.get("action_pair_audit_sha256"),
        "scientific_claim": receipt.get("scientific_claim"),
    }


def build(evidence_root: Path) -> dict[str, Any]:
    plan = load_json(PLAN)
    protocol = load_json(PROTOCOL)
    source = load_json(SOURCE)
    pre_root = load_json(PRE_ROOT)
    require(plan.get("status") == "STAGE_AA_AA2R2_ENGINEERING_CANARY_PLAN_FROZEN", "AC2R1_PLAN_NOT_FROZEN")
    canaries = [row for row in plan.get("canaries", []) if row.get("model_family") == M1]
    require({row.get("canonical_parent_key") for row in canaries} == EXPECTED_KEYS and len(canaries) == 3, "AC2R1_CANARY_SET_INVALID")
    require(protocol.get("status") == "STAGE_AC_AC2R1_PRE_GPU_REQUALIFICATION_AUTHORIZED", "AC2R1_PROTOCOL_NOT_AUTHORIZED")
    require(source.get("status") == "STAGE_AC_AC2R1_M1_RUNTIME_SOURCE_AUTHORITY_FROZEN", "AC2R1_SOURCE_NOT_FROZEN")
    require(pre_root.get("status") == "STAGE_AC_AC2R1_PRE_GPU_STATIC_REQUALIFICATION_PASS", "AC2R1_PRE_ROOT_NOT_PASS")
    receipt_paths = sorted((evidence_root / "receipts").glob("*.json"))
    require(len(receipt_paths) == 3, f"AC2R1_RECEIPT_COUNT:{len(receipt_paths)}")
    by_key = {}
    for path in receipt_paths:
        receipt = load_json(path)
        require(receipt.get("model_family") == M1, f"AC2R1_RECEIPT_MODEL:{path.name}")
        by_key[receipt.get("canonical_parent_key")] = path
    require(set(by_key) == EXPECTED_KEYS, "AC2R1_RECEIPT_KEY_SET_INVALID")
    expected_by_key = {row["canonical_parent_key"]: row for row in canaries}
    records = [audit_receipt(by_key[key], expected_by_key[key], evidence_root) for key in sorted(EXPECTED_KEYS)]
    aggregate = {key: sum(int(record[key]) for record in records) for key in ("env_step_calls", "model_inference_calls", "dummy_wait_env_step_calls", "physical_telemetry_reads", "action_pair_audit_count")}
    aggregate.update({key: 0 for key in FORBIDDEN})
    terminal = {
        "schema": "STAGE_AC_AC2R1_M1_CANARY_TERMINAL_V1",
        "status": "STAGE_AC_AC2R1_M1_CANARY_QUALIFICATION_PASS",
        "gate": protocol["gate"],
        "authorization_pi_comment_id": protocol["authorization_pi_comment_id"],
        "claim_boundary": "AC2R1 M1 manifest-byte repair and clean-only permanently-excluded canary qualification; no fresh AC2 parent or treatment exposure.",
        "source_authority": repo_artifact(SOURCE),
        "protocol": repo_artifact(PROTOCOL),
        "pre_gpu_root_seal": repo_artifact(PRE_ROOT),
        "evidence_root": str(evidence_root),
        "requalification": {"expected_cells": 3, "pass_cells": len(records), "records": records},
        "aggregate_runtime_counters": dict(sorted(aggregate.items())),
        "scientific_firewall": {key: aggregate[key] for key in FORBIDDEN},
        "ac2_resume_authorized_by_this_gate": True,
        "next_legal_action": "RESUME_FROZEN_AC2_CLEAN_ONLY_CENSUS",
    }
    index = {
        "schema": "STAGE_AC_AC2R1_M1_CANARY_RECEIPT_INDEX_V1",
        "status": terminal["status"],
        "gate": terminal["gate"],
        "evidence_root": str(evidence_root),
        "receipt_count": len(records),
        "receipts": records,
    }
    terminal_path = ROOT / "reports/STAGE_AC_AC2R1_M1_CANARY_TERMINAL_V1.json"
    index_path = ROOT / "reports/STAGE_AC_AC2R1_M1_CANARY_RECEIPT_INDEX_V1.json"
    root_path = ROOT / "reports/STAGE_AC_AC2R1_M1_CANARY_ROOT_SEAL_V1.json"
    write_json(terminal_path, terminal)
    index["terminal"] = repo_artifact(terminal_path)
    write_json(index_path, index)
    root_payload = {
        "schema": "STAGE_AC_AC2R1_M1_CANARY_ROOT_SEAL_V1",
        "status": terminal["status"],
        "gate": terminal["gate"],
        "authorization_pi_comment_id": terminal["authorization_pi_comment_id"],
        "terminal": repo_artifact(terminal_path),
        "receipt_index": repo_artifact(index_path),
        "source_authority": repo_artifact(SOURCE),
        "protocol": repo_artifact(PROTOCOL),
        "pre_gpu_root_seal": repo_artifact(PRE_ROOT),
        "receipt_count": len(records),
        "receipts": records,
        "aggregate_runtime_counters": terminal["aggregate_runtime_counters"],
        "scientific_firewall": terminal["scientific_firewall"],
        "ac2_resume_authorized_by_this_gate": True,
        "next_legal_action": "RESUME_FROZEN_AC2_CLEAN_ONLY_CENSUS",
    }
    root = {**root_payload, "root_payload_sha256": hashlib.sha256(canonical(root_payload)).hexdigest()}
    write_json(root_path, root)
    sidecar = root_path.with_suffix(".sha256")
    sidecar.write_text(f"{sha256_file(root_path)}  {root_path.name}\n", encoding="ascii")
    return {"status": terminal["status"], "receipt_count": len(records), "root_seal": repo_artifact(root_path), "root_payload_sha256": root["root_payload_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.evidence_root.resolve()), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
