from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/STAGE_X_X1R2_Q3_ARM_REPAIR_PROTOCOL_V1.json"
OUTPUT = ROOT / "reports/STAGE_X_X1R2_Q3_ARM_REPAIR_STATIC_AUDIT_V1.json"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    try:
        value = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        value = path.read_bytes()
    return sha256_bytes(value)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    protocol = read_json(PROTOCOL)
    require(protocol.get("schema") == "STAGE_X_X1R2_Q3_ARM_REPAIR_PROTOCOL_V1", "PROTOCOL_SCHEMA_INVALID")
    require(protocol.get("status") == "FROZEN_ARM_REPAIR_ENGINEERING_PRE_GPU", "PROTOCOL_STATUS_INVALID")
    require(protocol.get("scientific_authority") is False, "SCIENTIFIC_AUTHORITY_MUST_BE_FALSE")

    fixture_path = ROOT / str(protocol["fixture_report"]["path"])
    contract_path = ROOT / str(protocol["selective_attack_contract"]["path"])
    victim_path = ROOT / str(protocol["victim_contract"]["path"])
    fixtures = read_json(fixture_path)
    contract = read_json(contract_path)
    victim = read_json(victim_path)
    require(canonical_sha256(fixture_path) == protocol["fixture_report"]["sha256"], "FIXTURE_REPORT_SHA_MISMATCH")
    require(canonical_sha256(contract_path) == protocol["selective_attack_contract"]["sha256"], "SELECTIVE_CONTRACT_SHA_MISMATCH")
    require(canonical_sha256(victim_path) == protocol["victim_contract"]["sha256"], "VICTIM_CONTRACT_SHA_MISMATCH")
    require(fixtures.get("status") == "FROZEN_REPAIR_ENGINEERING_FIXTURE_PRE_GPU", "FIXTURE_STATUS_INVALID")
    require(fixtures.get("scientific_use") is False, "FIXTURE_SCIENTIFIC_USE_INVALID")
    fixture = fixtures.get("fixture")
    require(isinstance(fixture, dict), "SINGLE_FIXTURE_REQUIRED")
    require(fixture.get("fixture_id") == protocol["fixture_report"]["required_fixture_id"], "FIXTURE_ID_INVALID")
    require(fixture.get("fixture_id") == "Q3-AR-F01", "REPAIR_FIXTURE_ID_INVALID")
    require(fixture.get("permanent_exclusion") is True, "REPAIR_FIXTURE_NOT_PERMANENTLY_EXCLUDED")
    require(fixture.get("prior_attack_exposure") is False, "REPAIR_FIXTURE_PRIOR_EXPOSURE")
    require(fixture.get("protected_identity") is False, "REPAIR_FIXTURE_PROTECTED")

    order = fixtures.get("candidate_order", [])
    require(len(order) == 3, "COMPLETE_REMAINING_CANDIDATE_UNIVERSE_REQUIRED")
    require(order == sorted(order, key=lambda row: str(row["rank_key"])), "CANDIDATE_ORDER_NOT_RANK_SORTED")
    selected = [row for row in order if row.get("selected") is True]
    require(len(selected) == 1 and selected[0]["review_id"] == fixture["review_id"], "SELECTED_FIXTURE_NOT_DETERMINISTIC_MINIMUM")
    require(selected[0]["rank_key"] == order[0]["rank_key"], "SELECTED_FIXTURE_NOT_LOWEST_RANK")
    require(all(row.get("excluded_from_q3_v1") is True for row in order), "CANDIDATE_EXCLUSION_NOT_FROZEN")

    require(contract.get("schema") == "STAGE_X_X1R2_GRIPPER_SELECTIVE_ATTACK_CONTRACT_V1", "SELECTIVE_CONTRACT_SCHEMA_INVALID")
    require(contract.get("scientific_authority") is False, "SELECTIVE_CONTRACT_SCIENTIFIC_AUTHORITY_INVALID")
    required_gates = set(contract["required_gates_before_attacked_env_step"])
    require("arm token IDs exact equality at dimensions 0..5" in required_gates, "ARM_GATE_MISSING")
    require("clean direct-generated gripper execution class != NATIVE_OPEN" in required_gates, "CLEAN_GRIPPER_GATE_MISSING")
    require("adversarial direct-generated gripper execution class = NATIVE_OPEN" in required_gates, "ADV_GRIPPER_GATE_MISSING")
    require(contract["repair_candidate_policy"]["name"] == "STRICT_CANDIDATE_AUDIT_V1", "CANDIDATE_POLICY_INVALID")
    require(protocol.get("arm_isolation_candidate_policy") == "STRICT_CANDIDATE_AUDIT_V1", "PROTOCOL_CANDIDATE_POLICY_INVALID")
    require(victim.get("schema") == "STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1", "VICTIM_SCHEMA_INVALID")

    primary = (ROOT / "scripts/stage_x/run_stage_x1r_primary_matrix.py").read_text(encoding="utf-8")
    adapter = (ROOT / "src/gripper_attack/attack_adapter.py").read_text(encoding="utf-8")
    runner = (ROOT / "scripts/stage_x/run_stage_x1r2_q3_engineering.py").read_text(encoding="utf-8")
    for marker in ("def audit_direct_action_tokens", "arm_isolation_candidate_policy", "failure_context"):
        require(marker in primary, f"PRIMARY_REPAIR_MARKER_MISSING:{marker}")
    for marker in ("STRICT_CANDIDATE_AUDIT_V1", "STRUCTURAL_INVALID_NO_SELECTIVE_CANDIDATE", "def _select_strict_arm_candidate"):
        require(marker in adapter, f"ADAPTER_REPAIR_MARKER_MISSING:{marker}")
    for marker in ("--protocol", "--fixture-report", "arm_isolation_candidate_policy"):
        require(marker in runner, f"Q3_RUNNER_REPAIR_MARKER_MISSING:{marker}")

    audit = {
        "schema": "STAGE_X_X1R2_Q3_ARM_REPAIR_STATIC_AUDIT_V1",
        "status": "PASS_Q3_ARM_REPAIR_STATIC_AUDIT_PRE_GPU",
        "scientific_authority": False,
        "diagnosis": {
            "historical_q3_f01": "IMMUTABLE_RUNTIME_INVALID_ARM_TOKEN_ISOLATION_FAIL",
            "working_hypothesis": "AUTOREGRESSIVE_PREFIX_SPILLOVER_OR_OPTIMIZATION_SPILLOVER",
            "validator": "CANONICAL_DIRECT_GENERATION_ARM_GATE_RETAINED",
            "repair": "OUTCOME_BLIND_STRICT_CANDIDATE_AUDIT_OVER_DELTA0_AND_PGD_TRAJECTORY",
            "diagnosis_closed": False,
        },
        "fixture_report": {"path": str(fixture_path.relative_to(ROOT)).replace("\\", "/"), "sha256": canonical_sha256(fixture_path), "fixture_id": fixture["fixture_id"]},
        "selective_contract": {"path": str(contract_path.relative_to(ROOT)).replace("\\", "/"), "sha256": canonical_sha256(contract_path)},
        "victim_contract": {"path": str(victim_path.relative_to(ROOT)).replace("\\", "/"), "sha256": canonical_sha256(victim_path)},
        "source_checks": {"primary": True, "adapter": True, "runner": True},
        "gpu_or_model_exposure": {"model_inference_calls": 0, "env_step_calls": 0, "pgd_calls": 0, "physical_interventions": 0},
        "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "attack_outcome_reads": 0, "vphys_reads": 0, "protected_reads": 0},
        "next_gate": "STAGE_X1R2_Q3_ARM_REPAIR_REAL_FIXTURE_REVIEW",
    }
    OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "fixture_id": fixture["fixture_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
