"""Static/CPU-only gate for the prospective Stage-X primary matrix."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/STAGE_X_X1R_PRIMARY_MATRIX_PROTOCOL_V1.json"
COHORT = ROOT / "reports/STAGE_X_X1R_T1D1M1_FINAL_ATTACK_COHORT_V1.json"
RUNNER = ROOT / "scripts/stage_x/run_stage_x1r_primary_matrix.py"
ROUTE = ROOT / "src/gripper_attack/route_contract.py"
CONTROLS = ROOT / "src/gripper_attack/m3_controls.py"
ADAPTER = ROOT / "src/gripper_attack/attack_adapter.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def main() -> int:
    protocol = load(PROTOCOL)
    cohort = load(COHORT)
    route_text = ROUTE.read_text(encoding="utf-8")
    controls_text = CONTROLS.read_text(encoding="utf-8")
    adapter_text = ADAPTER.read_text(encoding="utf-8")
    runner_text = RUNNER.read_text(encoding="utf-8")
    errors: list[str] = []

    if protocol.get("status") != "FROZEN_PRE_LABEL_INGESTION":
        errors.append("PRIMARY_PROTOCOL_NOT_FROZEN")
    if protocol.get("arms") != ["CLEAN_EVAL", "TRUE_PGD_T5", "RAND_UNIFORM_T5", "SHUFFLED_GRAD_T5"]:
        errors.append("ARM_ORDER_OR_SET_MISMATCH")
    timing = protocol.get("timing", {})
    if timing.get("attack_start_step") != "t_emit" or timing.get("attack_window_offsets") != [0, 1, 2, 3, 4] or timing.get("physical_followup_offsets") != list(range(5, 15)):
        errors.append("TIMING_CONTRACT_MISMATCH")
    true_pgd = protocol.get("true_pgd", {})
    for key, expected in {
        "route": "token_prefix_pgd",
        "strict_route": True,
        "allow_fallback": False,
        "objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        "target_token_id": 31745,
        "target_execution_class": "NATIVE_OPEN",
        "epsilon_processor_pixel_values": 0.03,
        "step_size_processor_pixel_values": 0.006,
        "num_steps": 5,
        "random_start": False,
        "temporal_init": "none",
        "iterate_selection": "FINAL_ONLY",
    }.items():
        if true_pgd.get(key) != expected:
            errors.append(f"TRUE_PGD_PROTOCOL_MISMATCH:{key}")
    if protocol.get("arm_isolation", {}).get("primary_invariant") != "adv arm token IDs[0:6] == clean arm token IDs[0:6]":
        errors.append("ARM_ISOLATION_PROTOCOL_MISSING")
    if cohort.get("status") != "FROZEN_PRE_ATTACK_IMPLEMENTATION_AUDIT" or int(cohort.get("count", 0)) <= 0:
        errors.append("FINAL_COHORT_NOT_FROZEN_OR_EMPTY")
    cohort_rows = list(cohort.get("rows", []))
    if len({row.get("canonical_parent_key") for row in cohort_rows}) != len(cohort_rows):
        errors.append("COHORT_PARENT_DUPLICATE")
    if any(row.get("legal_horizon") is not True for row in cohort_rows):
        errors.append("COHORT_LEGAL_HORIZON_INVALID")

    required_route_strings = (
        "EXPECTED_M3_TARGET_TOKEN_ID = 31745",
        'EXPECTED_M3_TARGET_EXECUTION_CLASS = "NATIVE_OPEN"',
        "def validate_true_pgd_attack_result",
        "resolved_adapter_class",
    )
    for needle in required_route_strings:
        if needle not in route_text:
            errors.append(f"ROUTE_CONTRACT_MISSING:{needle}")
    for needle in ("def sample_processor_delta", "def project_and_cast_processor_values", "def shuffled_grad_direction"):
        if needle not in controls_text:
            errors.append(f"CONTROL_PRIMITIVE_MISSING:{needle}")
    for needle in ("class TokenPrefixPGDAttacker", "autoregressive_prefix_gripper_target_token_logratio_arm_v3", "cached_autoregressive_generate_v1", "gradient_transform"):
        if needle not in adapter_text:
            errors.append(f"ATTACK_ADAPTER_MISSING:{needle}")
    for needle in (
        "validate_true_pgd_attack_result(result, route)",
        "sample_processor_delta",
        "gradient_transform",
        "arm_equal = executed_tokens[:6] == clean_tokens[:6]",
        "claim_parent",
        "retry_authorized",
        "official_environment",
    ):
        if needle not in runner_text:
            errors.append(f"RUNNER_GUARD_MISSING:{needle}")
    if "ExistingDenseAttackAdapter" in runner_text or "Eval160" in runner_text or "V_phys" in runner_text:
        errors.append("RUNNER_CONTAINS_FORBIDDEN_FALLBACK_OR_PROTECTED_PATH")
    try:
        ast.parse(runner_text)
    except SyntaxError as exc:
        errors.append(f"RUNNER_SYNTAX_ERROR:{exc}")

    report = {
        "schema": "STAGE_X_X1R_G2_ATTACK_IMPLEMENTATION_AUDIT_V1",
        "status": "PASS_STATIC_ATTACK_IMPLEMENTATION" if not errors else "HOLD_STATIC_ATTACK_IMPLEMENTATION",
        "scope": "CPU/static only; no model inference, env.step, PGD, intervention, V_phys, Eval160, or protected read",
        "protocol": {"path": str(PROTOCOL), "sha256": sha256(PROTOCOL), "status": protocol.get("status")},
        "cohort": {"path": str(COHORT), "sha256": sha256(COHORT), "count": len(cohort_rows), "status": cohort.get("status")},
        "source": {
            "route_contract_sha256": sha256(ROUTE),
            "controls_sha256": sha256(CONTROLS),
            "attack_adapter_sha256": sha256(ADAPTER),
            "runner_sha256": sha256(RUNNER),
        },
        "checks": {
            "strict_token_prefix_route": not any(error.startswith("ROUTE_CONTRACT") for error in errors),
            "controls_share_processor_budget": not any(error.startswith("CONTROL_PRIMITIVE") for error in errors),
            "target_objective_and_cached_surrogate_bound": not any(error.startswith("ATTACK_ADAPTER") for error in errors),
            "runner_claims_atomic_parent": "claim_parent" in runner_text,
            "runner_has_no_fallback": "ExistingDenseAttackAdapter" not in runner_text,
            "runner_has_no_protected_path": "Eval160" not in runner_text and "V_phys" not in runner_text,
            "runner_syntax_valid": not any(error.startswith("RUNNER_SYNTAX") for error in errors),
        },
        "errors": errors,
        "protected_boundary": {
            "eval160": "UNREAD",
            "protected_evaluation": "UNREAD",
            "model_inference": 0,
            "env_step": 0,
            "pgd_calls": 0,
            "physical_interventions": 0,
            "vphys_reads": 0,
            "attack_outcome_reads": 0,
            "eval160_reads": 0,
            "protected_reads": 0,
        },
        "next_gate": "STAGE_X_X1R_G3_RUNTIME_CANARY_REQUIRED" if not errors else "STAGE_X_X1R_G2_STATIC_REPAIR_REQUIRED",
    }
    output = ROOT / "reports/STAGE_X_X1R_G2_ATTACK_IMPLEMENTATION_AUDIT_V1.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": errors, "output": str(output)}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
