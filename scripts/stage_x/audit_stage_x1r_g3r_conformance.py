"""CPU/static G3R-P0 conformance gate; never loads models or environments."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/STAGE_X_X1R_PRIMARY_MATRIX_PROTOCOL_V1.json"
COHORT = ROOT / "reports/STAGE_X_X1R_T1D1M1_FINAL_ATTACK_COHORT_V1.json"
RUNNER = ROOT / "scripts/stage_x/run_stage_x1r_primary_matrix.py"
OWNER_BINDING = ROOT / "reports/STAGE_X_X1R_T1D1M1_OWNER_BINDING_RECEIPT_V1.json"
OWNER_CSV = ROOT / "reports/STAGE_X_X1R_T1D1M1_OWNER_LABEL_SUBMISSION_V1.csv"
PREINGESTION = ROOT / "reports/STAGE_X_X1R_PRIMARY_MATRIX_PREINGESTION_AUDIT_V1.json"
G2_AUDIT = ROOT / "reports/STAGE_X_X1R_G2_ATTACK_IMPLEMENTATION_AUDIT_V1.json"
TOKEN_AUDIT = ROOT / "reports/STAGE_X_X1R_TARGET_TOKEN_CPU_SEMANTICS_V1.json"
M001_HOLD = ROOT / "reports/STAGE_X_X1R_PRIMARY_MATRIX_CANARY_RUNTIME_HOLD_V1.json"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT).strip()


def seed(namespace: str, key: str) -> int:
    return int(hashlib.sha256(f"{namespace}|{key}|PRIMARY_EMIT_T5".encode()).hexdigest()[:8], 16)


def main() -> int:
    protocol = load(PROTOCOL)
    cohort = load(COHORT)
    owner = load(OWNER_BINDING)
    preingestion = load(PREINGESTION)
    g2 = load(G2_AUDIT)
    token = load(TOKEN_AUDIT)
    m001 = load(M001_HOLD)
    runner_text = RUNNER.read_text(encoding="utf-8")
    errors: list[str] = []

    if protocol.get("status") != "FROZEN_PRE_LABEL_INGESTION":
        errors.append("PROTOCOL_NOT_FROZEN")
    if protocol.get("arms") != ["CLEAN_EVAL", "TRUE_PGD_T5", "RAND_UNIFORM_T5", "SHUFFLED_GRAD_T5"]:
        errors.append("ARM_SET_MISMATCH")
    timing = protocol.get("timing", {})
    if timing.get("attack_window_offsets") != [0, 1, 2, 3, 4] or timing.get("physical_followup_offsets") != list(range(5, 15)):
        errors.append("TIMING_WINDOW_MISMATCH")
    true_pgd = protocol.get("true_pgd", {})
    for key, expected in {
        "target_token_id": 31745,
        "target_execution_class": "NATIVE_OPEN",
        "epsilon_processor_pixel_values": 0.03,
        "step_size_processor_pixel_values": 0.006,
        "num_steps": 5,
        "iterate_selection": "FINAL_ONLY",
        "temporal_init": "none",
    }.items():
        if true_pgd.get(key) != expected:
            errors.append(f"TRUE_PROTOCOL_MISMATCH:{key}")

    rows = list(cohort.get("rows", []))
    if cohort.get("count") != 7 or len(rows) != 7:
        errors.append("COHORT_COUNT_NOT_SEVEN")
    if len({row.get("canonical_parent_key") for row in rows}) != len(rows):
        errors.append("COHORT_DUPLICATE")
    if owner.get("owner_submission", {}).get("raw_sha256") != "76c835c292c76190b2a764da7be746a568697deea1cb4009d63e306fdc610c2c":
        errors.append("OWNER_BINDING_SHA_MISMATCH")
    if sha256(OWNER_CSV) != "76c835c292c76190b2a764da7be746a568697deea1cb4009d63e306fdc610c2c":
        errors.append("OWNER_CSV_SHA_MISMATCH")
    if sha256(PROTOCOL) != preingestion.get("raw_bindings", {}).get("primary_protocol_sha256"):
        errors.append("PRIMARY_PROTOCOL_MUTATED")
    if g2.get("status") != "PASS_STATIC_ATTACK_IMPLEMENTATION":
        errors.append("G2_STATIC_AUDIT_NOT_PASS")
    if token.get("status") != "PASS_31745_NATIVE_OPEN":
        errors.append("TARGET_TOKEN_AUDIT_NOT_PASS")
    if m001.get("status") != "STAGE_X_X1R_PRIMARY_MATRIX_HOLD_CANARY_RUNTIME_INVALID" or m001.get("canary", {}).get("review_id") != "M001":
        errors.append("M001_HOLD_BINDING_INVALID")

    try:
        tree = ast.parse(runner_text)
        update_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "update_feature"
        ]
        unpacked_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "update_feature"
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Tuple)
            and len(node.targets[0].elts) == 2
        ]
        if len(update_calls) != 1 or len(unpacked_calls) != 1:
            errors.append("UPDATE_FEATURE_TUPLE_CONTRACT")
    except SyntaxError as exc:
        errors.append(f"RUNNER_SYNTAX:{exc}")

    required_strings = {
        "exposure": ("initial_exposure", "mark_policy_action_materialized", "mark_env_step_executed", "policy_action_materialized", "first_env_step_executed", "rows_materialized"),
        "seed": ("primary_seed_values(protocol, key)", "set_seed(seeds[\"eval_seed\"])", "seeds[\"perturb_seed\"]", "arm_order(key, protocol)"),
        "full_episode": ("official_task_success", "terminal_success_step", "official_horizon_reached", "final_policy_steps_executed", "physical_followup_complete", "attack_fully_delivered"),
        "arm_gate": ('if attack_summary.get("attack_executed") and not arm_equal:', "ARM_TOKEN_ISOLATION_FAIL"),
        "durable_telemetry": ("append_telemetry(telemetry_path, row)", "os.fsync(handle.fileno())", "persist_attack_tensor"),
        "route": ("validate_true_pgd_attack_result(result, route)", "TARGET_TOKEN", "TARGET_CLASS", "allow_fallback"),
        "rand_accounting": ('"optimizer_steps": 0', '"temporal_attack_budget_frames": ATTACK_WINDOW'),
    }
    for group, needles in required_strings.items():
        if any(needle not in runner_text for needle in needles):
            errors.append(f"RUNNER_CONFORMANCE_MISSING:{group}")
    for forbidden in ("expected_clean_seed", "student_trace", "load_student", "if clean_emit_verified and step >= expected_emit + ATTACK_WINDOW + H_PHYS - 1:"):
        if forbidden in runner_text:
            errors.append(f"RUNNER_FORBIDDEN_OR_OLD_PATH:{forbidden}")
    if "ExistingDenseAttackAdapter" in runner_text or "Eval160" in runner_text or "V_phys" in runner_text:
        errors.append("RUNNER_FORBIDDEN_PATH")

    seed_rows = []
    contract = protocol["seed_contract"]
    for row in rows:
        key = str(row["canonical_parent_key"])
        rotation = int(hashlib.sha256(f"{contract['arm_order_namespace']}|{key}|PRIMARY_EMIT_T5".encode()).hexdigest()[:2], 16) % 4
        base = list(contract["arm_order_base"])
        seed_rows.append({
            "review_id": row["review_id"],
            "ordinal": row["ordinal"],
            "canonical_parent_key": key,
            "eval_seed": seed(str(contract["eval_seed_namespace"]), key),
            "perturb_seed": seed(str(contract["perturb_seed_namespace"]), key),
            "arm_rotation": rotation,
            "arm_order": base[rotation:] + base[:rotation],
        })
    ledger_digest = hashlib.sha256(json.dumps(seed_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    source = {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"), "status_porcelain": git("status", "--porcelain")}
    if source["status_porcelain"]:
        errors.append("WORKTREE_NOT_CLEAN_BEFORE_G3R_SEAL")

    protected = {
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
    }
    ledger = {
        "schema": "STAGE_X_X1R_G3R_FROZEN_SEED_LEDGER_V1",
        "status": "PASS_FROZEN_SEED_LEDGER" if not errors else "HOLD_FROZEN_SEED_LEDGER",
        "protocol_sha256": sha256(PROTOCOL),
        "source": source,
        "probe_id": "PRIMARY_EMIT_T5",
        "rows": seed_rows,
        "ledger_sha256": ledger_digest,
        "protected_boundary": protected,
    }
    (ROOT / "reports/STAGE_X_X1R_G3R_FROZEN_SEED_LEDGER_V1.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checks = {
        "update_feature_tuple_contract": "UPDATE_FEATURE_TUPLE_CONTRACT" not in errors,
        "policy_action_exposure": all(x in runner_text for x in ("policy_action_materialized", "mark_policy_action_materialized")),
        "env_step_exposure": all(x in runner_text for x in ("first_env_step_executed", "mark_env_step_executed")),
        "seed_contract_exact": not any(error.startswith("RUNNER_FORBIDDEN_OR_OLD_PATH:expected_clean_seed") for error in errors),
        "arm_rotation_order_exact": "arm_order(key, protocol)" in runner_text,
        "full_episode_outcome_semantics": not any(error.startswith("RUNNER_FORBIDDEN_OR_OLD_PATH:if clean_emit_verified") for error in errors),
        "t5_h10_intermediate_window": all(x in runner_text for x in ("physical_followup_complete", "attack_fully_delivered")),
        "arm_isolation_hard_gate": all(x in runner_text for x in ("attack_summary.get(\"attack_executed\") and not arm_equal", "ARM_TOKEN_ISOLATION_FAIL")),
        "rand_no_gradient_accounting": '"optimizer_steps": 0' in runner_text,
        "strict_route_no_fallback": "validate_true_pgd_attack_result(result, route)" in runner_text and "ExistingDenseAttackAdapter" not in runner_text,
        "native_open_authority": token.get("status") == "PASS_31745_NATIVE_OPEN",
        "processor_epsilon_projection": "persist_attack_tensor" in runner_text and "project_and_cast_processor_values" in runner_text,
        "m001_immutable": m001.get("canary", {}).get("retry_authorized") is False and m001.get("canary", {}).get("replacement_authorized") is False,
        "cohort_protocol_owner_hashes": not any(error.endswith("SHA_MISMATCH") or error in {"PRIMARY_PROTOCOL_MUTATED", "COHORT_COUNT_NOT_SEVEN"} for error in errors),
        "protected_boundary": protected["eval160"] == "UNREAD" and protected["protected_evaluation"] == "UNREAD" and all(protected[key] == 0 for key in protected if key not in {"eval160", "protected_evaluation"}),
    }
    report = {
        "schema": "STAGE_X_X1R_G3R_PROTOCOL_RUNTIME_CONFORMANCE_AUDIT_V1",
        "status": "STAGE_X_X1R_G3R_PROTOCOL_RUNTIME_CONFORMANCE_PASS" if not errors else "STAGE_X_X1R_G3R_HOLD_PROTOCOL_RUNTIME_CONFORMANCE",
        "scope": "CPU/static only; no model load, inference, reset, env.step, PGD, intervention, V_phys, Eval160, or protected read",
        "source": source,
        "protocol_sha256": sha256(PROTOCOL),
        "cohort_sha256": sha256(COHORT),
        "owner_submission_sha256": sha256(OWNER_CSV),
        "seed_ledger_sha256": ledger_digest,
        "checks": checks,
        "errors": errors,
        "m001": {"status": m001.get("status"), "hold_report_sha256": sha256(M001_HOLD)},
        "protected_boundary": protected,
        "next_gate": "STAGE_X_X1R_G3R_M002_REPAIR_CANARY_REQUIRED" if not errors else "OWNER_PI_REVIEW_G3R_CONFORMANCE_REQUIRED",
    }
    (ROOT / "reports/STAGE_X_X1R_G3R_PROTOCOL_RUNTIME_CONFORMANCE_AUDIT_V1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": errors, "seed_ledger_sha256": ledger_digest}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
