#!/usr/bin/env python3
"""Aggregate and seal the Q3R3-D engineering-only five-arm matrix."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/STAGE_X_X1R2_Q3R3_ENGINEERING_MATRIX_PROTOCOL_V1.json"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
ARM_LABELS = (
    "CLEAN_ENGINEERING",
    "TRUE_PGD_T5_ENGINEERING",
    "RAND_UNIFORM_T5_ENGINEERING",
    "SHUFFLED_GRAD_T5_ENGINEERING",
    "TRUE_RANDOM_TIME_T5_ENGINEERING",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT).strip()


def source() -> dict[str, Any]:
    return {"branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"), "status_porcelain": git("status", "--porcelain")}


def fail(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def audit_attack_row(row: Mapping[str, Any], kind: str, errors: list[str]) -> None:
    attack = row.get("attack", {})
    if attack.get("attack_executed") is not True:
        errors.append(f"ATTACK_ROW_NOT_EXECUTED:{kind}:{row.get('step')}")
        return
    if row.get("direct_action_equals_env_step") is not True:
        errors.append(f"DIRECT_ACTION_ROUTE_INVALID:{kind}:{row.get('step')}")
    if row.get("direct_action_audit", {}).get("arm_token_ids_equal") is not True:
        errors.append(f"ARM_TOKEN_ISOLATION_INVALID:{kind}:{row.get('step')}")
    if kind in {"true_pgd", "shuffled_pgd"}:
        route = attack.get("route", {})
        if not isinstance(route, Mapping):
            errors.append(f"TRUE_ROUTE_MISSING:{row.get('step')}")
            return
        if route.get("strict_route") is not True or route.get("allow_fallback") is not False or route.get("fallback_used") is not False:
            errors.append(f"STRICT_ROUTE_INVALID:{row.get('step')}")
        if route.get("target_execution_class") != "NATIVE_OPEN" or int(route.get("target_token_id", -1)) != 31745:
            errors.append(f"TARGET_AUTHORITY_INVALID:{row.get('step')}")
        if float(route.get("pixel_budget_adv_inputs_linf", 1.0)) > 0.03 + 1e-6:
            errors.append(f"PROCESSOR_LINF_INVALID:{row.get('step')}")
        clean = row.get("clean_gripper_semantics", {})
        executed = row.get("executed_gripper_semantics", {})
        if clean.get("execution_class") == "NATIVE_OPEN" or executed.get("execution_class") != "NATIVE_OPEN":
            errors.append(f"GRIPPER_TRANSITION_INVALID:{row.get('step')}")
        if row.get("direct_action_audit", {}).get("gripper_token_changed") is not True:
            errors.append(f"GRIPPER_TOKEN_NOT_CHANGED:{row.get('step')}")
        selected = route.get("selected_candidate_index")
        audits = route.get("arm_isolation_candidate_audit")
        matches = [item for item in audits or [] if item.get("candidate_index") == selected]
        if selected is None or len(matches) != 1 or matches[0].get("clean_gripper_is_native_open") is not False or matches[0].get("gripper_token_changed") is not True or matches[0].get("direct_generated_gripper_is_native_open") is not True:
            errors.append(f"SELECTIVE_CANDIDATE_INVALID:{row.get('step')}")


def audit_arm(path: Path, expected_label: str, expected_kind: str, errors: list[str]) -> dict[str, Any] | None:
    receipt_path = path / "arm_receipt.json"
    telemetry_path = path / "step_telemetry.jsonl"
    if not receipt_path.is_file() or not telemetry_path.is_file():
        errors.append(f"ARM_FILES_MISSING:{expected_label}")
        return None
    receipt = load(receipt_path)
    fail(errors, receipt.get("status") == "PASS_Q3R3_D_ENGINEERING_ARM", f"ARM_STATUS:{expected_label}")
    fail(errors, receipt.get("structural_valid") is True, f"ARM_STRUCTURAL:{expected_label}")
    fail(errors, receipt.get("scientific_use") is False, f"ARM_SCIENTIFIC_USE:{expected_label}")
    fail(errors, receipt.get("arm") == expected_label, f"ARM_LABEL:{expected_label}")
    fail(errors, receipt.get("kind") == expected_kind, f"ARM_KIND:{expected_label}")
    fail(errors, receipt.get("state_audit", {}).get("equal") is True, f"ARM_STATE_AUDIT:{expected_label}")
    protected = receipt.get("protected_boundary", {})
    for key in ("physical_interventions", "vphys_reads", "attack_outcome_reads", "protected_reads", "eval160_reads"):
        fail(errors, int(protected.get(key, -1)) == 0, f"ARM_PROTECTED_COUNTER:{expected_label}:{key}")
    rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fail(errors, len(rows) == 15, f"ARM_ROW_COUNT:{expected_label}:{len(rows)}")
    attack_rows = [row for row in rows if row.get("attack", {}).get("attack_executed") is True]
    expected_attack_rows = 0 if expected_kind == "clean" else 5
    fail(errors, len(attack_rows) == expected_attack_rows, f"ARM_ATTACK_ROW_COUNT:{expected_label}:{len(attack_rows)}")
    for row in rows:
        fail(errors, row.get("direct_action_equals_env_step") is True, f"ARM_DIRECT_ACTION:{expected_label}:{row.get('step')}")
        if row.get("attack", {}).get("attack_executed"):
            audit_attack_row(row, expected_kind, errors)
    fail(errors, int(receipt.get("counters", {}).get("physical_interventions", -1)) == 0, f"ARM_COUNTER_PHYSICAL:{expected_label}")
    fail(errors, int(receipt.get("counters", {}).get("vphys_reads", -1)) == 0, f"ARM_COUNTER_VPHYS:{expected_label}")
    return {"receipt": receipt, "rows": rows, "receipt_sha256": sha256(receipt_path), "telemetry_sha256": sha256(telemetry_path)}


def file_rows(root: Path, excluded: set[str]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in excluded:
            continue
        rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    protocol = load(args.protocol)
    root = args.output_root or Path(str(protocol["resource"]["durable_output_root"]))
    errors: list[str] = []
    fail(errors, protocol.get("status") == "FROZEN_ENGINEERING_ONLY_PRE_GPU", "PROTOCOL_STATUS")
    fail(errors, protocol.get("scientific_authority") is False, "PROTOCOL_SCIENTIFIC_AUTHORITY")
    observed_source = source()
    fail(errors, not observed_source["status_porcelain"], "AUDIT_WORKTREE_DIRTY")
    suite_rows: list[dict[str, Any]] = []
    gpu_ids: list[int] = []
    protected_totals = {"physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "protected_reads": 0, "eval160_reads": 0, "pgd_calls": 0, "attacked_env_steps": 0}
    for fixture in protocol.get("fixtures", []):
        suite = str(fixture["suite"])
        suite_root = root / suite / str(fixture["fixture_id"])
        suite_receipt_path = suite_root / "suite_receipt.json"
        random_path = suite_root / "random_time_materialization.json"
        if not suite_receipt_path.is_file() or not random_path.is_file():
            errors.append(f"SUITE_FILES_MISSING:{suite}")
            continue
        suite_receipt = load(suite_receipt_path)
        fail(errors, suite_receipt.get("status") == "PASS_Q3R3_D_ENGINEERING_SUITE", f"SUITE_STATUS:{suite}")
        fail(errors, suite_receipt.get("structural_valid") is True, f"SUITE_STRUCTURAL:{suite}")
        fail(errors, suite_receipt.get("scientific_use") is False, f"SUITE_SCIENTIFIC_USE:{suite}")
        gpu = suite_receipt.get("gpu_before_model_load", {})
        gpu_ids.append(int(gpu.get("physical_gpu", -1)))
        fail(errors, int(gpu.get("free_memory_mib", 0)) > 20480, f"GPU_FREE_GATE:{suite}")
        random = load(random_path)
        fail(errors, random.get("replay_audit", {}).get("equal") is True, f"RANDOM_STATE_REPLAY:{suite}")
        fail(errors, random.get("observation_sha256") == suite_receipt.get("random_time", {}).get("observation_sha256"), f"RANDOM_OBSERVATION_BINDING:{suite}")
        arms = {}
        kind_by_label = {
            "CLEAN_ENGINEERING": "clean",
            "TRUE_PGD_T5_ENGINEERING": "true_pgd",
            "RAND_UNIFORM_T5_ENGINEERING": "random_uniform",
            "SHUFFLED_GRAD_T5_ENGINEERING": "shuffled_pgd",
            "TRUE_RANDOM_TIME_T5_ENGINEERING": "true_pgd",
        }
        for label, kind in kind_by_label.items():
            arm_audit = audit_arm(suite_root / label, label, kind, errors)
            if arm_audit is not None:
                arms[label] = arm_audit
                rec = arm_audit["receipt"]
                for key in protected_totals:
                    protected_totals[key] += int(rec.get("counters", {}).get(key, 0))
        emit_obs = arms.get("CLEAN_ENGINEERING", {}).get("receipt", {}).get("common_observation_sha256")
        if emit_obs is not None:
            for label in ("TRUE_PGD_T5_ENGINEERING", "RAND_UNIFORM_T5_ENGINEERING", "SHUFFLED_GRAD_T5_ENGINEERING"):
                if label in arms:
                    fail(errors, arms[label]["receipt"].get("common_observation_sha256") == emit_obs, f"COMMON_EMIT_OBSERVATION:{suite}:{label}")
        fail(errors, len(arms) == 5, f"SUITE_ARM_COUNT:{suite}:{len(arms)}")
        suite_rows.append({"suite": suite, "fixture_id": fixture["fixture_id"], "suite_receipt_sha256": sha256(suite_receipt_path), "random_time_sha256": sha256(random_path), "arms": {label: {"receipt_sha256": row["receipt_sha256"], "telemetry_sha256": row["telemetry_sha256"]} for label, row in arms.items()}})
    fail(errors, len(suite_rows) == 4, f"SUITE_COUNT:{len(suite_rows)}")
    fail(errors, len(gpu_ids) == len(set(gpu_ids)), "ONE_WORKER_PER_GPU")
    fail(errors, len(gpu_ids) <= 8, "MAX_WORKERS")
    fail(errors, protected_totals["physical_interventions"] == 0, "AGG_PHYSICAL_INTERVENTIONS")
    fail(errors, protected_totals["vphys_reads"] == 0, "AGG_VPHYS_READS")
    fail(errors, protected_totals["attack_outcome_reads"] == 0, "AGG_ATTACK_OUTCOME_READS")
    fail(errors, protected_totals["protected_reads"] == 0 and protected_totals["eval160_reads"] == 0, "AGG_PROTECTED_READS")
    report_status = "STAGE_X1R2_Q3R3_ENGINEERING_MATRIX_PASS_R0_STATIC_AUDIT" if not errors else "HOLD_Q3R3_D_ENGINEERING_MATRIX"
    root_seal = None
    seal_path = root / "STAGE_X1R2_Q3R3_ENGINEERING_MATRIX_ROOT_SEAL_V1.json"
    audit_path = root / "STAGE_X1R2_Q3R3_ENGINEERING_MATRIX_AUDIT_V1.json"
    if not errors:
        rows = file_rows(root, {seal_path.relative_to(root).as_posix(), audit_path.relative_to(root).as_posix()})
        root_seal = {
            "schema": "STAGE_X1R2_Q3R3_ENGINEERING_MATRIX_ROOT_SEAL_V1",
            "status": report_status,
            "scientific_use": False,
            "protocol_sha256": sha256(args.protocol),
            "source": observed_source,
            "files": rows,
            "file_count": len(rows),
            "protected_boundary": {"physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "protected_reads": 0, "eval160_reads": 0, "eval160": "UNREAD", "protected_evaluation": "UNREAD"},
        }
        write(seal_path, root_seal)
    report = {
        "schema": "STAGE_X1R2_Q3R3_ENGINEERING_MATRIX_AUDIT_V1",
        "status": report_status,
        "scientific_use": False,
        "source": observed_source,
        "protocol_sha256": sha256(args.protocol),
        "suite_rows": suite_rows,
        "gpu_ids": gpu_ids,
        "protected_totals": protected_totals,
        "errors": errors,
        "root_seal": {"path": str(seal_path), "sha256": sha256(seal_path), "file_count": root_seal["file_count"]} if root_seal is not None else None,
        "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "protected_reads": 0, "eval160_reads": 0},
        "next_gate": "STAGE_X1R2_Q3R3_ENGINEERING_MATRIX_PASS_R0_STATIC_AUDIT" if not errors else "OWNER_REVIEW_Q3R3_D_ENGINEERING_HOLD",
    }
    write(audit_path, report)
    print(json.dumps({"status": report_status, "errors": errors, "root_seal_sha256": report["root_seal"]["sha256"] if report["root_seal"] else None}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
