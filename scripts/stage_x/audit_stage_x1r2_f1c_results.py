#!/usr/bin/env python3
"""Aggregate and seal F1-C canary receipts without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/STAGE_X_X1R2_F1C_METHOD_FREEZE_T5_CANARY_PROTOCOL_V3.json"
F1C_ROOT = ROOT / "reports/STAGE_X_X1R2_F1C_METHOD_FREEZE_T5_CANARY_V3_20260821/F1C_ROOT_SEAL_V3.json"
CANARY = ROOT / "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_C_CANARY_V3_LEDGER_V3.json"
OUT = ROOT / "reports/STAGE_X_X1R2_F1C_T5_CANARY_RESULT_AGGREGATION_V3_20260821"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
ARMS = ("none", "prev_delta")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT).strip()


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def protected_ok(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("eval160") == "UNREAD"
        and value.get("protected_evaluation") == "UNREAD"
        and int(value.get("protected_reads", -1)) == 0
        and int(value.get("vphys_reads", -1)) == 0
        and int(value.get("physical_interventions", -1)) == 0
        and int(value.get("attack_outcome_reads", -1)) == 0
        and int(value.get("attacked_env_steps", -1)) == 0
    )


def audit_attack(receipt: dict[str, Any], step_row: dict[str, Any], errors: list[str], label: str) -> bool:
    status = receipt.get("status")
    if receipt.get("student_used") is not False or receipt.get("student_emit_used") is not False or not protected_ok(receipt.get("protected_boundary")):
        errors.append(f"BOUNDARY_INVALID:{label}")
    attacked = bool(step_row.get("attacked_action_executed"))
    if status == "PASS_F1C_STRICT_CANDIDATE":
        audit = receipt.get("candidate_audit")
        direct = receipt.get("direct_action_audit") or {}
        route = receipt.get("attack_route") or {}
        valid = bool(
            attacked
            and receipt.get("candidate_audit_complete") is True
            and isinstance(audit, list) and len(audit) == 11
            and route.get("strict_route") is True
            and route.get("allow_fallback") is False
            and route.get("fallback_used") is False
            and direct.get("arm_token_ids_equal") is True
            and receipt.get("selected_candidate_index") is not None
        )
        if not valid:
            errors.append(f"STRICT_RECEIPT_INVALID:{label}")
        return valid
    if status == "F1C_NO_STRICT_CANDIDATE":
        if attacked or receipt.get("candidate_audit_complete") is not True or not isinstance(receipt.get("candidate_audit"), list) or len(receipt["candidate_audit"]) != 11:
            errors.append(f"NO_STRICT_FALLBACK_INVALID:{label}")
            return False
        return False
    if status == "F1C_CLEAN_NATIVE_OPEN_NO_ATTACK":
        if attacked:
            errors.append(f"NATIVE_OPEN_ATTACKED:{label}")
        return False
    errors.append(f"ATTACK_STATUS_INVALID:{label}:{status}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    runtime_root = args.root.resolve()
    protocol = load(CONFIG)
    canary = load(CANARY)
    f1c_root = load(F1C_ROOT)
    errors: list[str] = []
    if protocol.get("status") != "FROZEN_F1C_T5_CANARY_V3" or f1c_root.get("status") != "PASS_F1C_PRE_GPU_STATIC_CONTRACT":
        errors.append("UPSTREAM_F1C_FREEZE_INVALID")
    if sha(CONFIG) != str(f1c_root.get("protocol_sha256")) or sha(F1C_ROOT) != str(protocol.get("upstream", {}).get("f1c_root_seal_sha256", sha(F1C_ROOT))):
        # The protocol intentionally does not duplicate this hash; the second clause is informational.
        if sha(CONFIG) != str(f1c_root.get("protocol_sha256")):
            errors.append("F1C_PROTOCOL_HASH_MISMATCH")
    rows = list(canary.get("rows", []))
    expected_keys = {str(row.get("canonical_parent_key")) for row in rows if row.get("role") == "C_CANARY_V3"}
    if len(expected_keys) != 8:
        errors.append("CANARY_EXPECTED_KEY_SET_INVALID")
    if not runtime_root.is_dir():
        raise SystemExit(f"F1C_RUNTIME_ROOT_MISSING:{runtime_root}")
    worker_dir = runtime_root / "workers"
    worker_receipts = sorted(worker_dir.glob("worker_*_receipt.json")) if worker_dir.is_dir() else []
    observed_keys: list[str] = []
    for path in worker_receipts:
        receipt = load(path)
        if receipt.get("status") != "PASS_F1C_WORKER_COMPLETED":
            errors.append(f"WORKER_NOT_PASS:{path.name}")
        if not protected_ok(receipt.get("protected_boundary")):
            errors.append(f"WORKER_BOUNDARY_INVALID:{path.name}")
        observed_keys.extend(str(key) for key in receipt.get("assigned_keys", []))
    if set(observed_keys) != expected_keys or len(observed_keys) != len(set(observed_keys)):
        errors.append("WORKER_COVERAGE_INVALID")
    per_arm: dict[str, dict[str, Any]] = {
        arm: {"parent_rows": [], "strict_valid_steps": 0, "parents_with_strict_step": 0, "per_suite_parent_success": {suite: 0 for suite in SUITES}, "runtime_error_count": 0}
        for arm in ARMS
    }
    parent_summaries: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        suite = key.split("/")[0]
        parent_dir = runtime_root / suite / safe_name(key)
        parent_path = parent_dir / "parent_receipt.json"
        if not parent_path.is_file():
            errors.append(f"PARENT_RECEIPT_MISSING:{key}")
            continue
        parent = load(parent_path)
        if parent.get("status") != "PASS_F1C_PARENT_COMPLETED" or parent.get("student_used") is not False or parent.get("student_emit_used") is not False or not protected_ok(parent.get("protected_boundary")):
            errors.append(f"PARENT_NOT_PASS:{key}")
        clean = parent.get("clean_probe") or {}
        probe = clean.get("selected_probe") or {}
        if clean.get("status") != "PASS_F1C_CLEAN_PROBE" or clean.get("selected_probe_count") != 1 or not probe.get("observation_path"):
            errors.append(f"CLEAN_PROBE_INVALID:{key}")
        else:
            probe_path = Path(str(probe["observation_path"]))
            if not probe_path.is_file() or sha(probe_path) != str(probe.get("observation_sha256")):
                errors.append(f"PROBE_HASH_INVALID:{key}")
        arm_rows = {str(row.get("temporal_init")): row for row in parent.get("temporal_arms", [])}
        if set(arm_rows) != set(ARMS):
            errors.append(f"TEMPORAL_ARM_SET_INVALID:{key}")
        parent_summary = {"canonical_parent_key": key, "suite": suite, "probe_step": probe.get("step"), "arms": {}}
        for arm in ARMS:
            arm_path = parent_dir / f"temporal_{arm}" / "arm_receipt.json"
            if not arm_path.is_file():
                errors.append(f"ARM_RECEIPT_MISSING:{key}:{arm}")
                continue
            arm_receipt = load(arm_path)
            if arm_receipt.get("status") != "PASS_F1C_ARM_COMPLETED":
                errors.append(f"ARM_NOT_PASS:{key}:{arm}")
                per_arm[arm]["runtime_error_count"] += 1
            if arm_receipt.get("student_used") is not False or arm_receipt.get("student_emit_used") is not False or not protected_ok(arm_receipt.get("protected_boundary")):
                errors.append(f"ARM_BOUNDARY_INVALID:{key}:{arm}")
            step_rows = list(arm_receipt.get("step_rows", []))
            if not step_rows or len(step_rows) > int(protocol["execution"]["attempted_steps"]):
                errors.append(f"ARM_STEP_COUNT_INVALID:{key}:{arm}")
            strict_count = 0
            for step_row in step_rows:
                attack_path = Path(str(step_row.get("attack_receipt_path", "")))
                if not attack_path.is_file():
                    errors.append(f"ATTACK_RECEIPT_MISSING:{key}:{arm}:{step_row.get('attempt')}")
                    continue
                attack = load(attack_path)
                strict_count += int(audit_attack(attack, step_row, errors, f"{key}:{arm}:{step_row.get('attempt')}"))
                if bool(step_row.get("attacked_action_executed")) != (attack.get("status") == "PASS_F1C_STRICT_CANDIDATE"):
                    errors.append(f"STEP_ACTION_STATUS_MISMATCH:{key}:{arm}:{step_row.get('attempt')}")
            if strict_count:
                per_arm[arm]["parents_with_strict_step"] += 1
                per_arm[arm]["per_suite_parent_success"][suite] += 1
            per_arm[arm]["strict_valid_steps"] += strict_count
            per_arm[arm]["parent_rows"].append({"canonical_parent_key": key, "suite": suite, "strict_valid_steps": strict_count, "attempted_steps": len(step_rows)})
            parent_summary["arms"][arm] = {"strict_valid_steps": strict_count, "attempted_steps": len(step_rows), "status": arm_receipt.get("status")}
        parent_summaries.append(parent_summary)
    ranking = {}
    for arm in ARMS:
        stats = per_arm[arm]
        ranking[arm] = {
            "minimum_per_suite_parent_success": min(stats["per_suite_parent_success"].values()),
            "total_parent_success": stats["parents_with_strict_step"],
            "total_strict_valid_steps": stats["strict_valid_steps"],
            "tie_simplicity": 1 if arm == "none" else 0,
        }
    chosen = max(ARMS, key=lambda arm: (ranking[arm]["minimum_per_suite_parent_success"], ranking[arm]["total_parent_success"], ranking[arm]["total_strict_valid_steps"], ranking[arm]["tie_simplicity"]))
    runtime_files: dict[str, str] = {}
    for path in sorted(path for path in runtime_root.rglob("*") if path.is_file()):
        runtime_files[path.relative_to(runtime_root).as_posix()] = sha(path)
    manifest = {"files": runtime_files}
    manifest_sha = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    status = "F1C_T5_CANARY_QUALIFICATION_PASS" if not errors else "HOLD_F1C_EXECUTABLE_EVIDENCE_INSUFFICIENT"
    ledger = {"schema": "STAGE_X1R2_F1C_T5_CANARY_PROBE_LEDGER_V3", "status": status, "rows": parent_summaries, "expected_parent_count": 8, "runtime_root": str(runtime_root), "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "protected_reads": 0, "vphys_reads": 0, "physical_interventions": 0, "attack_outcome_reads": 0, "attacked_env_steps": 0}}
    comparison = {"schema": "STAGE_X1R2_F1C_TEMPORAL_METHOD_COMPARISON_V3", "status": status, "arms": per_arm, "ranking": ranking, "selection_rule": protocol["temporal_selection"], "selected_temporal_init": chosen}
    decision = {"schema": "STAGE_X1R2_F1C_T5_CANARY_DECISION_V3", "status": status, "errors": errors, "selected_temporal_init": chosen, "ranking": ranking, "selected_method": {"method": "M1", "iterations": 10, "objective": protocol["method"]["objective"], "epsilon": protocol["method"]["epsilon_processor_pixel_values"], "step_size": protocol["method"]["step_size"], "temporal_init": chosen}, "protected_boundary": ledger["protected_boundary"]}
    ledger_path, comparison_path, decision_path = OUT / "F1C_T5_CANARY_PROBE_LEDGER_V3.json", OUT / "F1C_TEMPORAL_METHOD_COMPARISON_V3.json", OUT / "F1C_T5_CANARY_DECISION_V3.json"
    write(ledger_path, ledger); write(comparison_path, comparison); write(decision_path, decision)
    artifacts = {"configs/STAGE_X_X1R2_F1C_METHOD_FREEZE_T5_CANARY_PROTOCOL_V3.json": sha(CONFIG), "reports/STAGE_X_X1R2_F1C_METHOD_FREEZE_T5_CANARY_V3_20260821/F1C_ROOT_SEAL_V3.json": sha(F1C_ROOT), "reports/STAGE_X_X1R2_F1C_T5_CANARY_RESULT_AGGREGATION_V3_20260821/F1C_T5_CANARY_PROBE_LEDGER_V3.json": sha(ledger_path), "reports/STAGE_X_X1R2_F1C_T5_CANARY_RESULT_AGGREGATION_V3_20260821/F1C_TEMPORAL_METHOD_COMPARISON_V3.json": sha(comparison_path), "reports/STAGE_X_X1R2_F1C_T5_CANARY_RESULT_AGGREGATION_V3_20260821/F1C_T5_CANARY_DECISION_V3.json": sha(decision_path)}
    root = {"schema": "STAGE_X1R2_F1C_T5_CANARY_ROOT_SEAL_V3", "status": status, "errors": errors, "artifact_hashes": dict(sorted(artifacts.items())), "runtime_root": str(runtime_root), "runtime_file_count": len(runtime_files), "runtime_file_hashes": runtime_files, "runtime_manifest_sha256": manifest_sha, "source_commit": git("rev-parse", "HEAD"), "source_tree": git("show", "-s", "--format=%T", "HEAD"), "f1c_method_freeze_root_sha256": sha(F1C_ROOT), "protocol_sha256": sha(CONFIG), "selected_temporal_init": chosen, "protected_boundary": ledger["protected_boundary"], "seal_scope_excludes_sidecar": True}
    root_path = OUT / "F1C_T5_CANARY_ROOT_SEAL_V3.json"
    write(root_path, root)
    sidecar = OUT / "F1C_T5_CANARY_ROOT_SEAL_V3.sha256"
    sidecar.write_text(f"{sha(root_path)}  {root_path.name}\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": status, "errors": errors, "selected_temporal_init": chosen, "ranking": ranking, "runtime_file_count": len(runtime_files), "root_seal_sha256": sha(root_path)}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
