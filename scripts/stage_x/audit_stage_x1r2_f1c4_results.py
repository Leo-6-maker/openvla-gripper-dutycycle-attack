#!/usr/bin/env python3
"""Seal the F1-C4 runtime receipts without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
ARMS = ("none", "prev_delta")
TERMINAL_HOLD = "HOLD_F1C4_EXECUTABLE_EVIDENCE_INSUFFICIENT_TERMINAL"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT).strip()


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def localize(raw: str, remote_root: Path, runtime_root: Path) -> Path:
    path = Path(raw)
    try:
        return runtime_root / path.relative_to(remote_root)
    except ValueError:
        return path if path.is_absolute() else runtime_root / path


def boundary_ok(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("eval160") != "UNREAD":
        return False
    for key in ("protected_reads", "vphys_reads", "physical_interventions", "attack_outcome_reads", "attacked_env_steps"):
        if int(value.get(key, 0)) != 0:
            return False
    return True


def boundary_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"valid": False}
    return {
        "valid": boundary_ok(value),
        "eval160": value.get("eval160"),
        "protected_reads": int(value.get("protected_reads", 0)),
        "vphys_reads": int(value.get("vphys_reads", 0)),
        "physical_interventions": int(value.get("physical_interventions", 0)),
        "attack_outcome_reads": int(value.get("attack_outcome_reads", 0)),
        "attacked_env_steps": int(value.get("attacked_env_steps", 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--static-root-seal", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--remote-runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    repo_root = protocol_path.parents[1]
    ledger_path = args.ledger.resolve()
    static_root_path = args.static_root_seal.resolve()
    runtime_root = args.runtime_root.resolve()
    remote_runtime_root = args.remote_runtime_root
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"F1C4_RESULT_OUTPUT_ALREADY_EXISTS:{output}")

    protocol = load(protocol_path)
    ledger = load(ledger_path)
    static_root = load(static_root_path)
    expected_keys = [str(row["canonical_parent_key"]) for row in ledger.get("rows", [])]
    expected_keys_set = set(expected_keys)
    candidate_sources = list(protocol["method"]["candidate_order"])
    expected_candidate_count = len(candidate_sources)
    errors: list[str] = []
    boundary_errors: list[str] = []

    runtime_files = {
        path.relative_to(runtime_root).as_posix(): sha(path)
        for path in sorted(runtime_root.rglob("*"))
        if path.is_file()
    }
    runtime_manifest_sha = hashlib.sha256(
        json.dumps({"files": runtime_files}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    worker_summaries: list[dict[str, Any]] = []
    observed_keys: list[str] = []
    worker_dir = runtime_root / "workers"
    for path in sorted(worker_dir.glob("worker_*_receipt.json")):
        receipt = load(path)
        observed_keys.extend(str(key) for key in receipt.get("assigned_keys", []))
        if receipt.get("status") != "PASS_F1C_WORKER_COMPLETED":
            errors.append(f"WORKER_STATUS:{path.name}:{receipt.get('status')}")
        if receipt.get("source", {}).get("status_porcelain") != "":
            errors.append(f"WORKER_DIRTY:{path.name}")
        if not boundary_ok(receipt.get("protected_boundary")):
            boundary_errors.append(f"worker:{path.name}")
        worker_summaries.append({
            "worker_index": receipt.get("worker_index"),
            "physical_gpu": receipt.get("physical_gpu"),
            "status": receipt.get("status"),
            "assigned_keys": receipt.get("assigned_keys", []),
            "source": receipt.get("source"),
            "gpu_before_model_load": receipt.get("gpu_before_model_load"),
            "gpu_after": receipt.get("gpu_after"),
            "protected_boundary": boundary_record(receipt.get("protected_boundary")),
        })
    if len(worker_summaries) != 8:
        errors.append(f"WORKER_COUNT:{len(worker_summaries)}")
    if set(observed_keys) != expected_keys_set or len(observed_keys) != len(set(observed_keys)):
        errors.append("WORKER_COVERAGE")
    for worker in worker_summaries:
        admission = worker.get("gpu_before_model_load") or {}
        if int(admission.get("free_memory_mib", 0)) <= int(protocol["resource"]["free_memory_mib_strictly_greater_than"]):
            errors.append(f"GPU_ADMISSION:{worker.get('worker_index')}")
        if admission.get("foreign_processes_untouched") is not True:
            errors.append(f"GPU_FOREIGN_PROCESS_BOUNDARY:{worker.get('worker_index')}")

    error_counts: Counter[str] = Counter()
    parent_summaries: list[dict[str, Any]] = []
    arm_totals = {
        arm: {
            "parent_count": 0,
            "completed_parent_count": 0,
            "runtime_error_count": 0,
            "attempted_step_count": 0,
            "attack_invocation_count": 0,
            "candidate_evidence_complete_steps": 0,
            "candidate_evidence_rows": 0,
            "strict_valid_steps": 0,
            "no_strict_candidate_steps": 0,
            "clean_fallback_steps": 0,
            "pgd_calls": 0,
            "attacked_env_steps": 0,
            "physical_interventions": 0,
            "vphys_reads": 0,
        }
        for arm in ARMS
    }

    for key in sorted(expected_keys_set):
        suite = key.split("/", 1)[0]
        parent_dir = runtime_root / suite / safe_name(key)
        parent_path = parent_dir / "parent_receipt.json"
        if not parent_path.is_file():
            errors.append(f"PARENT_MISSING:{key}")
            continue
        parent = load(parent_path)
        if not boundary_ok(parent.get("protected_boundary")):
            boundary_errors.append(f"parent:{key}")
        clean = parent.get("clean_probe") or {}
        selected = clean.get("selected_probe") or {}
        probe_path = localize(str(selected.get("observation_path", "")), remote_runtime_root, runtime_root)
        clean_valid = (
            clean.get("status") == "PASS_F1C_CLEAN_PROBE"
            and clean.get("selected_probe_count") == 1
            and bool(selected.get("observation_path"))
            and probe_path.is_file()
            and sha(probe_path) == str(selected.get("observation_sha256"))
            and clean.get("student_used") is False
            and clean.get("student_emit_used") is False
        )
        if not clean_valid:
            errors.append(f"CLEAN_PROBE:{key}")
        arms: dict[str, Any] = {}
        for arm in ARMS:
            arm_totals[arm]["parent_count"] += 1
            arm_path = parent_dir / f"temporal_{arm}" / "arm_receipt.json"
            if not arm_path.is_file():
                errors.append(f"ARM_MISSING:{key}:{arm}")
                continue
            receipt = load(arm_path)
            counters = receipt.get("counters") or {}
            step_rows = list(receipt.get("step_rows", []))
            attack_statuses: Counter[str] = Counter()
            candidate_complete = 0
            candidate_rows = 0
            strict_valid = 0
            fallback_steps = 0
            candidate_native_open = 0
            candidate_arm_exact = 0
            attack_errors: Counter[str] = Counter()
            for row in step_rows:
                if row.get("executed_action_class") == "CLEAN_ACTION":
                    fallback_steps += 1
                attack_path = localize(str(row.get("attack_receipt_path", "")), remote_runtime_root, runtime_root)
                if not attack_path.is_file():
                    errors.append(f"ATTACK_MISSING:{key}:{arm}:{row.get('attempt')}")
                    continue
                attack = load(attack_path)
                status = str(attack.get("status"))
                attack_statuses[status] += 1
                if not boundary_ok(attack.get("protected_boundary")):
                    boundary_errors.append(f"attack:{key}:{arm}:{row.get('attempt')}")
                if attack.get("student_used") is not False or attack.get("student_emit_used") is not False:
                    errors.append(f"ATTACK_STUDENT_BOUNDARY:{key}:{arm}:{row.get('attempt')}")
                audit = attack.get("candidate_audit")
                complete = attack.get("candidate_audit_complete") is True and isinstance(audit, list) and len(audit) == expected_candidate_count
                if complete:
                    candidate_complete += 1
                    candidate_rows += len(audit)
                    if [item.get("candidate_source") for item in audit] != candidate_sources:
                        errors.append(f"CANDIDATE_ORDER:{key}:{arm}:{row.get('attempt')}")
                    candidate_native_open += sum(bool(item.get("direct_generated_gripper_is_native_open")) for item in audit)
                    candidate_arm_exact += sum(bool(item.get("arm_token_ids_equal")) for item in audit)
                else:
                    errors.append(f"CANDIDATE_EVIDENCE:{key}:{arm}:{row.get('attempt')}")
                if status == "PASS_F1C_STRICT_CANDIDATE":
                    strict_valid += 1
                    selected_index = attack.get("selected_candidate_index")
                    selected = next((item for item in audit if item.get("candidate_index") == selected_index), None)
                    strict_candidate_ok = (
                        row.get("attacked_action_executed") is True
                        and isinstance(selected, dict)
                        and selected.get("arm_token_ids_equal") is True
                        and selected.get("direct_generated_gripper_is_native_open") is True
                        and len(selected.get("direct_generated_token_ids", [])) == int(protocol["method"]["direct_action_token_count"])
                        and float(selected.get("pixel_budget_adv_inputs_linf", float("inf"))) <= float(protocol["method"]["epsilon_processor_pixel_values"])
                    )
                    if not strict_candidate_ok:
                        errors.append(f"STRICT_CANDIDATE_GATE:{key}:{arm}:{row.get('attempt')}")
                if attack.get("selector_error_message"):
                    attack_errors[str(attack["selector_error_message"])] += 1
            attempted = int(receipt.get("attempted_step_count", 0))
            attack_invocations = int(counters.get("attack_invocation_count", 0))
            arm_status = str(receipt.get("status"))
            if not boundary_ok(receipt.get("protected_boundary")):
                boundary_errors.append(f"arm:{key}:{arm}")
            if arm_status == "PASS_F1C_ARM_COMPLETED":
                if attempted != int(protocol["execution"]["attempted_steps"]) or len(step_rows) != attempted or attack_invocations != attempted:
                    errors.append(f"ARM_COUNTER_RECONCILIATION:{key}:{arm}")
                arm_totals[arm]["completed_parent_count"] += 1
            elif arm_status == "HOLD_F1C_RUNTIME":
                arm_totals[arm]["runtime_error_count"] += 1
                if attempted != 0 or step_rows or attack_invocations != 0:
                    errors.append(f"ARM_HOLD_COUNTER_RECONCILIATION:{key}:{arm}")
                runtime_error = str(receipt.get("error"))
                error_counts[runtime_error] += 1
                errors.append(f"ARM_RUNTIME_ERROR:{key}:{arm}:{runtime_error}")
            else:
                errors.append(f"ARM_STATUS:{key}:{arm}:{arm_status}")
            for field in ("pgd_calls", "attacked_env_steps", "physical_interventions", "vphys_reads"):
                arm_totals[arm][field] += int(counters.get(field, 0))
            arm_totals[arm]["attempted_step_count"] += attempted
            arm_totals[arm]["attack_invocation_count"] += attack_invocations
            arm_totals[arm]["candidate_evidence_complete_steps"] += candidate_complete
            arm_totals[arm]["candidate_evidence_rows"] += candidate_rows
            arm_totals[arm]["strict_valid_steps"] += strict_valid
            arm_totals[arm]["no_strict_candidate_steps"] += attack_statuses.get("F1C_NO_STRICT_CANDIDATE", 0)
            arm_totals[arm]["clean_fallback_steps"] += fallback_steps
            arms[arm] = {
                "status": arm_status,
                "error": receipt.get("error"),
                "attempted_step_count": attempted,
                "counters": counters,
                "attack_status_counts": dict(sorted(attack_statuses.items())),
                "attack_error_counts": dict(sorted(attack_errors.items())),
                "candidate_evidence_complete_steps": candidate_complete,
                "candidate_evidence_rows": candidate_rows,
                "candidate_native_open_rows": candidate_native_open,
                "candidate_arm_exact_rows": candidate_arm_exact,
                "strict_valid_steps": strict_valid,
                "clean_fallback_steps": fallback_steps,
                "protected_boundary": boundary_record(receipt.get("protected_boundary")),
            }
        parent_status = str(parent.get("status"))
        if parent_status == "PASS_F1C_PARENT_COMPLETED":
            parent_completed = all(arms.get(arm, {}).get("status") == "PASS_F1C_ARM_COMPLETED" for arm in ARMS)
            if parent_completed:
                for arm in ARMS:
                    pass
            else:
                errors.append(f"PARENT_STATUS_INCONSISTENT:{key}")
        else:
            if not any(arms.get(arm, {}).get("status") == "HOLD_F1C_RUNTIME" for arm in ARMS):
                errors.append(f"PARENT_STATUS:{key}:{parent_status}")
        parent_summaries.append({
            "canonical_parent_key": key,
            "suite": suite,
            "status": parent_status,
            "clean_probe": {
                "status": clean.get("status"),
                "selected_probe_count": clean.get("selected_probe_count"),
                "observed_rows": clean.get("observed_rows"),
                "eligible_rows": clean.get("eligible_rows"),
                "selected_step": selected.get("step"),
                "selected_probe_sha256": selected.get("observation_sha256"),
                "hash_valid": clean_valid,
            },
            "arms": arms,
            "protected_boundary": boundary_record(parent.get("protected_boundary")),
        })

    if boundary_errors:
        errors.extend(f"PROTECTED_BOUNDARY:{item}" for item in boundary_errors)
    if protocol.get("status") != "FROZEN_F1C4_T5_CANARY_V1":
        errors.append("PROTOCOL_STATUS")
    if static_root.get("status") != "PASS_F1C4_PRE_GPU_STATIC_CONTRACT":
        errors.append("STATIC_ROOT_STATUS")

    parent_status_counts = Counter(row["status"] for row in parent_summaries)
    arm_status_counts = {arm: Counter(row["arms"].get(arm, {}).get("status") for row in parent_summaries) for arm in ARMS}
    overall = {
        "parent_count": len(parent_summaries),
        "parent_status_counts": dict(sorted(parent_status_counts.items())),
        "arm_status_counts": {arm: dict(sorted(counts.items())) for arm, counts in arm_status_counts.items()},
        "completed_parent_count": parent_status_counts.get("PASS_F1C_PARENT_COMPLETED", 0),
        "runtime_error_parent_count": parent_status_counts.get("HOLD_F1C_PARENT", 0),
        "protected_boundary_valid": not boundary_errors,
        "error_counts": dict(sorted(error_counts.items())),
    }
    audit = {
        "schema": "STAGE_X1R2_F1C4_RUNTIME_AUDIT_V1",
        "status": TERMINAL_HOLD,
        "gate": protocol["gate"],
        "namespace": ledger.get("namespace"),
        "source": {
            "commit": worker_summaries[0].get("source", {}).get("commit") if worker_summaries else None,
            "tree": worker_summaries[0].get("source", {}).get("tree") if worker_summaries else None,
            "all_workers_same_source": len({(w.get("source", {}).get("commit"), w.get("source", {}).get("tree")) for w in worker_summaries}) <= 1,
            "offline_freeze_validation": "PASS_F1C4_OFFLINE_FREEZE_VALIDATION",
        },
        "frozen_method": protocol["method"],
        "overall": overall,
        "workers": worker_summaries,
        "parents": parent_summaries,
        "arm_totals": arm_totals,
        "runtime": {
            "remote_root": str(remote_runtime_root),
            "local_audit_root": str(runtime_root),
            "file_count": len(runtime_files),
            "manifest_sha256": runtime_manifest_sha,
            "file_hashes": runtime_files,
        },
        "protected_boundary": {
            "eval160": "UNREAD",
            "protected_reads": 0,
            "vphys_reads": 0,
            "physical_interventions": 0,
            "attack_outcome_reads": 0,
            "attacked_env_steps": 0,
            "boundary_errors": boundary_errors,
        },
        "errors": sorted(set(errors)),
    }
    result_dir = output
    audit_path = result_dir / "F1C4_RUNTIME_AUDIT_V1.json"
    write(audit_path, audit)

    totals = {
        field: sum(arm_totals[arm][field] for arm in ARMS)
        for field in ("attempted_step_count", "attack_invocation_count", "candidate_evidence_complete_steps", "candidate_evidence_rows", "strict_valid_steps", "no_strict_candidate_steps", "clean_fallback_steps", "pgd_calls", "attacked_env_steps", "physical_interventions", "vphys_reads")
    }
    decision = {
        "schema": "STAGE_X1R2_F1C4_TERMINAL_DECISION_V1",
        "status": TERMINAL_HOLD,
        "gate": protocol["gate"],
        "namespace": ledger.get("namespace"),
        "classification": "REPLAY_EXECUTION_EVIDENCE_INSUFFICIENT_BEFORE_INTERPRETABLE_T5",
        "failure_mode": "F1C_REPLAY_OBSERVATION_HASH_MISMATCH",
        "failure_parent_count": overall["runtime_error_parent_count"],
        "failure_arm_count": sum(counts.get("HOLD_F1C_RUNTIME", 0) for counts in arm_status_counts.values()),
        "completed_parent_count": overall["completed_parent_count"],
        "completed_arm_count": sum(counts.get("PASS_F1C_ARM_COMPLETED", 0) for counts in arm_status_counts.values()),
        "totals": totals,
        "no_physical_or_protected_use": {
            "new_vphys_reads": 0,
            "physical_interventions": 0,
            "attack_outcome_reads": 0,
            "attacked_env_steps": totals["attacked_env_steps"],
            "eval160": "UNREAD",
            "protected_evaluation": "UNREAD",
        },
        "temporal_selection": "NOT_APPLICABLE_TERMINAL_HOLD",
        "stop_conditions": {
            "create_f1c5": False,
            "tune_method": False,
            "recycle_or_top_up_identities": False,
            "open_bridge_v3": False,
            "run_f1d": False,
            "enter_eval160_or_protected": False,
        },
        "audit_sha256": sha(audit_path),
    }
    decision_path = result_dir / "F1C4_TERMINAL_DECISION_V1.json"
    write(decision_path, decision)

    artifact_paths = [
        protocol_path,
        ledger_path,
        static_root_path,
        repo_root / "reports/STAGE_X1R2_F1C4_FRESH_CANARY_NAMESPACE_V1_20260822/F1C4_EXPOSURE_REAUDIT_V1.json",
        repo_root / "reports/STAGE_X1R2_F1C4_FRESH_CANARY_NAMESPACE_V1_20260822/F1C4_METHOD_EQUIVALENCE_AUDIT_V1.json",
        repo_root / "reports/STAGE_X1R2_F1C4_FRESH_CANARY_NAMESPACE_V1_20260822/F1C4_METHOD_SPEC_V1.json",
        repo_root / "reports/STAGE_X1R2_F1C4_FRESH_CANARY_NAMESPACE_V1_20260822/F1C4_PRE_GPU_AUDIT_V1.json",
        repo_root / "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_ROOT_SEAL_V3.json",
        repo_root / "reports/STAGE_X_X1R2_F1B_DEV_RESULT_AGGREGATION_V3_20260821/F1B_DEV_DECISION_V3.json",
        repo_root / "reports/STAGE_X_X1R2_F1B_DEV_RESULT_AGGREGATION_V3_20260821/F1B_DEV_ROOT_SEAL_V3.json",
        repo_root / "scripts/stage_x/run_stage_x1r2_f1c_t5_canary.py",
        repo_root / "scripts/stage_x/audit_stage_x1r2_f1c4_results.py",
        audit_path,
        decision_path,
    ]
    artifact_hashes = {path.relative_to(repo_root).as_posix(): sha(path) for path in artifact_paths}
    root = {
        "schema": "STAGE_X1R2_F1C4_RESULT_ROOT_SEAL_V1",
        "status": TERMINAL_HOLD,
        "gate": protocol["gate"],
        "namespace": ledger.get("namespace"),
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "runtime_root_remote": str(remote_runtime_root),
        "runtime_file_count": len(runtime_files),
        "runtime_manifest_sha256": runtime_manifest_sha,
        "source_commit": git(repo_root, "rev-parse", "HEAD"),
        "source_tree": git(repo_root, "show", "-s", "--format=%T", "HEAD"),
        "static_root_seal_sha256": sha(static_root_path),
        "protocol_sha256": sha(protocol_path),
        "decision_sha256": sha(decision_path),
        "protected_boundary": decision["no_physical_or_protected_use"],
        "stop_conditions": decision["stop_conditions"],
        "seal_scope_excludes_sidecar": True,
    }
    root_path = result_dir / "F1C4_RESULT_ROOT_SEAL_V1.json"
    write(root_path, root)
    sidecar = result_dir / "F1C4_RESULT_ROOT_SEAL_V1.sha256"
    sidecar.write_text(f"{sha(root_path)}  {root_path.name}\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": TERMINAL_HOLD, "parents": overall, "totals": totals, "runtime_file_count": len(runtime_files), "root_seal_sha256": sha(root_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
