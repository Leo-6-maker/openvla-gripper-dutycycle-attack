#!/usr/bin/env python3
"""Aggregate and seal the outcome-blind F1-B DEV receipts."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/STAGE_X_X1R2_F1B_DEV_PROTOCOL_V3.json"
F1B_FREEZE_DIR = ROOT / "reports/STAGE_X_X1R2_F1B_DEV_METHOD_FREEZE_V3_20260821"
REPORT_DIR = ROOT / "reports/STAGE_X_X1R2_F1B_DEV_RESULT_AGGREGATION_V3_20260821"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
BOUNDARY_COUNTERS = (
    "attacked_env_steps",
    "physical_interventions",
    "vphys_reads",
    "protected_reads",
    "eval160_reads",
    "attack_outcome_reads",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT).strip()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def boundary_errors(value: Mapping[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    if value.get("student_used") is not False or value.get("student_emit_used") is not False:
        errors.append(f"{label}:STUDENT_USED")
    boundary = value.get("protected_boundary") or {}
    if boundary.get("eval160") != "UNREAD" or boundary.get("protected_evaluation") != "UNREAD":
        errors.append(f"{label}:PROTECTED_STATUS_INVALID")
    counters = value.get("counters") or {}
    for key in BOUNDARY_COUNTERS:
        if int(value.get(key, counters.get(key, boundary.get(key, 0))) or 0) != 0:
            errors.append(f"{label}:{key}_NONZERO")
    return errors


def candidate_source(index: int) -> str:
    return "delta0" if index == 0 else f"pgd_iteration_{index}"


def validate_candidate_audit(receipt: Mapping[str, Any]) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    expected = int(receipt.get("iterations", 0)) + 1
    audit = receipt.get("candidate_audit")
    if receipt.get("candidate_audit_complete") is not True or not isinstance(audit, list) or len(audit) != expected:
        return ["CANDIDATE_AUDIT_INCOMPLETE"], {"arm_exact": 0, "native_open": 0, "strict": 0}
    arm_exact = 0
    native_open = 0
    for index, row in enumerate(audit):
        if row.get("candidate_index") != index or row.get("candidate_source") != candidate_source(index):
            errors.append(f"CANDIDATE_ORDER_INVALID:{index}")
        if row.get("arm_token_ids_equal") is True:
            arm_exact += 1
        if row.get("direct_generated_gripper_is_native_open") is True:
            native_open += 1
        for key in (
            "direct_generated_token_ids",
            "direct_generated_arm_token_ids",
            "arm_mismatch_dimensions",
            "arm_token_ids_equal",
            "direct_generated_gripper_is_native_open",
            "pixel_budget_adv_inputs_linf",
            "processor_input_sha256",
        ):
            if row.get(key) is None:
                errors.append(f"CANDIDATE_FIELD_MISSING:{index}:{key}")
        if not isinstance(row.get("direct_generated_token_ids"), list) or len(row["direct_generated_token_ids"]) != 7:
            errors.append(f"DIRECT_TOKEN_COUNT_INVALID:{index}")
    strict = 1 if receipt.get("status") == "PASS_F1B_VALID_CANDIDATE" else 0
    return errors, {"arm_exact": arm_exact, "native_open": native_open, "strict": strict}


def failure_category(summary: Mapping[str, int], status: str) -> str:
    if status == "PASS_F1B_VALID_CANDIDATE":
        return "STRICT_REALIZABLE"
    if summary["arm_exact"] and not summary["native_open"]:
        return "TARGETABILITY_LIMITED"
    if summary["native_open"] and not summary["arm_exact"]:
        return "SELECTIVITY_LIMITED"
    if not summary["arm_exact"] and not summary["native_open"]:
        return "JOINT_LIMITED"
    return "STRICT_INTERSECTION_LIMITED"


def method_rank(stat: Mapping[str, Any]) -> tuple[float, ...]:
    mean_linf = stat.get("mean_selected_linf")
    return (
        -float(stat["min_per_suite_parent_success"]),
        -float(stat["parent_success_count"]),
        float(mean_linf) if mean_linf is not None else math.inf,
        float(stat["complexity_rank"]),
        float(stat["iterations"]),
    )


def validate_freeze(protocol: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    freeze_root = F1B_FREEZE_DIR / "F1B_ROOT_SEAL_V3.json"
    freeze_sidecar = F1B_FREEZE_DIR / "F1B_ROOT_SEAL_V3.sha256"
    if not freeze_root.is_file() or not freeze_sidecar.is_file():
        errors.append("F1B_METHOD_FREEZE_MISSING")
        return {}
    freeze_sha = sha256_file(freeze_root)
    if freeze_sidecar.read_text(encoding="utf-8").split()[0] != freeze_sha:
        errors.append("F1B_METHOD_FREEZE_SIDECAR_MISMATCH")
    freeze = load_json(freeze_root)
    if freeze.get("status") != "PASS_F1B_PRE_GPU_STATIC_CONTRACT":
        errors.append("F1B_METHOD_FREEZE_NOT_PASS")
    if freeze.get("protocol_sha256") != sha256_file(PROTOCOL):
        errors.append("F1B_PROTOCOL_HASH_MISMATCH")
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_json(protocol_path)
    output_root = (args.output_root or Path(str(protocol["runtime"]["durable_output_root"]))).resolve()
    errors: list[str] = []
    if REPORT_DIR.exists() and any(REPORT_DIR.iterdir()):
        raise SystemExit("F1B_RESULT_REPORT_ALREADY_EXISTS")
    if not output_root.is_dir():
        raise SystemExit("F1B_OUTPUT_ROOT_MISSING")
    if protocol.get("status") != "FROZEN_F1B_DEV_V3" or protocol.get("scientific_authority") is not False:
        errors.append("F1B_PROTOCOL_NOT_FROZEN")
    freeze = validate_freeze(protocol, errors)
    f1a3_root = ROOT / str(protocol["population"]["f1a3_root_seal_path"])
    f1a3 = load_json(f1a3_root)
    dev_path = ROOT / str(protocol["population"]["path"])
    dev_ledger = load_json(dev_path)
    expected_rows = [row for row in dev_ledger.get("rows", []) if row.get("role") == "DEV_V3"]
    expected_keys = {str(row["canonical_parent_key"]) for row in expected_rows}
    if len(expected_rows) != 24 or len(expected_keys) != 24:
        errors.append("DEV_LEDGER_INVALID")
    if f1a3.get("selected_hard_or_unresolved_count") != 0:
        errors.append("F1A3_SELECTED_EXPOSURE_INVALID")
    if sha256_file(f1a3_root) != str(protocol["population"]["f1a3_root_seal_sha256"]):
        errors.append("F1A3_ROOT_HASH_INVALID")

    parent_paths = sorted(output_root.glob("**/parent_receipt.json"))
    attack_paths = sorted(output_root.glob("**/attack_receipt.json"))
    worker_paths = sorted((output_root / "workers").glob("worker_*_receipt.json"))
    parent_by_key: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in parent_paths:
        receipt = load_json(path)
        key = str(receipt.get("canonical_parent_key", ""))
        if key in parent_by_key:
            errors.append(f"DUPLICATE_PARENT:{key}")
        parent_by_key[key] = (receipt, path)
        errors.extend(boundary_errors(receipt, f"parent:{key}"))
    if set(parent_by_key) != expected_keys:
        errors.append("PARENT_KEY_SET_MISMATCH")
    if len(parent_by_key) != 24:
        errors.append("PARENT_COUNT_INVALID")

    worker_statuses = []
    worker_gpus = []
    assigned_keys: list[str] = []
    for path in worker_paths:
        worker = load_json(path)
        worker_statuses.append(worker.get("status"))
        worker_gpus.append(int(worker.get("physical_gpu", -1)))
        assigned_keys.extend(str(key) for key in worker.get("assigned_keys", []))
        errors.extend(boundary_errors(worker, f"worker:{path.name}"))
        if worker.get("foreign_processes_untouched") is False:
            errors.append(f"worker:{path.name}:FOREIGN_PROCESS_TOUCH")
    if worker_statuses != ["PASS_F1B_WORKER_COMPLETED"] * 6:
        errors.append("WORKER_STATUS_INVALID")
    if sorted(worker_gpus) != [0, 1, 2, 3, 4, 7] or len(set(worker_gpus)) != 6:
        errors.append("WORKER_GPU_ASSIGNMENT_INVALID")
    if sorted(assigned_keys) != sorted(expected_keys):
        errors.append("WORKER_ASSIGNMENT_LEDGER_MISMATCH")

    attacks: dict[tuple[str, int, int, str, int], dict[str, Any]] = {}
    attack_paths_by_key: collections.defaultdict[str, list[Path]] = collections.defaultdict(list)
    for path in attack_paths:
        receipt = load_json(path)
        key = str(receipt.get("canonical_parent_key", ""))
        probe = receipt.get("probe") or {}
        config_key = (key, int(probe.get("probe_index", -1)), int(probe.get("step", -1)), str(receipt.get("method")), int(receipt.get("iterations", -1)))
        if config_key in attacks:
            errors.append(f"DUPLICATE_ATTACK:{config_key}")
        attacks[config_key] = {"receipt": receipt, "path": path}
        attack_paths_by_key[key].append(path)
        errors.extend(boundary_errors(receipt, f"attack:{path.name}"))
        audit_errors, audit_summary = validate_candidate_audit(receipt)
        if audit_errors:
            errors.extend(f"{key}:{probe.get('probe_index')}:{receipt.get('method')}:{receipt.get('iterations')}:{item}" for item in audit_errors)
        attacks[config_key]["audit_summary"] = audit_summary
        status = str(receipt.get("status"))
        if status not in {"PASS_F1B_VALID_CANDIDATE", "F1B_NO_STRICT_CANDIDATE"}:
            errors.append(f"ATTACK_STATUS_INVALID:{config_key}:{status}")
        if status == "F1B_NO_STRICT_CANDIDATE":
            if receipt.get("selector_error_type") != "STRUCTURAL_INVALID_NO_SELECTIVE_CANDIDATE":
                errors.append(f"FAILURE_TYPE_INVALID:{config_key}")
            if receipt.get("diagnostics_sources_equal") is not True:
                errors.append(f"FAILURE_DIAGNOSTICS_NOT_EQUAL:{config_key}")
            if receipt.get("selected_candidate_index") is not None:
                errors.append(f"FAILED_RECEIPT_SELECTED_CANDIDATE:{config_key}")
        if status == "PASS_F1B_VALID_CANDIDATE" and receipt.get("selected_candidate_index") is None:
            errors.append(f"VALID_RECEIPT_WITHOUT_SELECTION:{config_key}")

    methods = protocol.get("methods", {})
    method_steps = [(method, int(iterations)) for method, spec in methods.items() for iterations in spec.get("iterations", [])]
    probe_rows: list[dict[str, Any]] = []
    stats: dict[tuple[str, int], dict[str, Any]] = {}
    for method, iterations in method_steps:
        stats[(method, iterations)] = {
            "method": method,
            "iterations": iterations,
            "complexity_rank": int(methods[method].get("complexity_rank", 999)),
            "total_parent_count": 24,
            "parent_success_keys": set(),
            "suite_parent_success_keys": {suite: set() for suite in SUITES},
            "strict_valid_probe_count": 0,
            "status_counts": collections.Counter(),
            "failure_categories": collections.Counter(),
            "valid_linf": [],
            "candidate_audit_complete_count": 0,
            "expected_attack_count": 0,
        }

    for key in sorted(expected_keys):
        parent, parent_path = parent_by_key.get(key, ({}, output_root / "missing"))
        clean = parent.get("clean_probe") or {}
        probes = list(clean.get("selected_probes", []))
        if clean.get("status") != "PASS_F1B_CLEAN_RUNTIME":
            errors.append(f"CLEAN_RUNTIME_NOT_PASS:{key}")
        if len(probes) > int(protocol["probe"]["max_per_parent"]):
            errors.append(f"PROBE_COUNT_EXCEEDS_LIMIT:{key}")
        seen_probe_indices: set[int] = set()
        for probe in sorted(probes, key=lambda item: int(item.get("probe_index", -1))):
            probe_index = int(probe.get("probe_index", -1))
            step = int(probe.get("step", -1))
            if probe_index in seen_probe_indices:
                errors.append(f"DUPLICATE_PROBE_INDEX:{key}:{probe_index}")
            seen_probe_indices.add(probe_index)
            result_map: dict[str, Any] = {}
            for method, iterations in method_steps:
                lookup = (key, probe_index, step, method, iterations)
                item = attacks.get(lookup)
                stat = stats[(method, iterations)]
                stat["expected_attack_count"] += 1
                if item is None:
                    errors.append(f"MISSING_ATTACK_RECEIPT:{lookup}")
                    continue
                receipt = item["receipt"]
                summary = item["audit_summary"]
                status = str(receipt.get("status"))
                stat["status_counts"][status] += 1
                stat["failure_categories"][failure_category(summary, status)] += 1
                if receipt.get("candidate_audit_complete") is True:
                    stat["candidate_audit_complete_count"] += 1
                if status == "PASS_F1B_VALID_CANDIDATE":
                    stat["strict_valid_probe_count"] += 1
                    stat["parent_success_keys"].add(key)
                    stat["suite_parent_success_keys"][str(parent.get("suite"))].add(key)
                    selected = receipt.get("candidate_audit", [])[int(receipt["selected_candidate_index"])]
                    stat["valid_linf"].append(float(selected["pixel_budget_adv_inputs_linf"]))
                result_map.setdefault(method, {})[f"steps_{iterations}"] = {
                    "status": status,
                    "attack_receipt_path": item["path"].relative_to(output_root).as_posix(),
                    "selected_candidate_index": receipt.get("selected_candidate_index"),
                    "selected_candidate_source": receipt.get("selected_candidate_source"),
                    "candidate_audit_sha256": canonical_sha(receipt.get("candidate_audit")),
                    "candidate_summary": summary,
                }
            probe_rows.append({
                "suite": parent.get("suite"),
                "canonical_parent_key": key,
                "parent_receipt_path": parent_path.relative_to(output_root).as_posix(),
                "probe_index": probe_index,
                "step": step,
                "observation_sha256": probe.get("observation_sha256"),
                "clean_direct_token_ids": probe.get("clean_tokens"),
                "clean_gripper": probe.get("clean_gripper"),
                "results": result_map,
            })

    if len(attack_paths) != sum(stat["expected_attack_count"] for stat in stats.values()):
        errors.append("ATTACK_RECEIPT_COUNT_MISMATCH")

    serial_stats: dict[str, dict[str, Any]] = {}
    for config, stat in stats.items():
        method, iterations = config
        per_suite = {suite: len(stat["suite_parent_success_keys"][suite]) for suite in SUITES}
        serial_stats[f"{method}_steps_{iterations}"] = {
            "method": method,
            "iterations": iterations,
            "complexity_rank": stat["complexity_rank"],
            "total_parent_count": stat["total_parent_count"],
            "parent_success_count": len(stat["parent_success_keys"]),
            "per_suite_parent_success_count": per_suite,
            "min_per_suite_parent_success": min(per_suite.values()) if per_suite else 0,
            "strict_valid_probe_count": stat["strict_valid_probe_count"],
            "mean_strict_valid_probes_per_parent": stat["strict_valid_probe_count"] / 24.0,
            "mean_selected_linf": sum(stat["valid_linf"]) / len(stat["valid_linf"]) if stat["valid_linf"] else None,
            "status_counts": dict(sorted(stat["status_counts"].items())),
            "failure_categories": dict(sorted(stat["failure_categories"].items())),
            "candidate_audit_complete_count": stat["candidate_audit_complete_count"],
            "expected_attack_count": stat["expected_attack_count"],
        }

    best_by_method: dict[str, dict[str, Any]] = {}
    for method in methods:
        candidates = [stat for stat in serial_stats.values() if stat["method"] == method]
        best_by_method[method] = min(candidates, key=method_rank) if candidates else {}
    m0 = best_by_method.get("M0", {})
    strict_improvements = {}
    for method in ("M1", "M2"):
        candidate = best_by_method.get(method, {})
        strict_improvements[method] = bool(candidate) and (
            candidate["min_per_suite_parent_success"] > m0.get("min_per_suite_parent_success", -1)
            and candidate["parent_success_count"] > m0.get("parent_success_count", -1)
        )
    improving = [best_by_method[method] for method, improved in strict_improvements.items() if improved]
    decision_status = "F1B_NEW_METHOD_SELECTED_FOR_F1C" if improving else "F1_TARGETABILITY_DEVELOPMENT_NO_CLEAR_IMPROVEMENT"
    selected = min(improving, key=method_rank) if improving else None

    comparison = {
        "schema": "STAGE_X1R2_F1B_DEV_METHOD_COMPARISON_V3",
        "status": "PASS_F1B_DEV_RESULT_AGGREGATION" if not errors else "HOLD_F1B_DEV_RESULT_EVIDENCE_INSUFFICIENT",
        "scientific_authority": False,
        "population": {"role": "DEV_V3", "parent_count": 24, "per_suite_count": 6, "probe_count": len(probe_rows)},
        "configs": serial_stats,
        "best_by_method": best_by_method,
        "m0_baseline": m0,
        "strict_improvement_over_m0_on_first_two_criteria": strict_improvements,
        "errors": errors,
    }
    ledger = {
        "schema": "STAGE_X1R2_F1B_DEV_PROBE_LEDGER_V3",
        "status": comparison["status"],
        "scientific_authority": False,
        "outcome_blind": True,
        "parent_selection_unit": True,
        "rows": sorted(probe_rows, key=lambda row: (str(row["canonical_parent_key"]), int(row["probe_index"]))),
        "errors": errors,
    }
    decision = {
        "schema": "STAGE_X1R2_F1B_DEV_DECISION_V3",
        "status": decision_status if not errors else "HOLD_F1B_DEV_RESULT_EVIDENCE_INSUFFICIENT",
        "scientific_authority": False,
        "selection_rule": protocol["selection"],
        "selected_method": selected,
        "strict_improvement_over_m0_on_first_two_criteria": strict_improvements,
        "next_gate": "STAGE_X_X1R2_F1C_METHOD_FREEZE_AND_T5_CANARY" if selected else "STOP_F1B_AND_RETURN_TO_PI",
        "bridge_runtime_reads": 0,
        "physical_interventions": 0,
        "vphys_reads": 0,
        "protected_reads": 0,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
        "errors": errors,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    ledger_path = REPORT_DIR / "F1B_DEV_PROBE_LEDGER_V3.json"
    comparison_path = REPORT_DIR / "F1B_DEV_METHOD_COMPARISON_V3.json"
    decision_path = REPORT_DIR / "F1B_DEV_DECISION_V3.json"
    write_json(ledger_path, ledger)
    write_json(comparison_path, comparison)
    write_json(decision_path, decision)
    runtime_hashes = {path.relative_to(output_root).as_posix(): sha256_file(path) for path in sorted(output_root.glob("**/*")) if path.is_file()}
    runtime_manifest = canonical_sha(runtime_hashes)
    source_paths = [
        protocol_path,
        f1a3_root,
        dev_path,
        F1B_FREEZE_DIR / "F1B_METHOD_SPEC_V3.json",
        F1B_FREEZE_DIR / "F1B_PRE_GPU_AUDIT_V3.json",
        F1B_FREEZE_DIR / "F1B_ROOT_SEAL_V3.json",
        Path(__file__).resolve(),
        ledger_path,
        comparison_path,
        decision_path,
    ]
    artifact_hashes = {rel(path): sha256_file(path) for path in source_paths if path.is_file()}
    seal = {
        "schema": "STAGE_X1R2_F1B_DEV_ROOT_SEAL_V3",
        "status": comparison["status"],
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("show", "-s", "--format=%T", "HEAD"),
        "protocol_sha256": sha256_file(protocol_path),
        "f1a3_root_seal_sha256": sha256_file(f1a3_root),
        "f1b_method_freeze_root_sha256": sha256_file(F1B_FREEZE_DIR / "F1B_ROOT_SEAL_V3.json"),
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "runtime_output_root": str(output_root),
        "runtime_file_count": len(runtime_hashes),
        "runtime_manifest_sha256": runtime_manifest,
        "runtime_file_hashes": dict(sorted(runtime_hashes.items())),
        "workers": {"count": len(worker_paths), "physical_gpus": sorted(worker_gpus), "foreign_processes_untouched": True},
        "population": {"role": "DEV_V3", "parent_count": 24, "probe_count": len(probe_rows), "attack_receipt_count": len(attack_paths)},
        "protected_boundary": protocol["protected_boundary"],
        "bridge_runtime_reads": 0,
        "bridge_outcome_reads": 0,
        "seal_scope_excludes_sidecar": True,
        "errors": errors,
    }
    seal_path = REPORT_DIR / "F1B_DEV_ROOT_SEAL_V3.json"
    sidecar_path = REPORT_DIR / "F1B_DEV_ROOT_SEAL_V3.sha256"
    write_json(seal_path, seal)
    seal_sha = sha256_file(seal_path)
    sidecar_path.write_text(f"{seal_sha}  {seal_path.name}\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": decision["status"], "errors": errors, "selected_method": selected, "root_seal_sha256": seal_sha, "report_dir": str(REPORT_DIR)}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
