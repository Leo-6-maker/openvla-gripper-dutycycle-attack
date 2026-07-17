#!/usr/bin/env python3
"""Build the immutable Official V3 identity registry from audit reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gripper_attack.official_v3_contract import PASS_STATUSES, SUITES, canonical_key, load_contract, sha256_file
from gripper_attack.official_v3_recovery import FORMAL_METHODS


REGISTRY_FIELDS = [
    "canonical_parent_key", "suite", "task_idx", "state_id", "split", "ledger_status", "task_success",
    "selected_artifact_root", "selected_artifact_recursive_sha256", "artifact_audit_path", "artifact_audit_sha256",
    "provenance_class", "provenance_binding_mode", "provenance_binding_sha256",
    "external_manifest_registry_sha256", "worker_start_manifest_sha256", "worker_start_manifest_sidecar_sha256",
    "source_split_raw", "split_mapping_rule", "collector_head", "worker_id", "gpu_id",
    "worker_script_sha256", "adapter_sha256", "protocol_sha256", "model_tree_sha256", "processor_tree_sha256",
    "recovery_status", "recovery_method", "recovery_start_uuid", "recovery_manifest_sha256", "recovery_census_sha256",
    "formal_eligible", "formal_selected", "superseded_artifact_sha256", "remediation_required",
    "remediation_reason", "selection_reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return None


def _audit_map(audit_reports: dict[str, list[dict[str, Any]]] | dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, values in audit_reports.items():
        if isinstance(values, dict):
            values = [values]
        result[key].extend(values)
    return result


def build_registry(
    manifest_rows: list[dict[str, str]],
    ledger_rows: list[dict[str, str]],
    audit_reports: dict[str, list[dict[str, Any]]] | dict[str, dict[str, Any]],
    *,
    equivalence_status: str = "HOLD",
    remediation_rows: list[dict[str, str]] | None = None,
    expected_identity_count: int = 2000,
    stale_recovery_unresolved_count: int | None = None,
    stale_recovery_summary_sha256: str = "",
    recovery_rows: list[dict[str, str]] | None = None,
    recovery_census_sha256: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audits = _audit_map(audit_reports)
    remediation = {row.get("canonical_parent_key", ""): row for row in (remediation_rows or [])}
    recovery: dict[str, dict[str, str]] = {}
    if recovery_rows is not None:
        for recovery_row in recovery_rows:
            recovery_key = recovery_row.get("canonical_parent_key", "")
            if not recovery_key or recovery_key in recovery:
                raise ValueError(f"duplicate or empty recovery identity: {recovery_key!r}")
            recovery[recovery_key] = recovery_row
    ledger: dict[str, dict[str, str]] = {}
    for row in ledger_rows:
        key = row.get("canonical_parent_key", "")
        if not key and row.get("cell_id", "").startswith("CLEAN|"):
            key = row["cell_id"][len("CLEAN|"):]
        if key:
            ledger[key] = row
    keys = [row.get("canonical_parent_key", "") for row in manifest_rows]
    duplicate_manifest_keys = sorted(key for key, count in Counter(keys).items() if not key or count > 1)
    identity_column_mismatches: list[str] = []
    rows: list[dict[str, Any]] = []
    for manifest in manifest_rows:
        key = manifest.get("canonical_parent_key", "")
        try:
            expected_key = canonical_key(manifest.get("suite", ""), int(manifest.get("task_idx", "-1")), int(manifest.get("state_id", "-1")))
        except (TypeError, ValueError):
            expected_key = ""
        if key != expected_key:
            identity_column_mismatches.append(key or expected_key or "<empty>")
        ledger_row = ledger.get(key, {})
        recovery_row = recovery.get(key, {}) if recovery_rows is not None else {}
        candidates = list(audits.get(key, []))
        selected_hint = remediation.get(key, {}).get("selected_artifact_root", "")
        if selected_hint:
            candidates = [row for row in candidates if row.get("artifact_root") == selected_hint]
        base: dict[str, Any] = {field: "" for field in REGISTRY_FIELDS}
        base.update({
            "canonical_parent_key": key,
            "suite": manifest.get("suite", ""),
            "task_idx": manifest.get("task_idx", ""),
            "state_id": manifest.get("state_id", ""),
            "split": manifest.get("split", ""),
            "ledger_status": ledger_row.get("status", ""),
            "task_success": _bool(ledger_row.get("task_success")) if ledger_row.get("task_success") else ledger_row.get("status") == "PASS",
            "remediation_required": bool(remediation.get(key)),
            "remediation_reason": remediation.get(key, {}).get("reason", ""),
            "superseded_artifact_sha256": remediation.get(key, {}).get("superseded_artifact_sha256", ""),
            "formal_eligible": False,
            "formal_selected": False,
            "recovery_status": recovery_row.get("recovery_status", ""),
            "recovery_method": recovery_row.get("recovery_method", ""),
            "recovery_start_uuid": recovery_row.get("start_uuid", ""),
            "recovery_manifest_sha256": recovery_row.get("worker_start_manifest_sha256", ""),
            "recovery_census_sha256": recovery_census_sha256,
        })
        if key in identity_column_mismatches:
            base["selection_reason"] = "CANONICAL_IDENTITY_COLUMN_MISMATCH"
        elif key in duplicate_manifest_keys:
            base["selection_reason"] = "DUPLICATE_MANIFEST_CANONICAL_KEY"
        elif len(candidates) != 1:
            base["selection_reason"] = "NO_AUDIT_CANDIDATE" if not candidates else "DUPLICATE_AUDIT_CANDIDATE"
        else:
            audit = candidates[0]
            base.update({field: audit.get(field, "") for field in REGISTRY_FIELDS if field in audit})
            base["selected_artifact_root"] = audit.get("selected_artifact_root", audit.get("artifact_root", ""))
            base["selected_artifact_recursive_sha256"] = audit.get(
                "selected_artifact_recursive_sha256", audit.get("artifact_recursive_sha256", "")
            )
            base["canonical_parent_key"] = key
            base["task_success"] = base["task_success"] if base["task_success"] is not None else audit.get("task_success", "")
            recovery_ok = recovery_rows is None or (
                key in recovery and recovery[key].get("recovery_status") in FORMAL_METHODS
            )
            eligible = audit.get("status") in PASS_STATUSES and bool(audit.get("formal_eligible")) and recovery_ok
            if recovery_rows is not None and not recovery_ok:
                base["selection_reason"] = "PROVENANCE_RECOVERY_HOLD"
            elif audit.get("provenance_class") == "B_PREVIOUS_HEAD_EQUIVALENT" and equivalence_status != "PASS":
                eligible = False
                base["selection_reason"] = "OLD_HEAD_EQUIVALENCE_HOLD"
            elif eligible:
                base["selection_reason"] = "UNIQUE_AUDIT_AND_PROVENANCE_PASS"
            else:
                base["selection_reason"] = audit.get("status", "AUDIT_HOLD")
            base["formal_eligible"] = eligible
            base["formal_selected"] = eligible and key not in duplicate_manifest_keys
        rows.append(base)

    fit_rows = [row for row in rows if row["split"] == "FIT_TRAIN"]
    fit_split_selected = Counter(row["split"] for row in fit_rows if row["formal_selected"])
    fit_suite_selected = Counter(row["suite"] for row in fit_rows if row["formal_selected"])
    fit_task_selected = Counter((row["suite"], str(int(row["task_idx"]))) for row in fit_rows if row["formal_selected"])
    global_split_selected = Counter(row["split"] for row in rows if row["formal_selected"])
    global_suite_selected = Counter(row["suite"] for row in rows if row["formal_selected"])
    global_task_selected = Counter((row["suite"], str(int(row["task_idx"]))) for row in rows if row["formal_selected"])
    full_artifact_audit_pass_count = sum(bool(row["formal_selected"]) for row in fit_rows)
    unresolved_provenance_count = sum(
        row.get("provenance_class") in {"C_START_RECORD_MISSING", "D_DIRTY_START_QUARANTINE"}
        or row.get("selection_reason") == "OLD_HEAD_EQUIVALENCE_HOLD"
        for row in fit_rows
    )
    unfinished_remediation_count = sum(
        bool(row["remediation_required"]) and not bool(row["formal_selected"])
        for row in fit_rows
    )
    recovery_unresolved_count = sum(
        recovery_rows is not None and row.get("recovery_status") not in FORMAL_METHODS
        for row in rows
    )
    fit_recovery_unresolved_count = sum(
        recovery_rows is not None and row.get("split") == "FIT_TRAIN" and row.get("recovery_status") not in FORMAL_METHODS
        for row in rows
    )
    duplicate_selection_count = len(duplicate_manifest_keys)
    fit_ready = (
        len(rows) == expected_identity_count
        and len(set(keys)) == expected_identity_count
        and fit_split_selected["FIT_TRAIN"] == 800
        and all(fit_suite_selected[suite] == 200 for suite in SUITES)
        and all(fit_task_selected[(suite, str(task))] == 20 for suite in SUITES for task in range(10))
        and not duplicate_manifest_keys
        and not identity_column_mismatches
        and all(row["formal_selected"] or row["split"] != "FIT_TRAIN" for row in rows)
        and full_artifact_audit_pass_count == 800
        and unresolved_provenance_count == 0
        and recovery_unresolved_count == 0
        and unfinished_remediation_count == 0
        and duplicate_selection_count == 0
        and stale_recovery_unresolved_count == 0
    )
    summary = {
        "schema": "OFFICIAL_V3_FORMAL_REGISTRY_SUMMARY_V1",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "identity_count": len(rows),
        "unique_identity_count": len(set(keys)),
        "duplicate_manifest_keys": duplicate_manifest_keys,
        "identity_column_mismatches": sorted(set(identity_column_mismatches)),
        "raw_sealed_count": sum(bool(audits.get(key)) for key in set(keys)),
        "formal_selected_count": sum(bool(row["formal_selected"]) for row in rows),
        "global_formal_selected_count": sum(bool(row["formal_selected"]) for row in rows),
        "fit_formal_selected_count": sum(bool(row["formal_selected"]) for row in fit_rows),
        "task_success_count": sum(row["task_success"] is True for row in rows),
        "task_failure_count": sum(row["task_success"] is False for row in rows),
        "by_split_formal_selected": dict(global_split_selected),
        "by_suite_formal_selected": dict(global_suite_selected),
        "by_task_formal_selected": {f"{suite}/task_{task}": count for (suite, task), count in sorted(global_task_selected.items())},
        "fit_by_suite_formal_selected": dict(fit_suite_selected),
        "fit_by_task_formal_selected": {f"{suite}/task_{task}": count for (suite, task), count in sorted(fit_task_selected.items())},
        "fit_train_missing": max(0, 800 - fit_split_selected["FIT_TRAIN"]),
        "full_artifact_audit_pass_count": full_artifact_audit_pass_count,
        "unresolved_provenance_count": unresolved_provenance_count,
        "unfinished_remediation_count": unfinished_remediation_count,
        "recovery_unresolved_count": recovery_unresolved_count,
        "fit_recovery_unresolved_count": fit_recovery_unresolved_count,
        "recovery_census_sha256": recovery_census_sha256,
        "stale_recovery_unresolved_count": stale_recovery_unresolved_count,
        "stale_recovery_summary_sha256": stale_recovery_summary_sha256,
        "stale_recovery_audit_sha256": stale_recovery_summary_sha256,
        "duplicate_selection_count": duplicate_selection_count,
        "provenance_counts": dict(Counter(str(row.get("provenance_class", "")) for row in rows)),
        "remediation_required_count": sum(bool(row["remediation_required"]) for row in rows),
        "formal_fit_ready": fit_ready,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }
    return rows, summary


def write_registry(rows: list[dict[str, Any]], summary: dict[str, Any], output_root: Path) -> None:
    if output_root.exists():
        raise ValueError(f"refusing to overwrite registry root: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    if staging.exists():
        raise ValueError(f"registry staging root already exists: {staging}")
    try:
        staging.mkdir(parents=True)
        registry = staging / "OFFICIAL_V3_FORMAL_REGISTRY_V1.csv"
        with registry.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in REGISTRY_FIELDS} for row in rows)
        sealed_summary = dict(summary)
        summary_path = staging / "OFFICIAL_V3_FORMAL_REGISTRY_SUMMARY_V1.json"
        sealed_summary["registry_sha256"] = sha256_file(registry)
        summary_path.write_text(json.dumps(sealed_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        sums = staging / "SHA256SUMS"
        sums.write_text(
            f"{sha256_file(registry)}  {registry.name}\n{sha256_file(summary_path)}  {summary_path.name}\n",
            encoding="utf-8",
        )
        (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")
        os.replace(staging, output_root)
    except (OSError, TypeError, ValueError):
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_audit_reports(root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        key = report.get("canonical_parent_key")
        if key:
            report["artifact_audit_path"] = str(path.resolve())
            report["artifact_audit_sha256"] = sha256_file(path)
            result[str(key)].append(report)
    return result


def _read_stale_recovery_audit(path: Path) -> tuple[int, str]:
    """Validate the actual Sprint-0 stale audit schema, not a guessed count."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "OFFICIAL_V3_STALE_LEASE_RECOVERY_AUDIT_V1":
        raise ValueError("unexpected stale recovery audit schema")
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{sha256_file(path)}  {path.name}"
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != expected:
        raise ValueError("stale recovery audit SHA sidecar is missing or invalid")
    if payload.get("status") not in {"RECOVERY_NOT_REQUIRED", "RECOVERY_SAFE"}:
        raise ValueError(f"stale recovery audit is not closed: {payload.get('status')!r}")
    if not isinstance(payload.get("stale_keys"), list):
        raise ValueError("stale recovery audit stale_keys must be a list")
    list_fields = (
        "unexpected_stale_keys", "missing_expected_stale_keys",
        "missing_recovery_records", "unexpected_recovery_records", "duplicate_formal_result_keys",
        "missing_formal_result_keys", "duplicate_active_canonical_keys", "fence_violations",
        "late_result_violations",
    )
    if any(payload.get(field) not in ([], {}) for field in list_fields):
        raise ValueError("stale recovery audit contains unresolved findings")
    runner = payload.get("runner_binding")
    if payload.get("ledger_mutated") is not False or payload.get("formal_training_authorized") is not False or payload.get("formal_attack_authorized") is not False:
        raise ValueError("stale recovery audit authorization boundary is invalid")
    if "official_v3_decision_allowed" in payload and payload["official_v3_decision_allowed"] is not False:
        raise ValueError("stale recovery audit decision boundary is invalid")
    if (
        not isinstance(runner, dict)
        or runner.get("runner_worktree_clean") is not True
        or not re.fullmatch(r"[0-9a-fA-F]{40}", str(runner.get("runner_head", "")))
        or not re.fullmatch(r"[0-9a-fA-F]{64}", str(runner.get("runner_script_sha256", "")))
        or not re.fullmatch(r"[0-9a-fA-F]{64}", str(runner.get("config_sha256", "")))
    ):
        raise ValueError("stale recovery audit runner provenance is incomplete")
    return 0, sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--equivalence-status", choices=("PASS", "HOLD"), default="HOLD")
    parser.add_argument("--remediation", type=Path)
    parser.add_argument("--recovery-census", type=Path)
    parser.add_argument("--stale-recovery-audit", "--stale-recovery-summary", dest="stale_recovery_audit", type=Path)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    remediation = read_csv(args.remediation) if args.remediation else []
    recovery_rows = None
    recovery_census_sha256 = ""
    if args.recovery_census:
        recovery_rows = read_csv(args.recovery_census / "recovery_rows.csv")
        sums = args.recovery_census / "SHA256SUMS"
        if not sums.is_file() or not (args.recovery_census / "SHA256SUMS.sha256").is_file():
            raise ValueError("recovery census checksum bundle is incomplete")
        recovery_census_sha256 = sha256_file(sums)
    stale_count = None
    stale_summary_sha = ""
    if args.stale_recovery_audit:
        stale_count, stale_summary_sha = _read_stale_recovery_audit(args.stale_recovery_audit)
    rows, summary = build_registry(
        read_csv(args.manifest), read_csv(args.ledger), _load_audit_reports(args.audit_root),
        equivalence_status=args.equivalence_status, remediation_rows=remediation,
        expected_identity_count=int(contract["expected_identity_count"]),
        stale_recovery_unresolved_count=stale_count,
        stale_recovery_summary_sha256=stale_summary_sha,
        recovery_rows=recovery_rows,
        recovery_census_sha256=recovery_census_sha256,
    )
    write_registry(rows, summary, args.output_root)
    print(json.dumps({key: summary[key] for key in ("identity_count", "formal_selected_count", "formal_fit_ready")}, sort_keys=True))
    return 0 if summary["formal_fit_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
