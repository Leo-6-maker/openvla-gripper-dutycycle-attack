#!/usr/bin/env python3
"""Build the immutable Official V3 identity registry from audit reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gripper_attack.official_v3_contract import PASS_STATUSES, SUITES, load_contract, sha256_file


REGISTRY_FIELDS = [
    "canonical_parent_key", "suite", "task_idx", "state_id", "split", "ledger_status", "task_success",
    "selected_artifact_root", "selected_artifact_recursive_sha256", "artifact_audit_sha256",
    "provenance_class", "worker_start_manifest_sha256", "collector_head", "worker_id", "gpu_id",
    "worker_script_sha256", "adapter_sha256", "protocol_sha256", "model_tree_sha256", "processor_tree_sha256",
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audits = _audit_map(audit_reports)
    remediation = {row.get("canonical_parent_key", ""): row for row in (remediation_rows or [])}
    ledger: dict[str, dict[str, str]] = {}
    for row in ledger_rows:
        key = row.get("canonical_parent_key", "")
        if not key and row.get("cell_id", "").startswith("CLEAN|"):
            key = row["cell_id"][len("CLEAN|"):]
        if key:
            ledger[key] = row
    keys = [row.get("canonical_parent_key", "") for row in manifest_rows]
    duplicate_manifest_keys = sorted(key for key, count in Counter(keys).items() if not key or count > 1)
    rows: list[dict[str, Any]] = []
    for manifest in manifest_rows:
        key = manifest.get("canonical_parent_key", "")
        ledger_row = ledger.get(key, {})
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
        })
        if key in duplicate_manifest_keys:
            base["selection_reason"] = "DUPLICATE_MANIFEST_CANONICAL_KEY"
        elif len(candidates) != 1:
            base["selection_reason"] = "NO_AUDIT_CANDIDATE" if not candidates else "DUPLICATE_AUDIT_CANDIDATE"
        else:
            audit = candidates[0]
            base.update({field: audit.get(field, "") for field in REGISTRY_FIELDS if field in audit})
            base["canonical_parent_key"] = key
            base["task_success"] = base["task_success"] if base["task_success"] is not None else audit.get("task_success", "")
            eligible = audit.get("status") in PASS_STATUSES and bool(audit.get("formal_eligible"))
            if audit.get("provenance_class") == "B_PREVIOUS_HEAD_EQUIVALENT" and equivalence_status != "PASS":
                eligible = False
                base["selection_reason"] = "OLD_HEAD_EQUIVALENCE_HOLD"
            elif eligible:
                base["selection_reason"] = "UNIQUE_AUDIT_AND_PROVENANCE_PASS"
            else:
                base["selection_reason"] = audit.get("status", "AUDIT_HOLD")
            base["formal_eligible"] = eligible
            base["formal_selected"] = eligible and key not in duplicate_manifest_keys
        rows.append(base)

    split_selected = Counter(row["split"] for row in rows if row["formal_selected"])
    suite_selected = Counter(row["suite"] for row in rows if row["formal_selected"])
    task_selected = Counter((row["suite"], str(row["task_idx"])) for row in rows if row["formal_selected"])
    fit_rows = [row for row in rows if row["split"] == "FIT_TRAIN"]
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
    duplicate_selection_count = len(duplicate_manifest_keys)
    fit_ready = (
        len(rows) == expected_identity_count
        and len(set(keys)) == expected_identity_count
        and split_selected["FIT_TRAIN"] == 800
        and all(suite_selected[suite] == 200 for suite in SUITES)
        and all(task_selected[(suite, str(task))] == 20 for suite in SUITES for task in range(10))
        and not duplicate_manifest_keys
        and all(row["formal_selected"] or row["split"] != "FIT_TRAIN" for row in rows)
        and full_artifact_audit_pass_count == 800
        and unresolved_provenance_count == 0
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
        "raw_sealed_count": sum(bool(audits.get(key)) for key in set(keys)),
        "formal_selected_count": sum(bool(row["formal_selected"]) for row in rows),
        "task_success_count": sum(row["task_success"] is True for row in rows),
        "task_failure_count": sum(row["task_success"] is False for row in rows),
        "by_split_formal_selected": dict(split_selected),
        "by_suite_formal_selected": dict(suite_selected),
        "fit_train_missing": max(0, 800 - split_selected["FIT_TRAIN"]),
        "full_artifact_audit_pass_count": full_artifact_audit_pass_count,
        "unresolved_provenance_count": unresolved_provenance_count,
        "unfinished_remediation_count": unfinished_remediation_count,
        "stale_recovery_unresolved_count": stale_recovery_unresolved_count,
        "stale_recovery_summary_sha256": stale_recovery_summary_sha256,
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
    output_root.mkdir(parents=True)
    registry = output_root / "OFFICIAL_V3_FORMAL_REGISTRY_V1.csv"
    with registry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in REGISTRY_FIELDS} for row in rows)
    summary_path = output_root / "OFFICIAL_V3_FORMAL_REGISTRY_SUMMARY_V1.json"
    summary["registry_sha256"] = sha256_file(registry)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sums = output_root / "SHA256SUMS"
    sums.write_text(
        f"{sha256_file(registry)}  {registry.name}\n{sha256_file(summary_path)}  {summary_path.name}\n",
        encoding="utf-8",
    )
    (output_root / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def _load_audit_reports(root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        key = report.get("canonical_parent_key")
        if key:
            result[str(key)].append(report)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--equivalence-status", choices=("PASS", "HOLD"), default="HOLD")
    parser.add_argument("--remediation", type=Path)
    parser.add_argument("--stale-recovery-summary", type=Path)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    remediation = read_csv(args.remediation) if args.remediation else []
    stale_count = None
    stale_summary_sha = ""
    if args.stale_recovery_summary:
        stale_payload = json.loads(args.stale_recovery_summary.read_text(encoding="utf-8"))
        stale_count = int(stale_payload["unresolved_count"])
        stale_summary_sha = sha256_file(args.stale_recovery_summary)
    rows, summary = build_registry(
        read_csv(args.manifest), read_csv(args.ledger), _load_audit_reports(args.audit_root),
        equivalence_status=args.equivalence_status, remediation_rows=remediation,
        expected_identity_count=int(contract["expected_identity_count"]),
        stale_recovery_unresolved_count=stale_count,
        stale_recovery_summary_sha256=stale_summary_sha,
    )
    write_registry(rows, summary, args.output_root)
    print(json.dumps({key: summary[key] for key in ("identity_count", "formal_selected_count", "formal_fit_ready")}, sort_keys=True))
    return 0 if summary["formal_fit_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
