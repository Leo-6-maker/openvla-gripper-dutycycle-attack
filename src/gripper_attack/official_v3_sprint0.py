"""Provenance-only Sprint 0 audits for the Official V3 corpus.

This module deliberately does not open rollout artifacts, Teacher files,
Detector outputs, or attack results.  The bridge audit consumes a narrow
provenance inventory so that legacy execution evidence can be classified
without accidentally turning a historical metadata field into measured V3
generation evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


class Sprint0ContractViolation(ValueError):
    """Raised when a Sprint 0 input cannot be audited fail-closed."""


BRIDGE_SCHEMA = "OFFICIAL_V3_LEGACY_START_BRIDGE_AUDIT_V1"
REMEDIATION_SCHEMA = "OFFICIAL_V3_FIT_REMEDIATION_QUEUE_V1"
STALE_LEASE_SCHEMA = "OFFICIAL_V3_STALE_LEASE_RECOVERY_AUDIT_V1"

BRIDGE_PASS = "BRIDGE_PASS"
BRIDGE_HOLD = "HOLD"
EXACT_REMEDIATION_REQUIRED = "EXACT_REMEDIATION_REQUIRED"

_ALLOWED_PROVENANCE_FIELDS = {
    "canonical_parent_key",
    "artifact_recursive_sha256",
    "artifact_schema",
    "runtime_valid",
    "source_contract_pass",
    "no_teacher_attack_files",
    "official_action_adapter",
    "official_horizon",
    "num_steps_wait",
    "collector_head",
    "worker_script_sha256",
    "adapter_sha256",
    "protocol_sha256",
    "model_tree_sha256",
    "processor_sha256",
    "processor_tree_sha256",
    "feature_order_sha256",
    "action_postprocess_sha256",
    "env_init_sha256",
    "generation_evidence",
    "legacy_metadata_generation_passes_per_step",
    "worker_start_manifest_present",
    "provenance_class",
    "active",
    "reason",
    "source_artifact_root",
}

_FORBIDDEN_PROVENANCE_TERMS = (
    "teacher",
    "detector",
    "attack",
    "trigger",
    "label",
    "outcome",
    "success",
    "failure",
    "sr",
)
_ALLOWED_SCOPE_FIELDS = {"no_teacher_attack_files"}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "pass"}


def _sha_text(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_key(row: Mapping[str, Any]) -> str:
    key = str(row.get("canonical_parent_key", "")).strip()
    if not key or "/task_" not in key or "/state_" not in key:
        raise Sprint0ContractViolation(f"invalid canonical_parent_key: {key!r}")
    return key


def _validate_provenance_columns(rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        for name in row:
            lowered = str(name).lower()
            if name not in _ALLOWED_PROVENANCE_FIELDS:
                raise Sprint0ContractViolation(
                    f"non-provenance field is not allowed in bridge input: {name}"
                )
            if name not in _ALLOWED_SCOPE_FIELDS and any(term in lowered for term in _FORBIDDEN_PROVENANCE_TERMS):
                raise Sprint0ContractViolation(
                    f"bridge input contains forbidden scientific-result field: {name}"
                )


def _validate_baseline(baseline: Mapping[str, Any]) -> None:
    if baseline.get("schema") != "OFFICIAL_V3_LEGACY_BRIDGE_BASELINE_V1":
        raise Sprint0ContractViolation("unexpected legacy bridge baseline schema")
    if baseline.get("status") != "FROZEN_PROVENANCE_ONLY":
        raise Sprint0ContractViolation("legacy bridge baseline is not frozen provenance-only")
    expected = baseline.get("expected_values")
    fields = baseline.get("bridge_fields")
    if not isinstance(expected, dict) or not isinstance(fields, list) or not fields:
        raise Sprint0ContractViolation("bridge baseline requires expected_values and bridge_fields")
    for field in fields:
        if field not in _ALLOWED_PROVENANCE_FIELDS or field not in expected:
            raise Sprint0ContractViolation(f"invalid bridge comparison field: {field}")
        if field not in _ALLOWED_SCOPE_FIELDS and any(term in str(field).lower() for term in _FORBIDDEN_PROVENANCE_TERMS):
            raise Sprint0ContractViolation(f"forbidden bridge comparison field: {field}")


def _compare_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return actual in expected
    if isinstance(expected, bool):
        return _truthy(actual) is expected
    if isinstance(expected, int):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def audit_legacy_bridge(
    inventory_rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    *,
    expected_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Classify legacy execution evidence using provenance fields only.

    ``LEGACY_METADATA_ONLY`` may pass the 25D bridge, but it is explicitly not
    an Official V3 measured-generation candidate.  It remains eligible only
    for a separately named legacy 25D pilot.  Missing or contradictory
    execution provenance is never repaired by this function.
    """

    _validate_baseline(baseline)
    _validate_provenance_columns(inventory_rows)
    expected = set(expected_keys or ())
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    bridge_fields = list(baseline["bridge_fields"])
    expected_values = baseline["expected_values"]
    for source in inventory_rows:
        key = _require_key(source)
        if key in seen:
            raise Sprint0ContractViolation(f"duplicate bridge identity: {key}")
        seen.add(key)
        reasons: list[str] = []
        status = BRIDGE_PASS
        if expected and key not in expected:
            status = EXACT_REMEDIATION_REQUIRED
            reasons.append("IDENTITY_NOT_IN_FROZEN_BASELINE")
        if _truthy(source.get("active")):
            status = EXACT_REMEDIATION_REQUIRED
            reasons.append("ACTIVE_IDENTITY_CANNOT_BE_BRIDGED")
        if not _truthy(source.get("runtime_valid")):
            status = EXACT_REMEDIATION_REQUIRED
            reasons.append("RUNTIME_INVALID")
        for field in bridge_fields:
            if source.get(field, "") in (None, ""):
                if status == BRIDGE_PASS:
                    status = BRIDGE_HOLD
                reasons.append(f"MISSING_{field.upper()}")
            elif not _compare_value(source.get(field), expected_values[field]):
                status = EXACT_REMEDIATION_REQUIRED
                reasons.append(f"MISMATCH_{field.upper()}")
        evidence = str(source.get("generation_evidence", ""))
        if evidence not in {"MEASURED_SINGLE_GENERATION", "LEGACY_METADATA_ONLY"}:
            status = EXACT_REMEDIATION_REQUIRED
            reasons.append("GENERATION_EVIDENCE_NOT_CLASSIFIED")
        if status == BRIDGE_PASS and evidence == "LEGACY_METADATA_ONLY":
            reasons.append("LEGACY_GENERATION_METADATA_NOT_MEASURED")
        if status == BRIDGE_PASS and not _truthy(source.get("no_teacher_attack_files")):
            status = EXACT_REMEDIATION_REQUIRED
            reasons.append("SOURCE_SCOPE_NOT_CLEAN")
        formal_v3 = status == BRIDGE_PASS and evidence == "MEASURED_SINGLE_GENERATION" and _truthy(
            source.get("worker_start_manifest_present")
        )
        rows.append(
            {
                "canonical_parent_key": key,
                "bridge_status": status,
                "generation_evidence": evidence,
                "legacy_25d_pilot_eligible": status == BRIDGE_PASS,
                "official_v3_formal_eligible": formal_v3,
                "remediation_required": status == EXACT_REMEDIATION_REQUIRED,
                "reason": ";".join(dict.fromkeys(reasons)) or "PROVENANCE_FIELDS_MATCH_FROZEN_BASELINE",
                "source_artifact_recursive_sha256": source.get("artifact_recursive_sha256", ""),
                "provenance_class": source.get("provenance_class", ""),
            }
        )
    missing = sorted(expected - seen)
    for key in missing:
        rows.append(
            {
                "canonical_parent_key": key,
                "bridge_status": EXACT_REMEDIATION_REQUIRED,
                "generation_evidence": "MISSING",
                "legacy_25d_pilot_eligible": False,
                "official_v3_formal_eligible": False,
                "remediation_required": True,
                "reason": "IDENTITY_MISSING_FROM_BRIDGE_INVENTORY",
                "source_artifact_recursive_sha256": "",
                "provenance_class": "",
            }
        )
    counts = Counter(row["bridge_status"] for row in rows)
    overall = BRIDGE_PASS
    if counts[EXACT_REMEDIATION_REQUIRED]:
        overall = EXACT_REMEDIATION_REQUIRED
    elif counts[BRIDGE_HOLD]:
        overall = BRIDGE_HOLD
    return {
        "schema": BRIDGE_SCHEMA,
        "created_at": _now(),
        "policy": "PROVENANCE_ONLY_NO_TEACHER_NO_DETECTOR_NO_ATTACK",
        "overall_status": overall,
        "identity_count": len(rows),
        "bridge_pass_count": counts[BRIDGE_PASS],
        "hold_count": counts[BRIDGE_HOLD],
        "exact_remediation_required_count": counts[EXACT_REMEDIATION_REQUIRED],
        "legacy_25d_pilot_eligible_count": sum(row["legacy_25d_pilot_eligible"] for row in rows),
        "official_v3_formal_eligible_count": sum(row["official_v3_formal_eligible"] for row in rows),
        "records": rows,
        "teacher_labels_read": False,
        "detector_outputs_read": False,
        "attack_results_read": False,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }


def _manifest_rank(row: Mapping[str, Any]) -> tuple[int, str]:
    try:
        return int(row.get("queue_rank", row.get("rank", 10**12))), _require_key(row)
    except (TypeError, ValueError):
        return 10**12, _require_key(row)


def build_fit_remediation_queue(
    manifest_rows: list[dict[str, str]],
    bridge_report: dict[str, Any],
    ledger_rows: list[dict[str, str]],
    *,
    queue_epoch_id: str,
    formal_registry_rows: list[dict[str, Any]] | None = None,
    include_statuses: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build an exact-identity FIT remediation queue without result selection."""

    if not queue_epoch_id or queue_epoch_id.strip() != queue_epoch_id:
        raise Sprint0ContractViolation("queue_epoch_id must be non-empty and stable")
    manifest_by_key: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        key = _require_key(row)
        if key in manifest_by_key:
            raise Sprint0ContractViolation(f"duplicate canonical manifest identity: {key}")
        manifest_by_key[key] = row
    bridge_by_key: dict[str, dict[str, Any]] = {}
    for row in bridge_report.get("records", []):
        key = _require_key(row)
        if key in bridge_by_key:
            raise Sprint0ContractViolation(f"duplicate bridge identity: {key}")
        bridge_by_key[key] = row
    active = {
        _require_key(row)
        for row in ledger_rows
        if row.get("status") in {"LEASED", "RUNNING"} and (row.get("canonical_parent_key") or row.get("cell_id"))
    }
    formal = {
        _require_key(row)
        for row in (formal_registry_rows or [])
        if _truthy(row.get("formal_selected"))
    }
    statuses = include_statuses or {BRIDGE_HOLD, EXACT_REMEDIATION_REQUIRED}
    selected: list[tuple[dict[str, str], dict[str, Any]]] = []
    for key, bridge in bridge_by_key.items():
        manifest = manifest_by_key.get(key)
        if not manifest:
            raise Sprint0ContractViolation(f"remediation identity absent from canonical manifest: {key}")
        if manifest.get("split", "") != "FIT_TRAIN":
            continue
        if bridge.get("bridge_status") not in statuses:
            continue
        if key in active:
            raise Sprint0ContractViolation(f"active identity cannot enter remediation queue: {key}")
        if key in formal:
            raise Sprint0ContractViolation(f"formal-selected identity cannot require remediation: {key}")
        selected.append((manifest, bridge))
    selected.sort(key=lambda pair: _manifest_rank(pair[0]))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest, bridge in selected:
        key = _require_key(manifest)
        if key in seen:
            raise Sprint0ContractViolation(f"duplicate remediation identity: {key}")
        seen.add(key)
        rows.append(
            {
                "schema": REMEDIATION_SCHEMA,
                "queue_epoch_id": queue_epoch_id,
                "canonical_parent_key": key,
                "suite": manifest.get("suite", ""),
                "task_idx": manifest.get("task_idx", ""),
                "state_id": manifest.get("state_id", ""),
                "split": manifest.get("split", ""),
                "source_bridge_status": bridge.get("bridge_status", ""),
                "source_provenance_class": bridge.get("provenance_class", ""),
                "superseded_artifact_sha256": bridge.get("source_artifact_recursive_sha256", ""),
                "remediation_required": True,
                "replacement_identity_policy": "EXACT_SAME_CANONICAL_IDENTITY",
                "lease_status": "PENDING",
                "fencing_token": "",
                "selection_reason": "PROVENANCE_ONLY_REMEDIATION",
                "attack_outcome_considered": False,
                "teacher_outcome_considered": False,
            }
        )
    summary = {
        "schema": "OFFICIAL_V3_FIT_REMEDIATION_QUEUE_SUMMARY_V1",
        "created_at": _now(),
        "queue_epoch_id": queue_epoch_id,
        "identity_count": len(rows),
        "unique_identity_count": len(seen),
        "fit_only": True,
        "exact_identity_replacement_only": True,
        "active_identity_count": len(active & seen),
        "formal_selected_conflicts": len(formal & seen),
        "teacher_outcomes_read": False,
        "attack_outcomes_read": False,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }
    return rows, summary


def _timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "")
    try:
        return float(text)
    except ValueError:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def audit_stale_lease_recovery(
    ledger_rows: list[dict[str, str]],
    process_rows: list[dict[str, Any]],
    formal_result_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    *,
    now_epoch: float,
    stale_after_seconds: float = 600.0,
    expected_stale_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Audit stale-lease fencing without mutating the ledger.

    A recovery record must introduce a new lease UUID and a strictly newer
    integer lease epoch.  Late results are required to be quarantined.  The
    auditor does not reclaim a lease or write a result.
    """

    live_pids = {str(row.get("pid")) for row in process_rows if _truthy(row.get("alive"))}
    active_rows = [row for row in ledger_rows if row.get("status") in {"LEASED", "RUNNING"}]
    stale_rows: list[dict[str, str]] = []
    for row in active_rows:
        pid = str(row.get("pid", ""))
        try:
            age = float(now_epoch) - _timestamp(row.get("lease_timestamp"))
        except (TypeError, ValueError, OverflowError):
            age = -1.0
        if age > stale_after_seconds and pid not in live_pids:
            stale_rows.append(row)
    stale_keys = sorted({_require_key(row) for row in stale_rows})
    expected = set(expected_stale_keys or ())
    unexpected = sorted(set(stale_keys) - expected) if expected else []
    missing_expected = sorted(expected - set(stale_keys)) if expected else []
    formal_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in formal_result_rows:
        if _truthy(row.get("formal_selected")) or row.get("formal_result_sha256"):
            formal_by_key[_require_key(row)].append(row)
    duplicate_formal_results = sorted(key for key, rows in formal_by_key.items() if len(rows) != 1)
    missing_formal_results = sorted(key for key in stale_keys if key not in formal_by_key)
    recovery_by_key: dict[str, dict[str, Any]] = {}
    fence_violations: list[str] = []
    for row in recovery_rows:
        key = _require_key(row)
        if key in recovery_by_key:
            fence_violations.append(f"DUPLICATE_RECOVERY_RECORD:{key}")
            continue
        recovery_by_key[key] = row
        old = next((lease for lease in stale_rows if _require_key(lease) == key), None)
        try:
            old_epoch = int(row.get("old_lease_epoch_id"))
            new_epoch = int(row.get("new_lease_epoch_id"))
        except (TypeError, ValueError):
            fence_violations.append(f"INVALID_EPOCH:{key}")
            continue
        if old is None or str(row.get("old_lease_uuid")) != str(old.get("lease_uuid")):
            fence_violations.append(f"OLD_LEASE_MISMATCH:{key}")
        if not row.get("new_lease_uuid") or row.get("new_lease_uuid") == row.get("old_lease_uuid"):
            fence_violations.append(f"LEASE_UUID_NOT_ROTATED:{key}")
        if new_epoch <= old_epoch:
            fence_violations.append(f"LEASE_EPOCH_NOT_ADVANCED:{key}")
        if not row.get("fencing_token"):
            fence_violations.append(f"MISSING_FENCING_TOKEN:{key}")
        if str(row.get("late_result_policy", "")).upper() != "QUARANTINE":
            fence_violations.append(f"LATE_RESULT_NOT_QUARANTINED:{key}")
    missing_recovery = sorted(set(stale_keys) - set(recovery_by_key))
    unexpected_recovery = sorted(set(recovery_by_key) - set(stale_keys))
    hold = bool(
        unexpected
        or missing_expected
        or duplicate_formal_results
        or missing_formal_results
        or fence_violations
        or missing_recovery
        or unexpected_recovery
    )
    status = "RECOVERY_NOT_REQUIRED" if not stale_keys else ("HOLD" if hold else "RECOVERY_SAFE")
    return {
        "schema": STALE_LEASE_SCHEMA,
        "created_at": _now(),
        "status": status,
        "stale_after_seconds": stale_after_seconds,
        "stale_keys": stale_keys,
        "unexpected_stale_keys": unexpected,
        "missing_expected_stale_keys": missing_expected,
        "missing_recovery_records": missing_recovery,
        "unexpected_recovery_records": unexpected_recovery,
        "duplicate_formal_result_keys": duplicate_formal_results,
        "missing_formal_result_keys": missing_formal_results,
        "fence_violations": fence_violations,
        "late_result_policy": "QUARANTINE",
        "ledger_mutated": False,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Sprint0ContractViolation(f"JSON object required: {path}")
    return value


def write_sealed_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise Sprint0ContractViolation(f"refusing to overwrite sealed output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.exists():
        raise Sprint0ContractViolation(f"refusing to overwrite checksum sidecar: {sidecar}")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")


def write_sealed_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise Sprint0ContractViolation(f"refusing to overwrite sealed output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.exists():
        raise Sprint0ContractViolation(f"refusing to overwrite checksum sidecar: {sidecar}")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")


__all__ = [
    "BRIDGE_HOLD",
    "BRIDGE_PASS",
    "EXACT_REMEDIATION_REQUIRED",
    "Sprint0ContractViolation",
    "audit_legacy_bridge",
    "audit_stale_lease_recovery",
    "build_fit_remediation_queue",
    "read_csv_rows",
    "read_json",
    "write_sealed_csv",
    "write_sealed_json",
]
