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
import io
import json
import os
import re
import shutil
import subprocess
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .official_v3_contract import SUITES, expected_split


class Sprint0ContractViolation(ValueError):
    """Raised when a Sprint 0 input cannot be audited fail-closed."""


BRIDGE_SCHEMA = "OFFICIAL_V3_LEGACY_START_BRIDGE_AUDIT_V1"
REMEDIATION_SCHEMA = "OFFICIAL_V3_FIT_REMEDIATION_QUEUE_V1"
STALE_LEASE_SCHEMA = "OFFICIAL_V3_STALE_LEASE_RECOVERY_AUDIT_V1"
REMEDIATION_FIELDS = [
    "schema", "queue_epoch_id", "canonical_parent_key", "suite", "task_idx", "state_id", "split",
    "source_bridge_status", "official_v3_disposition", "source_provenance_class",
    "superseded_artifact_sha256", "remediation_required", "replacement_identity_policy", "lease_status",
    "fencing_token", "selection_reason", "attack_outcome_considered", "teacher_outcome_considered",
]

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_key(row: Mapping[str, Any]) -> str:
    key = str(row.get("canonical_parent_key", "")).strip()
    parse_canonical_parent_key(key)
    return key


def parse_canonical_parent_key(key: str) -> tuple[str, int, int]:
    """Parse and validate the only canonical identity form accepted by V3."""

    match = re.fullmatch(r"([^/]+)/task_(\d{2})/state_(\d{2})", str(key).strip())
    if not match:
        raise Sprint0ContractViolation(f"invalid canonical_parent_key: {key!r}")
    suite, task_text, state_text = match.groups()
    task_idx, state_id = int(task_text), int(state_text)
    if suite not in SUITES or not 0 <= task_idx < 10 or not 0 <= state_id < 50:
        raise Sprint0ContractViolation(f"canonical identity is outside Official V3 universe: {key!r}")
    return suite, task_idx, state_id


def _input_binding(path: Path, *, schema: str, row_count: int | None = None, identity_count: int | None = None) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "schema": schema,
        "row_count": row_count,
        "identity_count": identity_count,
    }


def _git_output(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Sprint0ContractViolation(f"Git provenance command failed in {repo}: {' '.join(args)}") from exc
    return result.stdout.strip()


def _repo_relative(repo: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise Sprint0ContractViolation(f"path is outside runner repository: {path}") from exc


def _runner_binding(
    *,
    runner_repo: Path,
    expected_runner_head: str,
    config_path: Path,
    runner_script_path: Path,
) -> dict[str, Any]:
    """Read runner provenance from Git; caller-supplied status is not trusted."""

    if not re.fullmatch(r"[0-9a-fA-F]{40}", expected_runner_head):
        raise Sprint0ContractViolation("expected_runner_head must be a full 40-character Git SHA")
    repo = runner_repo.resolve()
    actual_head = _git_output(repo, "rev-parse", "HEAD")
    if actual_head.lower() != expected_runner_head.lower():
        raise Sprint0ContractViolation(
            f"runner HEAD mismatch: expected {expected_runner_head}, actual {actual_head}"
        )
    status = _git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise Sprint0ContractViolation("formal Sprint 0 audit requires an actually clean runner worktree")
    script = runner_script_path.resolve()
    config = config_path.resolve()
    script_rel = _repo_relative(repo, script)
    config_rel = _repo_relative(repo, config)
    if not script.is_file() or not config.is_file():
        raise Sprint0ContractViolation("runner script and config must exist in the runner repository")
    try:
        tracked_script = _git_output(repo, "ls-files", "--error-unmatch", "--", script_rel)
        tracked_config = _git_output(repo, "ls-files", "--error-unmatch", "--", config_rel)
        script_blob = _git_output(repo, "rev-parse", f"HEAD:{script_rel}")
        config_blob = _git_output(repo, "rev-parse", f"HEAD:{config_rel}")
    except Sprint0ContractViolation as exc:
        raise Sprint0ContractViolation("runner script/config must be tracked at expected HEAD") from exc
    return {
        "runner_repo": str(repo),
        "runner_head": actual_head,
        "expected_runner_head": expected_runner_head,
        "runner_worktree_clean": True,
        "runner_script_path": str(script),
        "runner_script_git_path": tracked_script,
        "runner_script_sha256": sha256_file(script),
        "runner_script_git_blob_sha1": script_blob,
        "config_path": str(config),
        "config_git_path": tracked_config,
        "config_sha256": sha256_file(config),
        "config_git_blob_sha1": config_blob,
    }


def _attach_run_binding(report: dict[str, Any], *, inputs: dict[str, Any], runner: dict[str, Any]) -> dict[str, Any]:
    report["input_snapshots"] = inputs
    report["runner_binding"] = runner
    report["official_v3_decision_allowed"] = False
    return report


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
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Classify legacy execution evidence using provenance fields only.

    ``LEGACY_METADATA_ONLY`` may pass the 25D bridge, but it is explicitly not
    an Official V3 measured-generation candidate.  It remains eligible only
    for a separately named legacy 25D pilot.  Missing or contradictory
    execution provenance is never repaired by this function.
    """

    _validate_baseline(baseline)
    _validate_provenance_columns(inventory_rows)
    if expected_keys is None and not allow_partial:
        raise Sprint0ContractViolation("formal bridge audit requires a frozen expected identity set")
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
        if formal_v3:
            disposition = "PASS_FORMAL_CANDIDATE"
        elif status == BRIDGE_PASS and evidence == "LEGACY_METADATA_ONLY":
            disposition = EXACT_REMEDIATION_REQUIRED
            reasons.append("OFFICIAL_V3_REQUIRES_EXACT_REMEDIATION")
        else:
            disposition = status
        rows.append(
            {
                "canonical_parent_key": key,
                "bridge_status": status,
                "legacy_pilot_bridge_status": status,
                "official_v3_disposition": disposition,
                "generation_evidence": evidence,
                "legacy_25d_pilot_eligible": status == BRIDGE_PASS,
                "official_v3_formal_eligible": formal_v3,
                "remediation_required": disposition == EXACT_REMEDIATION_REQUIRED,
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
                "legacy_pilot_bridge_status": EXACT_REMEDIATION_REQUIRED,
                "official_v3_disposition": EXACT_REMEDIATION_REQUIRED,
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
    disposition_counts = Counter(row["official_v3_disposition"] for row in rows)
    overall = BRIDGE_PASS
    if counts[EXACT_REMEDIATION_REQUIRED]:
        overall = EXACT_REMEDIATION_REQUIRED
    elif counts[BRIDGE_HOLD]:
        overall = BRIDGE_HOLD
    if disposition_counts[EXACT_REMEDIATION_REQUIRED]:
        official_overall = EXACT_REMEDIATION_REQUIRED
    elif disposition_counts[BRIDGE_HOLD]:
        official_overall = BRIDGE_HOLD
    elif disposition_counts["PASS_FORMAL_CANDIDATE"] != len(rows) or allow_partial:
        official_overall = BRIDGE_HOLD
    else:
        official_overall = "PASS"
    return {
        "schema": BRIDGE_SCHEMA,
        "created_at": _now(),
        "policy": "PROVENANCE_ONLY_NO_TEACHER_NO_DETECTOR_NO_ATTACK",
        "overall_status": overall,
        "legacy_pilot_bridge_overall_status": overall,
        "official_v3_overall_status": official_overall,
        "identity_count": len(rows),
        "bridge_pass_count": counts[BRIDGE_PASS],
        "hold_count": counts[BRIDGE_HOLD],
        "legacy_bridge_exact_remediation_count": counts[EXACT_REMEDIATION_REQUIRED],
        "official_v3_exact_remediation_required_count": disposition_counts[EXACT_REMEDIATION_REQUIRED],
        "exact_remediation_required_count": disposition_counts[EXACT_REMEDIATION_REQUIRED],
        "official_v3_disposition_counts": dict(disposition_counts),
        "legacy_25d_pilot_eligible_count": sum(row["legacy_25d_pilot_eligible"] for row in rows),
        "official_v3_formal_eligible_count": sum(row["official_v3_formal_eligible"] for row in rows),
        "records": rows,
        "teacher_labels_read": False,
        "detector_outputs_read": False,
        "attack_results_read": False,
        "partial_inventory_allowed": allow_partial,
        "official_v3_decision_allowed": False,
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
    expected_identity_count: int = 2000,
    expected_fit_count: int = 800,
    expected_per_suite: int = 200,
    expected_per_task: int = 20,
    expected_suite_count: int = 4,
    expected_task_count: int = 40,
    input_snapshots: dict[str, Any] | None = None,
    runner_binding: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build an exact-identity FIT remediation queue without result selection."""

    if not queue_epoch_id or queue_epoch_id.strip() != queue_epoch_id:
        raise Sprint0ContractViolation("queue_epoch_id must be non-empty and stable")
    manifest_by_key: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        key = _require_key(row)
        suite, task_idx, state_id = parse_canonical_parent_key(key)
        try:
            row_identity = (str(row.get("suite")), int(row.get("task_idx")), int(row.get("state_id")))
        except (TypeError, ValueError) as exc:
            raise Sprint0ContractViolation(f"manifest execution columns are invalid: {key}") from exc
        if row_identity != (suite, task_idx, state_id):
            raise Sprint0ContractViolation(f"manifest columns do not match canonical identity: {key}")
        if row.get("split") != expected_split(state_id):
            raise Sprint0ContractViolation(f"manifest split does not match canonical state: {key}")
        if key in manifest_by_key:
            raise Sprint0ContractViolation(f"duplicate canonical manifest identity: {key}")
        manifest_by_key[key] = row
    if len(manifest_by_key) != expected_identity_count:
        raise Sprint0ContractViolation(
            f"canonical manifest must contain {expected_identity_count} identities, got {len(manifest_by_key)}"
        )
    fit_manifest = [row for row in manifest_by_key.values() if row.get("split") == "FIT_TRAIN"]
    if len(fit_manifest) != expected_fit_count:
        raise Sprint0ContractViolation(f"FIT manifest count mismatch: expected {expected_fit_count}, got {len(fit_manifest)}")
    suite_counts = Counter(str(row.get("suite")) for row in fit_manifest)
    if any(count != expected_per_suite for count in suite_counts.values()) or len(suite_counts) != expected_suite_count:
        raise Sprint0ContractViolation(f"FIT suite quota mismatch: {dict(suite_counts)}")
    task_counts = Counter((str(row.get("suite")), int(row.get("task_idx"))) for row in fit_manifest)
    if any(count != expected_per_task for count in task_counts.values()) or len(task_counts) != expected_task_count:
        raise Sprint0ContractViolation("FIT task quota mismatch")
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
    selected: list[tuple[dict[str, str], dict[str, Any], str]] = []
    for key, bridge in bridge_by_key.items():
        manifest = manifest_by_key.get(key)
        if not manifest:
            raise Sprint0ContractViolation(f"remediation identity absent from canonical manifest: {key}")
        if manifest.get("split", "") != "FIT_TRAIN":
            continue
        disposition = bridge.get("official_v3_disposition", bridge.get("bridge_status"))
        if disposition not in statuses:
            continue
        if key in active:
            raise Sprint0ContractViolation(f"active identity cannot enter remediation queue: {key}")
        if key in formal:
            raise Sprint0ContractViolation(f"formal-selected identity cannot require remediation: {key}")
        selected.append((manifest, bridge, disposition))
    selected.sort(key=lambda triple: _manifest_rank(triple[0]))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest, bridge, disposition in selected:
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
                "official_v3_disposition": disposition,
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
        "input_snapshots": input_snapshots or {},
        "runner_binding": runner_binding or {},
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
    late_quarantine_rows: list[dict[str, Any]] | None = None,
    input_snapshots: dict[str, Any] | None = None,
    runner_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit stale-lease fencing without mutating the ledger.

    A recovery record must introduce a new lease UUID and a strictly newer
    integer lease epoch.  Late results are required to be quarantined.  The
    auditor does not reclaim a lease or write a result.
    """

    live_pids = {str(row.get("pid")) for row in process_rows if _truthy(row.get("alive"))}
    active_rows = [row for row in ledger_rows if row.get("status") in {"LEASED", "RUNNING"}]
    active_key_counts = Counter(_require_key(row) for row in active_rows)
    duplicate_active_keys = sorted(key for key, count in active_key_counts.items() if count > 1)
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
    results_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in formal_result_rows:
        results_by_key[_require_key(row)].append(row)
    selected_by_key: dict[str, list[dict[str, Any]]] = {
        key: [row for row in rows if _truthy(row.get("formal_selected"))]
        for key, rows in results_by_key.items()
    }
    duplicate_formal_results = sorted(key for key in stale_keys if len(selected_by_key.get(key, [])) > 1)
    missing_formal_results = sorted(key for key in stale_keys if len(selected_by_key.get(key, [])) == 0)
    quarantine_by_key_sha: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in late_quarantine_rows or []:
        quarantine_by_key_sha[(_require_key(row), str(row.get("artifact_sha256", "")))].append(row)
    late_result_violations: list[str] = []
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
            ledger_old_epoch = int(old.get("lease_epoch_id")) if old is not None else -1
            record_old_epoch = int(row.get("old_lease_epoch_id"))
            new_epoch = int(row.get("new_lease_epoch_id"))
        except (TypeError, ValueError):
            fence_violations.append(f"INVALID_EPOCH:{key}")
            continue
        if old is None or str(row.get("old_lease_uuid")) != str(old.get("lease_uuid")):
            fence_violations.append(f"OLD_LEASE_MISMATCH:{key}")
        if old is None or record_old_epoch != ledger_old_epoch:
            fence_violations.append(f"OLD_LEASE_EPOCH_MISMATCH:{key}")
        if not row.get("new_lease_uuid") or row.get("new_lease_uuid") == row.get("old_lease_uuid"):
            fence_violations.append(f"LEASE_UUID_NOT_ROTATED:{key}")
        if new_epoch <= ledger_old_epoch:
            fence_violations.append(f"LEASE_EPOCH_NOT_ADVANCED:{key}")
        if not row.get("fencing_token"):
            fence_violations.append(f"MISSING_FENCING_TOKEN:{key}")
        if str(row.get("late_result_policy", "")).upper() != "QUARANTINE":
            fence_violations.append(f"LATE_RESULT_NOT_QUARANTINED:{key}")
        selected = selected_by_key.get(key, [])
        if len(selected) == 1:
            result = selected[0]
            if not result.get("formal_result_sha256"):
                late_result_violations.append(f"SELECTED_RESULT_SHA_MISSING:{key}")
            if str(result.get("lease_uuid")) != str(row.get("new_lease_uuid")):
                late_result_violations.append(f"SELECTED_RESULT_LEASE_UUID_MISMATCH:{key}")
            try:
                result_epoch = int(result.get("lease_epoch_id"))
                expected_epoch = int(row.get("new_lease_epoch_id"))
            except (TypeError, ValueError):
                late_result_violations.append(f"SELECTED_RESULT_EPOCH_INVALID:{key}")
            else:
                if result_epoch != expected_epoch:
                    late_result_violations.append(f"SELECTED_RESULT_EPOCH_MISMATCH:{key}")
            if str(result.get("fencing_token")) != str(row.get("fencing_token")):
                late_result_violations.append(f"SELECTED_RESULT_FENCING_TOKEN_MISMATCH:{key}")
        for result in results_by_key.get(key, []):
            if _truthy(result.get("formal_selected")) or not result.get("formal_result_sha256"):
                continue
            artifact_sha = str(result.get("formal_result_sha256"))
            old_lease = next((lease for lease in stale_rows if _require_key(lease) == key), None)
            if old_lease is None or str(result.get("lease_uuid")) != str(old_lease.get("lease_uuid")):
                late_result_violations.append(f"LATE_RESULT_OLD_LEASE_UUID_MISMATCH:{key}:{artifact_sha}")
            try:
                result_old_epoch = int(result.get("lease_epoch_id"))
                expected_old_epoch = int(old_lease.get("lease_epoch_id")) if old_lease is not None else -1
            except (TypeError, ValueError):
                late_result_violations.append(f"LATE_RESULT_OLD_EPOCH_INVALID:{key}:{artifact_sha}")
            else:
                if result_old_epoch != expected_old_epoch:
                    late_result_violations.append(f"LATE_RESULT_OLD_EPOCH_MISMATCH:{key}:{artifact_sha}")
            quarantine = quarantine_by_key_sha.get((key, artifact_sha), [])
            if len(quarantine) != 1:
                late_result_violations.append(f"LATE_RESULT_NOT_QUARANTINED:{key}:{artifact_sha}")
                continue
            record = quarantine[0]
            if _truthy(record.get("formal_selected")):
                late_result_violations.append(f"QUARANTINE_MARKED_FORMAL:{key}:{artifact_sha}")
            if str(record.get("old_lease_uuid")) != str(row.get("old_lease_uuid")):
                late_result_violations.append(f"QUARANTINE_OLD_UUID_MISMATCH:{key}:{artifact_sha}")
            if str(record.get("old_lease_epoch_id")) != str(row.get("old_lease_epoch_id")):
                late_result_violations.append(f"QUARANTINE_OLD_EPOCH_MISMATCH:{key}:{artifact_sha}")
            if not str(record.get("quarantine_reason", "")).strip():
                late_result_violations.append(f"QUARANTINE_REASON_MISSING:{key}:{artifact_sha}")
    missing_recovery = sorted(set(stale_keys) - set(recovery_by_key))
    unexpected_recovery = sorted(set(recovery_by_key) - set(stale_keys))
    hold = bool(
        unexpected
        or missing_expected
        or duplicate_formal_results
        or missing_formal_results
        or duplicate_active_keys
        or fence_violations
        or late_result_violations
        or missing_recovery
        or unexpected_recovery
    )
    if hold:
        status = "HOLD"
    elif not stale_keys:
        status = "RECOVERY_NOT_REQUIRED"
    else:
        status = "RECOVERY_SAFE"
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
        "duplicate_active_canonical_keys": duplicate_active_keys,
        "fence_violations": fence_violations,
        "late_result_violations": late_result_violations,
        "late_result_policy": "QUARANTINE",
        "ledger_mutated": False,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
        "input_snapshots": input_snapshots or {},
        "runner_binding": runner_binding or {},
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Sprint0ContractViolation(f"JSON object required: {path}")
    return value


def _atomic_write_pair(path: Path, body: bytes) -> None:
    """Write a file and checksum without overwriting or leaving our temp files."""

    sidecar = path.with_name(path.name + ".sha256")
    if path.exists() or sidecar.exists():
        raise Sprint0ContractViolation(f"refusing to overwrite sealed output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temp_body = path.with_name(f".{path.name}.{token}.tmp")
    temp_sidecar = sidecar.with_name(f".{sidecar.name}.{token}.tmp")
    digest = hashlib.sha256(body).hexdigest()
    moved_body = False
    try:
        with temp_body.open("wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        sidecar_body = f"{digest}  {path.name}\n".encode("utf-8")
        with temp_sidecar.open("wb") as handle:
            handle.write(sidecar_body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_body, path)
        moved_body = True
        os.replace(temp_sidecar, sidecar)
    except OSError as exc:
        for temp in (temp_body, temp_sidecar):
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
        if moved_body:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise Sprint0ContractViolation(f"atomic sealed output failed: {path}") from exc


def write_sealed_json(path: Path, payload: dict[str, Any]) -> None:
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_pair(path, body)


def write_sealed_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    fields = fieldnames or (list(rows[0]) if rows else [])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    _atomic_write_pair(path, buffer.getvalue().encode("utf-8"))


def write_sealed_remediation_bundle(output_root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Seal queue, summary, and aggregate checksums as one new directory."""

    if output_root.exists():
        raise Sprint0ContractViolation(f"refusing to overwrite sealed output root: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    if staging.exists():
        raise Sprint0ContractViolation(f"staging output already exists: {staging}")
    queue_name = "OFFICIAL_V3_FIT_REMEDIATION_QUEUE_V1.csv"
    summary_name = "OFFICIAL_V3_FIT_REMEDIATION_QUEUE_SUMMARY_V1.json"
    try:
        staging.mkdir()
        queue_path = staging / queue_name
        write_sealed_csv(queue_path, rows, fieldnames=REMEDIATION_FIELDS)
        sealed_summary = dict(summary)
        sealed_summary["queue_csv_sha256"] = sha256_file(queue_path)
        summary_path = staging / summary_name
        write_sealed_json(summary_path, sealed_summary)
        aggregate_names = [queue_name, f"{queue_name}.sha256", summary_name, f"{summary_name}.sha256"]
        aggregate_body = "".join(
            f"{sha256_file(staging / name)}  {name}\n" for name in aggregate_names
        ).encode("utf-8")
        _atomic_write_pair(staging / "SHA256SUMS", aggregate_body)
        os.replace(staging, output_root)
    except (OSError, Sprint0ContractViolation):
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "BRIDGE_HOLD",
    "BRIDGE_PASS",
    "EXACT_REMEDIATION_REQUIRED",
    "REMEDIATION_FIELDS",
    "Sprint0ContractViolation",
    "audit_legacy_bridge",
    "audit_stale_lease_recovery",
    "build_fit_remediation_queue",
    "read_csv_rows",
    "read_json",
    "parse_canonical_parent_key",
    "sha256_file",
    "write_sealed_csv",
    "write_sealed_json",
    "write_sealed_remediation_bundle",
]
