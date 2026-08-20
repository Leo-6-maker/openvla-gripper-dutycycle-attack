"""Compact, durable failure-path evidence for strict candidate audits."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_AUDIT_KEYS = (
    "candidate_index",
    "candidate_source",
    "direct_generated_token_ids",
    "clean_arm_token_ids",
    "direct_generated_arm_token_ids",
    "arm_token_ids_equal",
    "arm_mismatch_dimensions",
    "direct_generated_gripper_token_id",
    "direct_generated_gripper_is_native_open",
    "clean_gripper_token_id",
    "clean_gripper_is_native_open",
    "gripper_token_changed",
    "processor_input_sha256",
    "delta_sha256",
    "pixel_budget_adv_inputs_linf",
)


def _compact_diagnostics(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    raw_audit = value.get("candidate_audit")
    if not isinstance(raw_audit, list):
        return {
            "candidate_policy": value.get("candidate_policy"),
            "candidate_audit": [],
            "selected_candidate_index": value.get("selected_candidate_index"),
            "selected_candidate_source": value.get("selected_candidate_source"),
        }
    audit = []
    for row in raw_audit:
        if not isinstance(row, Mapping):
            return {
                "candidate_policy": value.get("candidate_policy"),
                "candidate_audit": [],
                "selected_candidate_index": value.get("selected_candidate_index"),
                "selected_candidate_source": value.get("selected_candidate_source"),
            }
        audit.append({key: row.get(key) for key in _AUDIT_KEYS})
    return {
        "candidate_policy": value.get("candidate_policy"),
        "candidate_audit": audit,
        "selected_candidate_index": value.get("selected_candidate_index"),
        "selected_candidate_source": value.get("selected_candidate_source"),
    }


def _audit_complete(diagnostics: Mapping[str, Any] | None, expected_count: int) -> bool:
    if diagnostics is None:
        return False
    audit = diagnostics.get("candidate_audit")
    if not isinstance(audit, list) or len(audit) != expected_count:
        return False
    if any(any(row.get(key) is None for key in _AUDIT_KEYS) for row in audit):
        return False
    expected_sources = ["delta0", *(f"pgd_iteration_{index}" for index in range(1, expected_count))]
    return [row.get("candidate_index") for row in audit] == list(range(expected_count)) and [
        row.get("candidate_source") for row in audit
    ] == expected_sources


def build_failure_evidence(exc: BaseException, attacker: Any, *, expected_count: int = 6) -> dict[str, Any]:
    exception_diagnostics = _compact_diagnostics(getattr(exc, "diagnostics", None))
    adapter = getattr(attacker, "adapter", None)
    attacker_raw = getattr(adapter, "last_attack_diagnostics", None) if adapter is not None else None
    if attacker_raw is None:
        attacker_raw = getattr(attacker, "last_attack_diagnostics", None)
    attacker_diagnostics = _compact_diagnostics(attacker_raw)

    both_sources = exception_diagnostics is not None and attacker_diagnostics is not None
    sources_equal = exception_diagnostics == attacker_diagnostics if both_sources else None
    if both_sources and not sources_equal:
        canonical = None
        consistency = "HOLD_DIAGNOSTICS_SOURCE_DISAGREEMENT"
    else:
        canonical = exception_diagnostics or attacker_diagnostics
        consistency = (
            "PASS"
            if both_sources
            else "EXCEPTION_ONLY"
            if exception_diagnostics is not None
            else "ATTACKER_ONLY"
            if attacker_diagnostics is not None
            else "NOT_AVAILABLE"
        )

    audit = canonical.get("candidate_audit", []) if canonical is not None else []
    return {
        "failure_path_evidence_persistence": "STAGE_X_X1R2_Q3R3_E1_V1",
        "selector_error_type": type(exc).__name__,
        "selector_error_message": str(exc),
        "candidate_policy": canonical.get("candidate_policy") if canonical is not None else None,
        "candidate_audit_complete": _audit_complete(canonical, expected_count),
        "expected_candidate_count": int(expected_count),
        "observed_candidate_count": len(audit),
        "candidate_audit": audit,
        "selected_candidate_index": canonical.get("selected_candidate_index") if canonical is not None else None,
        "selected_candidate_source": canonical.get("selected_candidate_source") if canonical is not None else None,
        "diagnostics_source_exception": exception_diagnostics,
        "diagnostics_source_attacker": attacker_diagnostics,
        "diagnostics_sources_equal": sources_equal,
        "diagnostics_consistency_status": consistency,
    }


def write_failure_receipt(path: Path, failure: Mapping[str, Any], exc: BaseException, attacker: Any) -> dict[str, Any]:
    receipt = dict(failure)
    receipt.update(build_failure_evidence(exc, attacker))
    if receipt["diagnostics_consistency_status"] == "HOLD_DIAGNOSTICS_SOURCE_DISAGREEMENT":
        receipt["status"] = "HOLD_Q3R3_D_FAILURE_DIAGNOSTICS_INCONSISTENT"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return receipt
