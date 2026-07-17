"""Preparation-only Official V3 attack manifests and audit contracts."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


ATTACK_CONDITIONS = (
    "CLEAN", "R9Q_DETECTOR_T10", "RAND_VALID_T10", "COMMAND_OPEN_ORACLE",
    "DETECTOR_SHUFFLED_GRAD_T10", "R9Q_GRIPPER_ONLY_T10",
)


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value.lower()


def build_attack_manifest(
    parent_rows: Iterable[dict[str, Any]], *, protocol_sha256: str, check_status: str,
    cs200_manifest_sha256: str, check_report_sha256: str, checkpoint_sha256: str, calibration_sha256: str,
) -> dict[str, Any]:
    rows = [dict(row) for row in parent_rows]
    if check_status != "CHECK_PASS":
        raise ValueError("attack manifest requires a sealed CHECK_PASS decision")
    protocol_sha256 = _sha(protocol_sha256, "protocol_sha256")
    cs200_manifest_sha256 = _sha(cs200_manifest_sha256, "cs200_manifest_sha256")
    check_report_sha256 = _sha(check_report_sha256, "check_report_sha256")
    checkpoint_sha256 = _sha(checkpoint_sha256, "checkpoint_sha256")
    calibration_sha256 = _sha(calibration_sha256, "calibration_sha256")
    if len(rows) != 200:
        raise ValueError(f"attack manifest requires exactly 200 CS200 parents, got {len(rows)}")
    keys = [str(row.get("canonical_parent_key", "")) for row in rows]
    if len(set(keys)) != len(keys) or any(not key for key in keys):
        raise ValueError("attack manifest contains duplicate parent identities")
    task_counts = Counter()
    for row, key in zip(rows, keys):
        parts = key.split("/")
        if len(parts) != 3 or not parts[1].startswith("task_") or not parts[2].startswith("state_"):
            raise ValueError(f"invalid CS200 parent identity: {key}")
        try:
            task = int(row.get("task_idx"))
            state = int(row.get("state_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CS200 parent identity columns are invalid: {key}") from exc
        if key != f"{row.get('suite')}/task_{task:02d}/state_{state:02d}" or not 0 <= task < 10 or not 30 <= state < 50:
            raise ValueError(f"CS200 parent identity columns do not match key: {key}")
        if str(row.get("task_success", row.get("success", ""))).lower() not in {"true", "1"}:
            raise ValueError(f"CS200 parent is not a verified clean success: {key}")
        task_counts[f"{row.get('suite')}/task_{task:02d}"] += 1
    expected_tasks = {f"{suite}/task_{task:02d}" for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10") for task in range(10)}
    if set(task_counts) != expected_tasks or any(count != 5 for count in task_counts.values()):
        raise ValueError("CS200 must contain exactly 5 state-30-49 parents per task")
    cells = [{"canonical_parent_key": key, "condition": condition, "exact_t10": condition != "CLEAN", "attack_enabled": False} for key in sorted(keys) for condition in ATTACK_CONDITIONS]
    return {
        "schema": "B3_OFFICIAL_V3_ATTACK_MANIFEST_V1",
        "parent_count": len(keys),
        "cell_count": len(cells),
        "conditions": list(ATTACK_CONDITIONS),
        "cells": cells,
        "protocol_sha256": protocol_sha256,
        "cs200_manifest_sha256": cs200_manifest_sha256,
        "check_report_sha256": check_report_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_sha256": calibration_sha256,
        "status": "PREPARATION_ONLY",
        "formal_attack_authorized": False,
        "attack_execution_authorized": False,
    }


def audit_attack_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") != "B3_OFFICIAL_V3_ATTACK_MANIFEST_V1" or manifest.get("formal_attack_authorized") is not False or manifest.get("attack_execution_authorized") is not False:
        raise ValueError("attack manifest authorization boundary failed")
    cells = manifest.get("cells", [])
    if manifest.get("parent_count") != 200 or not all(_sha(manifest.get(name), name) for name in ("protocol_sha256", "cs200_manifest_sha256", "check_report_sha256", "checkpoint_sha256", "calibration_sha256")):
        raise ValueError("attack manifest is not bound to the complete CS200/CHECK/CAL bundle")
    if len(cells) != manifest.get("parent_count", 0) * len(ATTACK_CONDITIONS):
        raise ValueError("attack manifest cell count mismatch")
    counts = Counter(str(cell.get("condition")) for cell in cells)
    if set(counts) != set(ATTACK_CONDITIONS) or len(set((cell.get("canonical_parent_key"), cell.get("condition")) for cell in cells)) != len(cells):
        raise ValueError("attack manifest condition/identity closure failed")
    parent_keys = {str(cell.get("canonical_parent_key")) for cell in cells}
    task_counts = Counter("/".join(key.split("/")[:2]) for key in parent_keys)
    if len(parent_keys) != 200 or set(task_counts) != {f"{suite}/task_{task:02d}" for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10") for task in range(10)} or any(count != 5 for count in task_counts.values()):
        raise ValueError("attack manifest parent task quota is not closed")
    return {"schema": "B3_OFFICIAL_V3_ATTACK_MANIFEST_AUDIT_V1", "status": "PASS_PREPARATION_ONLY", "cell_count": len(cells), "formal_attack_authorized": False}


__all__ = ["ATTACK_CONDITIONS", "build_attack_manifest", "audit_attack_manifest"]
