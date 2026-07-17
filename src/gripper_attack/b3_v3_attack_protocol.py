"""Preparation-only Official V3 attack manifests and audit contracts."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


ATTACK_CONDITIONS = (
    "CLEAN", "R9Q_DETECTOR_T10", "RAND_VALID_T10", "COMMAND_OPEN_ORACLE",
    "DETECTOR_SHUFFLED_GRAD_T10", "R9Q_GRIPPER_ONLY_T10",
)


def build_attack_manifest(parent_rows: Iterable[dict[str, Any]], *, protocol_sha256: str, check_status: str) -> dict[str, Any]:
    rows = [dict(row) for row in parent_rows]
    if check_status != "CHECK_PASS" or not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64 or any(char not in "0123456789abcdefABCDEF" for char in protocol_sha256):
        raise ValueError("attack manifest requires a sealed CHECK_PASS decision")
    if not rows:
        raise ValueError("attack manifest requires parents")
    keys = [str(row.get("canonical_parent_key", "")) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("attack manifest contains duplicate parent identities")
    cells = [{"canonical_parent_key": key, "condition": condition, "exact_t10": condition != "CLEAN", "attack_enabled": False} for key in sorted(keys) for condition in ATTACK_CONDITIONS]
    return {
        "schema": "B3_OFFICIAL_V3_ATTACK_MANIFEST_V1",
        "parent_count": len(keys),
        "cell_count": len(cells),
        "conditions": list(ATTACK_CONDITIONS),
        "cells": cells,
        "protocol_sha256": protocol_sha256,
        "status": "PREPARATION_ONLY",
        "formal_attack_authorized": False,
        "attack_execution_authorized": False,
    }


def audit_attack_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") != "B3_OFFICIAL_V3_ATTACK_MANIFEST_V1" or manifest.get("formal_attack_authorized") is not False or manifest.get("attack_execution_authorized") is not False:
        raise ValueError("attack manifest authorization boundary failed")
    cells = manifest.get("cells", [])
    if len(cells) != manifest.get("parent_count", 0) * len(ATTACK_CONDITIONS):
        raise ValueError("attack manifest cell count mismatch")
    counts = Counter(str(cell.get("condition")) for cell in cells)
    if set(counts) != set(ATTACK_CONDITIONS) or len(set((cell.get("canonical_parent_key"), cell.get("condition")) for cell in cells)) != len(cells):
        raise ValueError("attack manifest condition/identity closure failed")
    return {"schema": "B3_OFFICIAL_V3_ATTACK_MANIFEST_AUDIT_V1", "status": "PASS_PREPARATION_ONLY", "cell_count": len(cells), "formal_attack_authorized": False}


__all__ = ["ATTACK_CONDITIONS", "build_attack_manifest", "audit_attack_manifest"]
