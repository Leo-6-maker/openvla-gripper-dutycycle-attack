"""Preparation-only Official V3 attack manifests and audit contracts."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .b3_formal import json_sha


ATTACK_CONDITIONS = (
    "CLEAN", "R9Q_DETECTOR_T10", "RAND_VALID_T10", "COMMAND_OPEN_ORACLE",
    "DETECTOR_SHUFFLED_GRAD_T10", "R9Q_GRIPPER_ONLY_T10",
)


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value.lower()


CS200_SELECTION_ALGORITHM = "FIRST_5_VERIFIED_CLEAN_SUCCESS_BY_CANONICAL_KEY_V1"


def _load_cs200_source(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"CS200 source manifest is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"canonical_parent_key", "suite", "task_idx", "state_id", "task_success", "formal_selected", "selection_rank", "selection_algorithm", "selection_order_sha256"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("CS200 source manifest is missing frozen selection fields")
    if len(rows) != 200:
        raise ValueError("CS200 source manifest must contain exactly 200 selected rows")
    expected_tasks = {f"{suite}/task_{task:02d}" for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10") for task in range(10)}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row["canonical_parent_key"])
        task_key = f"{row['suite']}/task_{int(row['task_idx']):02d}"
        if task_key not in expected_tasks or key != f"{task_key}/state_{int(row['state_id']):02d}":
            raise ValueError(f"CS200 source identity columns do not match: {key}")
        if str(row["task_success"]).lower() not in {"true", "1"} or str(row["formal_selected"]).lower() not in {"true", "1"}:
            raise ValueError(f"CS200 source row is not a verified selected success: {key}")
        if not 30 <= int(row["state_id"]) < 50 or row["selection_algorithm"] != CS200_SELECTION_ALGORITHM or not _sha(row["selection_order_sha256"], "selection_order_sha256"):
            raise ValueError(f"CS200 source selection contract failed: {key}")
        groups.setdefault(task_key, []).append(row)
    if set(groups) != expected_tasks or any(sorted(int(row["selection_rank"]) for row in group) != list(range(5)) for group in groups.values()):
        raise ValueError("CS200 source must contain ranks 0..4 for every task")
    ordered_keys = [str(row["canonical_parent_key"]) for row in sorted(rows, key=lambda row: (str(row["suite"]), int(row["task_idx"]), int(row["selection_rank"]))) ]
    order_sha = json_sha(ordered_keys)
    if any(row["selection_order_sha256"].lower() != order_sha for row in rows):
        raise ValueError("CS200 source selection order SHA does not match the sealed rows")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows, digest, order_sha


def build_attack_manifest(
    parent_rows: Iterable[dict[str, Any]], *, protocol_sha256: str, check_status: str,
    cs200_manifest_sha256: str, check_report_sha256: str, checkpoint_sha256: str, calibration_sha256: str,
    cs200_manifest_path: Path | None = None,
) -> dict[str, Any]:
    rows = [dict(row) for row in parent_rows]
    if check_status != "CHECK_PASS":
        raise ValueError("attack manifest requires a sealed CHECK_PASS decision")
    protocol_sha256 = _sha(protocol_sha256, "protocol_sha256")
    cs200_manifest_sha256 = _sha(cs200_manifest_sha256, "cs200_manifest_sha256")
    check_report_sha256 = _sha(check_report_sha256, "check_report_sha256")
    checkpoint_sha256 = _sha(checkpoint_sha256, "checkpoint_sha256")
    calibration_sha256 = _sha(calibration_sha256, "calibration_sha256")
    if cs200_manifest_path is None:
        raise ValueError("strict CS200 manifest construction requires --cs200-manifest")
    source_rows, source_sha, order_sha = _load_cs200_source(cs200_manifest_path)
    if cs200_manifest_sha256.lower() != source_sha:
        raise ValueError("cs200_manifest_sha256 does not match the sealed CS200 source manifest")
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
    source_by_task = {f"{row['suite']}/task_{int(row['task_idx']):02d}": [] for row in source_rows}
    for row in source_rows:
        source_by_task[f"{row['suite']}/task_{int(row['task_idx']):02d}"].append(row)
    for task_key, source_group in source_by_task.items():
        selected_keys = {key for key in keys if key.startswith(task_key + "/")}
        expected_keys = {str(row["canonical_parent_key"]) for row in source_group}
        if selected_keys != expected_keys:
            raise ValueError(f"CS200 parents do not match sealed first-five source rows: {task_key}")
    cells = [{"canonical_parent_key": key, "condition": condition, "exact_t10": condition != "CLEAN", "attack_enabled": False} for key in sorted(keys) for condition in ATTACK_CONDITIONS]
    return {
        "schema": "B3_OFFICIAL_V3_ATTACK_MANIFEST_V1",
        "parent_count": len(keys),
        "cell_count": len(cells),
        "conditions": list(ATTACK_CONDITIONS),
        "cells": cells,
        "protocol_sha256": protocol_sha256,
        "cs200_manifest_sha256": cs200_manifest_sha256,
        "cs200_source_manifest_verified": True,
        "cs200_selection_algorithm": CS200_SELECTION_ALGORITHM,
        "cs200_selection_order_sha256": order_sha,
        "check_report_sha256": check_report_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_sha256": calibration_sha256,
        "status": "PREPARATION_ONLY",
        "formal_attack_authorized": False,
        "attack_execution_authorized": False,
    }


def audit_attack_manifest(manifest: dict[str, Any], *, cs200_manifest_path: Path | None = None) -> dict[str, Any]:
    if manifest.get("schema") != "B3_OFFICIAL_V3_ATTACK_MANIFEST_V1" or manifest.get("formal_attack_authorized") is not False or manifest.get("attack_execution_authorized") is not False:
        raise ValueError("attack manifest authorization boundary failed")
    cells = manifest.get("cells", [])
    if manifest.get("parent_count") != 200 or not all(_sha(manifest.get(name), name) for name in ("protocol_sha256", "cs200_manifest_sha256", "check_report_sha256", "checkpoint_sha256", "calibration_sha256")):
        raise ValueError("attack manifest is not bound to the complete CS200/CHECK/CAL bundle")
    if manifest.get("cs200_source_manifest_verified") is not True or manifest.get("cs200_selection_algorithm") != CS200_SELECTION_ALGORITHM or not _sha(manifest.get("cs200_selection_order_sha256"), "cs200_selection_order_sha256"):
        raise ValueError("attack manifest lacks strict CS200 selection provenance")
    if cs200_manifest_path is not None:
        _, source_sha, order_sha = _load_cs200_source(cs200_manifest_path)
        if source_sha != manifest["cs200_manifest_sha256"] or order_sha != manifest["cs200_selection_order_sha256"]:
            raise ValueError("attack manifest does not match the supplied sealed CS200 source")
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


__all__ = ["ATTACK_CONDITIONS", "CS200_SELECTION_ALGORITHM", "build_attack_manifest", "audit_attack_manifest"]
