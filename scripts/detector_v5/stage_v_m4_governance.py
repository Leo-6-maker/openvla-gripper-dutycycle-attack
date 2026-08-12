"""Shared fail-closed governance for formal M4 authorization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
SPLITS = ("TRAIN", "VAL", "TEST")
REQUIRED_SUITE_COUNTS = {suite: 10 for suite in SUITES}
REQUIRED_SPLIT_COUNTS = {"TRAIN": 24, "VAL": 8, "TEST": 8}
REQUIRED_PER_SUITE_SPLIT_COUNTS = {suite: {"TRAIN": 6, "VAL": 2, "TEST": 2} for suite in SUITES}


class M4GovernanceError(ValueError):
    """A formal M4 artifact is not consumable under current governance."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise M4GovernanceError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _bound_path(value: Any, protocol_path: Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (protocol_path.parents[1] / path).resolve()


def protocol_declares_corridor_gate(protocol: Mapping[str, Any]) -> bool:
    inputs = protocol.get("inputs")
    if not isinstance(inputs, Mapping):
        return False
    required = (
        "supersession_hold_path",
        "supersession_hold_sha256",
        "formal_parent_manifest_path",
        "formal_parent_manifest_sha256",
        "corridor_pass_receipt_path",
        "corridor_pass_receipt_sha256",
        "corridor_qualification_protocol_sha256",
        "corridor_qualification_authorization_sha256",
        "corridor_reconciliation_sha256",
    )
    return all(isinstance(inputs.get(key), str) and inputs[key] for key in required)


def _parent_keys(value: Mapping[str, Any], field: str) -> list[str]:
    rows = value.get(field)
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise M4GovernanceError(f"M4_{field.upper()}_INVALID")
    keys = [str(row.get("canonical_parent_key", "")) for row in rows]
    if len(keys) != len(set(keys)) or any(not key for key in keys):
        raise M4GovernanceError(f"M4_{field.upper()}_DUPLICATE_OR_EMPTY")
    return keys


def _counts(rows: list[Mapping[str, Any]]) -> tuple[dict[str, int], dict[str, int], dict[str, dict[str, int]]]:
    suites = {suite: 0 for suite in SUITES}
    splits = {split: 0 for split in SPLITS}
    by_suite = {suite: {split: 0 for split in SPLITS} for suite in SUITES}
    for row in rows:
        suite, split = str(row.get("suite", "")), str(row.get("split", ""))
        if suite not in suites or split not in splits:
            raise M4GovernanceError("M4_PARENT_SUITE_OR_SPLIT_INVALID")
        suites[suite] += 1
        splits[split] += 1
        by_suite[suite][split] += 1
    return suites, splits, by_suite


def validate_formal_m4_corridor_gate(
    protocol: Mapping[str, Any],
    *,
    protocol_path: Path,
    split_path: Path,
    source_commit: str,
    source_tree: str,
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require a new corridor PASS receipt before formal M4 is consumable."""
    inputs = protocol.get("inputs")
    if not isinstance(inputs, Mapping) or not inputs.get("supersession_hold_path"):
        raise M4GovernanceError("M4_FORMAL_AUTHORIZATION_SUPERSEDED_BY_CORRIDOR_HOLD")
    if not protocol_declares_corridor_gate(protocol):
        raise M4GovernanceError("M4_CORRIDOR_GATE_BINDING_INCOMPLETE")

    hold_path = _bound_path(inputs["supersession_hold_path"], protocol_path)
    if not hold_path.is_file() or sha256(hold_path) != inputs["supersession_hold_sha256"]:
        raise M4GovernanceError("M4_SUPERSESSION_HOLD_BINDING_INVALID")
    hold = _load(hold_path)
    if hold.get("schema") != "STAGE_V_M4_FORMAL_AUTHORIZATION_SUPERSESSION_HOLD_V1" or hold.get("status") != "HOLD_FORMAL_M4_CORRIDOR_INSUFFICIENT":
        raise M4GovernanceError("M4_SUPERSESSION_HOLD_STATUS_INVALID")
    if hold.get("stable_parent_count") != 29 or hold.get("required_parent_count") != 40 or hold.get("protected_counters") != COUNTERS:
        raise M4GovernanceError("M4_SUPERSESSION_HOLD_BINDING_INVALID")
    stable_keys = hold.get("exact_stable_parent_keys")
    if not isinstance(stable_keys, list) or len(stable_keys) != 29 or len(set(stable_keys)) != 29:
        raise M4GovernanceError("M4_SUPERSESSION_STABLE_KEYS_INVALID")
    if hold.get("required_suite_counts") != REQUIRED_SUITE_COUNTS or hold.get("required_split_counts") != REQUIRED_SPLIT_COUNTS or hold.get("required_per_suite_split_counts") != REQUIRED_PER_SUITE_SPLIT_COUNTS:
        raise M4GovernanceError("M4_SUPERSESSION_POPULATION_CONTRACT_INVALID")

    manifest_path = _bound_path(inputs["formal_parent_manifest_path"], protocol_path)
    if not manifest_path.is_file() or sha256(manifest_path) != inputs["formal_parent_manifest_sha256"]:
        raise M4GovernanceError("M4_FORMAL_MANIFEST_BINDING_INVALID")
    manifest = _load(manifest_path)
    manifest_keys = _parent_keys(manifest, "parents")
    if len(manifest_keys) != 40:
        raise M4GovernanceError("M4_FORMAL_MANIFEST_COUNT_INVALID")
    manifest_suites, manifest_splits, _ = _counts(manifest.get("parents", []))
    if manifest_suites != REQUIRED_SUITE_COUNTS or manifest_splits != REQUIRED_SPLIT_COUNTS:
        raise M4GovernanceError("M4_FORMAL_MANIFEST_SPLIT_INVALID")

    if not split_path.is_file() or sha256(split_path) != inputs.get("formal_parent_split_sha256"):
        raise M4GovernanceError("M4_FORMAL_SPLIT_BINDING_INVALID")
    split = _load(split_path)
    split_keys = _parent_keys(split, "parents")
    if split_keys != manifest_keys:
        raise M4GovernanceError("M4_FORMAL_MANIFEST_SPLIT_KEY_MISMATCH")
    split_suites, split_counts, per_suite = _counts(split.get("parents", []))
    if split_suites != REQUIRED_SUITE_COUNTS or split_counts != REQUIRED_SPLIT_COUNTS or per_suite != REQUIRED_PER_SUITE_SPLIT_COUNTS:
        raise M4GovernanceError("M4_FORMAL_SPLIT_POPULATION_INVALID")

    receipt_path = _bound_path(inputs["corridor_pass_receipt_path"], protocol_path)
    if not receipt_path.is_file() or sha256(receipt_path) != inputs["corridor_pass_receipt_sha256"]:
        raise M4GovernanceError("M4_CORRIDOR_PASS_RECEIPT_BINDING_INVALID")
    receipt = _load(receipt_path)
    if receipt.get("schema") != "STAGE_V_M4_CORRIDOR_PASS_RECEIPT_V1" or receipt.get("status") != "PASS_FORMAL_M4_CORRIDOR":
        raise M4GovernanceError("M4_CORRIDOR_PASS_RECEIPT_NOT_PASS")
    if receipt.get("parent_count") != 40 or receipt.get("parent_keys") != sorted(manifest_keys):
        raise M4GovernanceError("M4_CORRIDOR_PASS_PARENT_SET_INVALID")
    if receipt.get("formal_parent_manifest_sha256") != inputs["formal_parent_manifest_sha256"]:
        raise M4GovernanceError("M4_CORRIDOR_PASS_MANIFEST_BINDING_INVALID")
    if receipt.get("formal_split_sha256") != inputs["formal_parent_split_sha256"]:
        raise M4GovernanceError("M4_CORRIDOR_PASS_SPLIT_BINDING_INVALID")
    if receipt.get("suite_counts") != REQUIRED_SUITE_COUNTS or receipt.get("split_counts") != REQUIRED_SPLIT_COUNTS or receipt.get("per_suite_split_counts") != REQUIRED_PER_SUITE_SPLIT_COUNTS:
        raise M4GovernanceError("M4_CORRIDOR_PASS_POPULATION_INVALID")
    if receipt.get("protocol_sha256") != inputs["corridor_qualification_protocol_sha256"] or receipt.get("authorization_sha256") != inputs["corridor_qualification_authorization_sha256"] or receipt.get("reconciliation_sha256") != inputs["corridor_reconciliation_sha256"]:
        raise M4GovernanceError("M4_CORRIDOR_PASS_UPSTREAM_BINDING_INVALID")
    if receipt.get("source_commit") != source_commit or receipt.get("source_tree") != source_tree or receipt.get("protected_counters") != COUNTERS:
        raise M4GovernanceError("M4_CORRIDOR_PASS_SOURCE_OR_BOUNDARY_INVALID")
    if authorization is not None:
        if authorization.get("corridor_pass_receipt_sha256") != inputs["corridor_pass_receipt_sha256"]:
            raise M4GovernanceError("M4_AUTHORIZATION_CORRIDOR_PASS_BINDING_INVALID")
        if authorization.get("corridor_pass_receipt") != str(receipt_path):
            raise M4GovernanceError("M4_AUTHORIZATION_CORRIDOR_PASS_PATH_INVALID")

    return {
        "hold_path": str(hold_path),
        "hold_sha256": sha256(hold_path),
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
        "parent_keys": sorted(manifest_keys),
        "suite_counts": dict(REQUIRED_SUITE_COUNTS),
        "split_counts": dict(REQUIRED_SPLIT_COUNTS),
    }
