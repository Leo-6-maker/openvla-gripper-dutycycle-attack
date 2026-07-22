#!/usr/bin/env python3
"""Validate the non-self-referential Factorized V2 handoff binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_SPLITS = tuple(f"o{outer}_i{inner}" for outer in range(4) for inner in range(3))
RECEIPT_BINDING_DEFINITION = "SHA256(canonical receipt JSON with handoff_blob_sha256 omitted)"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"DUPLICATE_JSON_KEY:{key}")
        value[key] = item
    return value


def _strict_json(text: str) -> Any:
    """Parse JSON without allowing silent last-key-wins overwrites."""
    return json.loads(text, object_pairs_hook=_reject_duplicate_keys)


def _strict_read_json(path: Path) -> Any:
    return _strict_json(path.read_text(encoding="utf-8"))


def canonical_handoff_sha(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("handoff_blob_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V2":
        raise ValueError("HANDOFF_SCHEMA")
    if "full_head" in value:
        raise ValueError("SELF_REFERENTIAL_FULL_HEAD_FORBIDDEN")
    for key in ("code_snapshot_commit", "metadata_parent_commit"):
        if not SHA40.fullmatch(str(value.get(key, ""))):
            raise ValueError(f"{key.upper()}_MUST_BE_FULL_SHA")
    handoff_sha = str(value.get("handoff_blob_sha256", ""))
    if not SHA64.fullmatch(handoff_sha) or handoff_sha != canonical_handoff_sha(value):
        raise ValueError("HANDOFF_BLOB_SHA_MISMATCH")
    if value.get("status") != "READY_FOR_DEEPSEEK_STATIC_INTEGRATION":
        raise ValueError("HANDOFF_NOT_STATIC_READY")
    return {
        "status": "PASS",
        "schema": value["schema"],
        "code_snapshot_commit": value["code_snapshot_commit"],
        "metadata_parent_commit": value["metadata_parent_commit"],
        "handoff_blob_sha256": handoff_sha,
    }


def _repo_relative(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("UNSAFE_REFERENCE_PATH")
    target = (root / value).resolve(strict=True)
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("REFERENCE_OUTSIDE_REPO") from exc
    current = root.resolve()
    for part in Path(value).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("SYMLINK_REFERENCE_FORBIDDEN")
    if not target.is_file():
        raise ValueError("REFERENCE_NOT_REGULAR_FILE")
    return target


def _references(value: Any):
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            yield value
        for child in value.values():
            yield from _references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _references(child)


def _canonical_reference_sha(path: Path) -> str:
    """Hash text references with stable LF bytes across Windows and CI."""
    data = path.read_bytes()
    if path.suffix.lower() in {".json", ".csv", ".py", ".yml", ".yaml", ".md", ".schema"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def receipt_binding_sha(path: Path) -> str:
    """Hash receipt metadata without the one-way handoff payload binding."""
    receipt = _strict_read_json(path)
    if not isinstance(receipt, dict) or receipt.get("schema") != "FACTORIZED_V3_1_HANDOFF_RECEIPT_V1":
        raise ValueError("HANDOFF_RECEIPT_SCHEMA")
    handoff_sha = str(receipt.get("handoff_blob_sha256", ""))
    if not SHA64.fullmatch(handoff_sha):
        raise ValueError("HANDOFF_RECEIPT_BINDING_MISSING")
    payload = dict(receipt)
    payload.pop("handoff_blob_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_v3(path: Path) -> dict[str, Any]:
    path = path.resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3":
        raise ValueError("HANDOFF_V3_SCHEMA")
    if "full_head" in value or "branch_head_verified_externally" in value:
        raise ValueError("AMBIGUOUS_HEAD_BINDING")
    for key in ("code_snapshot_commit", "handoff_metadata_parent_commit"):
        if not SHA40.fullmatch(str(value.get(key, ""))):
            raise ValueError(f"{key.upper()}_MUST_BE_FULL_SHA")
    if value.get("status") != "READY_FOR_DEEPSEEK_STATIC_INTEGRATION":
        raise ValueError("HANDOFF_NOT_STATIC_READY")
    if value.get("expected_split_keys") != list(EXPECTED_SPLITS):
        raise ValueError("EXACT_SPLIT_KEYS_REQUIRED")
    forbidden = value.get("forbidden_runtime_fields")
    if not isinstance(forbidden, list) or not {"event_id", "known_mask", "teacher_phase", "utility_probability", "regrasp_probability"}.issubset(forbidden):
        raise ValueError("FORBIDDEN_RUNTIME_FIELDS_INCOMPLETE")
    refs = list(_references(value))
    if len(refs) < 8:
        raise ValueError("HANDOFF_REFERENCES_INCOMPLETE")
    root = path.parents[1]
    for ref in refs:
        digest = str(ref.get("sha256", ""))
        if not SHA64.fullmatch(digest):
            raise ValueError("REFERENCE_SHA_INVALID")
        target = _repo_relative(root, ref["path"])
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != digest.lower():
            raise ValueError(f"REFERENCE_SHA_MISMATCH:{ref['path']}")
    handoff_sha = str(value.get("handoff_blob_sha256", ""))
    if not SHA64.fullmatch(handoff_sha) or handoff_sha != canonical_handoff_sha(value):
        raise ValueError("HANDOFF_BLOB_SHA_MISMATCH")
    return {
        "status": "STATIC_INTEGRATION_PASS",
        "schema": value["schema"],
        "code_snapshot_commit": value["code_snapshot_commit"],
        "handoff_metadata_parent_commit": value["handoff_metadata_parent_commit"],
        "expected_split_keys": list(EXPECTED_SPLITS),
        "reference_count": len(refs),
        "handoff_blob_sha256": handoff_sha,
    }


def validate_v3_1(path: Path) -> dict[str, Any]:
    """Validate the nested V3.1 static handoff without production execution."""
    path = path.resolve()
    value = _strict_read_json(path)
    if not isinstance(value, dict) or value.get("schema") != "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3_1":
        raise ValueError("HANDOFF_V3_1_SCHEMA")
    if value.get("interface_revision") != "V3.1":
        raise ValueError("HANDOFF_V3_1_REVISION")
    if value.get("status") != "READY_FOR_DEEPSEEK_STATIC_INTEGRATION":
        raise ValueError("HANDOFF_NOT_STATIC_READY")
    if "execution_ready" in value or value.get("production_execution") is True:
        raise ValueError("EXECUTION_READY_FORBIDDEN")
    if "full_head" in value or "branch_head_verified_externally" in value:
        raise ValueError("AMBIGUOUS_HEAD_BINDING")
    for key in ("code_snapshot_commit", "handoff_metadata_parent_commit"):
        if not SHA40.fullmatch(str(value.get(key, ""))):
            raise ValueError(f"{key.upper()}_MUST_BE_FULL_SHA")
    if value.get("expected_split_keys") != list(EXPECTED_SPLITS):
        raise ValueError("EXACT_SPLIT_KEYS_REQUIRED")
    if not isinstance(value.get("metadata_commit_consumption_rule"), str) or "code_snapshot_commit" not in value["metadata_commit_consumption_rule"]:
        raise ValueError("METADATA_CONSUMPTION_RULE_MISSING")
    required_sections = (
        "scheduler_api", "runtime_adapter", "runtime_bundle", "offline_bundles",
        "calibration_contract", "structural_config", "handoff_validator",
        "production_receipt_requirements", "remaining_blockers",
    )
    if any(section not in value for section in required_sections):
        raise ValueError("HANDOFF_V3_1_SECTION_MISSING")
    forbidden = value.get("forbidden_runtime_fields")
    if not isinstance(forbidden, list) or not {"event_id", "known_mask", "teacher_phase", "utility_probability", "regrasp_probability"}.issubset(forbidden):
        raise ValueError("FORBIDDEN_RUNTIME_FIELDS_INCOMPLETE")
    refs = list(_references(value))
    if len(refs) < 10:
        raise ValueError("HANDOFF_REFERENCES_INCOMPLETE")
    root = path.parents[1]
    receipt_requirement = value["production_receipt_requirements"].get("handoff_receipt")
    receipt_ref_path = receipt_requirement.get("path") if isinstance(receipt_requirement, Mapping) else None
    for ref in refs:
        digest = str(ref.get("sha256", ""))
        if not SHA64.fullmatch(digest):
            raise ValueError("REFERENCE_SHA_INVALID")
        target = _repo_relative(root, ref["path"])
        actual = receipt_binding_sha(target) if ref.get("path") == receipt_ref_path else _canonical_reference_sha(target)
        if actual != digest.lower():
            raise ValueError(f"REFERENCE_SHA_MISMATCH:{ref['path']}")
    if value["runtime_bundle"].get("split_directory_name") != "{split}":
        raise ValueError("RUNTIME_SPLIT_DIRECTORY_CONTRACT")
    runtime_bundle = value["runtime_bundle"]
    if runtime_bundle.get("schema_name") != "FACTORIZED_V2_RUNTIME_SCHEDULER_INPUT_BUNDLE_V2":
        raise ValueError("RUNTIME_SCHEMA_NAME_CONTRACT")
    runtime_schema_file = runtime_bundle.get("schema_file", {})
    if runtime_schema_file.get("path") != "schemas/factorized_v2_runtime_scheduler_input.schema.json":
        raise ValueError("RUNTIME_SCHEMA_FILE_CONTRACT")
    receipt_requirement = value["production_receipt_requirements"].get("handoff_receipt")
    if not isinstance(receipt_requirement, dict) or receipt_requirement.get("path") != "reports/FACTORIZED_V3_1_HANDOFF_RECEIPT.json":
        raise ValueError("HANDOFF_RECEIPT_REFERENCE_CONTRACT")
    if value["production_receipt_requirements"].get("handoff_receipt_reference_sha_definition") != RECEIPT_BINDING_DEFINITION:
        raise ValueError("HANDOFF_RECEIPT_REFERENCE_DEFINITION")
    receipt_target = _repo_relative(root, receipt_requirement["path"])
    if receipt_binding_sha(receipt_target) != str(receipt_requirement.get("sha256", "")):
        raise ValueError("HANDOFF_RECEIPT_REFERENCE_SHA_MISMATCH")
    if value["runtime_bundle"].get("data_filename") != "runtime_scheduler_inputs.jsonl" or value["runtime_bundle"].get("manifest_filename") != "manifest.json":
        raise ValueError("RUNTIME_FILENAME_CONTRACT")
    if value["offline_bundles"].get("calibration", {}).get("data_filename") != "calibration_records.jsonl":
        raise ValueError("OFFLINE_CALIBRATION_FILENAME_CONTRACT")
    if value["offline_bundles"].get("evaluation", {}).get("data_filename") != "evaluation_records.jsonl":
        raise ValueError("OFFLINE_EVALUATION_FILENAME_CONTRACT")
    calibration_schema = value["calibration_contract"].get("schema", {})
    if calibration_schema.get("path", "") != "schemas/factorized_v2_calibration_and_threshold_contract_v3.schema.json":
        raise ValueError("CALIBRATION_V3_SCHEMA_REFERENCE")
    handoff_sha = str(value.get("handoff_blob_sha256", ""))
    if not SHA64.fullmatch(handoff_sha) or handoff_sha != canonical_handoff_sha(value):
        raise ValueError("HANDOFF_BLOB_SHA_MISMATCH")
    return {
        "status": "STATIC_INTEGRATION_PASS",
        "schema": value["schema"],
        "interface_revision": value["interface_revision"],
        "code_snapshot_commit": value["code_snapshot_commit"],
        "handoff_metadata_parent_commit": value["handoff_metadata_parent_commit"],
        "expected_split_keys": list(EXPECTED_SPLITS),
        "reference_count": len(refs),
        "handoff_blob_sha256": handoff_sha,
        "production_execution": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("handoff", type=Path)
    try:
        handoff = parser.parse_args().handoff
        schema = _strict_read_json(handoff).get("schema")
        if schema == "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3_1":
            result = validate_v3_1(handoff)
        elif schema == "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3":
            result = validate_v3(handoff)
        else:
            result = validate(handoff)
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        print(f"HOLD:{type(exc).__name__}:{exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
