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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("handoff", type=Path)
    try:
        handoff = parser.parse_args().handoff
        print(json.dumps(validate_v3(handoff) if json.loads(handoff.read_text(encoding="utf-8")).get("schema") == "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3" else validate(handoff), sort_keys=True))
    except Exception as exc:
        print(f"HOLD:{type(exc).__name__}:{exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
