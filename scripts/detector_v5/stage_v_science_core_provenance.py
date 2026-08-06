"""Freeze and verify external Stage V science-core snapshots."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Any

try:
    from .stage_v_dynamic_common import atomic_write_json, sha256_file, utc_now
except ImportError:
    from stage_v_dynamic_common import atomic_write_json, sha256_file, utc_now


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def build(paths: list[Path], *, source_commit: str, source_tree: str) -> dict[str, Any]:
    files = []
    for path in paths:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({
            "path": str(path), "raw_sha256": sha256_file(path),
            "git_blob_sha1": git_blob_sha1(path),
        })
    return {
        "schema": "STAGE_V_SCIENCE_CORE_PROVENANCE_V1",
        "status": "PASS", "science_source_commit": source_commit, "science_source_tree": source_tree,
        "files": files, "immutable_snapshot": True, "created_utc": utc_now(),
    }


def verify(path: Path, *, expected_commit: str | None = None, expected_tree: str | None = None) -> tuple[bool, list[str]]:
    import json
    value = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if value.get("schema") != "STAGE_V_SCIENCE_CORE_PROVENANCE_V1" or value.get("status") != "PASS":
        errors.append("PROVENANCE_SCHEMA_OR_STATUS")
    if expected_commit and value.get("science_source_commit") != expected_commit:
        errors.append("SCIENCE_SOURCE_COMMIT_MISMATCH")
    if expected_tree and value.get("science_source_tree") != expected_tree:
        errors.append("SCIENCE_SOURCE_TREE_MISMATCH")
    for item in value.get("files", []):
        target = Path(item.get("path", ""))
        if not target.is_file():
            errors.append(f"MISSING:{target}")
            continue
        if sha256_file(target) != item.get("raw_sha256"):
            errors.append(f"RAW_SHA256_MISMATCH:{target}")
        if git_blob_sha1(target) != item.get("git_blob_sha1"):
            errors.append(f"GIT_BLOB_SHA1_MISMATCH:{target}")
    return not errors, sorted(errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--file", action="append", type=Path, required=True)
    args = parser.parse_args(argv)
    atomic_write_json(args.output, build(args.file, source_commit=args.source_commit, source_tree=args.source_tree))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
