"""Build an append-only exposure union for a prospective Stage V selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

try:
    from .stage_v_dynamic_common import atomic_write_json, canonical_parent_key, read_json, sha256_file, utc_now
except ImportError:  # pragma: no cover
    from stage_v_dynamic_common import atomic_write_json, canonical_parent_key, read_json, sha256_file, utc_now


def _job_parent_key(job: dict[str, Any]) -> str:
    value = job.get("canonical_parent_key") or job.get("parent_key")
    if value:
        return str(value)
    return canonical_parent_key(job)


def _load_base(path: Path) -> tuple[set[str], dict[str, Any]]:
    value = read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("excluded_parent_keys"), list):
        raise ValueError("base manifest must contain excluded_parent_keys")
    keys = value["excluded_parent_keys"]
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("base manifest contains invalid excluded_parent_keys")
    if len(keys) != len(set(keys)):
        raise ValueError("base manifest contains duplicate excluded_parent_keys")
    return set(keys), value


def _attempted_keys(root: Path) -> set[str]:
    keys: set[str] = set()
    for path in sorted(root.glob("parents/*/attempt_*/JOB.json")):
        job = read_json(path)
        if not isinstance(job, dict):
            raise ValueError(f"invalid JOB.json: {path}")
        key = _job_parent_key(job)
        if not key:
            raise ValueError(f"JOB.json has no canonical parent key: {path}")
        keys.add(key)
    return keys


def build(base_manifest: Path, attempted_roots: list[Path], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite exposure manifest: {output}")
    base_keys, base_value = _load_base(base_manifest)
    attempted = {str(key) for root in attempted_roots for key in _attempted_keys(root)}
    union = sorted(base_keys | attempted)
    report = {
        "schema": "STAGE_V_EXPOSURE_MANIFEST_V2",
        "status": "PASS",
        "generated_utc": utc_now(),
        "base_manifest": {
            "path": str(base_manifest.resolve()),
            "sha256": sha256_file(base_manifest),
            "schema": base_value.get("schema"),
            "excluded_parent_count": len(base_keys),
        },
        "attempt_roots": [
            {"path": str(root.resolve()), "attempted_parent_count": len(_attempted_keys(root))}
            for root in attempted_roots
        ],
        "attempted_parent_keys": sorted(attempted),
        "attempted_parent_count": len(attempted),
        "newly_added_parent_keys": sorted(attempted - base_keys),
        "newly_added_parent_count": len(attempted - base_keys),
        "excluded_parent_keys": union,
        "excluded_parent_count": len(union),
    }
    atomic_write_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--attempted-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build(args.base_manifest, args.attempted_root, args.output)
    print(json.dumps({"status": report["status"], "excluded_parent_count": report["excluded_parent_count"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
