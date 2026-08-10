"""Freeze identities that already have a clean/control qualification result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

try:
    from .stage_v_dynamic_common import atomic_write_json, sha256_file, utc_now
except ImportError:  # pragma: no cover
    from stage_v_dynamic_common import atomic_write_json, sha256_file, utc_now


PARENT_KEY = re.compile(r"^libero_(?:10|goal|object|spatial)/task_\d+/state_\d+$")
RESULT_NAMES = {"CONTROL_RESULT.json", "QUALIFICATION_ROW.json", "CONTROL_QUALIFICATION_ROWS.jsonl"}


def _keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for field in ("canonical_parent_key", "parent_key", "parent_id"):
            candidate = value.get(field)
            if isinstance(candidate, str) and PARENT_KEY.fullmatch(candidate):
                found.add(candidate)
        for child in value.values():
            found.update(_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_keys(child))
    return found


def _read_result_file(path: Path) -> set[str]:
    if path.name.endswith(".jsonl"):
        values: list[Any] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                values.append(json.loads(line))
        return _keys(values)
    return _keys(json.loads(path.read_text(encoding="utf-8")))


def build(roots: Iterable[Path], output: Path, *, source_commit: str | None = None,
          source_tree: str | None = None) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite exclusion manifest: {output}")
    root_values = [Path(root).resolve() for root in roots]
    if not root_values or any(not root.is_dir() for root in root_values):
        raise ValueError("all source roots must be existing directories")
    files: list[Path] = []
    for root in root_values:
        files.extend(path for path in root.rglob("*") if path.is_file() and path.name in RESULT_NAMES)
    keys: set[str] = set()
    identity_sources: dict[str, list[str]] = {}
    for path in sorted(set(files)):
        found = _read_result_file(path)
        for key in found:
            keys.add(key)
            identity_sources.setdefault(key, []).append(str(path))
    if not keys:
        raise ValueError("no clean/control qualification identities found")
    report = {
        "schema": "STAGE_V_CLEAN_QUALIFICATION_ATTEMPT_EXCLUSION_V1",
        "status": "PASS",
        "semantics": "exclude every identity with a prior clean/control qualification result; do not rerun a valid result",
        "source_roots": [str(root) for root in root_values],
        "source_files": [{"path": str(path), "sha256": sha256_file(path)} for path in sorted(set(files))],
        "source_file_count": len(set(files)),
        "excluded_parent_keys": sorted(keys),
        "excluded_parent_count": len(keys),
        "identity_sources": {key: sorted(paths) for key, paths in sorted(identity_sources.items())},
        "old_artifacts_reused": False,
        "source_artifacts_modified": False,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "attack_rollouts": 0,
        "builder_source_commit": source_commit,
        "builder_source_tree": source_tree,
        "generated_utc": utc_now(),
    }
    atomic_write_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-tree")
    args = parser.parse_args(argv)
    report = build(args.root, args.output, source_commit=args.source_commit, source_tree=args.source_tree)
    print(json.dumps({"status": report["status"], "excluded_parent_count": report["excluded_parent_count"], "output": str(Path(args.output).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
