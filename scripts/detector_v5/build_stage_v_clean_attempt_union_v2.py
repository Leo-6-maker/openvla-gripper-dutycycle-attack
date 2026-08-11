"""Build the cumulative clean-attempt exclusion union for prospective V7."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

try:
    from .stage_v_dynamic_common import atomic_write_json, read_json, sha256_file, sha256_json, utc_now
except ImportError:  # pragma: no cover
    from stage_v_dynamic_common import atomic_write_json, read_json, sha256_file, sha256_json, utc_now


PARENT_PREFIXES = ("libero_10/", "libero_goal/", "libero_object/", "libero_spatial/")


def _is_parent_key(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith(PARENT_PREFIXES):
        return False
    parts = value.split("/")
    return len(parts) == 3 and parts[1].startswith("task_") and parts[2].startswith("state_")


def _collect_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for field in ("canonical_parent_key", "parent_key", "parent_id"):
            candidate = value.get(field)
            if _is_parent_key(candidate):
                found.add(candidate)
        for field in ("excluded_parent_keys", "parent_keys", "attempted_parent_keys"):
            candidates = value.get(field)
            if isinstance(candidates, list):
                found.update(item for item in candidates if _is_parent_key(item))
        for field in ("candidates", "parents", "qualified_parents", "selected_parents", "rows", "all_candidate_audits"):
            candidates = value.get(field)
            if isinstance(candidates, list):
                for item in candidates:
                    found.update(_collect_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_collect_keys(item))
    return found


def _role(schema: Any, path: Path) -> str:
    text = f"{schema} {path.name}".upper()
    if "CANDIDATE" in text and "V6" in text:
        return "v6_clean_qualification_attempts"
    if "CLEAN_QUALIFICATION_ATTEMPT" in text or "PRIOR" in text:
        return "prior_clean_control_attempts"
    return "future_failed_qualification_attempts"


def _source_record(path: Path) -> tuple[dict[str, Any], set[str]]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"source manifest is not an object: {path}")
    status = str(value.get("status", ""))
    if status not in {"PASS", "FROZEN", "COMPLETE_VALID", "FAIL_SEALED_NON_CONSUMABLE"}:
        raise ValueError(f"source manifest is not admissible: {path}:{status}")
    keys = _collect_keys(value)
    if not keys:
        raise ValueError(f"source manifest has no clean-attempt identities: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "schema": value.get("schema"),
        "status": status,
        "role": _role(value.get("schema"), Path(path)),
        "parent_count": len(keys),
    }, keys


def _write_sidecar(output: Path) -> str:
    digest = sha256_file(output)
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return digest


def build(source_manifests: list[Path], output: Path, *, source_commit: str | None = None,
          source_tree: str | None = None) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite clean-attempt union: {output}")
    if len(source_manifests) < 2:
        raise ValueError("at least prior and V6/future source manifests are required")
    records: list[dict[str, Any]] = []
    key_sets: list[set[str]] = []
    for path in source_manifests:
        record, keys = _source_record(Path(path))
        records.append(record)
        key_sets.append(keys)
    union = sorted(set().union(*key_sets))
    source_key_total = sum(len(keys) for keys in key_sets)
    overlap_count = source_key_total - len(union)
    report = {
        "schema": "STAGE_V_CUMULATIVE_CLEAN_ATTEMPT_EXCLUSION_V2",
        "status": "PASS",
        "union_role": "authoritative cumulative clean-attempt exclusion for prospective V7",
        "selection_rule": "candidate canonical_parent_key must have zero intersection with this union",
        "generated_utc": utc_now(),
        "source_manifests": records,
        "source_key_total": source_key_total,
        "overlap_count": overlap_count,
        "parent_count": len(union),
        "parent_keys": union,
        "excluded_parent_count": len(union),
        "excluded_parent_keys": union,
        "union_sha256": sha256_json(union),
        "future_failed_qualification_sources_included": any(record["role"] == "future_failed_qualification_attempts" for record in records),
        "future_parent_selection": "BLOCKED_UNTIL_POST_M3_5_RECOMPUTE",
        "old_artifacts_reused": False,
        "source_artifacts_modified": False,
        "protected_counters": {
            "protected_reads": 0,
            "eval160_reads": 0,
            "attack_rollouts": 0,
            "vis_pgd_attack_rollouts": 0,
        },
        "builder_source_commit": source_commit,
        "builder_source_tree": source_tree,
    }
    atomic_write_json(output, report)
    report["manifest_sha256"] = _write_sidecar(output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-tree")
    args = parser.parse_args(argv)
    report = build(args.source_manifest, args.output, source_commit=args.source_commit, source_tree=args.source_tree)
    print(json.dumps({"status": report["status"], "parent_count": report["parent_count"], "overlap_count": report["overlap_count"], "output": str(Path(args.output).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
