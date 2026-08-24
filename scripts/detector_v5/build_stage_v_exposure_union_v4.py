"""Build the prospective Stage V V4 exposure exclusion union.

This builder only reads manifests and identity metadata.  It never opens a
branch result, protected evaluation artifact, or historical run for replay.
"""
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
    if not isinstance(value, str) or not value:
        return False
    if not value.startswith(PARENT_PREFIXES):
        return False
    parts = value.split("/")
    return len(parts) == 3 and parts[1].startswith("task_") and parts[2].startswith("state_")


def _collect_keys(value: Any) -> set[str]:
    """Extract canonical identities from known manifest containers only."""
    found: set[str] = set()
    if isinstance(value, Mapping):
        for field in ("canonical_parent_key", "parent_key"):
            candidate = value.get(field)
            if _is_parent_key(candidate):
                found.add(candidate)
        for field in ("excluded_parent_keys", "parent_keys", "attempted_parent_keys"):
            candidates = value.get(field)
            if isinstance(candidates, list):
                found.update(item for item in candidates if _is_parent_key(item))
        for field in ("candidates", "parents", "qualified_parents", "selected_parents", "rows"):
            candidates = value.get(field)
            if isinstance(candidates, list):
                for item in candidates:
                    found.update(_collect_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_collect_keys(item))
    return found


def _source_record(path: Path, role: str) -> tuple[dict[str, Any], set[str]]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"source manifest is not an object: {path}")
    status = str(value.get("status", ""))
    if status not in {"PASS", "FROZEN", "COMPLETE_VALID"}:
        raise ValueError(f"source manifest is not admissible: {path}:{status}")
    keys = _collect_keys(value)
    if not keys:
        raise ValueError(f"source manifest has no canonical parent identities: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "schema": value.get("schema"),
        "status": status,
        "role": role,
        "parent_count": len(keys),
    }, keys


def _write_sidecar(output: Path) -> str:
    digest = sha256_file(output)
    sidecar = output.with_name(output.name + ".sha256")
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return digest


def build(
    source_manifests: list[Path],
    output: Path,
    *,
    diagnostic_manifests: list[Path] | None = None,
    m35_status: str = "NOT_STARTED",
    source_commit: str | None = None,
    source_tree: str | None = None,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite exposure union: {output}")
    if not source_manifests:
        raise ValueError("at least one exposure source manifest is required")
    diagnostic_manifests = list(diagnostic_manifests or [])
    if m35_status != "NOT_STARTED" and not diagnostic_manifests:
        raise ValueError("M3.5 status requires a diagnostic exposure manifest")

    records: list[dict[str, Any]] = []
    key_sets: list[set[str]] = []
    for path in source_manifests:
        record, keys = _source_record(Path(path), "historical_and_prior_exposure_union")
        records.append(record)
        key_sets.append(keys)
    diagnostic_records: list[dict[str, Any]] = []
    for path in diagnostic_manifests:
        record, keys = _source_record(Path(path), "m3_5_diagnostic_intervention_identities")
        diagnostic_records.append(record)
        key_sets.append(keys)

    union = sorted(set().union(*key_sets))
    source_key_total = sum(len(keys) for keys in key_sets)
    report = {
        "schema": "STAGE_V_COUNTERFACTUAL_EXPOSURE_UNION_V4",
        "status": "PASS",
        "union_role": "authoritative prospective exclusion for all prior and diagnostic exposure identities",
        "generated_utc": utc_now(),
        "source_manifests": records,
        "m35_diagnostic_sources": diagnostic_records,
        "m35_diagnostic_status": m35_status,
        "source_key_total": source_key_total,
        "overlap_count": source_key_total - len(union),
        "parent_count": len(union),
        "parent_keys": union,
        "excluded_parent_count": len(union),
        "excluded_parent_keys": union,
        "union_sha256": sha256_json(union),
        "branch_results_read": False,
        "source_artifacts_modified": False,
        "old_artifacts_reused": False,
        "protected_counters": {
            "protected_reads": 0,
            "eval160_reads": 0,
            "attack_rollouts": 0,
            "vis_pgd_attack_rollouts": 0,
        },
        "future_parent_selection": "BLOCKED_UNTIL_POST_M3_5_RECOMPUTE",
        "builder_source_commit": source_commit,
        "builder_source_tree": source_tree,
    }
    atomic_write_json(output, report)
    report["manifest_sha256"] = _write_sidecar(output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--diagnostic-manifest", type=Path, action="append", default=[])
    parser.add_argument("--m35-status", default="NOT_STARTED")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-tree")
    args = parser.parse_args(argv)
    report = build(
        args.source_manifest,
        args.output,
        diagnostic_manifests=args.diagnostic_manifest,
        m35_status=args.m35_status,
        source_commit=args.source_commit,
        source_tree=args.source_tree,
    )
    print(json.dumps({"status": report["status"], "parent_count": report["parent_count"], "output": str(Path(args.output).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
