"""Freeze a fresh clean-qualification candidate universe outside exposure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

try:
    from .stage_v_dynamic_common import atomic_write_json, exposure_binding, normalize_parent, read_json, sha256_file, utc_now
except ImportError:  # pragma: no cover
    from stage_v_dynamic_common import atomic_write_json, exposure_binding, normalize_parent, read_json, sha256_file, utc_now


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
PROTOCOL_ID = "STAGE_V_DYNAMIC_CLEAN_QUALIFICATION_V1"


def select(
    candidate_pool: Path,
    exposure_manifest: Path,
    output: Path,
    source_clean_root: str,
    salt: str,
    source_commit: str,
    source_tree: str,
    target_per_suite: int = 10,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate manifest: {output}")
    pool = read_json(candidate_pool)
    if not isinstance(pool, Mapping) or pool.get("schema") != "D8_STAGE_V_CLEAN_PROBE_CANDIDATE_POOL_V1":
        raise ValueError("candidate pool schema is not the frozen clean-only pool")
    gates = pool.get("gates")
    if not isinstance(gates, Mapping) or any(gates.get(field, 0) != 0 for field in ("eval160_reads", "protected_eval_reads", "attack_rollouts")):
        raise ValueError("candidate pool boundary is non-zero")
    if gates.get("attack_informed_tuning") is not False or pool.get("selection_frozen_before_new_rollouts") is not True:
        raise ValueError("candidate pool is not clean-only and pre-frozen")
    raw = pool.get("candidates")
    if not isinstance(raw, list):
        raise ValueError("candidate pool rows are missing")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("candidate row is not an object")
        row = normalize_parent(item)
        key = str(row["canonical_parent_key"])
        if key in seen or row.get("legacy_g10_test_only") is not True:
            raise ValueError(f"candidate identity invalid: {key}")
        seen.add(key)
        rows.append(row)
    excluded = set()
    exposure = read_json(exposure_manifest)
    if isinstance(exposure, Mapping):
        excluded = {str(key) for key in exposure.get("excluded_parent_keys", [])}
    fresh = [row for row in rows if str(row["canonical_parent_key"]) not in excluded]
    binding = exposure_binding((row["canonical_parent_key"] for row in fresh), exposure_manifest)
    if binding.get("status") != "PASS":
        raise ValueError(f"exposure-clean candidate set intersects exposure: {binding.get('reason')}")
    by_suite = {suite: [] for suite in SUITES}
    for row in fresh:
        by_suite[str(row["suite"])].append(row)
    if any(len(by_suite[suite]) < target_per_suite for suite in SUITES):
        raise ValueError("exposure-clean candidate pool cannot satisfy suite quotas")
    ranked: list[dict[str, Any]] = []
    root = source_clean_root.rstrip("/")
    for suite in SUITES:
        suite_rows = sorted(
            by_suite[suite],
            key=lambda row: (hashlib.sha256(f"{salt}::{row['canonical_parent_key']}".encode()).hexdigest(), row["canonical_parent_key"]),
        )
        for row in suite_rows:
            key = str(row["canonical_parent_key"])
            ranked.append({
                **row,
                "qualification_rank_sha256": hashlib.sha256(f"{salt}::{key}".encode()).hexdigest(),
                "source_artifact_root": f"{root}/{key}",
                "qualification_mode": "FRESH_CLEAN_AB_REPLAY",
                "selection_role": "fresh_exposure_clean_control_candidate",
                "old_artifacts_reused": False,
                "source_artifact_read": False,
            })
    report = {
        "schema": "STAGE_V_DYNAMIC_CLEAN_QUALIFICATION_CANDIDATE_MANIFEST_V1",
        "status": "FROZEN",
        "protocol_id": PROTOCOL_ID,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "salt": salt,
        "selection_rule": "hash-rank clean-only candidate pool after append-only exposure exclusion; no vulnerability outcomes read",
        "candidate_pool_path": str(candidate_pool.resolve()),
        "candidate_pool_sha256": sha256_file(candidate_pool),
        "candidate_pool_count": len(rows),
        "candidate_pool_exposed_count": len(seen & excluded),
        "exposure_manifest_path": str(exposure_manifest.resolve()),
        "exposure_manifest_sha256": sha256_file(exposure_manifest),
        "exposure_binding": binding,
        "fresh_candidate_count": len(ranked),
        "fresh_candidate_counts_by_suite": {suite: len(by_suite[suite]) for suite in SUITES},
        "target_per_suite": target_per_suite,
        "source_clean_root": root,
        "candidates": ranked,
        "old_artifacts_reused": False,
        "source_artifacts_modified": False,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "attack_rollouts": 0,
        "resource_policy": {
            "resource_contract_version": "STAGE_V_RESOURCE_CONTRACT_V2",
            "minimum_free_memory_mib": 20480,
            "required_gpu_count": 1,
            "minimum_gpu_count": 1,
            "maximum_gpu_count": 8,
            "strict_gpu_count": False,
            "partial_fleet_allowed": True,
            "maximum_project_workers_per_gpu": 1,
            "foreign_workload_allowed": True,
        },
        "generated_utc": utc_now(),
    }
    atomic_write_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--exposure-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-clean-root", required=True)
    parser.add_argument("--salt", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--target-per-suite", type=int, default=10)
    args = parser.parse_args(argv)
    report = select(**vars(args))
    print(json.dumps({
        "status": report["status"],
        "protocol_id": report["protocol_id"],
        "fresh_candidate_count": report["fresh_candidate_count"],
        "fresh_candidate_counts_by_suite": report["fresh_candidate_counts_by_suite"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
