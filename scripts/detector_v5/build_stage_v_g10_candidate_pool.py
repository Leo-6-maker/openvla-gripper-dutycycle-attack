"""Build a pre-outcome, exposure-clean candidate pool from G10 held-out identities."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

try:
    from .stage_v_dynamic_common import atomic_write_json, normalize_parent, read_json, sha256_file, utc_now
except ImportError:  # pragma: no cover
    from stage_v_dynamic_common import atomic_write_json, normalize_parent, read_json, sha256_file, utc_now


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
POOL_SCHEMA = "D8_STAGE_V_CLEAN_PROBE_CANDIDATE_POOL_V1"
EXPOSURE_SCHEMAS = {"STAGE_V_EXPOSURE_MANIFEST_V2", "STAGE_V_COUNTERFACTUAL_EXPOSURE_UNION_V4"}
ATTEMPT_SCHEMAS = {"STAGE_V_CLEAN_QUALIFICATION_ATTEMPT_EXCLUSION_V1", "STAGE_V_CUMULATIVE_CLEAN_ATTEMPT_EXCLUSION_V2"}


def _keys(value: Any, *, field: str) -> set[str]:
    raw = value.get(field) if isinstance(value, Mapping) else None
    if not isinstance(raw, list) or any(not isinstance(item, str) or not item for item in raw):
        raise ValueError(f"{field} must be a list of non-empty strings")
    if len(raw) != len(set(raw)):
        raise ValueError(f"{field} contains duplicate identities")
    return set(raw)


def _g10_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError("G10 manifest must be an object")
    identities = value.get("identities")
    if not isinstance(identities, list):
        raise ValueError("G10 manifest identities are missing")
    rows: dict[str, dict[str, Any]] = {}
    for identity in identities:
        if not isinstance(identity, str):
            raise ValueError("G10 identity is not a string")
        try:
            suite, task, state = identity.split("/")
            row = normalize_parent({
                "canonical_parent_key": identity,
                "legacy_g10_test_only": True,
            })
        except (ValueError, KeyError) as exc:
            raise ValueError(f"invalid G10 identity: {identity}") from exc
        if suite not in SUITES or not task.startswith("task_") or not state.startswith("state_"):
            raise ValueError(f"invalid G10 identity: {identity}")
        if identity in rows:
            raise ValueError(f"duplicate G10 identity: {identity}")
        rows[identity] = row
    return list(rows.values()), dict(value)


def build(
    g10_manifest: Path,
    exposure_manifest: Path,
    attempt_exclusion: Path,
    output: Path,
    *,
    salt: str,
    candidates_per_suite: int = 60,
    source_commit: str | None = None,
    source_tree: str | None = None,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate pool: {output}")
    if candidates_per_suite < 1:
        raise ValueError("candidates_per_suite must be positive")
    rows, g10_value = _g10_rows(g10_manifest)
    exposure = read_json(exposure_manifest)
    if not isinstance(exposure, Mapping) or exposure.get("schema") not in EXPOSURE_SCHEMAS or exposure.get("status") != "PASS":
        raise ValueError("exposure manifest is not an accepted PASS manifest")
    attempts = read_json(attempt_exclusion)
    if not isinstance(attempts, Mapping) or attempts.get("schema") not in ATTEMPT_SCHEMAS or attempts.get("status") != "PASS":
        raise ValueError("clean-attempt exclusion is not a PASS manifest")
    exposed = _keys(exposure, field="excluded_parent_keys")
    attempted = _keys(attempts, field="excluded_parent_keys")
    excluded = exposed | attempted
    g10_keys = {str(row["canonical_parent_key"]) for row in rows}
    by_suite = {suite: [] for suite in SUITES}
    for row in rows:
        key = str(row["canonical_parent_key"])
        if key not in excluded:
            by_suite[str(row["suite"])].append(row)
    if any(len(by_suite[suite]) < candidates_per_suite for suite in SUITES):
        counts = {suite: len(by_suite[suite]) for suite in SUITES}
        raise ValueError(f"fresh G10 pool cannot satisfy quota: {counts}")
    selected: list[dict[str, Any]] = []
    rank_by_key: dict[str, str] = {}
    for suite in SUITES:
        ranked = sorted(
            by_suite[suite],
            key=lambda row: (hashlib.sha256(f"{salt}::{row['canonical_parent_key']}".encode()).hexdigest(), row["canonical_parent_key"]),
        )
        for row in ranked[:candidates_per_suite]:
            key = str(row["canonical_parent_key"])
            rank_by_key[key] = hashlib.sha256(f"{salt}::{key}".encode()).hexdigest()
            selected.append(row)
    selected.sort(key=lambda row: (str(row["suite"]), rank_by_key[str(row["canonical_parent_key"])]))
    report = {
        "schema": POOL_SCHEMA,
        "pool_revision": "G10_HELDOUT_FRESH_V2",
        "candidate_count": len(selected),
        "candidates_per_suite": candidates_per_suite,
        "candidate_state_indices": sorted({int(row["state_index"]) for row in selected}),
        "candidates": selected,
        "clean_probe_selection_rule": "deterministic SHA rank over G10 held-out identities after exposure and prior clean-attempt exclusion; no clean or vulnerability outcome read",
        "selection_frozen_before_new_rollouts": True,
        "final_attack_test_parents_are_separate": True,
        "gates": {
            "attack_informed_tuning": False,
            "attack_rollouts": 0,
            "eval160_reads": 0,
            "protected_eval_reads": 0,
            "new_cohort_clean_only_until_freeze": True,
        },
        "exclusion_evidence": {
            "g10_manifest": {"path": str(g10_manifest.resolve()), "sha256": sha256_file(g10_manifest), "identity_count": len(g10_keys)},
            "exposure_manifest": {"path": str(exposure_manifest.resolve()), "sha256": sha256_file(exposure_manifest), "excluded_count": len(exposed)},
            "clean_attempt_exclusion": {"path": str(attempt_exclusion.resolve()), "sha256": sha256_file(attempt_exclusion), "excluded_count": len(attempted)},
            "union_excluded_g10_count": len(g10_keys & excluded),
        },
        "required_stage_v_cohort": {"per_suite_minimum": 10, "total_minimum": 40},
        "source_commit": source_commit,
        "source_tree": source_tree,
        "builder_source_commit": source_commit,
        "builder_source_tree": source_tree,
        "old_artifacts_reused": False,
        "source_artifacts_modified": False,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "attack_rollouts": 0,
        "generated_utc": utc_now(),
    }
    atomic_write_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g10-manifest", type=Path, required=True)
    parser.add_argument("--exposure-manifest", type=Path, required=True)
    parser.add_argument("--attempt-exclusion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--salt", required=True)
    parser.add_argument("--candidates-per-suite", type=int, default=60)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-tree")
    args = parser.parse_args(argv)
    report = build(**vars(args))
    print(json.dumps({"status": "PASS", "candidate_count": report["candidate_count"], "output": str(Path(args.output).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
