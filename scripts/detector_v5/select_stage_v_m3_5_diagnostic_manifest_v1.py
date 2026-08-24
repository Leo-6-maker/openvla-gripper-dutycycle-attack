"""Select the frozen, outcome-blind M3.5 diagnostic parent manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

try:
    from .stage_v_dynamic_common import atomic_write_json, read_json, sha256_file, utc_now
except ImportError:  # pragma: no cover
    from stage_v_dynamic_common import atomic_write_json, read_json, sha256_file, utc_now


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def _rank(salt: str, key: str) -> str:
    return hashlib.sha256(f"{salt}::{key}".encode("utf-8")).hexdigest()


def select(exposure_manifest: Path, output: Path, *, per_suite: int = 2,
           salt: str = "STAGE_V_M3_5_DIAGNOSTIC_SELECTION_V1",
           source_commit: str | None = None, source_tree: str | None = None) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic manifest: {output}")
    manifest = read_json(exposure_manifest)
    if not isinstance(manifest, dict) or manifest.get("schema") != "STAGE_V_COUNTERFACTUAL_EXPOSURE_UNION_V4" or manifest.get("status") != "PASS":
        raise ValueError("exposure V4 manifest is not admissible")
    keys = manifest.get("excluded_parent_keys")
    if not isinstance(keys, list) or len(keys) != len(set(keys)):
        raise ValueError("exposure V4 keys are missing or duplicated")
    by_suite = {suite: [] for suite in SUITES}
    for key in keys:
        suite = str(key).split("/", 1)[0]
        if suite in by_suite:
            by_suite[suite].append(str(key))
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for suite in SUITES:
        ranked = sorted((_rank(salt, key), key) for key in by_suite[suite])
        take = ranked[:min(per_suite, len(ranked))]
        counts[suite] = len(take)
        selected.extend({"canonical_parent_key": key, "suite": suite, "selection_rank_sha256": rank} for rank, key in take)
    if any(counts[suite] == 0 for suite in SUITES):
        raise ValueError(f"diagnostic suite coverage unavailable: {counts}")
    report = {
        "schema": "STAGE_V_M3_5_DIAGNOSTIC_PARENT_SELECTION_V1",
        "status": "FROZEN_FOR_VALIDATION",
        "selection_role": "outcome_blind_m3_5_diagnostic_only; exposed identities only",
        "generated_utc": utc_now(),
        "source_exposure_manifest": {
            "path": str(Path(exposure_manifest).resolve()),
            "sha256": sha256_file(exposure_manifest),
            "schema": manifest.get("schema"),
            "excluded_parent_count": manifest.get("excluded_parent_count"),
        },
        "selection_algorithm": "sha256(selection_salt + '::' + canonical_parent_key), ascending rank",
        "selection_salt": salt,
        "target_per_suite": per_suite,
        "selected_count": len(selected),
        "selected_counts_by_suite": counts,
        "selected_parents": sorted(selected, key=lambda row: (row["suite"], row["selection_rank_sha256"])),
        "selection_reads": {
            "branch_results_read": False,
            "outcomes_read": False,
            "teacher_student_predictions_read": False,
            "protected_reads": 0,
        },
        "source_artifacts_modified": False,
        "old_artifacts_reused": False,
        "protected_counters": {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0},
        "runtime_authorized": False,
        "builder_source_commit": source_commit,
        "builder_source_tree": source_tree,
    }
    atomic_write_json(output, report)
    digest = sha256_file(output)
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exposure-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-suite", type=int, default=2)
    parser.add_argument("--salt", default="STAGE_V_M3_5_DIAGNOSTIC_SELECTION_V1")
    parser.add_argument("--source-commit")
    parser.add_argument("--source-tree")
    args = parser.parse_args(argv)
    report = select(args.exposure_manifest, args.output, per_suite=args.per_suite, salt=args.salt, source_commit=args.source_commit, source_tree=args.source_tree)
    print(json.dumps({"status": report["status"], "selected_count": report["selected_count"], "selected_counts_by_suite": report["selected_counts_by_suite"], "output": str(Path(args.output).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
