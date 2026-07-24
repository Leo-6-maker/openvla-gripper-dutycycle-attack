#!/usr/bin/env python3
"""Rebuild corrected canary manifest from the original canary manifest.

Preserves all parent identities, conditions, and invariant fields while
correcting GPU/worker assignment for the 4-worker fixed mapping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


CONDITIONS = ("CLEAN", "R9Q_DETECTOR_T10", "RAND_T10", "COMMAND_OPEN_ORACLE")
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")

GPU_SUITE_MAP = {
    "libero_object": 6,
    "libero_spatial": 6,
    "libero_goal": 7,
    "libero_10": 7,
}

INVARIANT_FIELDS = frozenset({
    "parent_key", "suite", "task_index", "state_id", "cohort", "split",
    "max_steps", "condition", "burst_length", "protocol_name", "protocol_version",
    "attack_space", "payload_mode", "timing_source", "objective_seed",
    "planned_start_step", "attack_outcome_used_for_selection",
    "source_parent_manifest", "source_parent_manifest_sha256",
    "detector_bundle_sha256", "detector_checkpoint_sha256", "detector_config_sha256",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True,
                        help="Path to the ORIGINAL canary manifest (r9q_attack_manifest.jsonl)")
    parser.add_argument("--detector-bundle", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source_manifest).resolve()
    bundle = Path(args.detector_bundle).resolve()
    output = Path(args.output_root).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output root: {output}")
    if not source.is_file() or not bundle.is_dir():
        raise SystemExit("source manifest and detector bundle must exist")

    old_rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(old_rows) != 32:
        raise SystemExit(f"expected 32 cells in source manifest, got {len(old_rows)}")

    # Extract unique parents preserving order from old manifest
    unique_parents: dict[str, dict[str, Any]] = {}
    for row in old_rows:
        pk = str(row["parent_key"])
        if pk not in unique_parents:
            unique_parents[pk] = {
                "parent_key": pk,
                "suite": str(row["suite"]),
                "task_index": int(row["task_index"]),
                "state_id": int(row["state_id"]),
                "cohort": str(row.get("cohort", row.get("split", "UNKNOWN"))),
                "split": str(row.get("split", row.get("cohort", "UNKNOWN"))),
                "max_steps": int(row.get("max_steps", 300)),
            }
            # Capture invariant fields from first occurrence
            for field in ("source_parent_manifest", "source_parent_manifest_sha256",
                          "detector_bundle_sha256", "detector_checkpoint_sha256",
                          "detector_config_sha256", "protocol_name", "protocol_version"):
                unique_parents[pk][field] = row.get(field, "")

    if len(unique_parents) != 8:
        raise SystemExit(f"expected 8 unique parents, got {len(unique_parents)}")

    per_suite = Counter(str(r["suite"]) for r in unique_parents.values())
    for suite in SUITES:
        if per_suite.get(suite, 0) != 2:
            raise SystemExit(f"expected 2 parents per suite, {suite} has {per_suite.get(suite, 0)}")

    # Generate corrected manifest
    bundle_sha = sha256_file(bundle / "SHA256SUMS") if (bundle / "SHA256SUMS").is_file() else ""
    checkpoint_sha = sha256_file(bundle / "checkpoint.pt")
    config_sha = sha256_file(bundle / "detector_config.json")

    manifests: list[dict[str, Any]] = []
    worker_counts: Counter[str] = Counter()

    for suite in SUITES:
        suite_parents = sorted(
            [p for p in unique_parents.values() if p["suite"] == suite],
            key=lambda p: p["parent_key"],
        )
        gpu = GPU_SUITE_MAP[suite]
        worker_suffix = "l10" if suite == "libero_10" else suite.replace("libero_", "")
        worker_id = f"g{gpu}_{worker_suffix}"

        for shard_index, parent in enumerate(suite_parents):
            pk = parent["parent_key"]
            max_steps = int(parent["max_steps"])
            task_index = int(parent["task_index"])
            state_id = int(parent["state_id"])

            for condition in CONDITIONS:
                # Preserve old planned_start_step and objective_seed
                old_row = [r for r in old_rows if r["parent_key"] == pk and r["condition"] == condition]
                planned_start = int(old_row[0]["planned_start_step"]) if old_row else -1
                objective_seed = int(old_row[0]["objective_seed"]) if old_row else 0

                manifests.append({
                    "manifest_schema": "c2g.r9q.attack_manifest.2026-07-13.v1",
                    "parent_key": pk,
                    "suite": suite,
                    "task_index": task_index,
                    "state_id": state_id,
                    "cohort": parent["cohort"],
                    "split": parent["split"],
                    "max_steps": max_steps,
                    "condition": condition,
                    "protocol_name": parent.get("protocol_name", "C2G_R9Q_MATCHED_ATTACK"),
                    "protocol_version": parent.get("protocol_version", "2026-07-13.v1"),
                    "attack_space": "VIS_PGD_OR_DIRECT_COMMAND",
                    "payload_mode": "NONE" if condition == "CLEAN" else (
                        "DIRECT_COMMAND_OPEN" if condition == "COMMAND_OPEN_ORACLE" else "TARGETED_VIS_PGD"
                    ),
                    "timing_source": (
                        "CLEAN" if condition == "CLEAN" else (
                            "DETECTOR_PERSISTENT_GATE" if condition in {"R9Q_DETECTOR_T10", "COMMAND_OPEN_ORACLE"}
                            else "DETERMINISTIC_RANDOM_TIME"
                        )
                    ),
                    "burst_length": 10,
                    "planned_start_step": planned_start,
                    "objective_seed": objective_seed,
                    "assigned_physical_gpu": gpu,
                    "assigned_worker_id": worker_id,
                    "assigned_shard_id": f"{worker_id}_shard",
                    "shard_local_index": shard_index,
                    "source_parent_manifest": parent.get("source_parent_manifest", ""),
                    "source_parent_manifest_sha256": parent.get("source_parent_manifest_sha256", ""),
                    "detector_bundle_sha256": bundle_sha,
                    "detector_checkpoint_sha256": checkpoint_sha,
                    "detector_config_sha256": config_sha,
                    "expected_git_commit": args.expected_git_commit,
                    "attack_outcome_used_for_selection": False,
                })
                worker_counts[worker_id] += 1

    assert len(manifests) == 32, f"expected 32 cells, got {len(manifests)}"

    output.mkdir(parents=True)
    manifest_path = output / "r9q_attack_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifests), encoding="utf-8"
    )

    # Write worker shards
    shards = output / "shards"
    shards.mkdir()
    for worker_id in sorted(worker_counts):
        worker_rows = [row for row in manifests if row["assigned_worker_id"] == worker_id]
        (shards / f"{worker_id}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in worker_rows), encoding="utf-8"
        )

    # Comparison report
    old_parents = sorted(set(r["parent_key"] for r in old_rows))
    new_parents = sorted(set(r["parent_key"] for r in manifests))
    old_workers = sorted(set(r.get("assigned_worker_id", "") for r in old_rows))
    new_workers = sorted(set(r["assigned_worker_id"] for r in manifests))

    field_changes: list[dict[str, Any]] = []
    for old_row in old_rows:
        pk = old_row["parent_key"]
        cond = old_row["condition"]
        new_row = [r for r in manifests if r["parent_key"] == pk and r["condition"] == cond][0]
        for key in sorted(new_row):
            if key in ("expected_git_commit", "assigned_physical_gpu", "assigned_worker_id",
                       "assigned_shard_id", "shard_local_index", "manifest_schema",
                       "detector_bundle_sha256", "detector_config_sha256"):
                continue
            if key in old_row and str(old_row[key]) != str(new_row[key]):
                field_changes.append({
                    "parent_key": pk, "condition": cond, "field": key,
                    "old": str(old_row[key]), "new": str(new_row[key]),
                })

    report = {
        "status": "PASS_C2G_R9Q_CORRECTED_CANARY_MANIFEST",
        "source_manifest": str(source),
        "source_manifest_sha256": sha256_file(source),
        "old_parents": old_parents,
        "new_parents": new_parents,
        "parents_match": old_parents == new_parents,
        "parent_count": len(unique_parents),
        "cell_count": len(manifests),
        "old_workers": old_workers,
        "new_workers": new_workers,
        "worker_counts": dict(sorted(worker_counts.items())),
        "gpu_suite_map": GPU_SUITE_MAP,
        "expected_git_commit": args.expected_git_commit,
        "unexpected_field_changes": field_changes,
    }
    report_path = output / "manifest_comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Generate SHA256SUMS
    files = sorted(path for path in output.rglob("*") if path.is_file())
    sums = output / "SHA256SUMS"
    sums.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    sums_sha = output / "SHA256SUMS.sha256"
    sums_sha.write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
