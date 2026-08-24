#!/usr/bin/env python3
"""Freeze S0 identity/provenance without reading Teacher-derived content."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--compatibility-report", type=Path, required=True)
    parser.add_argument("--legacy-output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--runner-git-head", required=True)
    args = parser.parse_args()

    rows = []
    with args.selection_manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("runtime_valid", "").lower() == "true"]
    records = []
    state_counts = Counter()
    suite_task_counts = Counter()
    for row in rows:
        artifact = Path(row["artifact"])
        metadata = json.loads((artifact / "episode_metadata.json").read_text(encoding="utf-8"))
        key = metadata["canonical_parent_key"]
        child_matches = sorted((args.legacy_output_root / "episodes").glob(f"*_{key.replace('/', '__')}"))
        if len(child_matches) != 1:
            raise ValueError(f"expected one child output for {key}, got {len(child_matches)}")
        source_checksums = json.loads((artifact / "artifact_sha256.json").read_text(encoding="utf-8"))
        state_counts[(metadata["suite"], int(metadata["state_id"]))] += 1
        suite_task_counts[(metadata["suite"], int(metadata["task_idx"]))] += 1
        records.append({
            "canonical_parent_key": key,
            "suite": metadata["suite"],
            "task_idx": metadata["task_idx"],
            "state_id": metadata["state_id"],
            "split": metadata["split"],
            "source_artifact_sha256": source_checksums["recursive_sha256"],
            "materialization_manifest_sha256": sha256_file(child_matches[0] / "materialization_manifest.json"),
        })

    teacher_files_present = any(
        path.name in {"teacher_retention_records.jsonl", "retention_events.json"}
        for path in args.legacy_output_root.rglob("*")
        if path.is_file()
    )
    summary = {
        "schema": "B3_RETENTION_S0_CURRENT_SCHEMA_SUMMARY_V1",
        "status": "S0_COMPATIBILITY_ONLY_NOT_FOR_TEACHER_ANALYSIS",
        "source_root": str(args.source_root),
        "selection_manifest": str(args.selection_manifest),
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "compatibility_report": str(args.compatibility_report),
        "compatibility_report_sha256": sha256_file(args.compatibility_report),
        "protocol_config_sha256": sha256_file(args.protocol_config),
        "runner_git_head": args.runner_git_head,
        "selected_count": len(records),
        "unique_canonical_key_count": len({row["canonical_parent_key"] for row in records}),
        "selected_state_counts": {f"{suite}/state_{state:02d}": count for (suite, state), count in sorted(state_counts.items())},
        "suite_task_selected_counts": {f"{suite}/task_{task:02d}": count for (suite, task), count in sorted(suite_task_counts.items())},
        "fit_state_0_19_only": all(int(row["state_id"]) <= 19 for row in records),
        "freeze_read_teacher_content": False,
        "teacher_files_present": teacher_files_present,
        "human_teacher_content_exposure": "UNKNOWN",
        "eligible_for_teacher_analysis": False,
        "source_unchanged_during_original_run": "NOT_RECORDED",
        "formal_training_ready": False,
        "formal_attack_ready": False,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_name(args.output.name + ".sha256").write_text(
        f"{sha256_file(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("status", "selected_count", "fit_state_0_19_only", "freeze_read_teacher_content")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
