#!/usr/bin/env python3
"""Build a source-only, matched-condition R9Q attack manifest.

The input parent manifest is the only source of parent identities. Selection is
stable and never reads episode outcomes. The resulting rows are execution
instructions, not evidence that an attack has run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONDITIONS = ("CLEAN", "R9Q_DETECTOR_T10", "RAND_T10", "COMMAND_OPEN_ORACLE")
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
PROTOCOL_NAME = "C2G_R9Q_MATCHED_ATTACK"
PROTOCOL_VERSION = "2026-07-13.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(salt: str, value: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()


def stable_seed(parent_key: str, condition: str) -> int:
    material = f"{PROTOCOL_NAME}|{PROTOCOL_VERSION}|{parent_key}|{condition}"
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") % (2**32)


def random_start(parent_key: str, max_steps: int, burst_length: int) -> int:
    available = max(1, int(max_steps) - int(burst_length) + 1)
    return int(stable_seed(parent_key, "RAND_T10") % available)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--detector-bundle", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--selection-salt", default="R9Q_ATTACK_V1")
    parser.add_argument("--max-parents-per-suite", type=int, default=0)
    parser.add_argument("--burst-length", type=int, default=10)
    return parser.parse_args()


def _int(row: dict[str, Any], *names: str, default: int = 0) -> int:
    for name in names:
        if row.get(name) not in (None, ""):
            return int(row[name])
    return default


def main() -> int:
    args = parse_args()
    source = Path(args.source_manifest).resolve()
    bundle = Path(args.detector_bundle).resolve()
    output = Path(args.output_root).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output root: {output}")
    if not source.is_file() or not bundle.is_dir():
        raise SystemExit("source manifest and detector bundle must exist")
    if source.suffix.lower() == ".csv":
        rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    else:
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unique_source_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        suite = str(row.get("suite", ""))
        parent_key = str(row.get("parent_key", ""))
        if suite not in SUITES or not parent_key:
            raise SystemExit(f"invalid source parent row: {row}")
        previous = unique_source_rows.get(parent_key)
        if previous is None:
            unique_source_rows[parent_key] = row
        elif str(previous.get("suite")) != suite:
            raise SystemExit(f"parent identity maps to multiple suites: {parent_key}")
    for row in unique_source_rows.values():
        by_suite[str(row["suite"])].append(row)

    selected: list[dict[str, Any]] = []
    for suite in SUITES:
        candidates = sorted(
            by_suite[suite],
            key=lambda row: stable_rank(args.selection_salt, str(row["parent_key"])),
        )
        if args.max_parents_per_suite > 0:
            candidates = candidates[: args.max_parents_per_suite]
        selected.extend(candidates)
    if not selected:
        raise SystemExit("no parent identities selected")
    if len({str(row["parent_key"]) for row in selected}) != len(selected):
        raise SystemExit("duplicate parent identities in selected source manifest")

    GPU_SUITE_MAP = {
        "libero_object": 6,
        "libero_spatial": 6,
        "libero_goal": 7,
        "libero_10": 7,
    }
    bundle_sha = sha256_file(bundle / "SHA256SUMS") if (bundle / "SHA256SUMS").is_file() else ""
    checkpoint_sha = sha256_file(bundle / "checkpoint.pt")
    config_sha = sha256_file(bundle / "detector_config.json")
    manifests: list[dict[str, Any]] = []
    worker_counts: Counter[str] = Counter()
    gpu_counts: Counter[str] = Counter()
    parent_assignments: dict[str, tuple[int, str, int]] = {}
    for suite in SUITES:
        suite_rows = [row for row in selected if str(row["suite"]) == suite]
        suite_rows = sorted(suite_rows, key=lambda row: stable_rank(f"{args.selection_salt}|GPU", str(row["parent_key"])))
        gpu = GPU_SUITE_MAP[suite]
        worker_suffix = "l10" if suite == "libero_10" else suite.replace("libero_", "")
        worker_id = f"g{gpu}_{worker_suffix}"
        for index, row in enumerate(suite_rows):
            shard_index = index
            parent_key = str(row["parent_key"])
            parent_assignments[parent_key] = (gpu, worker_id, shard_index)

            max_steps = _int(row, "max_steps", "horizon", default=300)
            parsed_parts = parent_key.split("/")
            parsed_task = int(parsed_parts[1].replace("task_", "")) if len(parsed_parts) > 1 else 0
            parsed_state = int(parsed_parts[2].replace("state_", "")) if len(parsed_parts) > 2 else 0
            task_index = _int(row, "task_index", "task_idx", default=parsed_task)
            state_id = _int(row, "state_id", "init_state_id", default=parsed_state)
            cohort = str(row.get("cohort", row.get("split", "UNKNOWN")))
            split = str(row.get("split", cohort))
            for condition in CONDITIONS:
                planned = random_start(parent_key, max_steps, args.burst_length) if condition == "RAND_T10" else -1
                manifests.append(
                    {
                        "manifest_schema": "c2g.r9q.attack_manifest.2026-07-13.v1",
                        "parent_key": parent_key,
                        "suite": suite,
                        "task_index": task_index,
                        "state_id": state_id,
                        "cohort": cohort,
                        "split": split,
                        "max_steps": max_steps,
                        "condition": condition,
                        "protocol_name": PROTOCOL_NAME,
                        "protocol_version": PROTOCOL_VERSION,
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
                        "burst_length": args.burst_length,
                        "planned_start_step": planned,
                        "objective_seed": stable_seed(parent_key, condition),
                        "assigned_physical_gpu": gpu,
                        "assigned_worker_id": worker_id,
                        "assigned_shard_id": f"{worker_id}_shard",
                        "shard_local_index": shard_index,
                        "source_parent_manifest": str(source),
                        "source_parent_manifest_sha256": sha256_file(source),
                        "detector_bundle_sha256": bundle_sha,
                        "detector_checkpoint_sha256": checkpoint_sha,
                        "detector_config_sha256": config_sha,
                        "expected_git_commit": args.expected_git_commit,
                        "attack_outcome_used_for_selection": False,
                    }
                )
                worker_counts[worker_id] += 1
                gpu_counts[str(gpu)] += 1

    output.mkdir(parents=True)
    manifest_path = output / "r9q_attack_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifests), encoding="utf-8"
    )
    shards = output / "shards"
    shards.mkdir()
    for worker_id in sorted(worker_counts):
        worker_rows = [row for row in manifests if row["assigned_worker_id"] == worker_id]
        (shards / f"{worker_id}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in worker_rows), encoding="utf-8"
        )
    report = {
        "status": "PASS_C2G_R9Q_ATTACK_MANIFEST_MATERIALIZED",
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "expected_git_commit": args.expected_git_commit,
        "source_manifest": str(source),
        "source_manifest_sha256": sha256_file(source),
        "detector_bundle": str(bundle),
        "detector_bundle_sha256": bundle_sha,
        "detector_checkpoint_sha256": checkpoint_sha,
        "detector_config_sha256": config_sha,
        "selection_salt": args.selection_salt,
        "source_row_count": len(rows),
        "source_unique_parent_count": len(unique_source_rows),
        "total_parents": len(selected),
        "total_cells": len(manifests),
        "per_suite_parents": dict(Counter(str(row["suite"]) for row in selected)),
        "per_condition": dict(Counter(str(row["condition"]) for row in manifests)),
        "worker_counts": dict(sorted(worker_counts.items())),
        "gpu_counts": dict(sorted(gpu_counts.items())),
        "workers": sorted(worker_counts),
        "conditions": list(CONDITIONS),
        "attack_outcomes_used_for_selection": False,
    }
    report_path = output / "r9q_attack_manifest_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
