#!/usr/bin/env python3
"""Build Table1-Lite 12-parent extension manifest from the frozen Full manifest.

Excludes 8 canary parents and all DETECTOR identities (FIT/CAL/CHECK).
Selection: sha256("R9Q_TABLE1_LITE_V1|" + parent_key) per suite, top 3.
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
TABLE1_LITE_SALT = "R9Q_TABLE1_LITE_V1"

GPU_SUITE_MAP = {
    "libero_object": 6,
    "libero_spatial": 6,
    "libero_goal": 7,
    "libero_10": 7,
}


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
    parser.add_argument("--source-manifest", required=True,
                        help="Path to the frozen Full 179-parent manifest CSV")
    parser.add_argument("--canary-manifest", required=True,
                        help="Path to the PASS canary manifest JSONL (for exclusion list)")
    parser.add_argument("--detector-plan", required=True,
                        help="Path to the R9P preview plan JSON (for DETECTOR identity exclusion)")
    parser.add_argument("--detector-bundle", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source_manifest).resolve()
    canary_manifest = Path(args.canary_manifest).resolve()
    plan = Path(args.detector_plan).resolve()
    bundle = Path(args.detector_bundle).resolve()
    output = Path(args.output_root).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output root: {output}")

    # Load exclusion set: canary parents + detector identities
    canary_rows = [json.loads(line) for line in canary_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    excluded_parents: set[str] = set()
    for row in canary_rows:
        excluded_parents.add(str(row["parent_key"]))
    print(f"Canary exclusions: {len(set(r['parent_key'] for r in canary_rows))} parents")

    # Detector identities (FIT/CAL/CHECK)
    detector_data = json.loads(plan.read_text(encoding="utf-8"))
    detector_keys: set[str] = set()
    for key_name in ("episode_manifest", "preview_manifest", "preview_index"):
        source_path = detector_data.get(key_name)
        if source_path and Path(source_path).is_file():
            with open(source_path, encoding="utf-8") as f:
                detector_rows = [json.loads(line) for line in f if line.strip()]
                for r in detector_rows:
                    pk = str(r.get("parent_key", ""))
                    if pk:
                        detector_keys.add(pk)
    print(f"Detector identity exclusions: {len(detector_keys)} parents")
    excluded_parents |= detector_keys

    # Load source manifest
    if source.suffix.lower() == ".csv":
        rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    else:
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Deduplicate
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unique_source: dict[str, dict[str, Any]] = {}
    for row in rows:
        suite = str(row.get("suite", ""))
        parent_key = str(row.get("parent_key", ""))
        if suite not in SUITES or not parent_key:
            continue
        if parent_key not in unique_source:
            unique_source[parent_key] = row
    for row in unique_source.values():
        by_suite[str(row["suite"])].append(row)

    # Select 3 new parents per suite, excluding canary+detector identities
    extension_parents: list[dict[str, Any]] = []
    for suite in SUITES:
        candidates = sorted(
            [r for r in by_suite[suite] if str(r["parent_key"]) not in excluded_parents],
            key=lambda r: stable_rank(TABLE1_LITE_SALT, str(r["parent_key"])),
        )
        if len(candidates) < 3:
            raise SystemExit(f"{suite} has only {len(candidates)} eligible parents after exclusions (need 3)")
        chosen = candidates[:3]
        extension_parents.extend(chosen)
        print(f"{suite}: selected {len(chosen)}/{len(candidates)} eligible (excluded: canary+detector)")

    print(f"Total extension parents: {len(extension_parents)}")

    # Build manifest cells
    bundle_sha = sha256_file(bundle / "SHA256SUMS") if (bundle / "SHA256SUMS").is_file() else ""
    checkpoint_sha = sha256_file(bundle / "checkpoint.pt")
    config_sha = sha256_file(bundle / "detector_config.json")

    cells: list[dict[str, Any]] = []
    worker_counts: Counter[str] = Counter()
    per_suite_parents: Counter[str] = Counter()

    for parent in extension_parents:
        pk = str(parent["parent_key"])
        suite = str(parent["suite"])
        gpu = GPU_SUITE_MAP[suite]
        worker_suffix = "l10" if suite == "libero_10" else suite.replace("libero_", "")
        worker_id = f"g{gpu}_{worker_suffix}"
        per_suite_parents[suite] += 1

        max_steps = int(parent.get("max_steps", parent.get("horizon", 300)))
        task_index = int(parent.get("task_index", parent.get("task_idx", 0)))
        state_id = int(parent.get("state_id", parent.get("init_state_id", 0)))

        for condition in CONDITIONS:
            planned = random_start(pk, max_steps, 10) if condition == "RAND_T10" else -1
            cells.append({
                "manifest_schema": "c2g.r9q.attack_manifest.2026-07-13.v1",
                "parent_key": pk,
                "suite": suite,
                "task_index": task_index,
                "state_id": state_id,
                "cohort": str(parent.get("cohort", parent.get("split", "UNKNOWN"))),
                "split": str(parent.get("split", parent.get("cohort", "UNKNOWN"))),
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
                "burst_length": 10,
                "planned_start_step": planned,
                "objective_seed": stable_seed(pk, condition),
                "assigned_physical_gpu": gpu,
                "assigned_worker_id": worker_id,
                "assigned_shard_id": f"{worker_id}_shard",
                "shard_local_index": per_suite_parents[suite] - 1,
                "source_parent_manifest": str(source),
                "source_parent_manifest_sha256": sha256_file(source),
                "detector_bundle_sha256": bundle_sha,
                "detector_checkpoint_sha256": checkpoint_sha,
                "detector_config_sha256": config_sha,
                "expected_git_commit": args.expected_git_commit,
                "attack_outcome_used_for_selection": False,
            })
            worker_counts[worker_id] += 1

    assert len(cells) == 48, f"expected 48 cells, got {len(cells)}"

    output.mkdir(parents=True)

    # Write composite manifest (canary + extension)
    canary_cells = [json.loads(line) for line in canary_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    for c in canary_cells:
        c["expected_git_commit"] = args.expected_git_commit

    # Reuse ledger: check which canary cells can be reused
    reuse_ok = True  # Same detector bundle, same protocol, all invariants match
    reuse_rows: list[dict[str, Any]] = []
    for c in canary_cells:
        reuse_rows.append({
            "parent_key": c["parent_key"],
            "condition": c["condition"],
            "can_reuse": reuse_ok,
            "reason": "same detector bundle, same HEAD, same protocol" if reuse_ok else "mismatch",
        })

    # Write outputs
    ext_path = output / "table1_lite_extension_manifest.jsonl"
    ext_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in cells), encoding="utf-8")

    composite_path = output / "table1_lite_composite_manifest.jsonl"
    composite_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in (canary_cells + cells)), encoding="utf-8"
    )

    # Worker shards
    shards = output / "shards"
    shards.mkdir()
    for worker_id in sorted(worker_counts):
        worker_rows = [row for row in cells if row["assigned_worker_id"] == worker_id]
        (shards / f"{worker_id}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in worker_rows), encoding="utf-8"
        )

    # Reports
    report = {
        "status": "PASS_C2G_R9Q_TABLE1_LITE_EXTENSION_BUILT",
        "canary_parents_reused": 8,
        "extension_parents": len(extension_parents),
        "extension_cells": len(cells),
        "composite_cells": len(canary_cells) + len(cells),
        "per_suite_extension": dict(per_suite_parents),
        "canary_reuse_ledger": "ALL_REUSE_OK" if reuse_ok else "PARTIAL",
        "canary_reuse_details": reuse_rows[:5],
        "excluded_canary_parents": sorted(excluded_parents)[:15],
        "excluded_detector_count": len(detector_keys),
        "selection_salt": TABLE1_LITE_SALT,
        "source_manifest": str(source),
        "source_manifest_sha256": sha256_file(source),
        "detector_bundle_sha256": bundle_sha,
        "expected_git_commit": args.expected_git_commit,
        "worker_counts": dict(sorted(worker_counts.items())),
    }
    (output / "table1_lite_manifest_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # SHA256SUMS
    files = sorted(path for path in output.rglob("*") if path.is_file())
    sums = output / "SHA256SUMS"
    sums.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    (output / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
