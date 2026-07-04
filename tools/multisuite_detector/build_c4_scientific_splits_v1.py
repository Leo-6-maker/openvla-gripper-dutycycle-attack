#!/usr/bin/env python3
"""Build and validate C4 scientific detector splits.

This tool is CPU-only and metadata-only. It never trains a detector, runs
OpenVLA/LIBERO, performs rollouts, attacks, exact-prefix replay, or uses GPU.

The split builders here are intended for formal detector training after the
parent-random C4-1 candidate. They add validation folds to held-out test
protocols so thresholds remain validation-selected rather than test-selected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.multisuite_detector.detector_dataset_closure_v1 import (  # noqa: E402
    DetectorDatasetClosureError,
    connected_groups,
    load_dataset_manifest,
    sha256_file,
    write_csv,
)

SPLIT_COLUMNS = ["split_type", "fold_id", "group_id", "episode_key", "split"]
SPLITS = {"train", "val", "test"}
POPULATIONS = {"DETECTOR_ELIGIBLE", "DETECTOR_SAFETY"}


class C4ScientificSplitError(ValueError):
    pass


def fail(message: str) -> None:
    raise C4ScientificSplitError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_split_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != SPLIT_COLUMNS:
            fail(f"{path.name}: expected exact split header")
        rows = []
        for line_no, row in enumerate(reader, start=2):
            if None in row:
                fail(f"{path.name}:{line_no}: extra cells")
            if any(v is None or v == "" for v in row.values()):
                fail(f"{path.name}:{line_no}: empty field")
            if row["split"] not in SPLITS:
                fail(f"{path.name}:{line_no}: invalid split")
            rows.append(row)
        return rows


def group_rows(dataset_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    groups = connected_groups(dataset_rows)
    by_episode = {row["episode_key"]: row for row in dataset_rows}
    if not groups:
        fail("dataset has no connected groups")
    return groups, by_episode


def stable_group_order(group_ids: list[str], *, seed: int, fold_id: str) -> list[str]:
    ids = list(group_ids)
    random.Random(f"{seed}:{fold_id}").shuffle(ids)
    return ids


def assign_validation(
    assignments: dict[str, str],
    candidate_group_ids: list[str],
    *,
    seed: int,
    fold_id: str,
    val_ratio: float,
) -> None:
    if not (0 < val_ratio < 1):
        fail("val_ratio must be in (0, 1)")
    train_like = [gid for gid in candidate_group_ids if assignments[gid] == "train"]
    if len(train_like) < 2:
        fail(f"{fold_id}: not enough train-candidate groups for validation split")
    n_val = max(1, int(round(len(train_like) * val_ratio)))
    if n_val >= len(train_like):
        n_val = len(train_like) - 1
    for gid in stable_group_order(train_like, seed=seed, fold_id=fold_id)[:n_val]:
        assignments[gid] = "val"


def rows_from_assignments(groups: list[dict[str, Any]], assignments: dict[str, str], split_type: str, fold_id: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for group in groups:
        gid = str(group["group_id"])
        split = assignments[gid]
        for episode in group["episodes"]:
            out.append({
                "split_type": split_type,
                "fold_id": fold_id,
                "group_id": gid,
                "episode_key": str(episode),
                "split": split,
            })
    return sorted(out, key=lambda row: (row["split_type"], row["fold_id"], row["split"], row["episode_key"]))


def write_split_with_report(output: str | Path, rows: list[dict[str, str]], dataset_csv: str | Path, extra: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        fail("refusing to write empty split")
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, SPLIT_COLUMNS, rows)
    report = {
        "schema_version": "c4_scientific_split_manifest_v1",
        "split_types": sorted({row["split_type"] for row in rows}),
        "fold_ids": sorted({row["fold_id"] for row in rows}),
        "source_dataset_manifest_sha256": sha256_file(Path(dataset_csv)),
        "split_manifest_path": str(out),
        "split_manifest_sha256": sha256_file(out),
        "row_count": len(rows),
        "OpenVLA": "NOT_PERFORMED",
        "LIBERO": "NOT_PERFORMED",
        "rollout": "NOT_PERFORMED",
        "attack": "NOT_PERFORMED",
        "detector_training": "NOT_PERFORMED",
        **extra,
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_object_task_heldout_split(
    dataset_csv: str | Path,
    output: str | Path,
    *,
    seed: int = 2026070401,
    val_ratio: float = 0.15,
) -> dict[str, Any]:
    rows = load_dataset_manifest(dataset_csv)
    groups, by_episode = group_rows(rows)
    object_tasks = sorted({row["task_id"] for row in rows if row["suite"] == "Object"})
    if not object_tasks:
        fail("dataset contains no Object tasks")
    out: list[dict[str, str]] = []
    for task in object_tasks:
        fold_id = f"object_task_heldout_{task}"
        assignments: dict[str, str] = {}
        for group in groups:
            gid = str(group["group_id"])
            group_eps = [by_episode[str(ep)] for ep in group["episodes"]]
            is_held = any(row["suite"] == "Object" and row["task_id"] == task for row in group_eps)
            assignments[gid] = "test" if is_held else "train"
        assign_validation(assignments, [str(g["group_id"]) for g in groups], seed=seed, fold_id=fold_id, val_ratio=val_ratio)
        out.extend(rows_from_assignments(groups, assignments, "object_task_heldout_with_val_v1", fold_id))
    return write_split_with_report(output, out, dataset_csv, {"held_out_object_tasks": object_tasks, "seed": seed, "val_ratio": val_ratio})


def build_suite_loso_with_val_split(
    dataset_csv: str | Path,
    output: str | Path,
    *,
    seed: int = 2026070401,
    val_ratio: float = 0.15,
) -> dict[str, Any]:
    rows = load_dataset_manifest(dataset_csv)
    groups, by_episode = group_rows(rows)
    suites = sorted({row["suite"] for row in rows})
    if len(suites) < 2:
        fail("suite LOSO requires at least two suites")
    out: list[dict[str, str]] = []
    for suite in suites:
        fold_id = f"suite_loso_{suite}"
        assignments: dict[str, str] = {}
        for group in groups:
            gid = str(group["group_id"])
            group_eps = [by_episode[str(ep)] for ep in group["episodes"]]
            is_held = any(row["suite"] == suite for row in group_eps)
            assignments[gid] = "test" if is_held else "train"
        assign_validation(assignments, [str(g["group_id"]) for g in groups], seed=seed, fold_id=fold_id, val_ratio=val_ratio)
        out.extend(rows_from_assignments(groups, assignments, "suite_loso_with_val_v1", fold_id))
    return write_split_with_report(output, out, dataset_csv, {"held_out_suites": suites, "seed": seed, "val_ratio": val_ratio})


def validate_scientific_split(
    dataset_csv: str | Path,
    split_csv: str | Path,
    *,
    min_eligible_train: int = 1,
    min_eligible_val: int = 1,
    min_eligible_test: int = 1,
) -> dict[str, Any]:
    dataset_rows = load_dataset_manifest(dataset_csv)
    split_rows = read_split_csv(split_csv)
    groups, by_episode = group_rows(dataset_rows)
    group_by_episode = {str(ep): str(group["group_id"]) for group in groups for ep in group["episodes"]}
    group_episodes = {str(group["group_id"]): {str(ep) for ep in group["episodes"]} for group in groups}
    all_eps = set(by_episode)
    by_fold: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    split_types_by_fold: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        episode = row["episode_key"]
        if episode not in by_episode:
            fail(f"split references unknown episode: {episode}")
        if row["group_id"] != group_by_episode[episode]:
            fail(f"{episode}: split group_id mismatch")
        fold = by_fold[row["fold_id"]]
        if episode in fold:
            fail(f"{row['fold_id']}: duplicate split assignment for {episode}")
        fold[episode] = row
        split_types_by_fold[row["fold_id"]].add(row["split_type"])
    if not by_fold:
        fail("split has no folds")
    fold_reports: dict[str, Any] = {}
    for fold_id, assigned_rows in sorted(by_fold.items()):
        if set(assigned_rows) != all_eps:
            fail(f"{fold_id}: split coverage mismatch")
        if len(split_types_by_fold[fold_id]) != 1:
            fail(f"{fold_id}: mixed split types")
        for gid, eps in group_episodes.items():
            splits = {assigned_rows[ep]["split"] for ep in eps}
            if len(splits) != 1:
                fail(f"{fold_id}: parent/state group leakage: {gid}")
        split_type = next(iter(split_types_by_fold[fold_id]))
        counts = Counter(row["split"] for row in assigned_rows.values())
        pop_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
        for episode, split_row in assigned_rows.items():
            population = by_episode[episode]["population_id"]
            if population not in POPULATIONS:
                fail(f"{episode}: unknown population_id")
            pop_counts[split_row["split"]][population] += 1
        if counts["train"] <= 0 or counts["val"] <= 0 or counts["test"] <= 0:
            fail(f"{fold_id}: train/val/test must all be non-empty")
        if pop_counts["train"]["DETECTOR_ELIGIBLE"] < min_eligible_train:
            fail(f"{fold_id}: insufficient DETECTOR_ELIGIBLE train episodes")
        if pop_counts["val"]["DETECTOR_ELIGIBLE"] < min_eligible_val:
            fail(f"{fold_id}: insufficient DETECTOR_ELIGIBLE val episodes")
        if pop_counts["test"]["DETECTOR_ELIGIBLE"] < min_eligible_test:
            fail(f"{fold_id}: insufficient DETECTOR_ELIGIBLE test episodes")
        if split_type == "object_task_heldout_with_val_v1":
            held_task = fold_id.replace("object_task_heldout_", "")
            for episode, split_row in assigned_rows.items():
                row = by_episode[episode]
                is_held = row["suite"] == "Object" and row["task_id"] == held_task
                if is_held and split_row["split"] != "test":
                    fail(f"{fold_id}: held-out Object task leakage")
                if (not is_held) and split_row["split"] == "test":
                    fail(f"{fold_id}: non-held episode placed in test")
        elif split_type == "suite_loso_with_val_v1":
            held_suite = fold_id.replace("suite_loso_", "")
            for episode, split_row in assigned_rows.items():
                row = by_episode[episode]
                is_held = row["suite"] == held_suite
                if is_held and split_row["split"] != "test":
                    fail(f"{fold_id}: held-out suite leakage")
                if (not is_held) and split_row["split"] == "test":
                    fail(f"{fold_id}: non-held suite episode placed in test")
        else:
            fail(f"{fold_id}: unsupported scientific split type {split_type}")
        fold_reports[fold_id] = {
            "split_type": split_type,
            "counts": dict(counts),
            "population_counts": {split: dict(counter) for split, counter in pop_counts.items()},
        }
    return {
        "status": "PASS",
        "schema_version": "c4_scientific_split_validation_v1",
        "dataset_csv": str(dataset_csv),
        "dataset_csv_sha256": sha256_file(Path(dataset_csv)),
        "split_csv": str(split_csv),
        "split_csv_sha256": sha256_file(Path(split_csv)),
        "fold_count": len(fold_reports),
        "folds": fold_reports,
        "OpenVLA": "NOT_PERFORMED",
        "LIBERO": "NOT_PERFORMED",
        "rollout": "NOT_PERFORMED",
        "attack": "NOT_PERFORMED",
        "detector_training": "NOT_PERFORMED",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build-object-task-heldout")
    p.add_argument("--dataset-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=2026070401)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p = sub.add_parser("build-suite-loso")
    p.add_argument("--dataset-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=2026070401)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p = sub.add_parser("validate")
    p.add_argument("--dataset-csv", required=True)
    p.add_argument("--split-csv", required=True)
    p.add_argument("--min-eligible-train", type=int, default=1)
    p.add_argument("--min-eligible-val", type=int, default=1)
    p.add_argument("--min-eligible-test", type=int, default=1)
    p.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "build-object-task-heldout":
            report = build_object_task_heldout_split(args.dataset_csv, args.output, seed=args.seed, val_ratio=args.val_ratio)
        elif args.cmd == "build-suite-loso":
            report = build_suite_loso_with_val_split(args.dataset_csv, args.output, seed=args.seed, val_ratio=args.val_ratio)
        else:
            report = validate_scientific_split(
                args.dataset_csv,
                args.split_csv,
                min_eligible_train=args.min_eligible_train,
                min_eligible_val=args.min_eligible_val,
                min_eligible_test=args.min_eligible_test,
            )
            if args.output_json:
                out = Path(args.output_json)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        return 0
    except (OSError, json.JSONDecodeError, csv.Error, DetectorDatasetClosureError, C4ScientificSplitError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
