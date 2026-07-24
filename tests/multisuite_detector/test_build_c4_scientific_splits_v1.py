import csv
from pathlib import Path

import pytest

from tools.multisuite_detector.build_c4_scientific_splits_v1 import (
    C4ScientificSplitError,
    build_all_suite_stratified_split,
    build_object_task_heldout_split,
    build_suite_loso_with_val_split,
    read_split_csv,
    validate_scientific_split,
)
from tools.multisuite_detector.detector_dataset_closure_v1 import write_csv

DATASET_COLUMNS = ["episode_key", "parent_key", "suite", "task_id", "initial_state_hash", "trace_length", "population_id"]


def state(i: int) -> str:
    return f"{i:064x}"


def write_dataset(tmp_path: Path) -> Path:
    rows = []
    idx = 1
    suites_tasks = {
        "Object": ["task_a", "task_b", "task_c"],
        "Goal": ["goal_a", "goal_b"],
        "Spatial": ["spatial_a", "spatial_b"],
        "Libero10": ["libero_a", "libero_b"],
    }
    for suite, tasks in suites_tasks.items():
        for task in tasks:
            for rep in range(3):
                rows.append({
                    "episode_key": f"{suite}_{task}_{rep}",
                    "parent_key": f"parent_{suite}_{task}_{rep}",
                    "suite": suite,
                    "task_id": task,
                    "initial_state_hash": state(idx),
                    "trace_length": "4",
                    "population_id": "DETECTOR_ELIGIBLE" if rep < 2 else "DETECTOR_SAFETY",
                })
                idx += 1
    path = tmp_path / "detector_dataset_manifest_v1.csv"
    write_csv(path, DATASET_COLUMNS, rows)
    return path


def write_labels(tmp_path: Path, dataset: Path) -> Path:
    rows = []
    for row in csv.DictReader(dataset.open(newline="", encoding="utf-8")):
        is_positive = row["population_id"] == "DETECTOR_ELIGIBLE" and row["episode_key"].endswith("_0")
        rows.append({
            "episode_key": row["episode_key"],
            "event_present": "true" if is_positive else "false",
            "window_valid": "true" if is_positive else "false",
            "window_start": "1" if is_positive else "-1",
            "window_end": "3" if is_positive else "-1",
        })
    path = tmp_path / "label_v2.csv"
    write_csv(path, ["episode_key", "event_present", "window_valid", "window_start", "window_end"], rows)
    return path


def test_object_task_heldout_split_positive(tmp_path):
    dataset = write_dataset(tmp_path)
    split = tmp_path / "object_task_heldout.csv"
    report = build_object_task_heldout_split(dataset, split, seed=7, val_ratio=0.25)
    assert report["schema_version"] == "c4_scientific_split_manifest_v1"
    assert "object_task_heldout_with_val_v1" in report["split_types"]
    validation = validate_scientific_split(dataset, split)
    assert validation["status"] == "PASS"
    assert validation["fold_count"] == 3
    rows = read_split_csv(split)
    for fold_id in validation["folds"]:
        held = fold_id.replace("object_task_heldout_", "")
        test_eps = {r["episode_key"] for r in rows if r["fold_id"] == fold_id and r["split"] == "test"}
        assert all(f"Object_{held}_" in ep for ep in test_eps)


def test_object_task_heldout_accepts_formal_libero_object_suite_alias(tmp_path):
    dataset = write_dataset(tmp_path)
    text = dataset.read_text(encoding="utf-8").replace(",Object,", ",libero_object,")
    dataset.write_text(text, encoding="utf-8")
    split = tmp_path / "object_task_heldout.csv"
    build_object_task_heldout_split(dataset, split, seed=7, val_ratio=0.25)
    validation = validate_scientific_split(dataset, split)
    assert validation["status"] == "PASS"
    assert validation["fold_count"] == 3


def test_suite_loso_split_positive(tmp_path):
    dataset = write_dataset(tmp_path)
    split = tmp_path / "suite_loso.csv"
    report = build_suite_loso_with_val_split(dataset, split, seed=11, val_ratio=0.25)
    assert "suite_loso_with_val_v1" in report["split_types"]
    validation = validate_scientific_split(dataset, split)
    assert validation["status"] == "PASS"
    assert validation["fold_count"] == 4
    rows = read_split_csv(split)
    for fold_id in validation["folds"]:
        held = fold_id.replace("suite_loso_", "")
        test_eps = {r["episode_key"] for r in rows if r["fold_id"] == fold_id and r["split"] == "test"}
        assert test_eps
        assert all(ep.startswith(f"{held}_") for ep in test_eps)


def test_all_suite_stratified_split_positive(tmp_path):
    dataset = write_dataset(tmp_path)
    labels = write_labels(tmp_path, dataset)
    split = tmp_path / "all_suite_stratified.csv"
    report = build_all_suite_stratified_split(dataset, labels, split, seed=13, val_ratio=0.20, test_ratio=0.20)
    assert "all_suite_stratified_parent_split_v1" in report["split_types"]
    validation = validate_scientific_split(dataset, split, label_csv=labels)
    assert validation["status"] == "PASS"
    assert validation["fold_count"] == 1
    rows = read_split_csv(split)
    suites = {row["suite"] for row in csv.DictReader(dataset.open(newline="", encoding="utf-8"))}
    by_episode = {row["episode_key"]: row for row in csv.DictReader(dataset.open(newline="", encoding="utf-8"))}
    for split_name in {"train", "val", "test"}:
        split_suites = {by_episode[row["episode_key"]]["suite"] for row in rows if row["split"] == split_name}
        assert split_suites == suites


def test_scientific_split_rejects_group_leakage(tmp_path):
    dataset = write_dataset(tmp_path)
    split = tmp_path / "suite_loso.csv"
    build_suite_loso_with_val_split(dataset, split, seed=11, val_ratio=0.25)
    rows = read_split_csv(split)
    # Tamper one row's split within a fold/group to create leakage.
    target_group = rows[0]["group_id"]
    target_fold = rows[0]["fold_id"]
    for row in rows:
        if row["fold_id"] == target_fold and row["group_id"] == target_group:
            row["split"] = "train" if row["split"] != "train" else "val"
            break
    write_csv(split, ["split_type", "fold_id", "group_id", "episode_key", "split"], rows)
    with pytest.raises(C4ScientificSplitError, match="held-out suite leakage|non-held suite episode placed in test|parent/state group leakage"):
        validate_scientific_split(dataset, split)


def test_scientific_split_rejects_test_leakage(tmp_path):
    dataset = write_dataset(tmp_path)
    split = tmp_path / "object_task_heldout.csv"
    build_object_task_heldout_split(dataset, split, seed=7, val_ratio=0.25)
    rows = read_split_csv(split)
    for row in rows:
        if row["fold_id"] == "object_task_heldout_task_a" and row["episode_key"].startswith("Object_task_a_"):
            row["split"] = "train"
            break
    write_csv(split, ["split_type", "fold_id", "group_id", "episode_key", "split"], rows)
    with pytest.raises(C4ScientificSplitError, match="held-out Object task leakage"):
        validate_scientific_split(dataset, split)


def test_scientific_split_rejects_missing_val_eligible(tmp_path):
    dataset = write_dataset(tmp_path)
    split = tmp_path / "suite_loso.csv"
    build_suite_loso_with_val_split(dataset, split, seed=11, val_ratio=0.25)
    rows = read_split_csv(split)
    # Move all validation rows in one fold to train.
    fold = rows[0]["fold_id"]
    for row in rows:
        if row["fold_id"] == fold and row["split"] == "val":
            row["split"] = "train"
    write_csv(split, ["split_type", "fold_id", "group_id", "episode_key", "split"], rows)
    with pytest.raises(C4ScientificSplitError, match="train/val/test must all be non-empty"):
        validate_scientific_split(dataset, split)
