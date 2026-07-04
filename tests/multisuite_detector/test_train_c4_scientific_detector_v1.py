import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.gripper_attack.sc5mlp_v1 import SC5_FEATURES
from tools.multisuite_detector.build_c4_scientific_splits_v1 import (
    build_object_task_heldout_split,
    build_suite_loso_with_val_split,
)
from tools.multisuite_detector.detector_dataset_closure_v1 import sha256_file
from tools.multisuite_detector.train_c4_scientific_detector_v1 import C4ScientificTrainingError, run_training

DATASET_COLUMNS = ["episode_key", "parent_key", "suite", "task_id", "initial_state_hash", "trace_length", "population_id"]


def state(i: int) -> str:
    return f"{i:064x}"


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_fixture(tmp_path: Path):
    dataset_rows = []
    feature_rows = []
    label_rows = []
    idx = 1
    suites = {
        "Object": ["task_a", "task_b", "task_c"],
        "Goal": ["goal_a", "goal_b", "goal_c"],
        "Spatial": ["spatial_a", "spatial_b", "spatial_c"],
    }
    for suite, tasks in suites.items():
        for task in tasks:
            for rep in range(3):
                ep = f"{suite}_{task}_{rep}"
                dataset_rows.append({
                    "episode_key": ep,
                    "parent_key": f"parent_{suite}_{task}_{rep}",
                    "suite": suite,
                    "task_id": task,
                    "initial_state_hash": state(idx),
                    "trace_length": "4",
                    "population_id": "DETECTOR_ELIGIBLE" if rep < 2 else "DETECTOR_SAFETY",
                })
                label_rows.append({
                    "episode_key": ep,
                    "event_present": "true" if rep < 2 else "false",
                    "window_valid": "true" if rep < 2 else "false",
                    "window_start": "1" if rep < 2 else "-1",
                    "window_end": "3" if rep < 2 else "-1",
                })
                for step in range(4):
                    row = {"episode_key": ep, "step": str(step)}
                    for j, name in enumerate(SC5_FEATURES):
                        row[name] = f"{0.01 * idx + 0.1 * step + 0.001 * j:.6f}"
                    feature_rows.append(row)
                idx += 1
    dataset = tmp_path / "detector_dataset_manifest_v1.csv"
    features = tmp_path / "features.csv"
    labels = tmp_path / "labels.csv"
    write_csv(dataset, DATASET_COLUMNS, dataset_rows)
    write_csv(features, ["episode_key", "step", *SC5_FEATURES], feature_rows)
    write_csv(labels, ["episode_key", "event_present", "window_valid", "window_start", "window_end"], label_rows)
    return dataset, features, labels


def run_wrapper(dataset: Path, features: Path, labels: Path, split: Path, fold_id: str, output: Path, **overrides):
    args = dict(
        dataset_csv=str(dataset),
        feature_csv=str(features),
        split_csv=str(split),
        fold_id=fold_id,
        population="DETECTOR_ELIGIBLE",
        label_csv=str(labels),
        label_artifact_root=None,
        seed=7,
        epochs=2,
        batch_size=8,
        backend="numpy",
        expected_dataset_csv_sha256=sha256_file(dataset),
        expected_split_csv_sha256=sha256_file(split),
        expected_state_index_sha256="5" * 64,
        output_root=str(output),
    )
    args.update(overrides)
    return run_training(type("Args", (), args)())


def test_object_task_heldout_training_smoke(tmp_path):
    dataset, features, labels = make_fixture(tmp_path)
    split = tmp_path / "object_task_heldout_with_val_v1.csv"
    build_object_task_heldout_split(dataset, split, seed=7, val_ratio=0.25)
    report = run_wrapper(dataset, features, labels, split, "object_task_heldout_task_a", tmp_path / "out_object")
    assert report["status"] == "PASS"
    assert report["split_type"] == "object_task_heldout_with_val_v1"
    assert report["threshold_source"] == "validation"
    out = tmp_path / "out_object"
    for name in ["training_config.json", "dataset_identity.json", "split_identity.json", "normalization_identity.json", "threshold_selection.json", "metrics_summary.json", "metrics_by_suite.csv", "metrics_by_task.csv", "checkpoint_last.pt", "best_checkpoint.pt", "bundle_load_report.json", "SHA256SUMS", "SHA256SUMS.sha256"]:
        assert (out / name).is_file()
    norm = json.loads((out / "normalization_identity.json").read_text())
    assert norm["normalization_source"] == "train_only"


def test_suite_loso_training_smoke(tmp_path):
    dataset, features, labels = make_fixture(tmp_path)
    split = tmp_path / "suite_loso_with_val_v1.csv"
    build_suite_loso_with_val_split(dataset, split, seed=7, val_ratio=0.25)
    report = run_wrapper(dataset, features, labels, split, "suite_loso_Object", tmp_path / "out_suite")
    assert report["status"] == "PASS"
    assert report["split_type"] == "suite_loso_with_val_v1"
    assert report["test"]["count"] > 0


def test_training_wrapper_cli(tmp_path):
    dataset, features, labels = make_fixture(tmp_path)
    split = tmp_path / "object_task_heldout_with_val_v1.csv"
    build_object_task_heldout_split(dataset, split, seed=7, val_ratio=0.25)
    out = tmp_path / "cli_out"
    subprocess.run([
        sys.executable,
        "tools/multisuite_detector/train_c4_scientific_detector_v1.py",
        "--dataset-csv", str(dataset),
        "--feature-csv", str(features),
        "--split-csv", str(split),
        "--fold-id", "object_task_heldout_task_a",
        "--population", "DETECTOR_ELIGIBLE",
        "--label-csv", str(labels),
        "--backend", "numpy",
        "--epochs", "1",
        "--batch-size", "8",
        "--output-root", str(out),
    ], check=True)
    assert (out / "metrics_summary.json").is_file()


def test_rejects_missing_fold(tmp_path):
    dataset, features, labels = make_fixture(tmp_path)
    split = tmp_path / "suite_loso_with_val_v1.csv"
    build_suite_loso_with_val_split(dataset, split, seed=7, val_ratio=0.25)
    with pytest.raises(C4ScientificTrainingError, match="fold_id not found"):
        run_wrapper(dataset, features, labels, split, "missing_fold", tmp_path / "out")


def test_rejects_parent_random_split_type(tmp_path):
    dataset, features, labels = make_fixture(tmp_path)
    split = tmp_path / "bad_split.csv"
    rows = []
    for row in csv.DictReader(dataset.open(newline="", encoding="utf-8")):
        split_name = "train" if row["task_id"] != "task_a" else "test"
        if row["task_id"] == "goal_a":
            split_name = "val"
        rows.append({"split_type": "parent_random_split_v1", "fold_id": "parent_random", "group_id": row["episode_key"], "episode_key": row["episode_key"], "split": split_name})
    write_csv(split, ["split_type", "fold_id", "group_id", "episode_key", "split"], rows)
    with pytest.raises(C4ScientificTrainingError, match="unsupported scientific split type"):
        run_wrapper(dataset, features, labels, split, "parent_random", tmp_path / "out")


def test_rejects_dataset_identity_mismatch(tmp_path):
    dataset, features, labels = make_fixture(tmp_path)
    split = tmp_path / "object_task_heldout_with_val_v1.csv"
    build_object_task_heldout_split(dataset, split, seed=7, val_ratio=0.25)
    with pytest.raises(C4ScientificTrainingError, match="dataset identity mismatch"):
        run_wrapper(dataset, features, labels, split, "object_task_heldout_task_a", tmp_path / "out", expected_dataset_csv_sha256="0" * 64)


def test_rejects_nan_feature(tmp_path):
    dataset, features, labels = make_fixture(tmp_path)
    text = features.read_text()
    features.write_text(text.replace("0.010000", "nan", 1), encoding="utf-8")
    split = tmp_path / "object_task_heldout_with_val_v1.csv"
    build_object_task_heldout_split(dataset, split, seed=7, val_ratio=0.25)
    with pytest.raises(C4ScientificTrainingError, match="finite float"):
        run_wrapper(dataset, features, labels, split, "object_task_heldout_task_a", tmp_path / "out", expected_dataset_csv_sha256=sha256_file(dataset), expected_split_csv_sha256=sha256_file(split))
