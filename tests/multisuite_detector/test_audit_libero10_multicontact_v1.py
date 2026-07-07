import csv
import json
from pathlib import Path

from src.gripper_attack.sc5mlp_v1 import SC5_FEATURES
from tools.multisuite_detector.audit_libero10_multicontact_v1 import run_audit

DATASET_COLUMNS = ["episode_key", "parent_key", "suite", "task_id", "initial_state_hash", "trace_length", "population_id"]
SPLIT_COLUMNS = ["split_type", "fold_id", "group_id", "episode_key", "split"]


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_fixture(tmp_path: Path):
    dataset = tmp_path / "dataset.csv"
    labels = tmp_path / "label_v2.csv"
    features = tmp_path / "features.csv"
    split = tmp_path / "split.csv"
    write_csv(dataset, DATASET_COLUMNS, [
        {"episode_key": "l10_ep", "parent_key": "p1", "suite": "libero_10", "task_id": "long_task", "initial_state_hash": "1" * 64, "trace_length": "8", "population_id": "DETECTOR_ELIGIBLE"},
        {"episode_key": "obj_ep", "parent_key": "p2", "suite": "libero_object", "task_id": "obj_task", "initial_state_hash": "2" * 64, "trace_length": "4", "population_id": "DETECTOR_ELIGIBLE"},
    ])
    label_cols = ["episode_key", "suite", "task_id", "event_present", "window_valid", "window_start", "window_end"]
    write_csv(labels, label_cols, [
        {"episode_key": "l10_ep", "suite": "libero_10", "task_id": "long_task", "event_present": "false", "window_valid": "false", "window_start": "-1", "window_end": "-1"},
        {"episode_key": "obj_ep", "suite": "libero_object", "task_id": "obj_task", "event_present": "true", "window_valid": "true", "window_start": "1", "window_end": "3"},
    ])
    rows = []
    for ep, steps in [("l10_ep", 8), ("obj_ep", 4)]:
        for step in range(steps):
            row = {"episode_key": ep, "step": str(step)}
            for name in SC5_FEATURES:
                row[name] = "0.0"
            if ep == "l10_ep" and step in {1, 2, 5, 6}:
                row["gripper_command"] = "-1.0"
                row["action_gripper"] = "-1.0"
                row["recent_close_streak"] = str(1 + (step % 2))
                row["qpos_delta_1"] = "0.01"
            if ep == "l10_ep" and step in {3, 7}:
                row["gripper_command"] = "1.0"
                row["action_gripper"] = "1.0"
                row["recent_open_streak"] = "1"
            rows.append(row)
    write_csv(features, ["episode_key", "step", *SC5_FEATURES], rows)
    write_csv(split, SPLIT_COLUMNS, [
        {"split_type": "all_suite_stratified_parent_split_v1", "fold_id": "all_suite_stratified", "group_id": "g1", "episode_key": "l10_ep", "split": "test"},
        {"split_type": "all_suite_stratified_parent_split_v1", "fold_id": "all_suite_stratified", "group_id": "g2", "episode_key": "obj_ep", "split": "train"},
    ])
    return dataset, features, labels, split


def test_libero10_multicontact_audit_outputs(tmp_path):
    dataset, features, labels, split = make_fixture(tmp_path)
    out = tmp_path / "out"
    args = type("Args", (), {
        "dataset_csv": str(dataset),
        "feature_csv": str(features),
        "label_csv": str(labels),
        "split_csv": str(split),
        "fold_id": "all_suite_stratified",
        "target_suite": "libero_10",
        "min_segment_len": 2,
        "min_response_delta": 1e-6,
        "output_root": str(out),
    })()
    report = run_audit(args)
    assert report["status"] == "PASS"
    assert report["target_suite"] == "libero_10"
    assert report["episode_count"] == 1
    assert report["label_positive_episode_count"] == 0
    assert report["candidate_contact_episode_count"] == 1
    assert report["classification_counts"]["MULTI_CONTACT_LONG_HORIZON"] == 1
    for name in [
        "libero10_episode_support_summary.csv",
        "libero10_gripper_event_segments.csv",
        "libero10_label_v2_alignment.csv",
        "libero10_multicontact_candidate_windows.csv",
        "libero10_no_positive_reason_report.json",
        "libero10_detector_score_overlay.json",
        "SHA256SUMS",
        "SHA256SUMS.sha256",
    ]:
        assert (out / name).is_file()
    saved = json.loads((out / "libero10_no_positive_reason_report.json").read_text())
    assert saved["new_training"] == "NOT_PERFORMED"
    assert saved["label_mutation"] == "NOT_PERFORMED"
