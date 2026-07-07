import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.multisuite_detector.detector_dataset_closure_v1 import (
    DetectorDatasetClosureError,
    FEATURE_SCHEMA_SHA256,
    SC5_FEATURES,
    build_dataset_manifest,
    build_normalization,
    build_object_loto_split,
    build_parent_random_split,
    build_suite_loso_split,
    load_feature_artifact,
    validate_dataset_closure,
    validate_split,
)
from tools.multisuite_detector.load_label_v2_artifact import LABEL_COLUMNS, MANUAL_COLUMNS


VALIDATOR = ROOT / "tools" / "multisuite_detector" / "validate_detector_dataset_manifest_v1.py"
FEATURE_LOADER = ROOT / "tools" / "multisuite_detector" / "load_frozen_clean_features.py"
BUILDER_GIT = "a" * 40
BUILDER_SHA = "b" * 64
SOURCE_SHA = "c" * 64


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_sums(root):
    names = ["label_v2.csv", "build_manifest.json", "validation_summary.json", "manual_audit_sample_manifest.csv"]
    (root / "SHA256SUMS").write_text("\n".join(f"{sha256(root / name)}  {name}" for name in names) + "\n", encoding="utf-8")


def label_row(episode, parent, suite, task, trace, *, eligible=True, event=True):
    return {
        "episode_key": episode,
        "parent_key": parent,
        "suite": suite,
        "task_id": task,
        "cohort_class": "PRIMARY_SUCCESS_ELIGIBLE" if eligible else "MECHANISM_INELIGIBLE_ABSTENTION",
        "clean_success": "true",
        "mechanism_eligible": "true" if eligible else "false",
        "event_present": "true" if event else "false",
        "anchor_absolute_step": "1" if event else "-1",
        "window_start": "0" if event else "-1",
        "window_end": "2" if event else "-1",
        "event_source": "source_availability" if event else "",
        "source_path": f"ledger/{episode}.jsonl",
        "source_sha256": SOURCE_SHA,
        "builder_git_sha": BUILDER_GIT,
        "builder_sha256": BUILDER_SHA,
        "invalid_reason": "",
        "abstain_reason": "MECHANISM_INELIGIBLE" if not eligible else "",
        "mechanism_type": "GRIPPER_TRANSFER_ELIGIBLE" if eligible else "MECHANISM_UNSUPPORTED",
        "event_id": f"{episode}#event_1" if event else "NO_EVENT",
        "segment_id": f"{episode}#segment_1" if event else "NO_EVENT",
        "event_rank": "1" if event else "0",
        "coordinate_semantics": "zero_based_observation_before_action_start_inclusive_end_exclusive_full_trajectory",
        "trace_length": str(trace),
        "source_schema_version": "source_availability_ledger_v1",
        "teacher_confidence": "0.9" if event else "UNKNOWN",
        "confidence_available": "true" if event else "false",
        "confidence_provenance": "SOURCE_AVAILABILITY" if event else "UNAVAILABLE",
        "event_id_provenance": "SOURCE_AVAILABILITY" if event else "NOT_APPLICABLE",
        "source_semantics_authority": "SOURCE_AVAILABILITY_LEDGER",
        "source_jsonl_check_mode": "LEDGER_PROVENANCE_ONLY_NO_RUNTIME_READ",
        "window_valid": "true",
        "label_validity_status": "VALID",
        "manual_audit_status": "PENDING",
        "manual_audit_reason": "",
    }


def make_label_artifact(tmp_path):
    root = tmp_path / "label_artifact"
    root.mkdir(parents=True)
    specs = [
        ("ep_obj_a", "p_obj_a", "Object", "task_00", True, True),
        ("ep_obj_b", "p_obj_b", "Object", "task_01", True, False),
        ("ep_obj_c", "p_obj_c", "Object", "task_02", False, False),
        ("ep_sp_a", "p_sp_a", "Spatial", "task_00", True, True),
        ("ep_goal_a", "p_goal_a", "Goal", "task_00", True, False),
        ("ep_l10_a", "p_l10_a", "LIBERO_10", "task_00", True, True),
        ("ep_link_a", "p_link_a", "Spatial", "task_01", True, True),
        ("ep_link_b", "p_link_b", "Goal", "task_01", True, True),
    ]
    rows = [label_row(ep, parent, suite, task, 3, eligible=eligible, event=event) for ep, parent, suite, task, eligible, event in specs]
    manual = [{
        "suite": rows[0]["suite"],
        "task_id": rows[0]["task_id"],
        "episode_key": rows[0]["episode_key"],
        "cohort_class": rows[0]["cohort_class"],
        "clean_success": rows[0]["clean_success"],
        "mechanism_eligible": rows[0]["mechanism_eligible"],
        "event_present": rows[0]["event_present"],
        "label_validity_status": rows[0]["label_validity_status"],
        "requested_priority": "positive_clean_success",
        "actual_selected_category": "positive_clean_success",
        "fallback_used": "false",
        "fallback_reason": "",
        "sampling_seed": "20260703",
    }]
    counts = {
        "PRIMARY_SUCCESS_ELIGIBLE": {"positive": 4, "no_event": 2, "total": 6},
        "ELIGIBLE_CLEAN_FAILURE": {"positive": 0, "no_event": 0, "total": 0},
        "MECHANISM_INELIGIBLE_ABSTENTION": {"positive": 0, "no_event": 1, "total": 1},
    }
    # ep_link_b adds one more eligible positive.
    counts["PRIMARY_SUCCESS_ELIGIBLE"] = {"positive": 5, "no_event": 2, "total": 7}
    write_csv(root / "label_v2.csv", LABEL_COLUMNS, rows)
    write_csv(root / "manual_audit_sample_manifest.csv", MANUAL_COLUMNS, manual)
    (root / "validation_summary.json").write_text(json.dumps({
        "status": "PASS", "mode": "synthetic-dry-run", "row_count": len(rows),
        "counts": counts, "invalid_window_rows": 0, "manual_audit_sample_n": len(manual),
        "unexplained_disposition_rows": 0,
    }), encoding="utf-8")
    (root / "build_manifest.json").write_text(json.dumps({
        "schema_version": "clean2000_label_v2_episode_primary_event_v1",
        "mode": "synthetic-dry-run",
        "synthetic_only": True,
        "builder_git_sha": BUILDER_GIT,
        "builder_sha256": BUILDER_SHA,
        "source_semantics_authority": "SOURCE_AVAILABILITY_LEDGER",
        "source_jsonl_check_mode": "LEDGER_PROVENANCE_ONLY_NO_RUNTIME_READ",
        "atomic_publish": False,
        "inputs": {
            "source_manifest": {"path": "ledger/source.csv", "sha256": "1" * 64},
            "episode_census": {"path": "ledger/census.csv", "sha256": "2" * 64},
            "source_crosstab": {"path": "ledger/crosstab.csv", "sha256": "3" * 64},
        },
        "outputs": sorted(["label_v2.csv", "build_manifest.json", "validation_summary.json", "manual_audit_sample_manifest.csv", "SHA256SUMS"]),
    }), encoding="utf-8")
    write_sums(root)
    return root, rows


def make_feature_csv(tmp_path, label_rows, *, constant=False):
    path = tmp_path / "features.csv"
    shared = "f" * 64
    state_by_ep = {
        "ep_link_a": shared,
        "ep_link_b": shared,
    }
    rows = []
    for episode_i, label in enumerate(label_rows):
        state = state_by_ep.get(label["episode_key"], hashlib.sha256(label["episode_key"].encode()).hexdigest())
        for step in range(int(label["trace_length"])):
            row = {
                "episode_key": label["episode_key"],
                "parent_key": label["parent_key"],
                "suite": label["suite"],
                "task_id": label["task_id"],
                "initial_state_hash": state,
                "trace_length": label["trace_length"],
                "step": str(step),
            }
            for feature_i, name in enumerate(SC5_FEATURES):
                row[name] = "1.0" if constant else str(episode_i + step + feature_i / 100)
            rows.append(row)
    write_csv(path, ["episode_key", "parent_key", "suite", "task_id", "initial_state_hash", "trace_length", "step"] + SC5_FEATURES, rows)
    return path


def build_dataset(tmp_path):
    label_root, label_rows = make_label_artifact(tmp_path)
    feature_csv = make_feature_csv(tmp_path, label_rows)
    dataset_csv = tmp_path / "detector_dataset_manifest_v1.csv"
    manifest = build_dataset_manifest(label_root, feature_csv, dataset_csv)
    return label_root, feature_csv, dataset_csv, manifest


def test_feature_contract_and_dataset_join(tmp_path):
    _, feature_csv, dataset_csv, manifest = build_dataset(tmp_path)
    features = load_feature_artifact(feature_csv)
    assert features["feature_names"] == SC5_FEATURES
    assert features["feature_count"] == 25
    assert features["feature_schema_sha256"] == FEATURE_SCHEMA_SHA256
    assert manifest["population_counts"]["DETECTOR_ELIGIBLE"] == 7
    assert manifest["population_counts"]["DETECTOR_SAFETY"] == 1
    assert manifest["population_counts"]["DETECTOR_MULTI_EVENT"] == "UNAVAILABLE_SEPARATE_ARTIFACT_REQUIRED"
    assert len(read_csv(dataset_csv)) == 8


@pytest.mark.parametrize("mutate,message", [
    (lambda rows: [r for r in rows if r["episode_key"] != "ep_l10_a"], "episode sets differ"),
    (lambda rows: [dict(r, parent_key="wrong") if r["episode_key"] == "ep_obj_a" else r for r in rows], "parent_key mismatch"),
    (lambda rows: [dict(r, trace_length="2") if r["episode_key"] == "ep_obj_a" else r for r in rows], "invalid trace_length"),
])
def test_join_rejects_missing_identity_and_trace_mismatch(tmp_path, mutate, message):
    label_root, label_rows = make_label_artifact(tmp_path)
    feature_csv = make_feature_csv(tmp_path, label_rows)
    rows = read_csv(feature_csv)
    rows = mutate(rows)
    write_csv(feature_csv, rows[0].keys(), rows)
    with pytest.raises(DetectorDatasetClosureError, match=message):
        build_dataset_manifest(label_root, feature_csv, tmp_path / "dataset.csv")


@pytest.mark.parametrize("mutate,message", [
    (lambda rows: rows[0].update({"initial_state_hash": ""}), "empty field"),
    (lambda rows: rows[0].update({SC5_FEATURES[0]: "nan"}), "finite float"),
])
def test_feature_artifact_rejects_bad_state_and_nonfinite(tmp_path, mutate, message):
    _, label_rows = make_label_artifact(tmp_path)
    feature_csv = make_feature_csv(tmp_path, label_rows)
    rows = read_csv(feature_csv)
    mutate(rows)
    write_csv(feature_csv, rows[0].keys(), rows)
    with pytest.raises(DetectorDatasetClosureError, match=message):
        load_feature_artifact(feature_csv)


def test_feature_artifact_rejects_reordered_or_extra_feature_header(tmp_path):
    _, label_rows = make_label_artifact(tmp_path)
    feature_csv = make_feature_csv(tmp_path, label_rows)
    rows = read_csv(feature_csv)
    cols = ["episode_key", "parent_key", "suite", "task_id", "initial_state_hash", "trace_length", "step"] + list(reversed(SC5_FEATURES))
    write_csv(feature_csv, cols, rows)
    with pytest.raises(DetectorDatasetClosureError, match="header"):
        load_feature_artifact(feature_csv)

    cols = ["episode_key", "parent_key", "suite", "task_id", "initial_state_hash", "trace_length", "step"] + SC5_FEATURES + ["attack_condition"]
    for row in rows:
        row["attack_condition"] = "OURS"
    write_csv(feature_csv, cols, rows)
    with pytest.raises(DetectorDatasetClosureError, match="header"):
        load_feature_artifact(feature_csv)


def test_feature_artifact_rejects_duplicate_step(tmp_path):
    _, label_rows = make_label_artifact(tmp_path)
    feature_csv = make_feature_csv(tmp_path, label_rows)
    rows = read_csv(feature_csv)
    rows[1]["step"] = rows[0]["step"]
    write_csv(feature_csv, rows[0].keys(), rows)
    with pytest.raises(DetectorDatasetClosureError, match="duplicate feature step"):
        load_feature_artifact(feature_csv)


def test_splits_preserve_parent_and_state_components(tmp_path):
    _, _, dataset_csv, _ = build_dataset(tmp_path)
    split_csv = tmp_path / "parent_random_split_v1.csv"
    report = build_parent_random_split(dataset_csv, split_csv, seed=7, train_ratio=0.5, val_ratio=0.25)
    assert report["schema_version"] == "parent_random_split_v1"
    assert validate_split(dataset_csv, split_csv)["status"] == "PASS"
    rows = read_csv(split_csv)
    split_by_ep = {r["episode_key"]: r["split"] for r in rows}
    assert split_by_ep["ep_link_a"] == split_by_ep["ep_link_b"]


def test_split_validator_rejects_state_leakage(tmp_path):
    _, _, dataset_csv, _ = build_dataset(tmp_path)
    split_csv = tmp_path / "parent_random_split_v1.csv"
    build_parent_random_split(dataset_csv, split_csv, seed=7, train_ratio=0.5, val_ratio=0.25)
    rows = read_csv(split_csv)
    for row in rows:
        if row["episode_key"] == "ep_link_b":
            row["split"] = "test" if row["split"] != "test" else "train"
    write_csv(split_csv, rows[0].keys(), rows)
    with pytest.raises(DetectorDatasetClosureError, match="group leakage"):
        validate_split(dataset_csv, split_csv)

    build_parent_random_split(dataset_csv, split_csv, seed=7, train_ratio=0.5, val_ratio=0.25)
    rows = read_csv(split_csv)[:-1]
    write_csv(split_csv, rows[0].keys(), rows)
    with pytest.raises(DetectorDatasetClosureError, match="coverage mismatch"):
        validate_split(dataset_csv, split_csv)


def test_object_loto_and_suite_loso_exclusions(tmp_path):
    _, _, dataset_csv, _ = build_dataset(tmp_path)
    object_split = tmp_path / "object_leave_task_out_v1.csv"
    suite_split = tmp_path / "suite_loso_split_v1.csv"
    build_object_loto_split(dataset_csv, object_split)
    build_suite_loso_split(dataset_csv, suite_split)
    assert validate_split(dataset_csv, object_split)["status"] == "PASS"
    assert validate_split(dataset_csv, suite_split)["status"] == "PASS"
    object_rows = read_csv(object_split)
    assert all(r["split"] == "test" for r in object_rows if r["fold_id"] == "object_loto_task_00" and r["episode_key"] == "ep_obj_a")
    suite_rows = read_csv(suite_split)
    assert all(r["split"] == "test" for r in suite_rows if r["fold_id"] == "loso_Goal" and r["episode_key"] in {"ep_goal_a", "ep_link_b"})


def test_normalization_uses_train_only_and_rejects_zero_variance(tmp_path):
    _, feature_csv, dataset_csv, _ = build_dataset(tmp_path)
    split_csv = tmp_path / "parent_random_split_v1.csv"
    build_parent_random_split(dataset_csv, split_csv, seed=2, train_ratio=0.5, val_ratio=0.25)
    norm_json = tmp_path / "detector_normalization_v1.json"
    report = build_normalization(feature_csv, dataset_csv, split_csv, norm_json, population_id="DETECTOR_ELIGIBLE", fold_id="parent_random")
    assert report["normalization_source"] == "train_only"
    assert min(report["count_per_feature"]) > 0
    assert validate_dataset_closure(dataset_csv, feature_csv, split_csv, norm_json)["normalization_validation"] == "PASS"

    label_root, label_rows = make_label_artifact(tmp_path / "zero")
    constant_features = make_feature_csv(tmp_path / "zero", label_rows, constant=True)
    constant_dataset = tmp_path / "zero" / "dataset.csv"
    build_dataset_manifest(label_root, constant_features, constant_dataset)
    constant_split = tmp_path / "zero" / "split.csv"
    build_parent_random_split(constant_dataset, constant_split, seed=2, train_ratio=0.5, val_ratio=0.25)
    with pytest.raises(DetectorDatasetClosureError, match="zero"):
        build_normalization(constant_features, constant_dataset, constant_split, tmp_path / "zero" / "norm.json", population_id="DETECTOR_ELIGIBLE", fold_id="parent_random")


def test_closure_validator_rejects_normalization_sha_tamper(tmp_path):
    _, feature_csv, dataset_csv, _ = build_dataset(tmp_path)
    split_csv = tmp_path / "parent_random_split_v1.csv"
    build_parent_random_split(dataset_csv, split_csv, seed=2, train_ratio=0.5, val_ratio=0.25)
    norm_json = tmp_path / "detector_normalization_v1.json"
    build_normalization(feature_csv, dataset_csv, split_csv, norm_json, population_id="DETECTOR_ELIGIBLE", fold_id="parent_random")
    norm = json.loads(norm_json.read_text(encoding="utf-8"))
    norm["source_dataset_manifest_sha256"] = "0" * 64
    norm_json.write_text(json.dumps(norm), encoding="utf-8")
    with pytest.raises(DetectorDatasetClosureError, match="dataset SHA"):
        validate_dataset_closure(dataset_csv, feature_csv, split_csv, norm_json)


@pytest.mark.parametrize("field,value,message", [
    ("parent_key", "tampered", "parent_key"),
    ("suite", "tampered", "suite"),
    ("task_id", "tampered", "task_id"),
    ("initial_state_hash", "0" * 64, "initial_state_hash"),
    ("trace_length", "9", "trace_length"),
    ("population_id", "DETECTOR_SAFETY", "population_id"),
])
def test_closure_validator_rejects_dataset_row_tamper(tmp_path, field, value, message):
    _, feature_csv, dataset_csv, _ = build_dataset(tmp_path)
    rows = read_csv(dataset_csv)
    target = next(row for row in rows if row["population_id"] == "DETECTOR_ELIGIBLE")
    target[field] = value
    write_csv(dataset_csv, rows[0].keys(), rows)
    with pytest.raises(DetectorDatasetClosureError, match=message):
        validate_dataset_closure(dataset_csv, feature_csv)


def test_closure_validator_rejects_duplicate_or_unknown_population(tmp_path):
    _, feature_csv, dataset_csv, _ = build_dataset(tmp_path)
    rows = read_csv(dataset_csv)
    rows[1]["episode_key"] = rows[0]["episode_key"]
    write_csv(dataset_csv, rows[0].keys(), rows)
    with pytest.raises(DetectorDatasetClosureError, match="duplicate"):
        validate_dataset_closure(dataset_csv, feature_csv)

    _, feature_csv, dataset_csv, _ = build_dataset(tmp_path / "unknown")
    rows = read_csv(dataset_csv)
    rows[0]["population_id"] = "PRIMARY_ATTACK"
    write_csv(dataset_csv, rows[0].keys(), rows)
    with pytest.raises(DetectorDatasetClosureError, match="unknown population_id"):
        validate_dataset_closure(dataset_csv, feature_csv)


def test_split_validator_rejects_group_id_tamper(tmp_path):
    _, _, dataset_csv, _ = build_dataset(tmp_path)
    split_csv = tmp_path / "parent_random_split_v1.csv"
    build_parent_random_split(dataset_csv, split_csv, seed=7, train_ratio=0.5, val_ratio=0.25)
    rows = read_csv(split_csv)
    rows[0]["group_id"] = "badgroup"
    write_csv(split_csv, rows[0].keys(), rows)
    with pytest.raises(DetectorDatasetClosureError, match="group_id"):
        validate_split(dataset_csv, split_csv)


@pytest.mark.parametrize("mutate,message", [
    (lambda n: n.update({"schema_version": "wrong"}), "schema_version"),
    (lambda n: n.update({"feature_names": list(reversed(n["feature_names"]))}), "feature order"),
    (lambda n: n.update({"count_per_feature": n["count_per_feature"][:-1]}), "count_per_feature"),
    (lambda n: n.update({"mean": [float("nan")] + n["mean"][1:]}), "mean"),
    (lambda n: n.update({"std": [0.0] + n["std"][1:]}), "std"),
    (lambda n: n.update({"normalization_source": "all_rows"}), "normalization_source"),
    (lambda n: n.update({"population_id": "PRIMARY_ATTACK"}), "population_id"),
    (lambda n: n.update({"fold_id": ""}), "fold_id"),
    (lambda n: n.update({"source_feature_artifact_sha256": "0" * 64}), "feature artifact SHA"),
])
def test_closure_validator_rejects_normalization_tamper(tmp_path, mutate, message):
    _, feature_csv, dataset_csv, _ = build_dataset(tmp_path)
    split_csv = tmp_path / "parent_random_split_v1.csv"
    build_parent_random_split(dataset_csv, split_csv, seed=2, train_ratio=0.5, val_ratio=0.25)
    norm_json = tmp_path / "detector_normalization_v1.json"
    build_normalization(feature_csv, dataset_csv, split_csv, norm_json, population_id="DETECTOR_ELIGIBLE", fold_id="parent_random")
    norm = json.loads(norm_json.read_text(encoding="utf-8"))
    mutate(norm)
    norm_json.write_text(json.dumps(norm), encoding="utf-8")
    with pytest.raises(DetectorDatasetClosureError, match=message):
        validate_dataset_closure(dataset_csv, feature_csv, split_csv, norm_json)


def test_cli_json_success_and_concise_failure(tmp_path):
    _, feature_csv, dataset_csv, _ = build_dataset(tmp_path)
    ok = subprocess.run(
        [sys.executable, str(VALIDATOR), "--dataset-csv", str(dataset_csv), "--feature-csv", str(feature_csv)],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(ok.stdout)["status"] == "PASS"

    bad = subprocess.run(
        [sys.executable, str(FEATURE_LOADER), "--feature-csv", str(tmp_path / "missing.csv")],
        capture_output=True, text=True,
    )
    assert bad.returncode == 1
    assert "Traceback" not in bad.stderr
