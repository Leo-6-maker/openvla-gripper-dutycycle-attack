import csv
import json
from pathlib import Path

import pytest

from tests.multisuite_detector.test_detector_dataset_closure_v1 import make_label_artifact, write_csv
from tests.multisuite_detector.test_extract_formal_25d_features_v1 import make_sources
from tools.multisuite_detector.build_formal_25d_source_csv_v1 import (
    FEATURE_INPUT_COLUMNS,
    STATE_INPUT_COLUMNS,
    SourceCsvBuildError,
    build_source_csv,
)
from tools.multisuite_detector.extract_formal_25d_features_v1 import (
    FormalFeatureError,
    SC5_FEATURES,
    SOURCE_COLUMNS,
    load_source_rows,
    validate_against_label,
)


def split_inputs(tmp_path, source_rows):
    feature_csv = tmp_path / "per_step_features.csv"
    state_csv = tmp_path / "state_metadata.csv"
    feature_rows = []
    state_rows = []
    seen = set()
    for row in source_rows:
        feature_rows.append({col: row[col] for col in FEATURE_INPUT_COLUMNS})
        episode = row["episode_key"]
        if episode not in seen:
            seen.add(episode)
            state_rows.append({col: row[col] for col in STATE_INPUT_COLUMNS})
    write_csv(feature_csv, FEATURE_INPUT_COLUMNS, feature_rows)
    write_csv(state_csv, STATE_INPUT_COLUMNS, state_rows)
    return feature_csv, state_csv, feature_rows, state_rows


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_source_csv_builder_positive_round_trip(tmp_path):
    label_root, label_rows = make_label_artifact(tmp_path)
    approved_root, _, source_rows = make_sources(tmp_path, label_rows)
    feature_csv, state_csv, _, _ = split_inputs(tmp_path, source_rows)
    output_csv = tmp_path / "formal_source.csv"
    report_json = tmp_path / "source_report.json"
    report_sha = tmp_path / "source_report.sha256"

    report = build_source_csv(
        label_artifact_root=label_root,
        per_step_features_csv=feature_csv,
        state_metadata_csv=state_csv,
        approved_source_root=approved_root,
        output_csv=output_csv,
        report_json=report_json,
        sha256_output=report_sha,
        expected_label_mode="synthetic-dry-run",
    )

    assert report["status"] == "PASS"
    assert report["exact_source_csv_build"] == "PASS"
    assert report["formal_feature_extraction"] == "NOT_PERFORMED"
    assert report["formal_detector_dataset_build"] == "NOT_PERFORMED"
    assert report["training"] == "NOT_PERFORMED"
    assert report["gpu"] == "NOT_PERFORMED"
    assert read_rows(output_csv)[0].keys() == set(SOURCE_COLUMNS)
    validated = load_source_rows(output_csv, approved_root)
    validate_against_label(validated, label_root, "synthetic-dry-run")
    assert json.loads(report_json.read_text())["output_csv_sha256"] == report["output_csv_sha256"]
    assert report_sha.is_file()


def test_source_csv_builder_rejects_missing_state_metadata(tmp_path):
    label_root, label_rows = make_label_artifact(tmp_path)
    approved_root, _, source_rows = make_sources(tmp_path, label_rows)
    feature_csv, state_csv, _, state_rows = split_inputs(tmp_path, source_rows)
    write_csv(state_csv, STATE_INPUT_COLUMNS, state_rows[:-1])

    with pytest.raises(SourceCsvBuildError, match="state metadata episode set"):
        build_source_csv(
            label_artifact_root=label_root,
            per_step_features_csv=feature_csv,
            state_metadata_csv=state_csv,
            approved_source_root=approved_root,
            output_csv=tmp_path / "out.csv",
            expected_label_mode="synthetic-dry-run",
        )


def test_source_csv_builder_rejects_missing_step(tmp_path):
    label_root, label_rows = make_label_artifact(tmp_path)
    approved_root, _, source_rows = make_sources(tmp_path, label_rows)
    feature_csv, state_csv, feature_rows, _ = split_inputs(tmp_path, source_rows)
    feature_rows = [row for row in feature_rows if not (row["episode_key"] == "ep_obj_a" and row["step"] == "1")]
    write_csv(feature_csv, FEATURE_INPUT_COLUMNS, feature_rows)

    with pytest.raises(SourceCsvBuildError, match="steps must cover"):
        build_source_csv(
            label_artifact_root=label_root,
            per_step_features_csv=feature_csv,
            state_metadata_csv=state_csv,
            approved_source_root=approved_root,
            output_csv=tmp_path / "out.csv",
            expected_label_mode="synthetic-dry-run",
        )


def test_source_csv_builder_rejects_bad_state_provenance(tmp_path):
    label_root, label_rows = make_label_artifact(tmp_path)
    approved_root, _, source_rows = make_sources(tmp_path, label_rows)
    feature_csv, state_csv, _, state_rows = split_inputs(tmp_path, source_rows)
    state_rows[0]["initial_state_hash_provenance"] = "UNBOUND"
    write_csv(state_csv, STATE_INPUT_COLUMNS, state_rows)

    with pytest.raises(SourceCsvBuildError, match="unsupported initial_state_hash_provenance"):
        build_source_csv(
            label_artifact_root=label_root,
            per_step_features_csv=feature_csv,
            state_metadata_csv=state_csv,
            approved_source_root=approved_root,
            output_csv=tmp_path / "out.csv",
            expected_label_mode="synthetic-dry-run",
        )


def test_source_csv_builder_rejects_nonclean_condition(tmp_path):
    label_root, label_rows = make_label_artifact(tmp_path)
    approved_root, _, source_rows = make_sources(tmp_path, label_rows)
    feature_csv, state_csv, feature_rows, _ = split_inputs(tmp_path, source_rows)
    feature_rows[0]["source_condition"] = "ATTACK"
    write_csv(feature_csv, FEATURE_INPUT_COLUMNS, feature_rows)

    with pytest.raises(SourceCsvBuildError, match="source condition is not clean"):
        build_source_csv(
            label_artifact_root=label_root,
            per_step_features_csv=feature_csv,
            state_metadata_csv=state_csv,
            approved_source_root=approved_root,
            output_csv=tmp_path / "out.csv",
            expected_label_mode="synthetic-dry-run",
        )


def test_source_csv_builder_output_must_not_preexist(tmp_path):
    label_root, label_rows = make_label_artifact(tmp_path)
    approved_root, _, source_rows = make_sources(tmp_path, label_rows)
    feature_csv, state_csv, _, _ = split_inputs(tmp_path, source_rows)
    output_csv = tmp_path / "out.csv"
    output_csv.write_text("already here\n", encoding="utf-8")

    with pytest.raises(SourceCsvBuildError, match="already exists"):
        build_source_csv(
            label_artifact_root=label_root,
            per_step_features_csv=feature_csv,
            state_metadata_csv=state_csv,
            approved_source_root=approved_root,
            output_csv=output_csv,
            expected_label_mode="synthetic-dry-run",
        )
