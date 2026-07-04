import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.multisuite_detector.test_detector_dataset_closure_v1 import make_label_artifact, write_csv
from tools.multisuite_detector.feature_binding_manifest_v1 import (
    FeatureBindingError,
    SC5_FEATURES,
    build_binding,
    validate_binding_manifest,
    write_binding_manifest,
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_feature_csv(tmp_path, label_rows, *, include_meta=True):
    path = tmp_path / "clean2000_features.csv"
    columns = ["episode_key", "step"]
    if include_meta:
        columns += ["parent_key", "suite", "task_id", "trace_length"]
    columns += SC5_FEATURES
    rows = []
    for ep_i, label in enumerate(label_rows):
        for step in range(int(label["trace_length"])):
            row = {"episode_key": label["episode_key"], "step": str(step)}
            if include_meta:
                row.update({k: label[k] for k in ["parent_key", "suite", "task_id", "trace_length"]})
            for feat_i, name in enumerate(SC5_FEATURES):
                row[name] = str(ep_i + step + feat_i / 1000)
            rows.append(row)
    write_csv(path, columns, rows)
    return path


def make_binding(tmp_path, *, include_meta=True):
    label_root, label_rows = make_label_artifact(tmp_path)
    feature_csv = make_feature_csv(tmp_path, label_rows, include_meta=include_meta)
    manifest = tmp_path / "binding_manifest.json"
    report = write_binding_manifest(label_root, feature_csv, manifest, expected_label_mode="synthetic-dry-run")
    return label_root, label_rows, feature_csv, manifest, report


def test_feature_binding_positive_perfect_join(tmp_path):
    _, _, _, manifest, report = make_binding(tmp_path)
    assert report["episode_count"] == 8
    assert report["step_count"] == 24
    assert report["feature_names"] == SC5_FEATURES
    assert report["formal_detector_dataset_build"] == "NOT_PERFORMED"
    assert validate_binding_manifest(manifest, expected_label_mode="synthetic-dry-run")["status"] == "PASS"


def test_feature_binding_allows_feature_csv_without_label_metadata(tmp_path):
    _, _, _, manifest, report = make_binding(tmp_path, include_meta=False)
    assert report["episode_count"] == 8
    assert validate_binding_manifest(manifest, expected_label_mode="synthetic-dry-run")["status"] == "PASS"


@pytest.mark.parametrize("mutate,message", [
    (lambda rows: rows + [dict(rows[0], step=rows[0]["step"])], "duplicate step"),
    (lambda rows: [r for r in rows if not (r["episode_key"] == "ep_obj_a" and r["step"] == "1")], "steps must cover"),
    (lambda rows: rows + [dict(rows[0], episode_key="orphan", step="0")], "orphan feature episodes"),
    (lambda rows: [r for r in rows if r["episode_key"] != "ep_obj_a"], "missing feature episodes"),
    (lambda rows: [dict(r, parent_key="wrong") if r["episode_key"] == "ep_obj_a" else r for r in rows], "parent_key mismatch"),
    (lambda rows: [dict(r, suite="wrong") if r["episode_key"] == "ep_obj_a" else r for r in rows], "suite mismatch"),
    (lambda rows: [dict(r, task_id="wrong") if r["episode_key"] == "ep_obj_a" else r for r in rows], "task_id mismatch"),
    (lambda rows: [dict(r, trace_length="4") if r["episode_key"] == "ep_obj_a" else r for r in rows], "trace_length mismatch"),
    (lambda rows: [dict(r, **{SC5_FEATURES[0]: "nan"}) if r["episode_key"] == "ep_obj_a" else r for r in rows], "finite float"),
    (lambda rows: [dict(r, **{SC5_FEATURES[0]: "inf"}) if r["episode_key"] == "ep_obj_a" else r for r in rows], "finite float"),
])
def test_feature_binding_rejects_bad_rows(tmp_path, mutate, message):
    label_root, label_rows = make_label_artifact(tmp_path)
    feature_csv = make_feature_csv(tmp_path, label_rows)
    rows = mutate(read_csv(feature_csv))
    write_csv(feature_csv, rows[0].keys(), rows)
    with pytest.raises(FeatureBindingError, match=message):
        build_binding(label_root, feature_csv, expected_label_mode="synthetic-dry-run")


def test_feature_binding_rejects_feature_schema_mismatch_and_order(tmp_path):
    label_root, label_rows = make_label_artifact(tmp_path)
    feature_csv = make_feature_csv(tmp_path, label_rows)
    rows = read_csv(feature_csv)
    columns = ["episode_key", "step", "parent_key", "suite", "task_id", "trace_length"] + list(reversed(SC5_FEATURES))
    write_csv(feature_csv, columns, rows)
    with pytest.raises(FeatureBindingError, match="SC5 feature ordering"):
        build_binding(label_root, feature_csv, expected_label_mode="synthetic-dry-run")
    rows = read_csv(feature_csv)
    for row in rows:
        row.pop(SC5_FEATURES[-1], None)
    write_csv(feature_csv, [c for c in columns if c != SC5_FEATURES[-1]], rows)
    with pytest.raises(FeatureBindingError, match="missing feature column"):
        build_binding(label_root, feature_csv, expected_label_mode="synthetic-dry-run")


def test_validate_binding_rejects_wrong_sha(tmp_path):
    _, _, feature_csv, manifest, _ = make_binding(tmp_path)
    obj = json.loads(manifest.read_text())
    obj["feature_csv_sha256"] = "0" * 64
    manifest.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(FeatureBindingError, match="feature_csv_sha256 mismatch"):
        validate_binding_manifest(manifest, expected_label_mode="synthetic-dry-run")
    subprocess.run([
        sys.executable,
        str(ROOT / "tools/multisuite_detector/feature_binding_manifest_v1.py"),
        "build",
        "--label-artifact-root",
        obj["label_v2_artifact_root"],
        "--feature-csv",
        str(feature_csv),
        "--output-json",
        str(tmp_path / "cli_manifest.json"),
        "--expected-label-mode",
        "synthetic-dry-run",
    ], check=True)
