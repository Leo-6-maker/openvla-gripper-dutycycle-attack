import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.multisuite_detector.test_detector_dataset_closure_v1 import make_label_artifact, write_csv
from tools.multisuite_detector.detector_dataset_closure_v1 import SC5_FEATURES
from tools.multisuite_detector.extract_formal_25d_features_v1 import (
    FormalFeatureError,
    SOURCE_COLUMNS,
    audit_source_schema,
    build_feature_artifact,
    validate_feature_artifact,
)


def state_hash(name):
    return hashlib.sha256(("state:" + name).encode()).hexdigest()


def make_sources(tmp_path, label_rows):
    root = tmp_path / "clean_records"
    root.mkdir()
    rows = []
    for episode_i, label in enumerate(label_rows):
        (root / f"{label['episode_key']}.json").write_text('{"clean": true}\n', encoding="utf-8")
        for step in range(int(label["trace_length"])):
            row = {
                "episode_key": label["episode_key"],
                "parent_key": label["parent_key"],
                "suite": label["suite"],
                "task_id": label["task_id"],
                "initial_state_hash": state_hash(label["episode_key"]),
                "trace_length": label["trace_length"],
                "step": str(step),
                "source_record_path": f"{label['episode_key']}.json",
                "source_condition": "CLEAN",
                "initial_state_hash_provenance": "SIMULATOR_RESET_STATE_SERIALIZED",
            }
            for feature_i, name in enumerate(SC5_FEATURES):
                row[name] = str(episode_i + step + feature_i / 100)
            rows.append(row)
    source_csv = tmp_path / "source_features.csv"
    write_csv(source_csv, SOURCE_COLUMNS, rows)
    return root, source_csv, rows


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rewrite(path, rows, columns=SOURCE_COLUMNS):
    write_csv(path, columns, [{col: row[col] for col in columns if col in row} for row in rows])


def test_positive_end_to_end_extraction_and_validation(tmp_path):
    label_root, label_rows = make_label_artifact(tmp_path)
    approved_root, source_csv, _ = make_sources(tmp_path, label_rows)
    out = tmp_path / "feature_artifact"

    manifest = build_feature_artifact(source_csv, label_root, out, approved_root)
    report = validate_feature_artifact(out, label_root)

    assert manifest["schema_version"] == "formal_25d_feature_artifact_v1"
    assert manifest["episode_count"] == len(label_rows)
    assert report["status"] == "PASS"
    assert report["formal_detector_dataset_build"] == "NOT_PERFORMED"
    assert (out / "formal_25d_features_v1.csv").is_file()
    assert (out / "SHA256SUMS").is_file()


def test_audit_source_schema_reports_exact_header(tmp_path):
    label_root, label_rows = make_label_artifact(tmp_path)
    approved_root, source_csv, _ = make_sources(tmp_path, label_rows)
    report = audit_source_schema(source_csv, tmp_path / "audit.json")
    assert report["status"] == "PASS"
    assert report["has_exact_sc5_order"] is True
    assert json.loads((tmp_path / "audit.json").read_text())["formal_extraction"] == "NOT_PERFORMED"


@pytest.mark.parametrize("mutate,match", [
    (lambda rows: [dict(r, **{SC5_FEATURES[0]: "nan"}) if r["episode_key"] == "ep_obj_a" and r["step"] == "0" else r for r in rows], "finite float"),
    (lambda rows: [dict(r, **{SC5_FEATURES[0]: "inf"}) if r["episode_key"] == "ep_obj_a" and r["step"] == "0" else r for r in rows], "finite float"),
    (lambda rows: rows + [dict(rows[0])], "steps must cover|duplicate"),
    (lambda rows: [r for r in rows if not (r["episode_key"] == "ep_obj_a" and r["step"] == "1")], "steps must cover"),
    (lambda rows: [dict(r, trace_length="2") if r["episode_key"] == "ep_obj_a" else r for r in rows], "invalid step coverage|trace_length mismatch"),
    (lambda rows: [dict(r, episode_key="missing_ep") if r["episode_key"] == "ep_obj_a" else r for r in rows], "episode set"),
    (lambda rows: [dict(r, parent_key="wrong_parent") if r["episode_key"] == "ep_obj_a" else r for r in rows], "parent_key mismatch"),
    (lambda rows: [dict(r, suite="wrong_suite") if r["episode_key"] == "ep_obj_a" else r for r in rows], "suite mismatch"),
    (lambda rows: [dict(r, task_id="wrong_task") if r["episode_key"] == "ep_obj_a" else r for r in rows], "task_id mismatch"),
    (lambda rows: [dict(r, initial_state_hash_provenance="") if r["episode_key"] == "ep_obj_a" else r for r in rows], "empty field"),
    (lambda rows: [dict(r, initial_state_hash="abc") if r["episode_key"] == "ep_obj_a" else r for r in rows], "64 lowercase hex"),
    (lambda rows: [dict(r, initial_state_hash=hashlib.sha256(r["episode_key"].encode()).hexdigest()) if r["episode_key"] == "ep_obj_a" else r for r in rows], "forbidden episode_key"),
    (lambda rows: [dict(r, source_condition="ATTACK") if r["episode_key"] == "ep_obj_a" else r for r in rows], "not clean"),
    (lambda rows: [dict(r, source_record_path="../outside.json") if r["episode_key"] == "ep_obj_a" else r for r in rows], "outside approved root"),
])
def test_extractor_rejects_bad_source_rows(tmp_path, mutate, match):
    label_root, label_rows = make_label_artifact(tmp_path)
    approved_root, source_csv, rows = make_sources(tmp_path, label_rows)
    rewrite(source_csv, mutate(rows))
    with pytest.raises(FormalFeatureError, match=match):
        build_feature_artifact(source_csv, label_root, tmp_path / "out", approved_root)


def test_missing_or_reordered_sc5_feature_fails_closed(tmp_path):
    label_root, label_rows = make_label_artifact(tmp_path)
    approved_root, source_csv, rows = make_sources(tmp_path, label_rows)
    cols = SOURCE_COLUMNS.copy()
    cols[-1], cols[-2] = cols[-2], cols[-1]
    rewrite(source_csv, rows, cols)
    with pytest.raises(FormalFeatureError, match="expected exact header"):
        build_feature_artifact(source_csv, label_root, tmp_path / "out", approved_root)

    cols = [c for c in SOURCE_COLUMNS if c != SC5_FEATURES[-1]]
    rewrite(source_csv, rows, cols)
    with pytest.raises(FormalFeatureError, match="expected exact header"):
        build_feature_artifact(source_csv, label_root, tmp_path / "out2", approved_root)


def test_attack_or_future_column_rejected(tmp_path):
    label_root, label_rows = make_label_artifact(tmp_path)
    approved_root, source_csv, rows = make_sources(tmp_path, label_rows)
    cols = SOURCE_COLUMNS + ["future_qpos"]
    for row in rows:
        row["future_qpos"] = "0.0"
    rewrite(source_csv, rows, cols)
    with pytest.raises(FormalFeatureError, match="forbidden source column"):
        build_feature_artifact(source_csv, label_root, tmp_path / "out", approved_root)


def test_validator_catches_sha_manifest_and_output_mutation(tmp_path):
    label_root, label_rows = make_label_artifact(tmp_path)
    approved_root, source_csv, rows = make_sources(tmp_path, label_rows)
    out = tmp_path / "feature_artifact"
    build_feature_artifact(source_csv, label_root, out, approved_root)
    assert validate_feature_artifact(out, label_root)["status"] == "PASS"

    feature_csv = out / "formal_25d_features_v1.csv"
    feature_csv.write_text(feature_csv.read_text(encoding="utf-8").replace("0.0", "0.123", 1), encoding="utf-8")
    with pytest.raises(FormalFeatureError, match="SHA256SUMS mismatch"):
        validate_feature_artifact(out, label_root)

    build_feature_artifact(source_csv, label_root, tmp_path / "feature_artifact2", approved_root)
    out2 = tmp_path / "feature_artifact2"
    manifest_path = out2 / "extraction_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_csv_path"] = str(tmp_path / "missing.csv")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with (out2 / "SHA256SUMS").open("w", encoding="utf-8") as f:
        for name in ["formal_25d_features_v1.csv", "extraction_manifest.json"]:
            f.write(f"{hashlib.sha256((out2 / name).read_bytes()).hexdigest()}  {name}\n")
    with pytest.raises(FormalFeatureError, match="source CSV SHA mismatch"):
        validate_feature_artifact(out2, label_root)
