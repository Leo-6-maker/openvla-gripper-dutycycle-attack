import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.multisuite_detector.test_detector_dataset_closure_v1 import make_label_artifact
from tests.multisuite_detector.test_extract_formal_25d_features_v1 import make_sources, rewrite
from tools.multisuite_detector.prepare_formal_25d_feature_extraction_readiness_v1 import (
    build_readiness_report,
)

EXTRACTOR = ROOT / "tools/multisuite_detector/extract_formal_25d_features_v1.py"
VALIDATOR = ROOT / "tools/multisuite_detector/validate_formal_25d_features_v1.py"


def test_readiness_holds_when_no_exact_source_csv(tmp_path):
    label_root, _ = make_label_artifact(tmp_path)
    out_json = tmp_path / "readiness.json"
    out_sha = tmp_path / "readiness.sha256"

    report = build_readiness_report(
        label_artifact_root=label_root,
        repo_head="a" * 40,
        extractor_path=EXTRACTOR,
        validator_path=VALIDATOR,
        output_json=out_json,
        sha256_output=out_sha,
        expected_label_mode="synthetic-dry-run",
    )

    assert report["status"] == "HOLD_NEEDS_SOURCE_CSV_CONSTRUCTION"
    assert report["formal_feature_extraction"] == "NOT_PERFORMED"
    assert report["formal_detector_dataset_build"] == "NOT_PERFORMED"
    assert report["training"] == "NOT_PERFORMED"
    assert report["gpu"] == "NOT_PERFORMED"
    assert json.loads(out_json.read_text())["recommendation"] == "AUTHORIZE_SOURCE_CSV_CONSTRUCTION"
    assert out_sha.is_file()


def test_readiness_passes_with_bound_exact_source_csv(tmp_path):
    label_root, label_rows = make_label_artifact(tmp_path)
    approved_root, source_csv, _ = make_sources(tmp_path, label_rows)
    out_json = tmp_path / "readiness.json"

    report = build_readiness_report(
        label_artifact_root=label_root,
        repo_head="b" * 40,
        extractor_path=EXTRACTOR,
        validator_path=VALIDATOR,
        exact_source_csv=source_csv,
        approved_source_root=approved_root,
        output_json=out_json,
        expected_label_mode="synthetic-dry-run",
    )

    assert report["status"] == "READY_EXACT_SOURCE_CSV_BOUND"
    assert report["recommendation"] == "AUTHORIZE_EXTRACTION"
    assert report["exact_source_csv"]["has_exact_schema"] is True
    assert report["exact_source_csv"]["has_exact_sc5_order"] is True
    assert report["exact_source_csv"]["episode_count"] == len(label_rows)
    assert report["exact_source_csv"]["initial_state_hash_provenance"] == "BOUND"


def test_readiness_holds_on_missing_state_provenance(tmp_path):
    label_root, label_rows = make_label_artifact(tmp_path)
    approved_root, source_csv, rows = make_sources(tmp_path, label_rows)
    rows[0]["initial_state_hash_provenance"] = "UNBOUND"
    rewrite(source_csv, rows)
    out_json = tmp_path / "readiness.json"

    report = build_readiness_report(
        label_artifact_root=label_root,
        repo_head="c" * 40,
        extractor_path=EXTRACTOR,
        validator_path=VALIDATOR,
        exact_source_csv=source_csv,
        approved_source_root=approved_root,
        output_json=out_json,
        expected_label_mode="synthetic-dry-run",
    )

    assert report["status"] == "HOLD_MISSING_INITIAL_STATE_PROVENANCE"
    assert report["recommendation"] == "BLOCKED_NEEDS_SCIENTIFIC_DECISION"
    assert "provenance" in report["source_error"]


def test_readiness_holds_on_non_exact_source_schema(tmp_path):
    label_root, label_rows = make_label_artifact(tmp_path)
    approved_root, source_csv, rows = make_sources(tmp_path, label_rows)
    columns = [col for col in rows[0].keys() if col != "initial_state_hash"]
    rewrite(source_csv, rows, columns)
    out_json = tmp_path / "readiness.json"

    report = build_readiness_report(
        label_artifact_root=label_root,
        repo_head="d" * 40,
        extractor_path=EXTRACTOR,
        validator_path=VALIDATOR,
        exact_source_csv=source_csv,
        approved_source_root=approved_root,
        output_json=out_json,
        expected_label_mode="synthetic-dry-run",
    )

    assert report["status"] == "HOLD_MISSING_FEATURE_INPUTS"
    assert report["source_schema_audit"]["has_exact_source_schema"] is False
