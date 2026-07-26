import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "n5" / "phase3_student"))

from c3_g_predicate_evaluator import load_contract  # noqa: E402
from run_c3_g_stage3_synthetic import _canonical, _evaluate, build_predicate_cases, REQUIRED_INPUT_FIELDS  # noqa: E402
from run_c3_s3a_fresh_synthetic import build_relation_plan  # noqa: E402


def test_stage3_has_six_geometry_cases_for_all_44_relations():
    manifest = {"episodes": []}
    reference_poses = {}
    for relation in build_relation_plan():
        episode_id = f"c3s3a_{relation['relation_id']}"
        manifest["episodes"].append({"episode_id": episode_id, "relation_id": relation["relation_id"], "category": relation["category"]})
        reference_poses[episode_id] = {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}
    cases = build_predicate_cases(manifest, reference_poses, load_contract())
    records, summary = _evaluate(cases, load_contract())
    assert len(cases) == 264
    assert len(records) == 264
    assert summary["status"] == "PASS"
    assert summary["case_kinds"] == {kind: 44 for kind in ("TRUE", "FALSE", "BOUNDARY", "IDENTITY_MISMATCH", "POSE_HARD_NEGATIVE", "UNKNOWN")}


def test_stage3_case_input_and_output_closure_is_explicit():
    manifest = {"episodes": []}
    reference_poses = {}
    for relation in build_relation_plan():
        episode_id = f"c3s3a_{relation['relation_id']}"
        manifest["episodes"].append({
            "episode_id": episode_id,
            "relation_id": relation["relation_id"],
            "relation_index": relation["relation_index"],
            "task_key": relation["task_key"],
            "category": relation["category"],
        })
        reference_poses[episode_id] = {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}
    cases = build_predicate_cases(manifest, reference_poses, load_contract())
    records, summary = _evaluate(cases, load_contract())
    assert REQUIRED_INPUT_FIELDS.issubset(cases[0])
    assert all(row["case_input_sha256"] for row in cases)
    assert all(row["record_sha256"] and "raw_measurements" in row for row in records)
    assert summary["case_input_count"] == summary["predicate_record_count"] == 264
    assert summary["relation_count"] == 44
    assert set(summary["per_relation_count"].values()) == {6}


def test_nonfinite_values_are_tagged_not_emitted_as_invalid_json():
    encoded = _canonical({"value": float("nan")}).decode("utf-8")
    assert "NaN" in encoded
    assert ":NaN" not in encoded
