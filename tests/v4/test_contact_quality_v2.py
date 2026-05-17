import json

from evaluate_cq_v2_manual_calibration import read_mapping_with_trailing_notes
from extract_contact_quality_metrics import main, summarize_run


def _write_run(root, name, *, success=True, rows=None):
    run = root / name
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"task_id": "libero_spatial_black_bowl"}), encoding="utf-8")
    (run / "episode_records.jsonl").write_text(json.dumps({"success": success}) + "\n", encoding="utf-8")
    (run / "step_records.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return run


def _rows(*, active_open=False, no_pose=False, lifted=True, dropped=True):
    out = []
    base = 0.02
    for step in range(8):
        z = "" if no_pose else base
        if not no_pose and lifted and step in {2, 3, 4}:
            z = base + 0.03
        if not no_pose and lifted and dropped and step >= 5:
            z = base + 0.002
        out.append(
            {
                "step": step,
                "attack_active": active_open and step in {2, 3, 4},
                "executed_gripper_open_ok": active_open and step in {2, 3, 4},
                "bowl_z_after": z,
                "qpos_abs_after_max": 0.07,
            }
        )
    return out


def test_cq_v2_official_success_does_not_override_premature_drop(tmp_path):
    run = _write_run(tmp_path, "anything_state9_seed4_vis_like", success=True, rows=_rows(active_open=True))
    row = summarize_run(run)
    assert row["official_success"] is True
    assert row["premature_release"] is True
    assert row["drop_after_lift"] is True
    assert row["contact_quality_failure"] is True
    assert row["contact_quality_success"] is False
    assert row["sr_cq_mismatch"] is True
    assert row["uncontrolled_final_drop"] is True
    assert row["cq_failure_reason_v2"] == "uncontrolled_final_drop_to_target"


def test_cq_v2_stable_controlled_success_stays_success(tmp_path):
    run = _write_run(tmp_path, "renamed_run_without_condition", success=True, rows=_rows(active_open=False))
    row = summarize_run(run)
    assert row["contact_quality_failure"] is False
    assert row["contact_quality_success"] is True
    assert row["stable_controlled_place"] is True
    assert row["cq_failure_reason_v2"] == "stable_controlled_place"


def test_cq_v2_official_failure_with_premature_drop_is_failure(tmp_path):
    run = _write_run(tmp_path, "failure_metadata_is_irrelevant", success=False, rows=_rows(active_open=True))
    row = summarize_run(run)
    assert row["contact_quality_failure"] is True
    assert row["contact_quality_success"] is False
    assert row["sr_cq_mismatch"] is False
    assert row["cq_failure_reason_v2"] == "premature_release_plus_drop"


def test_cq_v2_official_failure_with_drop_after_lift_is_failure_even_if_active_open_proxy_misses(tmp_path):
    run = _write_run(tmp_path, "official_failure_drop_proxy", success=False, rows=_rows(active_open=False))
    row = summarize_run(run)
    assert row["premature_release"] is False
    assert row["drop_after_lift"] is True
    assert row["contact_quality_failure"] is True
    assert row["cq_failure_reason_v2"] == "premature_release_plus_drop"


def test_cq_v2_missing_pose_is_low_confidence_na(tmp_path):
    run = _write_run(tmp_path, "missing_pose_run", success=True, rows=_rows(active_open=True, no_pose=True))
    row = summarize_run(run)
    assert row["contact_quality_failure"] == "NA"
    assert row["contact_quality_success"] == "NA"
    assert row["cq_confidence_v2"] == "low"
    assert row["cq_failure_reason_v2"] == "missing_pose_low_confidence"


def test_cq_v2_metadata_does_not_control_decision(tmp_path):
    rows = _rows(active_open=True)
    run_a = _write_run(tmp_path, "state5_seed1_vis_arm_clean", success=True, rows=rows)
    run_b = _write_run(tmp_path, "state7_seed99_random_gripper_clean", success=True, rows=rows)
    a = summarize_run(run_a)
    b = summarize_run(run_b)
    for key in (
        "contact_quality_failure",
        "contact_quality_success",
        "sr_cq_mismatch",
        "uncontrolled_final_drop",
        "stable_controlled_place",
        "cq_failure_reason_v2",
    ):
        assert a[key] == b[key]


def test_mapping_parser_merges_trailing_notes(tmp_path):
    path = tmp_path / "mapping.csv"
    path.write_text("review_id,run_id,notes\nR1,run_a,video fetched,SHA256 computed\n", encoding="utf-8")
    rows = read_mapping_with_trailing_notes(path)
    assert rows == [{"review_id": "R1", "run_id": "run_a", "notes": "video fetched SHA256 computed"}]


def test_production_cq_extractor_does_not_reference_manual_review_files():
    import inspect
    import extract_contact_quality_metrics

    source = inspect.getsource(extract_contact_quality_metrics)
    assert "manual_review" not in source
    assert "manual_contact_quality_failure" not in source


def test_cq_extractor_includes_non_black_bowl_runs(tmp_path, monkeypatch):
    run = _write_run(tmp_path, "generic_object_task_clean", success=True, rows=_rows(active_open=False))
    (run / "run_manifest.json").write_text(json.dumps({"task_id": "not_black_bowl"}), encoding="utf-8")
    out = tmp_path / "cq.csv"
    monkeypatch.setattr("sys.argv", ["extract_contact_quality_metrics.py", "--input_root", str(tmp_path), "--output_csv", str(out)])
    assert main() == 0
    assert "not_black_bowl" in out.read_text(encoding="utf-8")
