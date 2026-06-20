import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stageb.prepare_h2_human_review_forms import (  # noqa: E402
    ACCEPTED_STATUS,
    EVENT_JUDGMENT_FIELDS,
    HUMAN_FIELDS,
    REVIEW_FIELDS,
    base_blank_row,
    build_reviewer_b_rows,
    forbidden_columns,
    validate_review_rows,
)


def _row(status=ACCEPTED_STATUS):
    row = base_blank_row()
    row.update(
        {
            "review_round_id": "round_v2",
            "review_stratum": "DEV_CANARY",
            "proposal_version": "h2_diagnostic_review_package_v2",
            "resolver_commit": "abc123",
            "ontology_sha256": "o",
            "teacher_schema_sha256": "s",
            "physics_config_sha256": "p",
            "timing_contract_sha256": "t",
            "source_queue_sha256": "q",
            "teacher_overlay_manifest_sha256": "m",
            "review_id": "review_000_event_00",
            "episode_key": "libero_spatial|0|10|0|CLEAN",
            "suite": "libero_spatial",
            "task_idx": "0",
            "state_id": "10",
            "mechanism_type": "single_object_pick_place",
            "teacher_status": status,
            "object_binding_status": "BOUND_STRUCTURED_FALLBACK",
            "target_binding_status": "BOUND_STRUCTURED_FALLBACK",
            "event_id": "libero_spatial|0|10|0|CLEAN|event0",
            "proposed_object_body": "akita_black_bowl_1_main",
            "proposed_target_body_or_site": "plate_1_default_site",
            "proposed_close_onset": "10",
            "proposed_grasp_established": "11",
            "proposed_lift_onset": "12",
            "proposed_stable_carry_start": "13",
            "proposed_window_start": "10",
            "proposed_anchor": "12",
            "proposed_window_end": "20",
            "proposed_release_onset": "21",
            "raw_video_path": "/server/video.mp4",
            "teacher_only_timeline_path": "/server/timeline.csv",
            "teacher_only_timeline_status": "WROTE",
            "teacher_only_overlay_path": "/server/overlay.mp4",
            "teacher_only_overlay_status": "WROTE",
            "video_status": "source_path",
        }
    )
    return row


def test_template_rows_preserve_proposals_and_leave_human_fields_blank():
    row = _row()
    assert row["review_id"] == "review_000_event_00"
    assert row["raw_video_path"] == "/server/video.mp4"
    assert "blind_video_path" not in REVIEW_FIELDS
    for field in HUMAN_FIELDS:
        assert row[field] == ""


def test_empty_template_validates_before_human_completion():
    rows = [_row(), _row("TARGET_BINDING_AMBIGUOUS")]
    rows[1]["event_id"] = ""
    rows[1]["proposed_object_body"] = ""
    rows[1]["proposed_target_body_or_site"] = ""
    errors, summary = validate_review_rows(rows)
    assert errors == []
    assert summary["accepted_event_rows"] == 1
    assert summary["abstain_or_fail_closed_rows"] == 1
    assert summary["completed_review_rows"] == 0
    assert summary["nonempty_human_field_count"] == 0
    assert summary["validation_mode"] == "template_or_partial"


def test_completed_accepted_event_requires_all_event_judgments_and_abstain_na():
    row = _row()
    row["reviewer_id"] = "human_a"
    row["review_timestamp"] = "2026-06-20T20:00:00+08:00"
    row["object_identity_valid"] = "YES"
    errors, summary = validate_review_rows([row], require_completed=True)
    assert summary["validation_status"] == "FAIL"
    assert any("target_identity_valid:required" in err for err in errors)
    for field in EVENT_JUDGMENT_FIELDS:
        row[field] = "YES"
    row["false_positive_carry"] = "NO"
    row["reviewer_notes"] = "carry is not false positive"
    row["abstain_or_fail_closed_correct"] = "NA"
    errors, summary = validate_review_rows([row], require_completed=True)
    assert errors == []
    assert summary["completed_review_rows"] == 1


def test_completed_nonaccepted_row_requires_na_event_fields_and_fail_closed_judgment():
    row = _row("CORRECT_SEMANTIC_ABSTAIN")
    row["event_id"] = ""
    row["proposed_object_body"] = ""
    row["proposed_target_body_or_site"] = ""
    row["reviewer_id"] = "human_a"
    row["review_timestamp"] = "2026-06-20T20:00:00+08:00"
    errors, _ = validate_review_rows([row], require_completed=True)
    assert any("abstain_or_fail_closed_correct:required" in err for err in errors)
    for field in EVENT_JUDGMENT_FIELDS:
        row[field] = "NA"
    row["abstain_or_fail_closed_correct"] = "YES"
    errors, summary = validate_review_rows([row], require_completed=True)
    assert errors == []
    assert summary["completed_review_rows"] == 1


def test_no_or_uncertain_requires_correction_or_notes():
    row = _row()
    row["reviewer_id"] = "human_a"
    row["review_timestamp"] = "2026-06-20T20:00:00+08:00"
    for field in EVENT_JUDGMENT_FIELDS:
        row[field] = "YES"
    row["target_identity_valid"] = "NO"
    row["false_positive_carry"] = "NA"
    row["abstain_or_fail_closed_correct"] = "NA"
    errors, _ = validate_review_rows([row], require_completed=True)
    assert any("target_identity_valid:NO_requires_corrected_target_body_or_site_or_notes" in err for err in errors)
    row["corrected_target_body_or_site"] = "plate_2_default_site"
    row["window_end_valid"] = "UNCERTAIN"
    errors, _ = validate_review_rows([row], require_completed=True)
    assert any("window_end_valid:UNCERTAIN_requires_notes" in err for err in errors)


def test_duplicate_completed_reviewer_key_rejected():
    row1 = _row()
    row2 = _row()
    for row in [row1, row2]:
        row["reviewer_id"] = "human_a"
        row["review_timestamp"] = "2026-06-20T20:00:00+08:00"
        for field in EVENT_JUDGMENT_FIELDS:
            row[field] = "YES"
        row["false_positive_carry"] = "NO"
        row["reviewer_notes"] = "not false positive"
        row["abstain_or_fail_closed_correct"] = "NA"
    errors, _ = validate_review_rows([row1, row2], require_completed=True)
    assert any("duplicate_completed_review_key" in err for err in errors)


def test_forbidden_reviewer_fields_detect_detector_and_attack_leakage():
    assert "task_success" in forbidden_columns(["review_id", "task_success"])
    assert "detector_emit_probability" in forbidden_columns(["detector_emit_probability"])
    assert "VIS_result" in forbidden_columns(["VIS_result"])


def test_reviewer_b_initial_subset_contains_accepted_and_binding_problem_rows():
    accepted = _row()
    ambiguous = _row("TARGET_BINDING_AMBIGUOUS")
    ambiguous["target_binding_status"] = "FAILED"
    abstain = _row("CORRECT_SEMANTIC_ABSTAIN")
    selected = build_reviewer_b_rows([accepted, ambiguous, abstain])
    assert [row["teacher_status"] for row in selected] == [ACCEPTED_STATUS, "TARGET_BINDING_AMBIGUOUS"]
