import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stageb.prepare_h2_human_review_forms import (  # noqa: E402
    ACCEPTED_STATUS,
    HUMAN_FIELDS,
    build_review_template,
    validate_review_rows,
)


def _queue_row(status=ACCEPTED_STATUS):
    return {
        "review_id": "review_000_event_00",
        "episode_key": "libero_spatial|0|10|0|CLEAN",
        "suite": "libero_spatial",
        "task_idx": "0",
        "state_id": "10",
        "mechanism_type": "single_object_pick_place",
        "teacher_status": status,
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
        "blind_video_path": "/server/video.mp4",
        "teacher_only_timeline_path": "/server/timeline.csv",
        "teacher_only_overlay_path": "/server/overlay.mp4",
    }


def test_review_template_preserves_proposals_and_leaves_human_fields_blank():
    rows = build_review_template([_queue_row()])
    assert rows[0]["review_id"] == "review_000_event_00"
    assert rows[0]["proposed_object_body"] == "akita_black_bowl_1_main"
    for field in HUMAN_FIELDS:
        assert rows[0][field] == ""


def test_empty_template_validates_before_human_completion():
    rows = build_review_template([_queue_row(), _queue_row("TARGET_BINDING_AMBIGUOUS")])
    errors, summary = validate_review_rows(rows)
    assert errors == []
    assert summary["accepted_event_rows"] == 1
    assert summary["abstain_or_fail_closed_rows"] == 1
    assert summary["completed_review_rows"] == 0


def test_completed_accepted_event_requires_all_event_judgments():
    row = build_review_template([_queue_row()])[0]
    row["reviewer_id"] = "human_a"
    row["object_identity_valid"] = "YES"
    errors, summary = validate_review_rows([row], require_completed=True)
    assert summary["validation_status"] == "FAIL"
    assert any("target_identity_valid:required" in err for err in errors)


def test_completed_abstain_row_requires_fail_closed_judgment():
    row = build_review_template([_queue_row("CORRECT_SEMANTIC_ABSTAIN")])[0]
    row["reviewer_id"] = "human_a"
    errors, _ = validate_review_rows([row], require_completed=True)
    assert any("abstain_or_fail_closed_correct:required" in err for err in errors)
    row["abstain_or_fail_closed_correct"] = "YES"
    errors, summary = validate_review_rows([row], require_completed=True)
    assert errors == []
    assert summary["completed_review_rows"] == 1


def test_invalid_judgment_enum_rejected():
    row = build_review_template([_queue_row()])[0]
    row["object_identity_valid"] = "MAYBE"
    errors, _ = validate_review_rows([row])
    assert any("invalid_enum:MAYBE" in err for err in errors)
