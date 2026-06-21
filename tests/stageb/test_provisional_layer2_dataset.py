import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.stageb.build_provisional_layer2_dataset import (  # noqa: E402
    IGNORE_STATUSES,
    SC5_FEATURES,
    SUPPLEMENTARY_POSITIVE_STATUS,
    build_rows_for_episode,
    leakage_audit,
    label_role_for_status,
    phase_for_step,
)


def test_eligible_event_phase_and_corridor_labels():
    label = {"teacher_status": "ELIGIBLE_EVENT"}
    event = {
        "close_onset_step": "10",
        "grasp_established_step": "12",
        "lift_onset_step": "14",
        "stable_carry_start": "16",
        "teacher_window_start": "8",
        "teacher_window_end": "20",
        "release_onset_step": "19",
    }
    assert phase_for_step(7, label, event) == ("approach", 0, 0)
    assert phase_for_step(10, label, event) == ("grasp_close", 1, 0)
    assert phase_for_step(12, label, event) == ("stable_grasp", 1, 0)
    assert phase_for_step(14, label, event) == ("first_lift", 1, 0)
    assert phase_for_step(16, label, event) == ("stable_carry", 1, 0)
    assert phase_for_step(19, label, event) == ("release_safe", 1, 1)
    assert phase_for_step(25, label, event) == ("recovery_or_regrasp", 0, 0)


def test_non_eligible_statuses_never_create_positive_corridor():
    for status in ["CORRECT_SEMANTIC_ABSTAIN", "NO_RELEVANT_GRASP_EVENT", *sorted(IGNORE_STATUSES)]:
        phase, corridor, release = phase_for_step(10, {"teacher_status": status}, None)
        assert phase == "abstain_unsupported"
        assert corridor == 0
        assert release == 0


def test_supplementary_event_uses_positive_phase_labels():
    label = {"teacher_status": SUPPLEMENTARY_POSITIVE_STATUS}
    event = {
        "close_onset_step": "10",
        "grasp_established_step": "12",
        "lift_onset_step": "14",
        "stable_carry_start": "16",
        "teacher_window_start": "8",
        "teacher_window_end": "20",
        "release_onset_step": "",
    }
    assert phase_for_step(16, label, event) == ("stable_carry", 1, 0)
    assert label_role_for_status(SUPPLEMENTARY_POSITIVE_STATUS) == (
        "supplementary_multievent_grasp_carry_bridge",
        "supplementary",
    )


def test_frame_dataset_excludes_task_success_and_uses_primary_supplementary_event(tmp_path):
    ep = tmp_path / "episode"
    ep.mkdir()
    fields = ["step", *SC5_FEATURES]
    with (ep / "step_telemetry.csv").open("w", newline="", encoding="utf-8") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"step": "16", **{name: "0.0" for name in SC5_FEATURES}})
    manifest_row = {
        "episode_path": str(ep),
        "canonical_key": "libero_10|4|10|0|CLEAN",
        "suite": "libero_10",
        "task_idx": "4",
        "state_id": "10",
        "eval_seed": "0",
        "task_success": "True",
    }
    label = {
        "teacher_status": SUPPLEMENTARY_POSITIVE_STATUS,
        "mechanism_type": "multi_object_transfer",
        "primary_supplementary_event_id": "event_b",
    }
    events = [
        {
            "event_id": "event_a",
            "close_onset_step": "1",
            "grasp_established_step": "2",
            "lift_onset_step": "3",
            "stable_carry_start": "4",
            "teacher_window_start": "0",
            "teacher_window_end": "5",
            "teacher_anchor_step": "2",
            "release_onset_step": "",
        },
        {
            "event_id": "event_b",
            "close_onset_step": "10",
            "grasp_established_step": "12",
            "lift_onset_step": "14",
            "stable_carry_start": "16",
            "teacher_window_start": "8",
            "teacher_window_end": "20",
            "teacher_anchor_step": "12",
            "release_onset_step": "",
        },
    ]
    rows, problems = build_rows_for_episode(manifest_row=manifest_row, label=label, events=events, split="train")
    assert problems == []
    assert rows[0]["event_id"] == "event_b"
    assert rows[0]["label_role"] == "supplementary_multievent_grasp_carry_bridge"
    assert "task_success" not in rows[0]


def test_leakage_audit_rejects_split_state_overlap_and_duplicate_frames():
    rows = [
        {"dataset_split": "train", "episode_key": "a", "suite": "libero_spatial", "task_idx": 0, "state_id": 10, "step": 0, "features_finite": 1},
        {"dataset_split": "val", "episode_key": "b", "suite": "libero_spatial", "task_idx": 0, "state_id": 10, "step": 0, "features_finite": 1},
        {"dataset_split": "train", "episode_key": "a", "suite": "libero_spatial", "task_idx": 0, "state_id": 10, "step": 0, "features_finite": 1},
    ]
    audit = leakage_audit(rows)
    assert audit["status"] == "FAIL"
    assert audit["duplicate_frame_count"] == 1
    assert audit["split_overlap"]["train__val"]["state_overlap"] == 1


def test_leakage_audit_accepts_disjoint_splits():
    rows = [
        {"dataset_split": "train", "episode_key": "a", "suite": "libero_spatial", "task_idx": 0, "state_id": 10, "step": 0, "features_finite": 1},
        {"dataset_split": "val", "episode_key": "b", "suite": "libero_spatial", "task_idx": 0, "state_id": 18, "step": 0, "features_finite": 1},
        {"dataset_split": "test", "episode_key": "c", "suite": "libero_spatial", "task_idx": 0, "state_id": 0, "step": 0, "features_finite": 1},
    ]
    audit = leakage_audit(rows)
    assert audit["status"] == "PASS"
    assert audit["failures"] == []
