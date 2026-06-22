import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.stageb.build_provisional_layer2_dataset import (  # noqa: E402
    IGNORE_STATUSES,
    leakage_audit,
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
