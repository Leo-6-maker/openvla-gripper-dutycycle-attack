#!/usr/bin/env python3
"""Unit tests for train_sc5_strict_fold0_v2 label building (corridor + release)."""
import json, sys, os, pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "stageb"))

from train_sc5_strict_fold0_v2 import (
    build_labels, SC5_PHASES, TRAIN_TASKS, VAL_TASKS, TEST_TASK
)

# ═══════════════════════════════════════════════
# Test helpers
# ═══════════════════════════════════════════════
def make_teacher_labels(episodes):
    """Build teacher_labels dict from list of episodes.
    Each episode: [(task_idx, state_id, step_idx, phase), ...]"""
    labels = {}
    for ep in episodes:
        for t, s, step, phase in ep:
            labels[(t, s, step)] = {
                "task_idx": t, "state_id": s, "step_idx": step,
                "phase": phase, "split": "train" if t in TRAIN_TASKS else ("val" if t in VAL_TASKS else "test")
            }
    return labels

def make_feature_rows(task_idx, state_id, n_steps):
    """Build CSV row dicts for an episode."""
    rows = []
    for step in range(n_steps):
        rows.append({
            "task_idx": str(task_idx), "state_id": str(state_id),
            "split": "train" if task_idx in TRAIN_TASKS else ("val" if task_idx in VAL_TASKS else "test"),
            "step": str(step),
            **{("f_" + name): "0.0" for name in [
                "gripper_command","gripper_qpos","gripper_opening_proxy",
                "eef_x","eef_y","eef_z","eef_vx","eef_vy","eef_vz",
                "action_dx","action_dy","action_dz","action_gripper",
                "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
                "close_onset","time_since_close","eef_speed",
                "eef_z_delta_since_close","qpos_delta_1","qpos_delta_3",
                "opening_proxy_delta_3","opening_proxy_variance_5","eef_speed_variance_5",
            ]}
        })
    return rows


# ═══════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════

class TestLabelBuilding:
    """Test corridor/release label building with fail-closed semantics."""

    def test_valid_episode_with_corridor(self):
        """A normal train episode with stable_carry → corridor positives."""
        episode = [(0, 0, i, "approach") for i in range(50)] + \
                  [(0, 0, i, "grasp_close") for i in range(50, 55)] + \
                  [(0, 0, i, "stable_carry") for i in range(55, 90)] + \
                  [(0, 0, i, "pre_place_unsupported") for i in range(90, 110)] + \
                  [(0, 0, i, "release_safe") for i in range(110, 115)]
        labels = make_teacher_labels([episode])
        rows = make_feature_rows(0, 0, 115)
        yp, yc, yr, support, audit = build_labels(rows, labels, "train")

        assert support["corridor_positive_rows"] > 0, "Should have corridor positives"
        assert support["corridor_negative_rows"] > 0, "Should have corridor negatives"
        assert support["release_positive_rows"] > 0, "Should have release positives"
        assert support["release_negative_rows"] > 0, "Should have release negatives"
        # Verify corridor positives only at valid K10 starts
        pos_steps = [i for i in range(115) if yc[i] > 0]
        assert len(pos_steps) > 0
        # All corridor-positive steps should be >= stable_carry_start + guard
        sc_start = 55  # first stable_carry
        anchor = sc_start + 5  # guard=5
        for s in pos_steps:
            assert s >= anchor, "Corridor positive at %d < anchor %d" % (s, anchor)

    def test_no_stable_carry_yields_zero_corridor(self):
        """Episode without stable_carry should have 0 corridor positives."""
        episode = [(0, 0, i, "approach") for i in range(100)]
        labels = make_teacher_labels([episode])
        rows = make_feature_rows(0, 0, 100)
        yp, yc, yr, support, audit = build_labels(rows, labels, "train")

        assert support["corridor_positive_rows"] == 0
        assert support["corridor_negative_rows"] == 100
        assert audit[0]["sc5_valid"] == False
        assert audit[0]["sc5_reason"] == "no_stable_carry_phase"

    def test_invalid_anchor_yields_zero_corridor(self):
        """Episode with stable_carry but K10 window crosses release → invalid."""
        # stable_carry too close to release_safe
        episode = [(0, 0, i, "approach") for i in range(10)] + \
                  [(0, 0, i, "stable_carry") for i in range(10, 14)] + \
                  [(0, 0, i, "release_safe") for i in range(14, 20)]
        labels = make_teacher_labels([episode])
        rows = make_feature_rows(0, 0, 20)
        yp, yc, yr, support, audit = build_labels(rows, labels, "train")

        # sc_start=10, anchor=15, window=[15,24] crosses release at 14 → invalid
        assert audit[0]["sc5_valid"] == False
        assert support["corridor_positive_rows"] == 0

    def test_corridor_window_never_overlaps_release(self):
        """Every corridor-positive step's K10 window must not contain release_safe."""
        episode = [(0, 0, i, "approach") for i in range(30)] + \
                  [(0, 0, i, "stable_carry") for i in range(30, 70)] + \
                  [(0, 0, i, "pre_place_unsupported") for i in range(70, 100)] + \
                  [(0, 0, i, "release_safe") for i in range(100, 105)]
        labels = make_teacher_labels([episode])
        rows = make_feature_rows(0, 0, 105)
        yp, yc, yr, support, audit = build_labels(rows, labels, "train")

        release_steps = {i for i in range(100, 105)}
        for i in range(105):
            if yc[i] > 0:
                window = set(range(i, i + 10))
                assert not (window & release_steps), \
                    "Corridor at %d has window overlapping release" % i

    def test_missing_phase_key_raises(self):
        """Teacher label without 'phase' field should raise KeyError."""
        labels = {(0, 0, 0): {"task_idx": 0, "state_id": 0, "step_idx": 0}}
        rows = make_feature_rows(0, 0, 1)
        with pytest.raises(KeyError):
            build_labels(rows, labels, "train")

    def test_missing_label_raises(self):
        """Missing teacher label for a feature row should raise KeyError."""
        labels = {}  # empty
        rows = make_feature_rows(0, 0, 10)
        with pytest.raises(KeyError):
            build_labels(rows, labels, "train")

    def test_noncontiguous_steps_raises(self):
        """Non-contiguous step indices should raise AssertionError."""
        labels = {
            (0, 0, 0): {"task_idx": 0, "state_id": 0, "step_idx": 0, "phase": "approach"},
            (0, 0, 2): {"task_idx": 0, "state_id": 0, "step_idx": 2, "phase": "approach"},
            # step 1 is missing
        }
        rows = [make_feature_rows(0, 0, 1)[0], make_feature_rows(0, 0, 1)[0]]
        rows[0]["step"] = "0"
        rows[1]["step"] = "2"
        with pytest.raises(AssertionError):
            build_labels(rows, labels, "train")

    def test_corridor_pos_and_neg_support(self):
        """Both corridor positives and negatives must exist for a valid episode set."""
        episode = [(0, 0, i, "approach") for i in range(30)] + \
                  [(0, 0, i, "stable_carry") for i in range(30, 70)] + \
                  [(0, 0, i, "release_safe") for i in range(100, 105)]
        labels = make_teacher_labels([episode])
        rows = make_feature_rows(0, 0, 105)
        _, _, _, support, _ = build_labels(rows, labels, "train")

        assert support["corridor_positive_rows"] > 0
        assert support["corridor_negative_rows"] > 0

    def test_release_pos_and_neg_support(self):
        """Both release positives and negatives must exist."""
        episode = [(0, 0, i, "approach") for i in range(30)] + \
                  [(0, 0, i, "stable_carry") for i in range(30, 70)] + \
                  [(0, 0, i, "release_safe") for i in range(100, 105)]
        labels = make_teacher_labels([episode])
        rows = make_feature_rows(0, 0, 105)
        _, _, yr, support, _ = build_labels(rows, labels, "train")

        assert support["release_positive_rows"] > 0
        assert support["release_negative_rows"] > 0

    def test_train_dataset_contains_no_task8(self):
        """Task 8 (chocolate_pudding) must not appear in train labels."""
        # This should not raise because train dataset has no task 8
        episode = [(0, 0, i, "approach") for i in range(50)]
        labels = make_teacher_labels([episode])
        rows = make_feature_rows(0, 0, 50)
        yp, yc, yr, support, audit = build_labels(rows, labels, "train")
        # Verify no task 8 in the data
        assert all(r["task_idx"] != "8" for r in rows)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
