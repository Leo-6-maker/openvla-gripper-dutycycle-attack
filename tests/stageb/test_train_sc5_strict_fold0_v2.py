#!/usr/bin/env python3
"""Unit tests for train_sc5_strict_fold0_v2 label building (corridor + release).
P0-2 FIX: build_labels(enforce_support_gates=False) for single-episode tests."""
import json, sys, os, pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "stageb"))

from train_sc5_strict_fold0_v2 import (
    build_labels, validate_label_support,
    SC5_PHASES, TRAIN_TASKS, VAL_TASKS, TEST_TASK
)

FEATURE_NAMES = [
    "gripper_command","gripper_qpos","gripper_opening_proxy",
    "eef_x","eef_y","eef_z","eef_vx","eef_vy","eef_vz",
    "action_dx","action_dy","action_dz","action_gripper",
    "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
    "close_onset","time_since_close","eef_speed",
    "eef_z_delta_since_close","qpos_delta_1","qpos_delta_3",
    "opening_proxy_delta_3","opening_proxy_variance_5","eef_speed_variance_5",
]

def make_teacher_labels(episodes):
    labels = {}
    for ep in episodes:
        for t, s, step, phase in ep:
            labels[(t, s, step)] = {
                "task_idx": t, "state_id": s, "step_idx": step,
                "phase": phase, "split": "train" if t in TRAIN_TASKS else ("val" if t in VAL_TASKS else "test")
            }
    return labels

def make_feature_rows(task_idx, state_id, n_steps):
    rows = []
    for step in range(n_steps):
        d = {"task_idx": str(task_idx), "state_id": str(state_id),
             "split": "train" if task_idx in TRAIN_TASKS else ("val" if task_idx in VAL_TASKS else "test"),
             "step": str(step)}
        for name in FEATURE_NAMES:
            d["f_" + name] = "0.0"
        rows.append(d)
    return rows


class TestLabelBuilding:

    def test_valid_episode_with_corridor(self):
        """Normal train episode: approach→grasp→stable_carry→pre_place→release."""
        ep = ([(0,0,i,"approach") for i in range(50)] +
              [(0,0,i,"grasp_close") for i in range(50,55)] +
              [(0,0,i,"stable_carry") for i in range(55,95)] +
              [(0,0,i,"pre_place_unsupported") for i in range(95,110)] +
              [(0,0,i,"release_safe") for i in range(110,115)])
        labels = make_teacher_labels([ep])
        rows = make_feature_rows(0, 0, 115)
        yp, yc, yr, support, audit = build_labels(rows, labels, "test_ep", enforce_support_gates=False)

        assert support["corridor_positive_rows"] > 0
        assert support["corridor_negative_rows"] > 0
        assert support["release_positive_rows"] > 0
        assert support["release_negative_rows"] > 0
        # Corridor positives must be >= anchor (sc_start + guard = 55+5=60)
        pos_steps = [i for i in range(115) if yc[i] > 0]
        assert len(pos_steps) > 0
        anchor = 55 + 5
        for s in pos_steps:
            assert s >= anchor, "Corridor positive at %d < anchor %d" % (s, anchor)

    def test_no_stable_carry_yields_zero_corridor(self):
        """Episode without stable_carry → 0 corridor positives."""
        labels = make_teacher_labels([[(0,0,i,"approach") for i in range(100)]])
        rows = make_feature_rows(0, 0, 100)
        yp, yc, yr, support, audit = build_labels(rows, labels, "test_ep", enforce_support_gates=False)

        assert support["corridor_positive_rows"] == 0
        assert support["corridor_negative_rows"] == 100
        assert audit[0]["sc5_valid"] == False
        assert audit[0]["sc5_reason"] == "no_stable_carry_phase"

    def test_invalid_anchor_yields_zero_corridor(self):
        """stable_carry exists but K10 window crosses release → invalid SC5 → 0 corridor."""
        ep = ([(0,0,i,"approach") for i in range(8)] +
              [(0,0,i,"stable_carry") for i in range(8,14)] +
              [(0,0,i,"release_safe") for i in range(14,20)])
        labels = make_teacher_labels([ep])
        rows = make_feature_rows(0, 0, 20)
        yp, yc, yr, support, audit = build_labels(rows, labels, "test_ep", enforce_support_gates=False)

        assert audit[0]["sc5_valid"] == False
        assert audit[0]["sc5_anchor"] == 8 + 5  # sc_start=8, guard=5
        assert support["corridor_positive_rows"] == 0

    def test_corridor_window_never_overlaps_release(self):
        """Every corridor-positive step's K10 window must not contain release_safe."""
        ep = ([(0,0,i,"approach") for i in range(30)] +
              [(0,0,i,"stable_carry") for i in range(30,70)] +
              [(0,0,i,"pre_place_unsupported") for i in range(70,100)] +
              [(0,0,i,"release_safe") for i in range(100,105)])
        labels = make_teacher_labels([ep])
        rows = make_feature_rows(0, 0, 105)
        yp, yc, yr, support, audit = build_labels(rows, labels, "test_ep", enforce_support_gates=False)

        release_steps = set(range(100, 105))
        for i in range(105):
            if yc[i] > 0:
                window = set(range(i, i + 10))
                assert not (window & release_steps), \
                    "Corridor at %d has window overlapping release" % i

    def test_missing_phase_key_raises(self):
        """Teacher label without 'phase' → KeyError."""
        labels = {(0, 0, 0): {"task_idx": 0, "state_id": 0, "step_idx": 0}}
        rows = make_feature_rows(0, 0, 1)
        with pytest.raises(KeyError):
            build_labels(rows, labels, "test_ep", enforce_support_gates=False)

    def test_missing_label_raises(self):
        """Missing teacher label for a feature row → KeyError."""
        labels = {}
        rows = make_feature_rows(0, 0, 10)
        with pytest.raises(KeyError):
            build_labels(rows, labels, "test_ep", enforce_support_gates=False)

    def test_noncontiguous_steps_raises(self):
        """Non-contiguous step indices → AssertionError."""
        labels = {
            (0, 0, 0): {"task_idx": 0, "state_id": 0, "step_idx": 0, "phase": "approach"},
            (0, 0, 2): {"task_idx": 0, "state_id": 0, "step_idx": 2, "phase": "approach"},
        }
        rows = [make_feature_rows(0, 0, 1)[0], make_feature_rows(0, 0, 1)[0]]
        rows[0]["step"] = "0"; rows[1]["step"] = "2"
        with pytest.raises(AssertionError):
            build_labels(rows, labels, "test_ep", enforce_support_gates=False)

    def test_corridor_pos_and_neg_support(self):
        """Dataset with corridor episodes → both pos and neg non-zero."""
        ep = ([(0,0,i,"approach") for i in range(30)] +
              [(0,0,i,"stable_carry") for i in range(30,70)] +
              [(0,0,i,"pre_place_unsupported") for i in range(70,100)] +
              [(0,0,i,"release_safe") for i in range(100,105)])
        labels = make_teacher_labels([ep])
        rows = make_feature_rows(0, 0, 105)
        _, _, _, support, _ = build_labels(rows, labels, "test_ep", enforce_support_gates=False)
        assert support["corridor_positive_rows"] > 0
        assert support["corridor_negative_rows"] > 0

    def test_release_pos_and_neg_support(self):
        """Dataset with release_safe → both pos and neg non-zero."""
        ep = ([(0,0,i,"approach") for i in range(30)] +
              [(0,0,i,"stable_carry") for i in range(30,70)] +
              [(0,0,i,"pre_place_unsupported") for i in range(70,100)] +
              [(0,0,i,"release_safe") for i in range(100,105)])
        labels = make_teacher_labels([ep])
        rows = make_feature_rows(0, 0, 105)
        _, _, yr, support, _ = build_labels(rows, labels, "test_ep", enforce_support_gates=False)
        assert support["release_positive_rows"] > 0
        assert support["release_negative_rows"] > 0

    def test_dataset_gate_rejects_all_zero_corridor(self):
        """validate_label_support must reject all-zero corridor."""
        support = {"corridor_positive_rows": 0, "corridor_negative_rows": 100,
                   "release_positive_rows": 5, "release_negative_rows": 95,
                   "phase_unique_classes": 5}
        with pytest.raises(AssertionError):
            validate_label_support(support, "test")

    def test_dataset_gate_rejects_all_zero_release(self):
        """validate_label_support must reject all-zero release."""
        support = {"corridor_positive_rows": 10, "corridor_negative_rows": 90,
                   "release_positive_rows": 0, "release_negative_rows": 100,
                   "phase_unique_classes": 5}
        with pytest.raises(AssertionError):
            validate_label_support(support, "test")

    def test_dataset_gate_passes_valid_support(self):
        """validate_label_support must pass valid support counts."""
        support = {"corridor_positive_rows": 10, "corridor_negative_rows": 90,
                   "release_positive_rows": 5, "release_negative_rows": 95,
                   "phase_unique_classes": 5}
        validate_label_support(support, "test")  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
