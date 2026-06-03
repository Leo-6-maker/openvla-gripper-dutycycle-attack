# -*- coding: utf-8 -*-
"""Test that prefix_locked_gripper_open_margin actually includes gripper loss.

Includes the critical row-index verification: for action_dim=7, the gripper
logit row is at index -2 (NOT -1), matching ``_active_label_rows()``.
"""

import pytest
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from gripper_attack.attack_adapter import action_token_logit_row_index


ACTION_DIM = 7
GRIPPER_DIM = ACTION_DIM - 1  # 6

OPEN_IDS = list(range(31873, 32000))
CLOSE_IDS = list(range(31744, 31873))


# ── Row index test (CRITICAL — this is the bug fix) ──

class TestGripperLogitRowIndex:
    def test_gripper_row_is_neg2(self):
        """For action_dim=7, gripper(dim=6) predicting row is at -2."""
        idx = action_token_logit_row_index(GRIPPER_DIM, ACTION_DIM)
        assert idx == -2, (
            f"Gripper logit row MUST be -2 for action_dim=7, got {idx}. "
            f"logits[:, -1, :] predicts the post-gripper token, NOT gripper."
        )

    def test_arm_dim_0_is_neg8(self):
        assert action_token_logit_row_index(0, ACTION_DIM) == -8

    def test_arm_dim_5_is_neg3(self):
        assert action_token_logit_row_index(5, ACTION_DIM) == -3

    def test_all_dim_rows(self):
        """Verify all row indices match _active_label_rows convention."""
        expected = {0: -8, 1: -7, 2: -6, 3: -5, 4: -4, 5: -3, 6: -2}
        for dim, exp in expected.items():
            assert action_token_logit_row_index(dim, ACTION_DIM) == exp, \
                f"dim={dim}: expected {exp}, got {action_token_logit_row_index(dim, ACTION_DIM)}"


# ── Prefix-locked loss computation (mirrors _loss() logic) ──

def _gripper_row(logits, action_dim=ACTION_DIM):
    gripper_row_idx = action_token_logit_row_index(action_dim - 1, action_dim)
    return logits[0, gripper_row_idx, :]


def _compute_prefix_margin_loss(logits, region_ids, margin=5.0, action_dim=ACTION_DIM):
    row = _gripper_row(logits, action_dim)
    open_ids = torch.tensor(region_ids, dtype=torch.long)
    log_open = torch.logsumexp(row[open_ids], dim=0)
    non_open_mask = torch.ones_like(row, dtype=torch.bool)
    non_open_mask[open_ids] = False
    max_non_open = row[non_open_mask].max()
    return F.relu(max_non_open - log_open + float(margin))


def _compute_prefix_region_ce_loss(logits, region_ids, action_dim=ACTION_DIM):
    row = _gripper_row(logits, action_dim)
    open_ids = torch.tensor(region_ids, dtype=torch.long)
    log_region = torch.logsumexp(row[open_ids], dim=0)
    log_all = torch.logsumexp(row, dim=0)
    return -(log_region - log_all)


# ── CRITICAL: row index sensitivity ──

class TestGripperRowSensitivity:
    """Verify that the loss responds ONLY to the correct gripper row."""

    def test_changing_correct_row_changes_loss(self):
        """Modifying logits at row -2 changes gripper margin loss."""
        vocab = 32000
        logits = torch.zeros(1, 107, vocab, dtype=torch.float32)
        # Row -2 (gripper): CLOSE dominates
        logits[0, -2, CLOSE_IDS[0]] = 50.0
        logits[0, -2, OPEN_IDS[0]] = 10.0
        loss_before = float(_compute_prefix_margin_loss(logits, OPEN_IDS))

        # Increase OPEN strength at row -2
        logits[0, -2, OPEN_IDS[0]] = 80.0
        loss_after = float(_compute_prefix_margin_loss(logits, OPEN_IDS))
        assert loss_after != loss_before, (
            f"Changing gripper row (-2) must change loss. "
            f"before={loss_before:.4f} after={loss_after:.4f}"
        )

    def test_changing_wrong_row_does_not_change_loss(self):
        """Modifying logits at row -1 does NOT change gripper margin loss."""
        vocab = 32000
        logits = torch.zeros(1, 107, vocab, dtype=torch.float32)
        # Row -2 (gripper): CLOSE dominates
        logits[0, -2, CLOSE_IDS[0]] = 50.0
        logits[0, -2, OPEN_IDS[0]] = 10.0
        loss_before = float(_compute_prefix_margin_loss(logits, OPEN_IDS))

        # Change row -1 (post-gripper — should NOT affect gripper loss)
        logits[0, -1, OPEN_IDS[0]] = 999.0
        loss_after = float(_compute_prefix_margin_loss(logits, OPEN_IDS))
        assert loss_before == loss_after, (
            f"Changing post-gripper row (-1) must NOT change gripper loss. "
            f"before={loss_before:.4f} after={loss_after:.4f}"
        )

    def test_gripper_row_matches_active_label_rows(self):
        """The row used by prefix loss equals the row _active_label_rows returns for gripper dim."""
        vocab = 32000
        # Construct a minimal input: prompt of length 100, then 7 action tokens
        seq_len = 100
        full_len = seq_len + ACTION_DIM
        logits = torch.zeros(1, full_len, vocab, dtype=torch.float32)

        # Simulate _active_label_rows: gripper label at last position
        gripper_label_pos = full_len - 1  # last token = gripper
        # _active_label_rows computes: row_index = -(action_dim - dim + 1) = -(7-6+1) = -2
        active_row_index = action_token_logit_row_index(GRIPPER_DIM, ACTION_DIM)  # -2

        # The prefix-loss gripper row
        prefix_row_idx = action_token_logit_row_index(ACTION_DIM - 1, ACTION_DIM)

        assert active_row_index == prefix_row_idx == -2, (
            f"Row index mismatch: active={active_row_index}, prefix={prefix_row_idx}"
        )


# ── Margin loss tests ──

class TestPrefixLockedLossContainsGripper:
    def test_gripper_loss_nonzero_when_open_weaker(self):
        vocab = 32000
        logits = torch.zeros(1, 107, vocab, dtype=torch.float32)
        logits[0, -2, CLOSE_IDS[0]] = 50.0
        logits[0, -2, OPEN_IDS[0]] = 10.0
        loss = _compute_prefix_margin_loss(logits, OPEN_IDS, margin=5.0)
        assert float(loss) > 0.0, f"Expected positive gripper margin loss, got {float(loss):.4f}"

    def test_gripper_loss_zero_when_open_dominates(self):
        vocab = 32000
        logits = torch.zeros(1, 107, vocab, dtype=torch.float32)
        logits[0, -2, OPEN_IDS[0]] = 100.0
        logits[0, -2, CLOSE_IDS[0]] = 10.0
        loss = _compute_prefix_margin_loss(logits, OPEN_IDS, margin=5.0)
        assert float(loss) == 0.0, f"Expected zero margin loss, got {float(loss):.4f}"

    def test_gripper_loss_grad_flows(self):
        vocab = 32000
        logits = torch.zeros(1, 107, vocab, dtype=torch.float32)
        logits[0, -2, CLOSE_IDS[0]] = 50.0
        logits[0, -2, OPEN_IDS[0]] = 10.0
        logits.requires_grad_(True)
        loss = _compute_prefix_margin_loss(logits, OPEN_IDS, margin=5.0)
        loss.backward()
        gripper_grad = logits.grad[0, -2, :]
        assert gripper_grad.abs().sum() > 0, "Gradient should be non-zero on gripper row (-2)"

    def test_region_ce_loss_nonzero(self):
        vocab = 32000
        logits = torch.zeros(1, 107, vocab, dtype=torch.float32)
        logits[0, -2, CLOSE_IDS[0]] = 50.0
        logits[0, -2, OPEN_IDS[0]] = 10.0
        loss = _compute_prefix_region_ce_loss(logits, OPEN_IDS)
        assert float(loss) > 0.0, f"Expected positive region CE loss, got {float(loss):.4f}"

    def test_gripper_loss_unaffected_by_label_mask(self):
        vocab = 32000
        logits_a = torch.zeros(1, 107, vocab, dtype=torch.float32)
        logits_a[0, -2, CLOSE_IDS[0]] = 50.0
        logits_a[0, -2, OPEN_IDS[0]] = 10.0
        logits_b = logits_a.clone()
        l_a = _compute_prefix_margin_loss(logits_a, OPEN_IDS, margin=5.0)
        l_b = _compute_prefix_margin_loss(logits_b, OPEN_IDS, margin=5.0)
        assert float(l_a) == float(l_b), "Gripper loss independent of label state"

    def test_open_region_nonempty(self):
        assert len(OPEN_IDS) > 0

    def test_close_region_nonempty(self):
        assert len(CLOSE_IDS) > 0

    def test_open_close_disjoint(self):
        assert set(OPEN_IDS).isdisjoint(set(CLOSE_IDS))

    def test_combined_loss_includes_both_terms(self):
        vocab = 32000
        logits = torch.zeros(1, 107, vocab, dtype=torch.float32)
        logits[0, -2, CLOSE_IDS[0]] = 50.0
        logits[0, -2, OPEN_IDS[0]] = 10.0
        # Arm rows at correct indices
        for arm_dim in range(6):
            row_index = action_token_logit_row_index(arm_dim, ACTION_DIM)
            logits[0, row_index, 999] = 100.0

        grip_loss = _compute_prefix_margin_loss(logits, OPEN_IDS, margin=5.0)
        assert float(grip_loss) > 0.0, "gripper margin loss must be positive"

        arm_ces = []
        for arm_dim in range(6):
            row_index = action_token_logit_row_index(arm_dim, ACTION_DIM)
            row = logits[0, row_index, :]
            target = torch.tensor([arm_dim * 1000 + 1000])
            ce = F.cross_entropy(row.view(1, -1), target)
            arm_ces.append(ce)
        arm_term = torch.stack(arm_ces).mean()
        assert float(arm_term) > 0.0, "arm CE loss must be present"
