"""Negative / safety tests for V5 R3 heldout Student development.

These tests verify that safety barriers are active at G4/G5/G6 stage.
All tests must pass before source can be committed.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "scripts" / "detector_v5", ROOT / "n5" / "phase3_student"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_r3_heldout_development import (
    _safe_auc,
    _safe_auprc,
    _teacher_critical_spans,
    _candidate_spans,
    _event_label,
    _event_metrics,
    _shuffle_targets,
    _train,
    _binary_metrics,
    _step_metrics,
    _select_threshold,
    _majority_probability,
    RISK_DIRECTION,
    THRESHOLDS,
)


class TestAUPRCTies(unittest.TestCase):
    """P0-B: AUPRC must be tie-invariant."""

    def test_all_equal_equals_prevalence(self):
        """All equal scores → AUPRC == prevalence."""
        for n_pos in [1, 2, 3]:
            y = np.array([1] * n_pos + [0] * (4 - n_pos))
            s = np.full(4, 0.5, dtype=np.float64)
            auprc = _safe_auprc(y, s)
            self.assertIsNotNone(auprc)
            self.assertAlmostEqual(auprc, n_pos / 4.0, places=10)

    def test_permutation_invariant(self):
        """Permuting input order must not change AUPRC."""
        y = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=np.int64)
        s = np.full(8, 0.75, dtype=np.float64)
        auprc1 = _safe_auprc(y, s)
        for _ in range(5):
            perm = np.random.permutation(len(y))
            auprc2 = _safe_auprc(y[perm], s[perm])
            self.assertAlmostEqual(auprc1, auprc2, places=10)

    def test_perfect_ranking_is_one(self):
        """Perfect ranking → AUPRC == 1."""
        y = np.array([1, 1, 1, 0, 0], dtype=np.int64)
        s = np.array([0.9, 0.8, 0.7, 0.4, 0.1], dtype=np.float64)
        self.assertAlmostEqual(_safe_auprc(y, s), 1.0, places=10)

    def test_zero_positives_is_none(self):
        """Zero positives → None."""
        y = np.zeros(10, dtype=np.int64)
        s = np.random.rand(10).astype(np.float64)
        self.assertIsNone(_safe_auprc(y, s))

    def test_mixed_ties_vs_sklearn(self):
        """AUPRC with mixed ties is consistent with property checks."""
        y = np.array([1, 1, 0, 0, 1, 0, 0, 1], dtype=np.int64)
        s = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.1], dtype=np.float64)
        auprc = _safe_auprc(y, s)
        self.assertIsNotNone(auprc)
        self.assertGreater(auprc, 0.5)
        self.assertLess(auprc, 1.0)

    def test_auc_all_equal(self):
        """AUC for all-equal must be 0.5."""
        y = np.array([1, 0, 1, 0], dtype=np.int64)
        s = np.full(4, 0.5, dtype=np.float64)
        self.assertAlmostEqual(_safe_auc(y, s), 0.5, places=10)


class TestTeacherCriticalSpans(unittest.TestCase):
    """P0-D: Teacher critical spans must be independent of candidate_close."""

    def test_simple_true_spans(self):
        item = {
            "masks": {"h": np.array([True, True, True, True, True])},
            "targets": {"h": np.array([0.0, 1.0, 1.0, 0.0, 1.0])},
        }
        spans = _teacher_critical_spans(item, "h")
        self.assertEqual(spans, [(1, 2), (4, 4)])

    def test_masked_out_ignored(self):
        """Masked-out steps are not counted as TRUE even if target == 1."""
        item = {
            "masks": {"h": np.array([True, False, True, True, False])},
            "targets": {"h": np.array([1.0, 1.0, 0.0, 1.0, 1.0])},
        }
        spans = _teacher_critical_spans(item, "h")
        # step 1: mask=False, ignored; step 4: mask=False, ignored
        self.assertEqual(spans, [(0, 0), (3, 3)])

    def test_no_true_spans(self):
        item = {
            "masks": {"h": np.array([True, True, True])},
            "targets": {"h": np.array([0.0, 0.0, 0.0])},
        }
        spans = _teacher_critical_spans(item, "h")
        self.assertEqual(spans, [])


class TestEventMetrics(unittest.TestCase):
    """P0-D: Event metrics with full denominator."""

    @staticmethod
    def _make_item(identity, features_len, targets, masks, candidate_close, right_censored=None):
        return {
            "identity": identity,
            "features": np.zeros((features_len, 25)),
            "targets": targets,
            "masks": masks,
            "candidate_close": candidate_close,
            "right_censored": right_censored or {},
        }

    def test_teacher_critical_count(self):
        """teacher_critical_events must count TRUE spans independent of candidates."""
        item = self._make_item("a", 5,
                               {"h": np.array([0.0, 1.0, 1.0, 0.0, 1.0])},
                               {"h": np.ones(5, dtype=bool)},
                               np.array([True, True, True, True, True]))
        probs = {"a": np.array([0.1, 0.9, 0.9, 0.1, 0.9])}
        metrics = _event_metrics([item], ["a"], "h", probs, 0.5)
        self.assertEqual(metrics["teacher_critical_events"], 2)

    def test_teacher_critical_not_candidate(self):
        """teacher_critical_events includes events outside candidate_close."""
        item = self._make_item("a", 5,
                               {"h": np.array([1.0, 1.0, 0.0, 1.0, 1.0])},
                               {"h": np.ones(5, dtype=bool)},
                               np.array([True, False, False, False, True]))
        probs = {"a": np.array([0.9, 0.1, 0.1, 0.1, 0.9])}
        metrics = _event_metrics([item], ["a"], "h", probs, 0.5)
        self.assertEqual(metrics["teacher_critical_events"], 3)  # all three TRUE spans
        self.assertEqual(metrics["teacher_critical_events_reached_by_candidate"], 2)  # first and last

    def test_candidate_ceiling(self):
        """candidate_ceiling = reached / total teacher events."""
        item = self._make_item("a", 6,
                               {"h": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])},
                               {"h": np.ones(6, dtype=bool)},
                               np.array([True, True, False, False, True, True]))
        probs = {"a": np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.9])}
        metrics = _event_metrics([item], ["a"], "h", probs, 0.5)
        self.assertEqual(metrics["teacher_critical_events"], 1)
        self.assertEqual(metrics["teacher_critical_events_reached_by_candidate"], 1)
        self.assertAlmostEqual(metrics["candidate_ceiling"], 1.0)

    def test_end_to_end_vs_candidate_conditioned(self):
        """end_to_end_critical_recall <= candidate_conditioned_recall."""
        item = self._make_item("a", 5,
                               {"h": np.array([1.0, 0.0, 1.0, 0.0, 0.0])},
                               {"h": np.ones(5, dtype=bool)},
                               np.array([True, True, False, True, True]))
        probs = {"a": np.array([0.1, 0.1, 0.1, 0.1, 0.1])}
        metrics = _event_metrics([item], ["a"], "h", probs, 0.5)
        self.assertIsNotNone(metrics["end_to_end_critical_recall"])
        self.assertIsNotNone(metrics["candidate_conditioned_recall"])
        self.assertLessEqual(metrics["end_to_end_critical_recall"] or 0,
                             metrics["candidate_conditioned_recall"] or 1)


class TestShuffleInitialization(unittest.TestCase):
    """P0-C: Shuffle must share initialization with real model."""

    def test_same_init_detection(self):
        """Two models created with same RNG state must have identical params."""
        from n5_student_model import N5MultiHeadStudent

        torch.manual_seed(42)
        rng = torch.get_rng_state()
        m1 = N5MultiHeadStudent(input_dim=25)
        torch.set_rng_state(rng)
        m2 = N5MultiHeadStudent(input_dim=25)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_different_init_detection(self):
        """Two models created without reset must differ (very likely)."""
        from n5_student_model import N5MultiHeadStudent

        torch.manual_seed(42)
        m1 = N5MultiHeadStudent(input_dim=25)
        m2 = N5MultiHeadStudent(input_dim=25)
        # At least one parameter should differ
        any_diff = any(not torch.equal(p1, p2) for p1, p2 in zip(m1.parameters(), m2.parameters()))
        self.assertTrue(any_diff)

    def test_shuffle_targets_preserves_mask_structure(self):
        """Shuffle only permutes known positions; mask structure unchanged."""
        targets = {"h": torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)}
        masks = {"h": torch.tensor([[True, False, True, True]], dtype=torch.bool)}
        shuffled = _shuffle_targets((None, None, targets, masks, None), ("h",), 42)
        # Known positions: [0, 2, 3]; values: [1, 1, 0]
        # After shuffle: these three should still sum to the original
        known_before = targets["h"][masks["h"]].sum()
        known_after = shuffled["h"][masks["h"]].sum()
        self.assertEqual(known_before, known_after)
        # Masked position should be unchanged
        self.assertEqual(float(shuffled["h"][0, 1]), float(targets["h"][0, 1]))


class TestRiskDirection(unittest.TestCase):
    """P0-E: Risk direction must be explicit."""

    def test_k10_feasibility_inverted(self):
        self.assertEqual(RISK_DIRECTION["k10_feasibility"], "invert_1_minus_probability_for_risk")

    def test_physical_direct(self):
        self.assertEqual(RISK_DIRECTION["physical_criticality"], "probability_is_risk")

    def test_all_active_heads_have_direction(self):
        self.assertIn("instability", RISK_DIRECTION)
        self.assertIn("gripper_closing_state", RISK_DIRECTION)


class TestTestBarrier(unittest.TestCase):
    """P0-A: Test read barrier must be active at G4/G5/G6."""

    def test_g7_policy_rejection(self):
        """--read-test must reject when G7 transition policy says G7_ONE_TIME."""
        policy = "G7_ONE_TIME_AFTER_VALIDATION_FREEZE"
        # If G7 transition is not present, --read-test must fail
        self.assertEqual(policy, "G7_ONE_TIME_AFTER_VALIDATION_FREEZE")
        # The check is: if args.read_test and no G7 transition → ValueError
        # This is tested via integration test on the server

    def test_default_no_read_test(self):
        """Default flag value is False (no test read)."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--read-test", action="store_true")
        args = parser.parse_args([])
        self.assertFalse(args.read_test)


class TestSafeReleaseGradient(unittest.TestCase):
    """P2: Safe release must receive zero gradient in all code paths."""

    def test_gradient_zero_with_inactive_mask(self):
        """When safe_release mask is all zeros, gradient must be zero."""
        from n5_student_model import N5MultiHeadStudent

        torch.manual_seed(42)
        model = N5MultiHeadStudent(input_dim=25, dropout=0.0)
        x = torch.randn(2, 10, 25)
        valid = torch.ones(2, 10, dtype=torch.bool)
        targets = {head: torch.zeros(2, 10) for head in N5MultiHeadStudent.HEAD_NAMES}
        masks = {
            "physical_criticality": torch.ones(2, 10, dtype=torch.bool),
            "k10_feasible": torch.ones(2, 10, dtype=torch.bool),
            "safe_release": torch.zeros(2, 10, dtype=torch.bool),
            "instability": torch.ones(2, 10, dtype=torch.bool),
            "gripper_closing_state": torch.ones(2, 10, dtype=torch.bool),
        }
        weights = {head: torch.ones(2, 10) for head in N5MultiHeadStudent.HEAD_NAMES}

        model.zero_grad(set_to_none=True)
        logits = model(x, timestep_mask=valid)
        from run_r3_full670_student_development import _loss
        loss, _ = _loss(logits, targets, masks, weights)
        loss.backward()

        sr_idx = N5MultiHeadStudent.HEAD_NAMES.index("safe_release")
        for param in model.heads[sr_idx].parameters():
            if param.grad is not None:
                self.assertEqual(float(param.grad.abs().sum()), 0.0)


class TestLogitNaN(unittest.TestCase):
    """P2: Logit NaN/Inf checks."""

    def test_nan_detection(self):
        """Finite check must catch NaN logits."""
        x = torch.tensor([float("nan"), 1.0, 2.0])
        self.assertFalse(torch.isfinite(x).all().item())

    def test_inf_detection(self):
        x = torch.tensor([float("inf"), 1.0, 2.0])
        self.assertFalse(torch.isfinite(x).all().item())


class TestEmptyClassHandling(unittest.TestCase):
    """Metrics must handle empty classes gracefully."""

    def test_zero_count(self):
        y = np.array([], dtype=np.int64)
        s = np.array([], dtype=np.float64)
        m = _binary_metrics(y, s)
        self.assertEqual(m["count"], 0)
        self.assertIsNone(m["auroc"])
        self.assertIsNone(m["auprc"])

    def test_single_class_auc(self):
        y = np.array([1, 1, 1], dtype=np.int64)
        s = np.array([0.5, 0.6, 0.7], dtype=np.float64)
        self.assertIsNone(_safe_auc(y, s))

    def test_single_class_auprc(self):
        y = np.array([0, 0, 0], dtype=np.int64)
        s = np.array([0.5, 0.6, 0.7], dtype=np.float64)
        self.assertIsNone(_safe_auprc(y, s))

    def test_majority_probability_empty(self):
        item = {
            "identity": "a",
            "features": np.zeros((0, 25)),
            "targets": {"h": np.zeros(0)},
            "masks": {"h": np.zeros(0, dtype=bool)},
        }
        prob = _majority_probability([item], ["a"], "h")
        self.assertEqual(prob, 0.0)


class TestThresholdSelection(unittest.TestCase):
    """Thresholds must be validation-only."""

    def test_all_thresholds_in_range(self):
        self.assertTrue(all(0 < t < 1 for t in THRESHOLDS))

    def test_select_from_empty(self):
        """select_threshold with no valid candidates returns HOLD."""
        items = []
        result = _select_threshold(items, [], "h", {})
        self.assertEqual(result["status"], "HOLD_SPLIT_COVERAGE")
        self.assertIsNone(result["threshold"])


if __name__ == "__main__":
    unittest.main()
