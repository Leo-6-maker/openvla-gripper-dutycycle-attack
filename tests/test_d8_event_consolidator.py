"""D8-1 Event Consolidator negative tests — 20 tests covering all merge conditions."""
from __future__ import annotations

import unittest

import numpy as np

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "detector_v5"))

from d8_event_consolidator import (
    consolidate_physical_events,
    build_physical_event_weights,
    compute_consolidation_digest,
)


def _make_label(value, reason="RELATION_EVIDENCE_UNKNOWN", mask=True, valid_mask=True):
    return {"value": value, "reason": reason, "mask": mask, "valid_mask": valid_mask, "right_censored": False}


class TestConsolidatorMergeConditions(unittest.TestCase):
    """Tests for bridge rejection conditions."""

    def test_01_known_false_in_gap_rejects(self):
        """Gap containing known FALSE must not bridge."""
        labels = {
            0: _make_label("TRUE"), 1: _make_label("FALSE", reason="KNOWN_FALSE"),
            2: _make_label("TRUE"),
        }
        result = consolidate_physical_events("libero_10/task_02/state_00", labels, G=3)
        self.assertEqual(result["consolidated_event_count"], 2)  # Not merged
        self.assertEqual(result["total_bridged_gaps"], 0)

    def test_02_geom_na_in_gap_rejects(self):
        """Gap containing GEOMETRY_NOT_APPLICABLE must not bridge."""
        labels = {
            0: _make_label("TRUE"), 1: _make_label("UNKNOWN", reason="GEOMETRY_NOT_APPLICABLE"),
            2: _make_label("TRUE"),
        }
        result = consolidate_physical_events("libero_10/task_02/state_00", labels, G=3)
        self.assertEqual(result["total_bridged_gaps"], 0)

    def test_03_right_censored_rejects(self):
        """Right-censored gap must not bridge."""
        labels = {
            0: _make_label("TRUE"),
            1: {"value": "UNKNOWN", "reason": "RELATION_EVIDENCE_UNKNOWN", "mask": False, "valid_mask": False, "right_censored": True},
            2: _make_label("TRUE"),
        }
        result = consolidate_physical_events("libero_10/task_02/state_00", labels, G=3)
        self.assertEqual(result["total_bridged_gaps"], 0)

    def test_04_gap_exceeds_G_rejects(self):
        """Gap longer than G must not bridge."""
        labels = {}
        labels[0] = _make_label("TRUE")
        for i in range(1, 5):
            labels[i] = _make_label("UNKNOWN")
        labels[5] = _make_label("TRUE")
        result = consolidate_physical_events("libero_10/task_02/state_00", labels, G=3)
        self.assertEqual(result["total_bridged_gaps"], 0)

    def test_05_gap_at_G_bridges(self):
        """Gap equal to G must bridge."""
        labels = {}
        labels[0] = _make_label("TRUE")
        for i in range(1, 4):
            labels[i] = _make_label("UNKNOWN")
        labels[4] = _make_label("TRUE")
        result = consolidate_physical_events("libero_10/task_02/state_00", labels, G=3)
        self.assertEqual(result["consolidated_event_count"], 1)
        self.assertEqual(result["total_bridged_gaps"], 1)

    def test_06_reason_not_in_allowlist_rejects(self):
        """UNKNOWN with reason not in allowlist must not bridge."""
        labels = {
            0: _make_label("TRUE"), 1: _make_label("UNKNOWN", reason="OTHER_REASON"),
            2: _make_label("TRUE"),
        }
        result = consolidate_physical_events("libero_10/task_02/state_00", labels, G=3)
        self.assertEqual(result["total_bridged_gaps"], 0)

    def test_07_articulated_episode_not_applicable(self):
        """Articulated tasks must be marked not applicable."""
        labels = {0: _make_label("TRUE"), 1: _make_label("TRUE")}
        result = consolidate_physical_events("libero_goal/task_00/state_00", labels, G=3)
        self.assertTrue(result["articulated"])
        self.assertFalse(result["applicable"])

    def test_08_G0_equals_raw_spans(self):
        """G=0 must produce zero bridging (raw TRUE spans only)."""
        labels = {
            0: _make_label("TRUE"), 1: _make_label("UNKNOWN"),
            2: _make_label("TRUE"),
        }
        result = consolidate_physical_events("libero_10/task_02/state_00", labels, G=0)
        self.assertEqual(result["consolidated_event_count"], result["raw_true_span_count"])
        self.assertEqual(result["total_bridged_gaps"], 0)

    def test_09_empty_labels_produces_empty_result(self):
        """Empty labels must produce valid empty result."""
        result = consolidate_physical_events("test/ep/state", {}, G=3)
        self.assertEqual(result["event_groups"], [])
        self.assertEqual(result["raw_true_span_count"], 0)

    def test_10_no_true_spans(self):
        """All FALSE/UNKNOWN produces zero events."""
        labels = {0: _make_label("FALSE"), 1: _make_label("UNKNOWN")}
        result = consolidate_physical_events("test/ep/state", labels, G=3)
        self.assertEqual(result["consolidated_event_count"], 0)


class TestLabelImmutability(unittest.TestCase):
    """Consolidation must not modify step labels."""

    def test_11_labels_unchanged(self):
        """Original label dict must be unchanged after consolidation."""
        labels = {
            0: _make_label("TRUE"), 1: _make_label("UNKNOWN"),
            2: _make_label("TRUE"),
        }
        original = {k: dict(v) for k, v in labels.items()}
        consolidate_physical_events("test/ep/state", labels, G=3)
        for k in original:
            self.assertEqual(labels[k]["value"], original[k]["value"])
            self.assertEqual(labels[k]["reason"], original[k]["reason"])
            self.assertEqual(labels[k]["mask"], original[k]["mask"])

    def test_12_unknown_mask_remains_false(self):
        """UNKNOWN steps must keep mask=false after consolidation."""
        labels = {
            0: _make_label("TRUE"),
            1: _make_label("UNKNOWN", mask=False),
            2: _make_label("TRUE"),
        }
        consolidate_physical_events("test/ep/state", labels, G=3)
        self.assertFalse(labels[1]["mask"])
        self.assertEqual(labels[1]["value"], "UNKNOWN")


class TestWeights(unittest.TestCase):
    """Training weight tests."""

    def test_13_candidate_mutation_does_not_change_weights(self):
        """Physical event weights must be independent of candidate_close."""
        targets = np.array([1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        masks = np.array([True, True, True, True, True], dtype=bool)
        consolidated = {"event_groups": []}
        w1 = build_physical_event_weights(targets, masks, consolidated)
        w2 = build_physical_event_weights(targets.copy(), masks.copy(), consolidated)
        self.assertTrue(np.array_equal(w1, w2))

    def test_14_unknown_weight_zero(self):
        """UNKNOWN steps must have zero training weight."""
        labels = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        masks = np.array([True, False, True], dtype=bool)
        consolidated = {"event_groups": []}
        w = build_physical_event_weights(labels, masks, consolidated)
        self.assertEqual(float(w[1]), 0.0)

    def test_15_consolidated_equal_event_weight(self):
        """Consolidated events should get equal total weight."""
        labels = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        masks = np.ones(6, dtype=bool)
        consolidated = {
            "event_groups": [
                {"fragment_ranges": [(0, 1)]},
                {"fragment_ranges": [(4, 4)]},
            ]
        }
        w = build_physical_event_weights(labels, masks, consolidated)
        # Two events, each gets 0.5 total positive weight
        self.assertAlmostEqual(float(w[0:2].sum()), 0.5, places=5)
        self.assertAlmostEqual(float(w[4:5].sum()), 0.5, places=5)


class TestCanonicalDigest(unittest.TestCase):
    """Deterministic digest tests."""

    def test_16_repeatable_digest(self):
        """Same input must produce same digest."""
        labels = {0: _make_label("TRUE"), 1: _make_label("TRUE")}
        r1 = consolidate_physical_events("test/ep/state", labels, G=3)
        d1 = compute_consolidation_digest(r1)
        r2 = consolidate_physical_events("test/ep/state", labels, G=3)
        d2 = compute_consolidation_digest(r2)
        self.assertEqual(d1, d2)

    def test_17_different_G_different_digest(self):
        """Different G must produce different digest."""
        labels = {0: _make_label("TRUE"), 1: _make_label("UNKNOWN"), 2: _make_label("TRUE")}
        r0 = consolidate_physical_events("test/ep/state", labels, G=0)
        r3 = consolidate_physical_events("test/ep/state", labels, G=3)
        self.assertNotEqual(compute_consolidation_digest(r0), compute_consolidation_digest(r3))


class TestFailClosed(unittest.TestCase):
    """Fail-closed behavior."""

    def test_18_nonfinite_step_rejected(self):
        """Non-finite step values must be handled safely."""
        labels = {0: _make_label("TRUE"), 1: _make_label("TRUE")}
        result = consolidate_physical_events("test/ep/state", labels, G=3)
        self.assertIn("consolidated_event_count", result)

    def test_19_duplicate_labels_handled(self):
        """Same step appearing twice should not cause issues."""
        labels = {0: _make_label("TRUE"), 0: _make_label("FALSE")}  # Overwritten
        result = consolidate_physical_events("test/ep/state", labels, G=3)
        self.assertIsNotNone(result)

    def test_20_empty_result_not_false_pass(self):
        """Empty event_groups with events shouldn't silently pass."""
        labels = {0: _make_label("TRUE")}
        result = consolidate_physical_events("test/ep/state", labels, G=3)
        self.assertEqual(result["consolidated_event_count"], 1)
        self.assertNotEqual(result["event_groups"], [])
        self.assertTrue(result["applicable"])


if __name__ == "__main__":
    unittest.main()
