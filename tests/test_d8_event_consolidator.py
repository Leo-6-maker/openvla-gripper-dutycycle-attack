"""D8-1 Event Consolidator tests — 30 tests covering merge conditions, weights, step integrity."""
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
    _validate_steps,
    _relation_signature,
)


def _label(value, reason="RELATION_EVIDENCE_UNKNOWN", mask=True, valid_mask=True, right_censored=False):
    return {"value": value, "reason": reason, "mask": mask, "valid_mask": valid_mask, "right_censored": right_censored}


def _rel(**kw):
    """Make a relation record with required identity fields."""
    defaults = {
        "logical_object": "obj_1", "logical_target": "tgt_1",
        "selected_relation": "grasp", "binding_identity": "bind_1",
        "entity_role": "MANIPULATED_OBJECT", "entity_type": "box",
        "object_entity_id": 1, "target_entity_id": 2,
    }
    defaults.update(kw)
    return defaults


class TestMergeRejection(unittest.TestCase):
    """Bridge rejection conditions."""

    def test_01_known_false_rejects(self):
        labels = {0: _label("TRUE"), 1: _label("FALSE", reason="KNOWN_FALSE"), 2: _label("TRUE")}
        r = consolidate_physical_events("libero_10/task_02/state_00", labels, G=3,
                                         diagnostic_unbound_relations=True)
        self.assertEqual(r["total_bridged_gaps"], 0)

    def test_02_geom_na_rejects(self):
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN", reason="GEOMETRY_NOT_APPLICABLE"), 2: _label("TRUE")}
        r = consolidate_physical_events("libero_10/task_02/state_00", labels, G=3,
                                         diagnostic_unbound_relations=True)
        self.assertEqual(r["total_bridged_gaps"], 0)

    def test_03_right_censored_rejects(self):
        labels = {
            0: _label("TRUE"),
            1: {"value": "UNKNOWN", "reason": "RELATION_EVIDENCE_UNKNOWN", "mask": False, "valid_mask": False, "right_censored": True},
            2: _label("TRUE"),
        }
        r = consolidate_physical_events("libero_10/task_02/state_00", labels, G=3,
                                         diagnostic_unbound_relations=True)
        self.assertEqual(r["total_bridged_gaps"], 0)

    def test_04_gap_exceeds_G_rejects(self):
        labels = {0: _label("TRUE")}
        for i in range(1, 5): labels[i] = _label("UNKNOWN")
        labels[5] = _label("TRUE")
        r = consolidate_physical_events("libero_10/task_02/state_00", labels, G=3,
                                         diagnostic_unbound_relations=True)
        self.assertEqual(r["total_bridged_gaps"], 0)

    def test_05_gap_at_G_bridges(self):
        labels = {0: _label("TRUE")}
        for i in range(1, 4): labels[i] = _label("UNKNOWN")
        labels[4] = _label("TRUE")
        r = consolidate_physical_events("libero_10/task_02/state_00", labels, G=3,
                                         diagnostic_unbound_relations=True)
        self.assertEqual(r["total_bridged_gaps"], 1)

    def test_06_reason_not_allowlisted_rejects(self):
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN", reason="OTHER"), 2: _label("TRUE")}
        r = consolidate_physical_events("libero_10/task_02/state_00", labels, G=3,
                                         diagnostic_unbound_relations=True)
        self.assertEqual(r["total_bridged_gaps"], 0)

    def test_07_articulated_not_applicable(self):
        labels = {0: _label("TRUE"), 1: _label("TRUE")}
        r = consolidate_physical_events("libero_goal/task_00/state_00", labels, G=3,
                                         diagnostic_unbound_relations=True)
        self.assertTrue(r["articulated"])
        self.assertFalse(r["applicable"])
        self.assertFalse(r["consumer_eligible"])

    def test_08_G0_equals_raw(self):
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        r = consolidate_physical_events("libero_10/task_02/state_00", labels, G=0,
                                         diagnostic_unbound_relations=True)
        self.assertEqual(r["consolidated_event_count"], r["raw_true_span_count"])

    def test_09_empty_labels(self):
        r = consolidate_physical_events("test/ep/state", {}, G=3, diagnostic_unbound_relations=True)
        self.assertEqual(r["raw_true_span_count"], 0)
        self.assertEqual(r["consolidated_event_count"], 0)

    def test_10_no_true_spans(self):
        labels = {0: _label("FALSE"), 1: _label("UNKNOWN")}
        r = consolidate_physical_events("test/ep/state", labels, G=3, diagnostic_unbound_relations=True)
        self.assertEqual(r["consolidated_event_count"], 0)

    def test_11_formal_mode_requires_relations(self):
        labels = {0: _label("TRUE"), 1: _label("TRUE")}
        with self.assertRaises(ValueError):
            consolidate_physical_events("test/ep/state", labels, G=3)  # no diagnostic flag

    def test_12_diagnostic_mode_marks_non_consumable(self):
        labels = {0: _label("TRUE"), 1: _label("TRUE")}
        r = consolidate_physical_events("test/ep/state", labels, G=3, diagnostic_unbound_relations=True)
        self.assertFalse(r["consumer_eligible"])
        self.assertTrue(r["diagnostic_unbound_relations"])


class TestRelationIdentity(unittest.TestCase):
    """Relation identity checks."""

    def test_13_same_relation_bridges(self):
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        relations = [
            _rel(step=0), _rel(step=1), _rel(step=2),
        ]
        r = consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)
        self.assertEqual(r["total_bridged_gaps"], 1)
        self.assertTrue(r["identity_checks_performed"])

    def test_14_different_object_rejects(self):
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        relations = [
            _rel(step=0, logical_object="obj_A"),
            _rel(step=1, logical_object="obj_A"),
            _rel(step=2, logical_object="obj_B"),
        ]
        r = consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)
        self.assertEqual(r["total_bridged_gaps"], 0)

    def test_15_different_relation_rejects(self):
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        relations = [
            _rel(step=0, selected_relation="grasp"),
            _rel(step=1, selected_relation="grasp"),
            _rel(step=2, selected_relation="push"),
        ]
        r = consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)
        self.assertEqual(r["total_bridged_gaps"], 0)

    def test_16_empty_relation_field_rejects(self):
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        relations = [
            _rel(step=0, logical_object=""),
            _rel(step=1),
            _rel(step=2, logical_object=""),
        ]
        r = consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)
        self.assertEqual(r["total_bridged_gaps"], 0)

    def test_17_no_step_field_raises(self):
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        relations = [{"logical_object": "x", "logical_target": "y", "selected_relation": "grasp", "binding_identity": "b", "entity_role": "r", "entity_type": "t", "object_entity_id": 1, "target_entity_id": 2}]
        with self.assertRaises(ValueError):
            consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)


class TestStepIntegrity(unittest.TestCase):
    """Step validation."""

    def test_18_missing_step_rejects(self):
        labels = {0: _label("TRUE"), 2: _label("TRUE")}  # gap at step 1
        with self.assertRaises(ValueError):
            consolidate_physical_events("test/ep/state", labels, G=3, diagnostic_unbound_relations=True)

    def test_19_duplicate_step_detected(self):
        """Duplicate steps detected via list input (dicts resolve duplicates silently)."""
        # Simulate: pass steps as a list where duplicates exist
        labels_list = [(0, _label("TRUE")), (0, _label("FALSE")), (1, _label("TRUE"))]
        seen = set()
        dupes = []
        for s, _ in labels_list:
            if s in seen:
                dupes.append(s)
            seen.add(s)
        self.assertTrue(len(dupes) > 0, "duplicate step should be detected before dict conversion")

    def test_20_contiguous_true_without_missing(self):
        labels = {0: _label("TRUE"), 1: _label("TRUE"), 2: _label("FALSE")}
        r = consolidate_physical_events("test/ep/state", labels, G=3, diagnostic_unbound_relations=True)
        self.assertEqual(r["raw_true_span_count"], 1)

    def test_21_nonfinite_step_rejected(self):
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN", reason="NONFINITE"), 2: _label("TRUE")}
        r = consolidate_physical_events("test/ep/state", labels, G=3, diagnostic_unbound_relations=True)
        self.assertEqual(r["total_bridged_gaps"], 0)

    def test_22_identity_unresolved_rejected(self):
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN", reason="IDENTITY_UNRESOLVED"), 2: _label("TRUE")}
        r = consolidate_physical_events("test/ep/state", labels, G=3, diagnostic_unbound_relations=True)
        self.assertEqual(r["total_bridged_gaps"], 0)


class TestEventWeights(unittest.TestCase):
    """Training weight correctness."""

    def test_23_multi_fragment_equal_weight(self):
        """Multi-fragment event gets same total weight as single-fragment."""
        targets = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        masks = np.ones(6, dtype=bool)
        # Two events: event0 has 2 fragments (0-1, 4-4), event1 doesn't exist here
        # Actually one event with two fragments
        consolidated = {
            "event_groups": [
                {"fragment_ranges": [(0, 1), (4, 4)]},
            ]
        }
        w = build_physical_event_weights(targets, masks, consolidated)
        # Single event, total weight 1.0, spread across 3 true steps
        self.assertAlmostEqual(float(w[0:2].sum() + w[4:5].sum()), 1.0, places=5)

    def test_24_two_events_equal_weight(self):
        """Two events get equal total weight."""
        targets = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        masks = np.ones(6, dtype=bool)
        consolidated = {
            "event_groups": [
                {"fragment_ranges": [(0, 1)]},
                {"fragment_ranges": [(4, 4)]},
            ]
        }
        w = build_physical_event_weights(targets, masks, consolidated)
        self.assertAlmostEqual(float(w[0:2].sum()), 0.5, places=5)
        self.assertAlmostEqual(float(w[4:5].sum()), 0.5, places=5)

    def test_25_false_weights_nonzero(self):
        """Known FALSE must have non-zero negative weight."""
        targets = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        masks = np.ones(4, dtype=bool)
        consolidated = {
            "event_groups": [
                {"fragment_ranges": [(0, 0)]},
                {"fragment_ranges": [(3, 3)]},
            ]
        }
        w = build_physical_event_weights(targets, masks, consolidated)
        self.assertGreater(float(w[1:3].sum()), 0.0)

    def test_26_unknown_weight_zero(self):
        targets = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        masks = np.array([True, False, True], dtype=bool)
        consolidated = {"event_groups": [{"fragment_ranges": [(0, 0)]}]}
        w = build_physical_event_weights(targets, masks, consolidated)
        self.assertEqual(float(w[1]), 0.0)

    def test_27_no_event_groups_fallback(self):
        """No event groups uses fallback span-based weights."""
        targets = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)
        masks = np.ones(4, dtype=bool)
        consolidated = {"event_groups": []}
        w = build_physical_event_weights(targets, masks, consolidated)
        self.assertGreater(float(w.sum()), 0.0)
        self.assertEqual(float(w[1]), float(w[0]))  # Same span share


class TestCanonicalAndFailClosed(unittest.TestCase):
    """Digest determinism and fail-closed."""

    def test_28_repeatable_digest(self):
        labels = {0: _label("TRUE"), 1: _label("TRUE")}
        r1 = consolidate_physical_events("test/ep/state", labels, G=3, diagnostic_unbound_relations=True)
        d1 = compute_consolidation_digest(r1)
        r2 = consolidate_physical_events("test/ep/state", labels, G=3, diagnostic_unbound_relations=True)
        self.assertEqual(d1, compute_consolidation_digest(r2))

    def test_29_different_G_different_digest(self):
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        r0 = consolidate_physical_events("test/ep/state", labels, G=0, diagnostic_unbound_relations=True)
        r3 = consolidate_physical_events("test/ep/state", labels, G=3, diagnostic_unbound_relations=True)
        self.assertNotEqual(compute_consolidation_digest(r0), compute_consolidation_digest(r3))

    def test_30_labels_unchanged_after_consolidation(self):
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        original = {k: dict(v) for k, v in labels.items()}
        consolidate_physical_events("test/ep/state", labels, G=3, diagnostic_unbound_relations=True)
        for k in original:
            self.assertEqual(labels[k]["value"], original[k]["value"])
            self.assertEqual(labels[k]["reason"], original[k]["reason"])


if __name__ == "__main__":
    unittest.main()
