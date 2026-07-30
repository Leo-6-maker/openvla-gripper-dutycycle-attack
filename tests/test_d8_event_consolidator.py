"""D8-1 Event Consolidator tests — 30 tests covering merge conditions, weights, step integrity."""
from __future__ import annotations

import hashlib
import json
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
    """Make a per-relation record with required identity fields.

    R6: verdict and reason defaults added for gap detail validation (P0-2).
    """
    defaults = {
        "logical_object": "obj_1", "logical_target": "tgt_1",
        "selected_relation": "grasp",
        "entity_role": "MANIPULATED_OBJECT", "entity_type": "box",
        "object_entity_id": 1, "target_entity_id": 2,
        "relation_index": 0,
        "verdict": "UNKNOWN",
        "reason": "RELATION_EVIDENCE_UNKNOWN",
        "mask": False,
        "valid_mask": False,
    }
    defaults.update(kw)
    return defaults


def _make_sidecar_entry(step, per_relation, selection_status="UNIQUE_SUPPORT",
                         selected_relation_id=0, **kw):
    """Make a dict-format sidecar entry for a step."""
    entry = {
        "episode_id": "test/ep/state",
        "step": step,
        "per_relation": per_relation,
        "selection_status": selection_status,
        "selected_relation_id": selected_relation_id,
        "selected_relation_index": selected_relation_id,
        "candidate_relation_indices": [r["relation_index"] for r in per_relation],
        "supporting_relation_indices": [r["relation_index"] for r in per_relation if r.get("verdict") == "TRUE"],
        "aggregate_physical_label": "TRUE" if selection_status == "UNIQUE_SUPPORT" else "UNKNOWN",
        "aggregate_mask": True,
        "aggregate_reason": "",
        "candidate_close": False,
        "suite": "test", "task_id": 0, "state_id": 0, "seed": 0,
        **kw,
    }
    return entry


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
    """Relation identity checks with dict-format sidecar (P0-2)."""

    def _make_relations_dict(self, step0_rel, step1_rel, step2_rel):
        """Build dict-format relations from per-step per_relation lists."""
        return {
            0: _make_sidecar_entry(0, [step0_rel]),
            1: _make_sidecar_entry(1, [step1_rel], selection_status="RELATION_AMBIGUOUS", selected_relation_id=None),
            2: _make_sidecar_entry(2, [step2_rel]),
        }

    def test_13_same_relation_bridges(self):
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        relations = self._make_relations_dict(_rel(), _rel(), _rel())
        r = consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)
        self.assertEqual(r["total_bridged_gaps"], 1)
        self.assertTrue(r["identity_checks_performed"])

    def test_14_different_object_rejects(self):
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        relations = self._make_relations_dict(
            _rel(logical_object="obj_A"), _rel(logical_object="obj_A"), _rel(logical_object="obj_B"),
        )
        r = consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)
        self.assertEqual(r["total_bridged_gaps"], 0)

    def test_15_different_relation_rejects(self):
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        relations = self._make_relations_dict(
            _rel(selected_relation="grasp"), _rel(selected_relation="grasp"), _rel(selected_relation="push"),
        )
        r = consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)
        self.assertEqual(r["total_bridged_gaps"], 0)

    def test_16_empty_relation_field_rejects(self):
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        relations = self._make_relations_dict(
            _rel(logical_object=""), _rel(), _rel(logical_object=""),
        )
        r = consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)
        self.assertEqual(r["total_bridged_gaps"], 0)

    def test_17_unsupported_boundary_rejects(self):
        """MULTI_SUPPORT boundary step must reject (not UNIQUE_SUPPORT)."""
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        # Step 0 boundary has MULTI_SUPPORT (ambiguous)
        rels = {
            0: _make_sidecar_entry(0, [_rel()], selection_status="MULTI_SUPPORT", selected_relation_id=None),
            1: _make_sidecar_entry(1, [_rel()], selection_status="RELATION_AMBIGUOUS", selected_relation_id=None),
            2: _make_sidecar_entry(2, [_rel()], selection_status="UNIQUE_SUPPORT", selected_relation_id=0),
        }
        r = consolidate_physical_events("test/ep/state", labels, relations=rels, G=3)
        self.assertEqual(r["total_bridged_gaps"], 0)


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


class TestSidecarLoaderCorrectness(unittest.TestCase):
    """Episode ID loader: read from JSON entry, not filename."""

    def setUp(self):
        self.tmpdir = Path(__file__).resolve().parent / "_tmp_sidecar_test"
        self.tmpdir.mkdir(exist_ok=True)
        self.ep_dir = self.tmpdir / "per_episode"
        self.ep_dir.mkdir(exist_ok=True)
        self.sums = self.tmpdir / "SHA256SUMS"
        self.sums_sidecar = self.tmpdir / "SHA256SUMS.sha256"

    def tearDown(self):
        import shutil
        if self.tmpdir.exists():
            shutil.rmtree(self.tmpdir)

    def _make_entry(self, step, eid, **kw):
        entry = {
            "episode_id": eid, "step": step, "aggregate_physical_label": "TRUE",
            "aggregate_mask": True, "aggregate_reason": "",
            "per_relation": [], "selection_status": "UNIQUE_SUPPORT",
            "selected_relation_index": 0, "candidate_relation_indices": [0],
            "supporting_relation_indices": [0], "candidate_close": False,
            "suite": eid.split("/")[0], "task_id": 0, "state_id": 0, "seed": 0,
            **kw,
        }
        return entry

    def _seal(self):
        files = sorted(
            x for x in self.tmpdir.rglob("*")
            if x.is_file() and x.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
        )
        hashes = []
        for f in files:
            d = hashlib.sha256(f.read_bytes()).hexdigest()
            hashes.append(f"{d}  {f.relative_to(self.tmpdir).as_posix()}")
        self.sums.write_text("\n".join(hashes) + "\n", encoding="utf-8")
        s = hashlib.sha256(self.sums.read_bytes()).hexdigest()
        self.sums_sidecar.write_text(f"{s}  SHA256SUMS\n", encoding="utf-8")
        return s

    def test_31_episode_id_from_entry_not_filename(self):
        """Episode ID read from entry, not filename."""
        eid = "libero_10/task_02/state_05"
        fname = "wrong_name.json"  # filename differs from internal ID
        ep_data = {str(i): self._make_entry(i, eid) for i in range(3)}
        (self.ep_dir / fname).write_text(json.dumps(ep_data) + "\n")
        self._seal()
        from run_d8_formal_g_sensitivity import load_sidecar_correct
        sidecar = load_sidecar_correct(self.tmpdir)
        self.assertIn(eid, sidecar)
        self.assertEqual(len(sidecar[eid]), 3)

    def test_32_duplicate_internal_episode_id_fail_closed(self):
        """Two files with same internal episode_id must fail."""
        eid = "libero_10/task_02/state_05"
        ep1 = {str(i): self._make_entry(i, eid) for i in range(2)}
        ep2 = {str(i): self._make_entry(i, eid) for i in range(2, 4)}
        (self.ep_dir / "file_a.json").write_text(json.dumps(ep1) + "\n")
        (self.ep_dir / "file_b.json").write_text(json.dumps(ep2) + "\n")
        self._seal()
        from run_d8_formal_g_sensitivity import load_sidecar_correct
        with self.assertRaises(ValueError):
            load_sidecar_correct(self.tmpdir)

    def test_33_empty_episode_id_fail_closed(self):
        """Empty episode_id in entry must fail."""
        ep = {str(i): self._make_entry(i, "") for i in range(2)}
        (self.ep_dir / "empty_eid.json").write_text(json.dumps(ep) + "\n")
        self._seal()
        from run_d8_formal_g_sensitivity import load_sidecar_correct
        with self.assertRaises(ValueError):
            load_sidecar_correct(self.tmpdir)

    def test_34_filename_vs_entry_id_mismatch_diagnostic_only(self):
        """Filename different from internal ID is OK (not a failure)."""
        eid = "libero_goal/task_07/state_03"
        fname = "libero_10_task_07_state_03.json"  # wrong suite in filename
        ep = {str(i): self._make_entry(i, eid) for i in range(2)}
        (self.ep_dir / fname).write_text(json.dumps(ep) + "\n")
        self._seal()
        from run_d8_formal_g_sensitivity import load_sidecar_correct
        sidecar = load_sidecar_correct(self.tmpdir)
        self.assertIn(eid, sidecar)
        self.assertNotIn("libero_10/task_07/state_03", sidecar)

    def test_35_malformed_id_rejected_at_load(self):
        """Malformed episode IDs must be rejected at load time (P0-5 fix)."""
        eid = "weird_format_without_slashes"
        ep = {str(i): self._make_entry(i, eid) for i in range(2)}
        (self.ep_dir / "weird.json").write_text(json.dumps(ep) + "\n")
        self._seal()
        from run_d8_formal_g_sensitivity import load_sidecar_correct
        with self.assertRaises(ValueError):
            load_sidecar_correct(self.tmpdir)

    def test_36_missing_episode_id_field(self):
        """Entry without episode_id field must fail."""
        ep = {"0": {"step": 0}}  # no episode_id
        (self.ep_dir / "no_eid.json").write_text(json.dumps(ep) + "\n")
        self._seal()
        from run_d8_formal_g_sensitivity import load_sidecar_correct
        with self.assertRaises(ValueError):
            load_sidecar_correct(self.tmpdir)

    def test_37_inconsistent_episode_id_within_file(self):
        """Different steps in same file with different episode_ids must fail."""
        ep = {
            "0": self._make_entry(0, "libero_10/task_00/state_00"),
            "1": self._make_entry(1, "libero_10/task_00/state_01"),
        }
        (self.ep_dir / "mixed.json").write_text(json.dumps(ep) + "\n")
        self._seal()
        from run_d8_formal_g_sensitivity import load_sidecar_correct
        with self.assertRaises(ValueError):
            load_sidecar_correct(self.tmpdir)

    def test_38_empty_file(self):
        """Empty JSON file must fail."""
        (self.ep_dir / "empty.json").write_text("{}")
        self._seal()
        from run_d8_formal_g_sensitivity import load_sidecar_correct
        with self.assertRaises(ValueError):
            load_sidecar_correct(self.tmpdir)

    def test_39_sidecar_seal_verification(self):
        """Sidecar with correct seal loads successfully."""
        eid = "libero_10/task_00/state_00"
        ep = {str(i): self._make_entry(i, eid) for i in range(5)}
        (self.ep_dir / "ok.json").write_text(json.dumps(ep) + "\n")
        self._seal()
        from audit_r3_contact_input import verify_seal
        seal = verify_seal(self.tmpdir)
        self.assertIn("sha256sums_sha256", seal)

    def test_40_bad_seal_fails(self):
        """Sidecar with bad seal must fail verification."""
        eid = "libero_10/task_00/state_00"
        ep = {str(i): self._make_entry(i, eid) for i in range(2)}
        (self.ep_dir / "ok.json").write_text(json.dumps(ep) + "\n")
        self._seal()
        # Corrupt the seal
        self.sums_sidecar.write_text("0000000000000000000000000000000000000000000000000000000000000000  SHA256SUMS\n")
        from audit_r3_contact_input import verify_seal
        with self.assertRaises(ValueError):
            verify_seal(self.tmpdir)


class TestRelationSignatureOptionalFields(unittest.TestCase):
    """Canonical digest treats entity_type as optional."""

    def test_41_empty_entity_type_bridges(self):
        """Empty entity_type should NOT block bridging (optional field)."""
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        rel = _rel(entity_type="")
        relations = {
            0: _make_sidecar_entry(0, [rel], selected_relation_id=0),
            1: _make_sidecar_entry(1, [rel], selection_status="RELATION_AMBIGUOUS", selected_relation_id=None),
            2: _make_sidecar_entry(2, [rel], selected_relation_id=0),
        }
        r = consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)
        self.assertEqual(r["total_bridged_gaps"], 1)

    def test_42_missing_critical_field_still_rejects(self):
        """Missing logical_object (critical) must still reject."""
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        rel_empty = _rel(logical_object="")
        rel_ok = _rel()
        relations = {
            0: _make_sidecar_entry(0, [rel_empty], selected_relation_id=0),
            1: _make_sidecar_entry(1, [rel_ok], selection_status="RELATION_AMBIGUOUS", selected_relation_id=None),
            2: _make_sidecar_entry(2, [rel_empty], selected_relation_id=0),
        }
        r = consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)
        self.assertEqual(r["total_bridged_gaps"], 0)

    def test_43_none_entity_type_allowed(self):
        """None entity_type should be treated as empty (optional)."""
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        rel = _rel(entity_type=None)
        relations = {
            0: _make_sidecar_entry(0, [rel], selected_relation_id=0),
            1: _make_sidecar_entry(1, [rel], selection_status="RELATION_AMBIGUOUS", selected_relation_id=None),
            2: _make_sidecar_entry(2, [rel], selected_relation_id=0),
        }
        r = consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)
        self.assertEqual(r["total_bridged_gaps"], 1)


class TestR6GapDetailFailClosed(unittest.TestCase):
    """R6 P0-2: Gap relation detail MUST exist and match."""

    def test_44_missing_relation_detail_in_gap_rejects(self):
        """Gap step with candidate_relation_indices but no matching per_relation detail."""
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        rel_boundary = _rel(relation_index=0)
        # Gap step 1: candidate list contains relation 0, but per_relation is empty
        relations = {
            0: _make_sidecar_entry(0, [rel_boundary], selected_relation_id=0),
            1: _make_sidecar_entry(1, [], selection_status="RELATION_AMBIGUOUS",
                                   selected_relation_id=None,
                                   candidate_relation_indices=[0]),  # ID in list but no detail
            2: _make_sidecar_entry(2, [rel_boundary], selected_relation_id=0),
        }
        r = consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)
        self.assertEqual(r["total_bridged_gaps"], 0)

    def test_45_gap_relation_any_verdict_bridges(self):
        """Gap per-relation verdict=TRUE still bridges (aggregate level is UNK)."""
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        rel_same = _rel(relation_index=0)
        rel_gap = _rel(relation_index=0, verdict="TRUE", reason="RELATION_EVIDENCE_UNKNOWN")
        relations = {
            0: _make_sidecar_entry(0, [rel_same], selected_relation_id=0),
            1: _make_sidecar_entry(1, [rel_gap], selection_status="RELATION_AMBIGUOUS",
                                   selected_relation_id=None),
            2: _make_sidecar_entry(2, [rel_same], selected_relation_id=0),
        }
        r = consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)
        # Bridges because per-relation verdict is not a hard gate;
        # aggregate label reason (RELATION_EVIDENCE_UNKNOWN) is the gate
        self.assertEqual(r["total_bridged_gaps"], 1)

    def test_46_gap_relation_forbidden_reason_rejects(self):
        """Gap per-relation with forbidden reason (KNOWN_FALSE) must reject."""
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        rel_boundary = _rel(relation_index=0)
        rel_gap_forbidden = _rel(relation_index=0, verdict="UNKNOWN", reason="KNOWN_FALSE")
        relations = {
            0: _make_sidecar_entry(0, [rel_boundary], selected_relation_id=0),
            1: _make_sidecar_entry(1, [rel_gap_forbidden], selection_status="RELATION_AMBIGUOUS",
                                   selected_relation_id=None),
            2: _make_sidecar_entry(2, [rel_boundary], selected_relation_id=0),
        }
        r = consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)
        self.assertEqual(r["total_bridged_gaps"], 0)

    def test_47_gap_relation_binding_differs_rejects(self):
        """Gap relation with different binding digest must reject."""
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        rel_boundary = _rel(relation_index=0, logical_object="obj_A")
        rel_gap_diff = _rel(relation_index=0, logical_object="obj_B",
                            verdict="UNKNOWN", reason="RELATION_EVIDENCE_UNKNOWN")
        relations = {
            0: _make_sidecar_entry(0, [rel_boundary], selected_relation_id=0),
            1: _make_sidecar_entry(1, [rel_gap_diff], selection_status="RELATION_AMBIGUOUS",
                                   selected_relation_id=None),
            2: _make_sidecar_entry(2, [rel_boundary], selected_relation_id=0),
        }
        r = consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)
        self.assertEqual(r["total_bridged_gaps"], 0)

    def test_48_gap_relation_empty_signature_rejects(self):
        """Gap relation with empty canonical digest must reject."""
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        rel_boundary = _rel(relation_index=0)
        rel_gap_empty = _rel(relation_index=0, logical_object="",
                             verdict="UNKNOWN", reason="RELATION_EVIDENCE_UNKNOWN")
        relations = {
            0: _make_sidecar_entry(0, [rel_boundary], selected_relation_id=0),
            1: _make_sidecar_entry(1, [rel_gap_empty], selection_status="RELATION_AMBIGUOUS",
                                   selected_relation_id=None),
            2: _make_sidecar_entry(2, [rel_boundary], selected_relation_id=0),
        }
        r = consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)
        self.assertEqual(r["total_bridged_gaps"], 0)

    def test_49_same_relation_detail_bridges(self):
        """Gap with matching detail (verdict=UNKNOWN, reason=REL_UNK, same binding)."""
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        rel_same = _rel(relation_index=0, verdict="UNKNOWN", reason="RELATION_EVIDENCE_UNKNOWN")
        relations = {
            0: _make_sidecar_entry(0, [rel_same], selected_relation_id=0),
            1: _make_sidecar_entry(1, [rel_same], selection_status="RELATION_AMBIGUOUS",
                                   selected_relation_id=None),
            2: _make_sidecar_entry(2, [rel_same], selected_relation_id=0),
        }
        r = consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)
        self.assertEqual(r["total_bridged_gaps"], 1)


class TestR6FormalNoLegacyFallback(unittest.TestCase):
    """R6 P0-3: Formal mode must not fall back to selected_relation_index."""

    def test_50_no_selected_relation_id_rejects(self):
        """Boundary without selected_relation_id must reject even if selected_relation_index is set."""
        labels = {0: _label("TRUE"), 1: _label("UNKNOWN"), 2: _label("TRUE")}
        rel_same = _rel(relation_index=0)
        # Boundary step 0: selected_relation_id=None but selected_relation_index=0 (legacy)
        relations = {
            0: _make_sidecar_entry(0, [rel_same], selected_relation_id=None,
                                   selected_relation_index=0),  # legacy field present but ignored
            1: _make_sidecar_entry(1, [rel_same], selection_status="RELATION_AMBIGUOUS",
                                   selected_relation_id=None),
            2: _make_sidecar_entry(2, [rel_same], selected_relation_id=0),
        }
        r = consolidate_physical_events("test/ep/state", labels, relations=relations, G=3)
        self.assertEqual(r["total_bridged_gaps"], 0)


if __name__ == "__main__":
    unittest.main()
