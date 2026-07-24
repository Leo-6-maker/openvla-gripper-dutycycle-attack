import unittest

from tools.multisuite_detector import audit_c2g_r8w_full_clean_2000 as audit
from tools.multisuite_detector.build_c2g_r8w_full_clean_2000_plan import expected_flags


class FullCleanAuditTests(unittest.TestCase):
    def test_feature_action_and_persistence_contracts(self):
        self.assertTrue(audit.finite_vector([0.0] * 25, 25))
        self.assertFalse(audit.finite_vector([0.0] * 24, 25))
        self.assertTrue(audit.triggerable([True, False, True]))
        self.assertFalse(audit.triggerable([True, False, False]))

    def test_post_step_schema_requires_canonical_fields(self):
        row = {key: None for key in audit.STEP_REQUIRED_KEYS}
        row.update({
            "reward_after_step": 0.0,
            "done_after_step": False,
            "env_check_success_after_step": False,
        })
        self.assertTrue(audit.post_step_complete(row))
        del row["env_check_success_after_step"]
        self.assertFalse(audit.post_step_complete(row))

    def test_metadata_provenance_and_cohort_gates(self):
        expected = {
            "suite": "libero_object",
            "task_index": 1,
            "state_id": 2,
            "parent_key": "parent",
            "cohort": "DETECTOR_VAL",
            "split": "val",
            "shard_manifest_sha256": "b" * 64,
        }
        metadata = {key: "bound" for key in audit.META_REQUIRED_KEYS}
        metadata.update(expected)
        metadata.update(expected_flags("DETECTOR_VAL"))
        metadata.update({
            "runtime_valid": True,
            "condition": "CLEAN",
            "clean_success_observed": False,
            "post_step_outcome_complete": True,
            "post_step_outcome_schema_version": audit.POST_STEP_SCHEMA,
            "git_commit": "a" * 40,
            "git_clean": True,
        })
        ok, missing = audit.metadata_complete(metadata, expected, "a" * 40)
        self.assertTrue(ok, missing)
        metadata["eligible_for_detector_fit"] = True
        ok, missing = audit.metadata_complete(metadata, expected, "a" * 40)
        self.assertFalse(ok)
        self.assertIn("eligibility.eligible_for_detector_fit", missing)

    def test_fail_closed_status_precedence(self):
        base = dict(
            cardinality_pass=True,
            identity_failure_count=0,
            teacher_failure_count=0,
            l10_closure=True,
            worker_failure_count=0,
            complete_count=2000,
        )
        self.assertEqual(audit.classify_final_status(**base), audit.PASS_STATUS)
        self.assertEqual(
            audit.classify_final_status(**{**base, "identity_failure_count": 1}),
            audit.HOLD_IDENTITY,
        )
        self.assertEqual(
            audit.classify_final_status(**{**base, "teacher_failure_count": 1}),
            audit.HOLD_TEACHER,
        )
        self.assertEqual(
            audit.classify_final_status(**{**base, "worker_failure_count": 1}),
            audit.HOLD_COLLECTION,
        )

    def test_public_ledger_excludes_outcomes(self):
        self.assertNotIn("clean_success_observed", audit.PUBLIC_LEDGER_FIELDS)
        self.assertNotIn("reward_sum", audit.PUBLIC_LEDGER_FIELDS)

    def test_canary_prefix_comparison(self):
        self.assertTrue(audit.vectors_exact([0.0] * 7, [0.0] * 7, 7))
        self.assertFalse(audit.vectors_exact([0.0] * 7, [1e-8] + [0.0] * 6, 7))
        self.assertTrue(audit.vectors_equivalent([0.0] * 25, [1e-8] + [0.0] * 24, 25))


if __name__ == "__main__":
    unittest.main()
