import tempfile
import unittest
from pathlib import Path

from tools.multisuite_detector.rebuild_c2g_r8z_teacher_v2_labels import (
    assert_sha256,
    derive_official_prefix,
    require_new_output_root,
    sha256_file,
    validate_manifest_rows,
)


def steps(count, success_step=None):
    return [
        {
            "step": step,
            "env_check_success_after_step": step == success_step,
            "done_after_step": step == success_step,
            "reward_after_step": 1.0 if step == success_step else 0.0,
        }
        for step in range(count)
    ]


def full_manifest():
    rows = []
    cohorts = (
        ("DETECTOR_TRAIN", "train", 300),
        ("DETECTOR_VAL", "val", 50),
        ("DETECTOR_TEST_WITHIN_TASK", "test", 50),
        ("ATTACK_EVAL_PREREGISTERED", "attack_eval", 100),
    )
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        index = 0
        for cohort, split, count in cohorts:
            for _ in range(count):
                rows.append(
                    {
                        "suite": suite,
                        "task_index": index % 10,
                        "state_id": index,
                        "parent_key": f"{suite}/parent_{index:03d}",
                        "cohort": cohort,
                        "split": split,
                        "max_steps": 300,
                    }
                )
                index += 1
    return rows


class OfficialPrefixTests(unittest.TestCase):
    def test_spatial_keeps_219_and_drops_220(self):
        result = derive_official_prefix(steps(300), official_horizon=220)
        self.assertEqual(len(result.rows), 220)
        self.assertEqual(result.rows[-1]["step"], 219)
        self.assertFalse(result.canonical_success)
        self.assertEqual(result.termination_reason, "MAX_POLICY_STEPS_AT_220")

    def test_object_keeps_279_and_drops_280(self):
        result = derive_official_prefix(steps(300), official_horizon=280)
        self.assertEqual((len(result.rows), result.rows[-1]["step"]), (280, 279))

    def test_goal_keeps_0_through_299(self):
        result = derive_official_prefix(steps(300), official_horizon=300)
        self.assertEqual((len(result.rows), result.rows[-1]["step"]), (300, 299))

    def test_early_success_is_retained(self):
        result = derive_official_prefix(steps(151, success_step=150), official_horizon=220)
        self.assertTrue(result.canonical_success)
        self.assertEqual(result.first_success_step, 150)
        self.assertEqual(len(result.rows), 151)

    def test_late_success_becomes_official_timeout(self):
        result = derive_official_prefix(steps(231, success_step=230), official_horizon=220)
        self.assertFalse(result.canonical_success)
        self.assertTrue(result.late_success_in_extended_source)
        self.assertEqual(result.termination_reason, "MAX_POLICY_STEPS_AT_220")

    def test_source_final_success_does_not_contaminate_prefix(self):
        result = derive_official_prefix(steps(300, success_step=299), official_horizon=280)
        self.assertFalse(result.canonical_success)
        self.assertTrue(result.late_success_in_extended_source)

    def test_incomplete_failure_prefix_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "ended before"):
            derive_official_prefix(steps(219), official_horizon=220)


class SourceAndOutputGateTests(unittest.TestCase):
    def test_duplicate_source_identity_fails_closed(self):
        rows = full_manifest()
        rows[-1] = dict(rows[-2])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_manifest_rows(rows)

    def test_source_sha_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            path.write_text("{}\n", encoding="utf-8")
            self.assertEqual(assert_sha256(path, sha256_file(path), "source"), sha256_file(path))
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                assert_sha256(path, "0" * 64, "source")

    def test_existing_output_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                require_new_output_root(Path(tmp))


if __name__ == "__main__":
    unittest.main()

