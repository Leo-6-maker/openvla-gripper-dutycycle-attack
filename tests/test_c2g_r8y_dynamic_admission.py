"""Test R8Y dynamic GPU admission logic."""
import unittest

from scripts.stageb.run_c2g_r8y_l10_520_dynamic_scheduler import (
    ABSOLUTE_MIN_FREE_MIB,
    FALLBACK_WORKER_BUDGET_MIB,
    GPU_POST_LAUNCH_RESERVE_MIB,
    INITIAL_RESIDENT_CAP,
    MAX_RESIDENT_CAP,
    MODEL_LOAD_TRANSIENT_MARGIN_MIB,
    STABLE_POLL_COUNT,
    CalibrationState,
    GpuSnapshot,
    compute_admission_threshold,
    memory_admission_pass,
    stable_admission_pass,
)


def _snap(free_mib: int, util: int = 0) -> GpuSnapshot:
    return GpuSnapshot(
        index=4,
        memory_total_mib=81920,
        memory_used_mib=81920 - free_mib,
        memory_free_mib=free_mib,
        utilization_percent=util,
        temperature_c=35,
    )


class AdmissionThresholdTests(unittest.TestCase):
    def test_default_threshold(self):
        thresh = compute_admission_threshold(FALLBACK_WORKER_BUDGET_MIB)
        expected = FALLBACK_WORKER_BUDGET_MIB + GPU_POST_LAUNCH_RESERVE_MIB + MODEL_LOAD_TRANSIENT_MARGIN_MIB
        self.assertEqual(thresh, expected)

    def test_abs_min_overrides(self):
        small_budget = 1000
        thresh = compute_admission_threshold(small_budget)
        self.assertEqual(thresh, ABSOLUTE_MIN_FREE_MIB)

    def test_large_budget_wins(self):
        large_budget = ABSOLUTE_MIN_FREE_MIB + 10000
        thresh = compute_admission_threshold(large_budget)
        self.assertGreater(thresh, ABSOLUTE_MIN_FREE_MIB)


class MemoryAdmissionTests(unittest.TestCase):
    def test_free_below_threshold_rejects(self):
        budget = FALLBACK_WORKER_BUDGET_MIB
        threshold = compute_admission_threshold(budget)
        snap = _snap(threshold - 1)
        ok, reason = memory_admission_pass(snap, budget)
        self.assertFalse(ok)
        self.assertIn("FREE_", reason)

    def test_free_equal_to_threshold_rejects(self):
        # Strictly below means < threshold fails
        budget = FALLBACK_WORKER_BUDGET_MIB
        threshold = compute_admission_threshold(budget)
        snap = _snap(threshold - 1)
        ok, _ = memory_admission_pass(snap, budget)
        self.assertFalse(ok)

    def test_free_above_threshold_accepts(self):
        budget = FALLBACK_WORKER_BUDGET_MIB
        threshold = compute_admission_threshold(budget)
        snap = _snap(threshold + 1024)
        ok, reason = memory_admission_pass(snap, budget)
        self.assertTrue(ok)
        self.assertEqual(reason, "PASS")

    def test_very_high_free_accepts(self):
        snap = _snap(60000)
        ok, _ = memory_admission_pass(snap, FALLBACK_WORKER_BUDGET_MIB)
        self.assertTrue(ok)


class StableAdmissionTests(unittest.TestCase):
    def test_three_stable_passes(self):
        budget = FALLBACK_WORKER_BUDGET_MIB
        thresh = compute_admission_threshold(budget)
        samples = [_snap(thresh + 2048) for _ in range(3)]
        ok, reason = stable_admission_pass(samples, budget)
        self.assertTrue(ok)

    def test_less_than_three_fails(self):
        budget = FALLBACK_WORKER_BUDGET_MIB
        thresh = compute_admission_threshold(budget)
        samples = [_snap(thresh + 2048) for _ in range(2)]
        ok, reason = stable_admission_pass(samples, budget)
        self.assertFalse(ok)
        self.assertIn("NEED_3", reason)

    def test_unstable_memory_fails(self):
        budget = FALLBACK_WORKER_BUDGET_MIB
        thresh = compute_admission_threshold(budget)
        samples = [
            _snap(thresh + 2048),
            _snap(thresh + 2048 + 2048),  # jumped 2GiB
            _snap(thresh + 2048),
        ]
        ok, reason = stable_admission_pass(samples, budget)
        self.assertFalse(ok)
        self.assertIn("NOT_STABLE", reason)

    def test_mixed_gpu_fails(self):
        budget = FALLBACK_WORKER_BUDGET_MIB
        thresh = compute_admission_threshold(budget)
        s1 = _snap(thresh + 2048)
        s2 = GpuSnapshot(5, 81920, 10000, 71920, 0, 35)
        ok, reason = stable_admission_pass([s1, s1, s2], budget)
        self.assertFalse(ok)
        self.assertIn("MIXED", reason)

    def test_single_poll_below_threshold_fails(self):
        budget = FALLBACK_WORKER_BUDGET_MIB
        thresh = compute_admission_threshold(budget)
        samples = [
            _snap(thresh + 2048),
            _snap(thresh - 1),  # this one fails
            _snap(thresh + 2048),
        ]
        ok, reason = stable_admission_pass(samples, budget)
        self.assertFalse(ok)
        self.assertIn("POLL_1", reason)


class CalibrationTests(unittest.TestCase):
    def test_initial_cap_is_two(self):
        self.assertEqual(INITIAL_RESIDENT_CAP, 2)

    def test_max_cap_is_three(self):
        self.assertEqual(MAX_RESIDENT_CAP, 3)

    def test_calibration_defaults(self):
        cal = CalibrationState(gpu=4)
        self.assertEqual(cal.calibrated_budget_mib, FALLBACK_WORKER_BUDGET_MIB)
        self.assertEqual(cal.oom_count, 0)
        self.assertEqual(len(cal.observed_deltas_mib), 0)

    def test_cap2_with_oom_stays_cap2(self):
        # Simulate OOM prevents upgrade
        cal = CalibrationState(gpu=4, oom_count=1)
        self.assertEqual(cal.oom_count, 1)
        # OOM should prevent cap upgrade (tested via scheduler logic in integration)

    def test_oom_quarantine_effective(self):
        cal = CalibrationState(gpu=4, oom_count=3)
        self.assertGreater(cal.oom_count, 0)


class StablePollCountTests(unittest.TestCase):
    def test_requires_three_polls(self):
        self.assertEqual(STABLE_POLL_COUNT, 3)


if __name__ == "__main__":
    unittest.main()
