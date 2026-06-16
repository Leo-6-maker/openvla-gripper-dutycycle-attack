"""G1-R2: Non-vacuous adapter tests with correct OPEN/CLOSE convention.

Convention:
  raw > 0.5 → env = -1 → physical OPEN
  raw < 0.5 → env = +1 → physical CLOSE

Candidate triggers: raw crossing (raw>0.5→raw≤0.5), close_onset, close_streak==1.
"""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "stageb"))
from gripper_attack.d5_frozen_feature_adapter_v1 import D5FrozenFeatureAdapter


# ── Fixture helpers ──
def OPEN(step=0):
    """raw=0.9, env=-1 → physical OPEN, decoded_open=1"""
    return (step, 0.9, -1.0, 0.0, 0.0, 0.0, 0.0, 1, True, True, True, True, True)


def CLOSE(step=0, qpos=0.0):
    """raw=0.4, env=+1 → physical CLOSE, decoded_open=0"""
    return (step, 0.4, 1.0, qpos, 0.0, 0.0, 0.0, 0, True, True, True, True, True)


def CROSSING(step=0):
    """Step t after OPEN at t-1: raw crossing trigger."""
    return (step, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True, True, True)


def with_valid(args, **kw):
    s, r, e, q, x, y, z, d, rv, ev, qv, ef, gs = args
    defaults = dict(raw_valid=rv, env_valid=ev, qpos_valid=qv, eef_valid=ef, sem_valid=gs)
    defaults.update(kw)
    return (s, r, e, q, x, y, z, d,
            defaults["raw_valid"], defaults["env_valid"],
            defaults["qpos_valid"], defaults["eef_valid"],
            defaults["sem_valid"])


def do_update(adapter, args):
    return adapter.update(*args)


class TestAdapterStepSequence(unittest.TestCase):
    def setUp(self):
        self.a = D5FrozenFeatureAdapter()

    def test_duplicate_step_raises(self):
        do_update(self.a, OPEN(0))
        with self.assertRaises(ValueError):
            do_update(self.a, OPEN(0))

    def test_skipped_step_raises(self):
        do_update(self.a, OPEN(0))
        with self.assertRaises(ValueError):
            do_update(self.a, CLOSE(2))

    def test_reset_clears_state(self):
        do_update(self.a, OPEN(0))
        do_update(self.a, CLOSE(1))
        self.assertEqual(self.a.next_expected_step, 2)
        self.a.reset()
        self.assertEqual(self.a.next_expected_step, 0)


class TestAdapterValidityFlags(unittest.TestCase):
    def setUp(self):
        self.a = D5FrozenFeatureAdapter()

    def test_raw_valid_false_allows_close_onset_candidate(self):
        # raw_valid=False blocks raw_crossing but NOT close_onset
        r = do_update(self.a, with_valid(CLOSE(0), raw_valid=False))
        self.assertIsNotNone(r, "close_onset still fires with raw_valid=False")
        self.assertTrue(r["abstained"])  # too_early at step 0

    def test_env_valid_false_blocks_all(self):
        # env_valid=False → env_ok=False → no close_onset, no candidate
        r = do_update(self.a, with_valid(CLOSE(0), env_valid=False))
        self.assertIsNone(r)

    def test_eef_valid_false_allows_close_onset(self):
        # eef_valid=False blocks EEF features but NOT close_onset
        r = do_update(self.a, with_valid(CLOSE(0), eef_valid=False))
        self.assertIsNotNone(r, "close_onset fires regardless of eef_valid")

    def test_semantics_valid_false_blocks_all(self):
        r = do_update(self.a, with_valid(CLOSE(0), sem_valid=False))
        self.assertIsNone(r)

    def test_decoded_open_invalid_no_candidate(self):
        s, r, e, q, x, y, z, d, rv, ev, qv, ef, gs = CLOSE(0)
        r = do_update(self.a, (s, r, e, q, x, y, z, 2, rv, ev, qv, ef, gs))
        self.assertIsNone(r)


class TestAdapterNaN(unittest.TestCase):
    def setUp(self):
        self.a = D5FrozenFeatureAdapter()

    def test_nan_qpos_allows_close_onset(self):
        # NaN qpos → qpos_ok=False, but close_onset still fires
        s, r, e, q, x, y, z, d, rv, ev, qv, ef, gs = CLOSE(0)
        r = do_update(self.a, (s, r, e, float("nan"), x, y, z, d, rv, ev, qv, ef, gs))
        self.assertIsNotNone(r, "close_onset fires despite NaN qpos")
        self.assertTrue(r["abstained"])

    def test_nan_eef_allows_close_onset(self):
        s, r, e, q, x, y, z, d, rv, ev, qv, ef, gs = CLOSE(0)
        r = do_update(self.a, (s, r, e, q, float("nan"), y, z, d, rv, ev, qv, ef, gs))
        self.assertIsNotNone(r, "close_onset fires despite NaN EEF")


class TestAdapterAbstainReasons(unittest.TestCase):
    def setUp(self):
        self.a = D5FrozenFeatureAdapter()

    def _warmup_to_step(self, n):
        """Feed n OPEN steps to accumulate history and avoid too_early."""
        for i in range(n):
            do_update(self.a, OPEN(i))

    def test_too_early_abstain(self):
        # Step 0 crossing → candidate but too_early
        do_update(self.a, OPEN(0))
        r = do_update(self.a, CROSSING(1))
        self.assertIsNotNone(r, "expected candidate from raw crossing at step 1")
        self.assertTrue(r["abstained"])
        self.assertIn("too_early", r["abstain"])

    def test_gripper_already_open_abstain(self):
        self._warmup_to_step(5)
        # CLOSE but decoded_open=1 → gripper_already_open
        s, rv, e, q, x, y, z, d, rv2, ev2, qv2, ef2, gs2 = CLOSE(5)
        r = do_update(self.a, (s, rv, e, q, x, y, z, 1, rv2, ev2, qv2, ef2, gs2))
        self.assertIsNotNone(r, "expected candidate from close_onset")
        self.assertTrue(r["abstained"])
        self.assertIn("gripper_already_open", r["abstain"])

    def test_decoded_open_abstain_with_crossing(self):
        # OPEN → CLOSE crossing but decoded_open=1 → gripper_already_open
        self._warmup_to_step(4)
        do_update(self.a, OPEN(4))
        s, rv, e, q, x, y, z, d, rv2, ev2, qv2, ef2, gs2 = CLOSE(5)
        r = do_update(self.a, (s, rv, e, q, x, y, z, 1, rv2, ev2, qv2, ef2, gs2))
        self.assertIsNotNone(r, "expected candidate from close_onset at step 5")
        self.assertTrue(r["abstained"])
        self.assertIn("gripper_already_open", r["abstain"])


class TestAdapterValidCandidates(unittest.TestCase):
    def setUp(self):
        self.a = D5FrozenFeatureAdapter()

    def _warmup_to_step(self, n):
        for i in range(n):
            do_update(self.a, OPEN(i))

    def test_valid_close_candidate_has_16_features(self):
        self._warmup_to_step(5)
        # OPEN→CLOSE crossing: step 5 is CLOSE after OPEN at step 4
        do_update(self.a, OPEN(5))
        r = do_update(self.a, CLOSE(6))
        self.assertIsNotNone(r, "expected candidate from close_onset at step 6")
        self.assertIn("features", r)
        self.assertEqual(len(r["features"]), 16)
        self.assertIn("total_score", r["features"])
        self.assertIn("close_streak", r["features"])

    def test_candidate_reason_present(self):
        self._warmup_to_step(5)
        do_update(self.a, OPEN(5))
        r = do_update(self.a, CLOSE(6))
        self.assertIsNotNone(r)
        self.assertIn("candidate_reason", r)
        self.assertGreater(len(r["candidate_reason"]), 0)
        self.assertIn("close_onset", r["candidate_reason"])

    def test_schema_and_commit_frozen(self):
        self._warmup_to_step(5)
        do_update(self.a, OPEN(5))
        r = do_update(self.a, CLOSE(6))
        self.assertIsNotNone(r)
        self.assertEqual(r["feature_schema_version"], "d5_frozen_v1")
        self.assertTrue(r["source_commit"].startswith("44bf7b86"))

    def test_qpos_valid_false_still_can_generate_candidate(self):
        # qpos_valid=False → qpos stored as 0.0, close_onset still fires
        s, rv2, e, q, x, y, z, d, rv3, ev3, qv3, ef3, gs3 = CLOSE(0)
        r = do_update(self.a, (s, rv2, e, q, x, y, z, d, rv3, ev3, False, ef3, gs3))
        self.assertIsNotNone(r, "close_onset fires despite qpos_valid=False")
        self.assertEqual(r["features"]["qpos"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
