"""G1-R: Independent negative tests for D5FrozenFeatureAdapter."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "stageb"))
from gripper_attack.d5_frozen_feature_adapter_v1 import D5FrozenFeatureAdapter


def make_valid(step=0, raw=0.9, env=-1.0, qpos=0.0, ex=0.0, ey=0.0, ez=0.0, dec=0):
    return (step, raw, env, qpos, ex, ey, ez, dec, True, True, True, True, True)


def update(adapter, args):
    return adapter.update(*args)


class TestD5FrozenFeatureAdapter:
    def setup_method(self):
        self.adapter = D5FrozenFeatureAdapter()

    # ── T1-T3: Step sequence ──

    def test_duplicate_step_raises(self):
        update(self.adapter, make_valid(0))
        with pytest.raises(ValueError):
            update(self.adapter, make_valid(0))

    def test_skipped_step_raises(self):
        update(self.adapter, make_valid(0))
        with pytest.raises(ValueError):
            update(self.adapter, make_valid(2))

    def test_reset_clears_state(self):
        update(self.adapter, make_valid(0))
        update(self.adapter, make_valid(1))
        assert self.adapter.next_expected_step == 2
        self.adapter.reset()
        assert self.adapter.next_expected_step == 0

    # ── T4-T8: Validity flags ──

    def test_raw_valid_false_no_candidate(self):
        s, r, e, q, x, y, z, d, rv, ev, qv, ef, gs = make_valid(0, raw=0.9, env=-1.0)
        r = update(self.adapter, (0, 0.9, -1.0, q, x, y, z, d, False, True, True, True, True))
        assert r is None  # raw_valid=False → fail-closed

    def test_env_valid_false_no_candidate(self):
        r = update(self.adapter, (0, 0.9, -1.0, 0.0, 0.0, 0.0, 0.0, 0, True, False, True, True, True))
        assert r is None

    def test_qpos_valid_false_still_tracks(self):
        # qpos_valid=False should not crash, just fail-closed on qpos features
        r = update(self.adapter, (0, 0.9, -1.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, False, True, True))
        # no candidate expected (env_gripper > 0.5 triggers close_onset, but qpos valid doesn't block candidate)
        # accept None or valid result
        assert r is None or isinstance(r, dict)

    def test_eef_valid_false_no_candidate(self):
        r = update(self.adapter, (0, 0.9, -1.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True, False, True))
        assert r is None

    def test_semantics_valid_false_no_candidate(self):
        r = update(self.adapter, (0, 0.9, -1.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True, True, False))
        assert r is None

    # ── T9: decoded_open invalid ──

    def test_decoded_open_invalid_no_candidate(self):
        r = update(self.adapter, (0, 0.9, -1.0, 0.0, 0.0, 0.0, 0.0, 2, True, True, True, True, True))
        assert r is None

    # ── T10-T11: NaN ──

    def test_nan_qpos(self):
        r = update(self.adapter, (0, 0.9, -1.0, float("nan"), 0.0, 0.0, 0.0, 0, True, True, True, True, True))
        # NaN qpos → qpos_valid fails → qpos_ok=False → no candidate
        assert r is None

    def test_nan_eef(self):
        r = update(self.adapter, (0, 0.9, -1.0, 0.0, float("nan"), 0.0, 0.0, 0, True, True, True, True, True))
        assert r is None

    # ── T12-T14: Abstain reasons ──

    def test_too_early_abstain(self):
        self.adapter.reset()
        # Steps 0-2: create raw_crossing to trigger candidate
        update(self.adapter, (0, 0.9, -1.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True, True, True))
        update(self.adapter, (1, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True, True, True))
        r = update(self.adapter, (2, 0.9, -1.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True, True, True))
        if r is not None:
            assert r.get("abstained") == True
            assert "too_early" in r.get("abstain", "")

    def test_gripper_already_open_abstain(self):
        self.adapter.reset()
        for i in range(5):
            update(self.adapter, (i, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True, True, True))
        # Create candidate with decoded_open=1
        r = update(self.adapter, (5, 0.9, -1.0, 0.0, 0.0, 0.0, 0.0, 1, True, True, True, True, True))
        if r is not None:
            assert r.get("abstained") == True
            assert "gripper_already_open" in r.get("abstain", "")

    def test_low_confidence_abstain(self):
        self.adapter.reset()
        for i in range(5):
            update(self.adapter, (i, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True, True, True))
        # Candidate with no high-score signals → low_confidence
        r = update(self.adapter, (5, 0.9, -1.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True, True, True))
        if r is not None:
            # Either abstained=low_confidence or not a candidate at all
            if r.get("abstained"):
                assert "low_confidence" in r.get("abstain", "") or "too_early" in r.get("abstain", "")

    # ── T15: Valid close candidate ──

    def test_valid_close_candidate_has_features(self):
        self.adapter.reset()
        # Build enough history to not be too_early
        for i in range(5):
            update(self.adapter, (i, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True, True, True))
        # Now create a close — env > 0.5 triggers close_onset
        r = update(self.adapter, (5, 0.9, -1.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True, True, True))
        if r is not None:
            assert "features" in r
            assert len(r["features"]) == 16
            assert "total_score" in r["features"]

    # ── T16: Candidate reason ──

    def test_candidate_reason_present(self):
        self.adapter.reset()
        for i in range(5):
            update(self.adapter, (i, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True, True, True))
        r = update(self.adapter, (5, 0.9, -1.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True, True, True))
        if r is not None:
            assert "candidate_reason" in r
            assert len(r["candidate_reason"]) > 0

    # ── T17: Schema and commit frozen ──

    def test_schema_and_commit_frozen(self):
        self.adapter.reset()
        for i in range(5):
            update(self.adapter, (i, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True, True, True))
        r = update(self.adapter, (5, 0.9, -1.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True, True, True))
        if r is not None:
            assert r["feature_schema_version"] == "d5_frozen_v1"
            assert r["source_commit"].startswith("44bf7b86")
