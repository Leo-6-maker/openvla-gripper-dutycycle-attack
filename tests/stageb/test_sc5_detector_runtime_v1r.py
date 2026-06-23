#!/usr/bin/env python3
"""Unit tests for SC5DetectorRuntimeV1R — all FSM versions.

Tests FSM logic in isolation via update_from_scores (no model, no GPU).
Does NOT require GPU or MuJoCo.
"""
import os, sys, tempfile
from pathlib import Path
import numpy as np
import torch
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

from gripper_attack.sc5_detector_runtime_v1r import (
    SC5DetectorRuntimeV1R, SC5MLP, SC5_FEATURES, SC5_PHASES,
)


# ── Helpers ────────────────────────────────────────────────────────

def _make_ckpt():
    sd = SC5MLP(n_feat=25).state_dict()
    return {
        "model_state": sd,
        "mean": np.zeros(25, dtype=np.float32),
        "std": np.ones(25, dtype=np.float32),
        "feature_names": SC5_FEATURES,
        "phase_classes": SC5_PHASES,
        "dataset_sha256": "0" * 64,
        "split_mode": "frozen",
    }


def _make_detector(fsm_version="legacy_v1", **kwargs):
    ckpt = _make_ckpt()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(ckpt, f.name)
        path = f.name
    detector = SC5DetectorRuntimeV1R(path, fsm_version=fsm_version, **kwargs)
    os.unlink(path)
    return detector


# ── Config validation ──────────────────────────────────────────────

def test_config_rejects_tau_on_le_tau_off():
    with pytest.raises(ValueError, match="tau_on"):
        _make_detector("v1r_r2", tau_on=0.3, tau_off=0.5)

def test_config_rejects_n_candidate_zero():
    with pytest.raises(ValueError, match="n_candidate"):
        _make_detector("v1r_r2", n_candidate=0)

def test_config_rejects_guard_negative():
    with pytest.raises(ValueError, match="guard"):
        _make_detector("legacy_v1", guard=-1)

def test_config_rejects_max_arm_age_zero():
    with pytest.raises(ValueError, match="max_arm_age"):
        _make_detector("v1r_r2", max_arm_age=0)

def test_config_unknown_version():
    with pytest.raises(ValueError, match="Unknown fsm_version"):
        _make_detector("v1r_r3")


# ── Legacy v1 regression ───────────────────────────────────────────

def test_legacy_idle_to_armed():
    d = _make_detector("legacy_v1")
    r = d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    assert r["state"] == "ARMED"
    assert r["arm_step"] == 0

def test_legacy_idle_stays_idle_low_cp():
    d = _make_detector("legacy_v1")
    r = d.update_from_scores(0.1, 0.001, "stable_carry", 0)
    assert r["state"] == "IDLE"

def test_legacy_armed_to_emitted():
    d = _make_detector("legacy_v1")
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    for s in range(1, 5):
        d.update_from_scores(0.9, 0.001, "stable_carry", s)
    r = d.update_from_scores(0.9, 0.001, "stable_carry", 5)
    assert r["state"] == "EMITTED"
    assert r["emitted"] is True
    assert r["emit_step"] == 5

def test_legacy_no_disarm():
    d = _make_detector("legacy_v1")
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    r = d.update_from_scores(0.05, 0.8, "approach", 1)
    assert r["state"] == "ARMED"

def test_legacy_one_shot():
    d = _make_detector("legacy_v1")
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    for s in range(1, 5):
        d.update_from_scores(0.9, 0.001, "stable_carry", s)
    d.update_from_scores(0.9, 0.001, "stable_carry", 5)
    r = d.update_from_scores(0.9, 0.001, "stable_carry", 6)
    assert r["state"] == "EMITTED"

def test_legacy_reset():
    d = _make_detector("legacy_v1")
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    d.update_from_scores(0.9, 0.001, "stable_carry", 5)
    assert d.emitted is True
    d.reset()
    assert d.state == "IDLE"
    assert d.arm_step == -1
    assert d.emit_step == -1
    assert d.emitted is False


# ── R1: Minimal disarm ─────────────────────────────────────────────

def test_r1_armed_to_idle_phase_change():
    d = _make_detector("v1r_r1")
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    r = d.update_from_scores(0.9, 0.001, "approach", 1)
    assert r["state"] == "IDLE"
    assert r["disarm_count"] == 1
    assert r["disarm_reason"] == "PHASE_EXIT"

def test_r1_armed_to_idle_cp_drop():
    d = _make_detector("v1r_r1")
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    r = d.update_from_scores(0.1, 0.001, "stable_carry", 1)
    assert r["state"] == "IDLE"
    assert r["disarm_reason"] == "CORRIDOR_DROP"

def test_r1_armed_to_idle_rp_rise():
    d = _make_detector("v1r_r1")
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    r = d.update_from_scores(0.9, 0.9, "stable_carry", 1)
    assert r["state"] == "IDLE"
    assert r["disarm_reason"] == "RELEASE_RISE"

def test_r1_feature_invalid_disarm():
    d = _make_detector("v1r_r1")
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    r = d.update_from_scores(float("nan"), float("nan"), "stable_carry", 1, feat_valid=False)
    assert r["state"] == "IDLE"
    assert r["disarm_reason"] == "FEATURE_INVALID"

def test_r1_rearm_after_disarm():
    d = _make_detector("v1r_r1")
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    d.update_from_scores(0.9, 0.001, "approach", 1)
    assert d.state == "IDLE"
    assert d.disarm_count == 1
    r = d.update_from_scores(0.9, 0.001, "stable_carry", 2)
    assert r["state"] == "ARMED"

def test_r1_telemetry_clean_after_disarm():
    """arm_step cleared, last_arm_step recorded."""
    d = _make_detector("v1r_r1")
    d.update_from_scores(0.9, 0.001, "stable_carry", 10)
    assert d.arm_step == 10
    d.update_from_scores(0.1, 0.001, "stable_carry", 11)
    assert d.state == "IDLE"
    assert d.arm_step == -1
    assert d.last_arm_step == 10
    assert d.arm_age == 0

def test_r1_still_emits_on_sustained():
    d = _make_detector("v1r_r1")
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    for s in range(1, 5):
        d.update_from_scores(0.9, 0.001, "stable_carry", s)
    r = d.update_from_scores(0.9, 0.001, "stable_carry", 5)
    assert r["state"] == "EMITTED"
    assert r["disarm_count"] == 0

def test_r1_reset_includes_telemetry():
    d = _make_detector("v1r_r1")
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    d.update_from_scores(0.1, 0.9, "approach", 1)
    assert d.disarm_count == 1
    d.reset()
    assert d.disarm_count == 0
    assert d.last_disarm_step == -1
    assert d.disarm_reason == ""


# ── R2: Full candidate machine ─────────────────────────────────────

def test_r2_idle_to_candidate():
    d = _make_detector("v1r_r2")
    r = d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    assert r["state"] == "CANDIDATE"
    assert r["candidate_step"] == 0
    assert r["candidate_streak"] == 1

def test_r2_idle_stays_idle_below_tau_on():
    d = _make_detector("v1r_r2")
    r = d.update_from_scores(0.4, 0.001, "stable_carry", 0)
    assert r["state"] == "IDLE"

def test_r2_candidate_accumulates():
    d = _make_detector("v1r_r2")
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    d.update_from_scores(0.9, 0.001, "stable_carry", 1)
    assert d.candidate_streak == 2

def test_r2_candidate_to_armed():
    d = _make_detector("v1r_r2")
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    d.update_from_scores(0.9, 0.001, "stable_carry", 1)
    r = d.update_from_scores(0.9, 0.001, "stable_carry", 2)
    assert r["state"] == "ARMED"
    assert d.arm_step == 2
    assert d.candidate_streak == 0
    assert d.disarm_reason == ""  # cleared on arm

def test_r2_candidate_break_to_idle():
    d = _make_detector("v1r_r2")
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    d.update_from_scores(0.9, 0.001, "stable_carry", 1)
    r = d.update_from_scores(0.1, 0.9, "approach", 2)
    assert r["state"] == "IDLE"
    assert r["disarm_reason"] == "CANDIDATE_BREAK"
    assert d.candidate_step == -1
    assert d.last_candidate_step == 0
    assert d.candidate_streak == 0

def test_r2_armed_disarm_to_idle():
    d = _make_detector("v1r_r2")
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    d.update_from_scores(0.9, 0.001, "stable_carry", 1)
    d.update_from_scores(0.9, 0.001, "stable_carry", 2)  # armed
    r = d.update_from_scores(0.1, 0.9, "approach", 3)
    assert r["state"] == "IDLE"

def test_r2_arm_timeout():
    """Timeout fires when guard is large: guard=100, max_arm_age=5 → timeout at step 7."""
    d = _make_detector("v1r_r2", guard=100, max_arm_age=5)
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    d.update_from_scores(0.9, 0.001, "stable_carry", 1)
    d.update_from_scores(0.9, 0.001, "stable_carry", 2)  # armed at 2
    for s in range(3, 7):
        r = d.update_from_scores(0.9, 0.001, "stable_carry", s)
        assert r["state"] == "ARMED"
    r = d.update_from_scores(0.9, 0.001, "stable_carry", 7)
    assert r["state"] == "IDLE"
    assert r["disarm_reason"] == "ARM_TIMEOUT"

def test_r2_emits_after_guard():
    d = _make_detector("v1r_r2")
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    d.update_from_scores(0.9, 0.001, "stable_carry", 1)
    d.update_from_scores(0.9, 0.001, "stable_carry", 2)  # armed
    for s in range(3, 7):
        d.update_from_scores(0.9, 0.001, "stable_carry", s)
    r = d.update_from_scores(0.9, 0.001, "stable_carry", 7)
    assert r["state"] == "EMITTED"
    assert r["emit_step"] == 7

def test_r2_guard_loses_evidence_then_rearms():
    d = _make_detector("v1r_r2")
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    d.update_from_scores(0.9, 0.001, "stable_carry", 1)
    d.update_from_scores(0.9, 0.001, "stable_carry", 2)  # armed
    d.update_from_scores(0.9, 0.001, "stable_carry", 3)
    d.update_from_scores(0.9, 0.001, "stable_carry", 4)
    # Evidence lost — disarm
    d.update_from_scores(0.1, 0.9, "approach", 5)
    assert d.state == "IDLE"
    # Re-arm: need N_min=3 candidate frames again
    d.update_from_scores(0.9, 0.001, "stable_carry", 6)
    assert d.state == "CANDIDATE"
    d.update_from_scores(0.9, 0.001, "stable_carry", 7)
    d.update_from_scores(0.9, 0.001, "stable_carry", 8)  # armed again at 8
    assert d.state == "ARMED"
    # Guard restarts from re-arm
    for s in range(9, 13):
        d.update_from_scores(0.9, 0.001, "stable_carry", s)
    r = d.update_from_scores(0.9, 0.001, "stable_carry", 13)
    assert r["state"] == "EMITTED"

def test_r2_reset():
    d = _make_detector("v1r_r2")
    d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    assert d.state == "CANDIDATE"
    d.reset()
    assert d.state == "IDLE"
    assert d.candidate_step == -1
    assert d.candidate_streak == 0
    assert d.disarm_count == 0
    assert d.last_candidate_step == -1
    assert d.last_arm_step == -1


# ── Cross-version: legacy v1 unchanged ─────────────────────────────

def test_legacy_v1_output_fields():
    d = _make_detector("legacy_v1")
    r = d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    for key in ["state", "arm_step", "emit_step", "emitted", "corridor_p",
                "release_p", "pred_phase", "step", "fsm_version"]:
        assert key in r

def test_all_versions_report_fsm_version():
    for ver in ["legacy_v1", "v1r_r1", "v1r_r2"]:
        d = _make_detector(ver)
        r = d.update_from_scores(0.9, 0.001, "stable_carry", 0)
        assert r["fsm_version"] == ver

def test_legacy_disarm_fields_are_zero():
    d = _make_detector("legacy_v1")
    r = d.update_from_scores(0.9, 0.001, "stable_carry", 0)
    assert r["disarm_count"] == 0
    assert r["disarm_reason"] == ""

def test_update_deprecated_still_works():
    """Online update() should still work and delegate to FSM."""
    d = _make_detector("legacy_v1")
    feats = {fn: 0.0 for fn in SC5_FEATURES}
    feats["eef_z"] = 10.0  # arbitrary nonzero
    r = d.update(feats, 0)
    assert "state" in r


# ── Known case fixtures ────────────────────────────────────────────

def test_butter_s1_scenario_r1_disarms():
    d = _make_detector("v1r_r1")
    d.update_from_scores(0.62, 0.0002, "stable_carry", 105)
    assert d.state == "ARMED"
    r = d.update_from_scores(0.25, 0.0002, "stable_carry", 106)
    assert r["state"] == "IDLE"
    assert r["disarm_reason"] == "CORRIDOR_DROP"

def test_chocolate_pudding_s1_r1_blocks_emit():
    d = _make_detector("v1r_r1")
    d.update_from_scores(0.44, 0.035, "stable_carry", 37)
    for s in range(38, 42):
        d.update_from_scores(0.44, 0.035, "stable_carry", s)
    r = d.update_from_scores(0.81, 0.176, "pre_place_unsupported", 42)
    assert r["state"] == "IDLE"
    assert r["emitted"] is False

def test_orange_juice_s2_r2_disarms_immediately():
    d = _make_detector("v1r_r2", max_arm_age=200)
    d.update_from_scores(0.9, 0.001, "stable_carry", 91)
    d.update_from_scores(0.9, 0.001, "stable_carry", 92)
    d.update_from_scores(0.9, 0.001, "stable_carry", 93)
    assert d.state == "ARMED"
    r = d.update_from_scores(0.1, 0.9, "stable_carry", 94)
    assert r["state"] == "IDLE"
    assert r["disarm_reason"] in ("CORRIDOR_DROP", "RELEASE_RISE")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
