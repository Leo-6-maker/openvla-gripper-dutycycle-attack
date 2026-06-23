#!/usr/bin/env python3
"""Unit tests for SC5DetectorRuntimeV1R — all FSM versions.

Tests FSM logic in isolation by monkeypatching model inference.
Does NOT require GPU or MuJoCo.
"""
import json, sys, os, tempfile
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
    """Create a minimal valid checkpoint for testing."""
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


def _make_features(phase="stable_carry", cp=0.9, rp=0.001):
    """Build 25D feature dict. Phase/cp/rp are used by the monkeypatch only.
    The actual values here don't matter because we patch model inference."""
    return {fn: 0.0 for fn in SC5_FEATURES}


def _patch_inference(detector, phase, cp, rp):
    """Monkeypatch model to return controlled predictions."""
    phase_idx = SC5_PHASES.index(phase)
    logits = torch.zeros(1, len(SC5_PHASES))
    logits[0, phase_idx] = 100.0

    def _fake_forward(x):
        return {
            "phase_logits": logits,
            "corridor_logit": torch.tensor([[float(np.log(cp / (1 - cp)))]]),
            "release_logit": torch.tensor([[float(np.log(rp / (1 - rp)))]]),
        }

    detector.model.forward = _fake_forward


def _make_detector(fsm_version="legacy_v1", **kwargs):
    """Create a detector with a temp checkpoint file. Extra kwargs forwarded."""
    ckpt = _make_ckpt()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(ckpt, f.name)
        path = f.name
    detector = SC5DetectorRuntimeV1R(path, fsm_version=fsm_version, **kwargs)
    os.unlink(path)
    return detector


# ── Legacy v1 regression ───────────────────────────────────────────

def test_legacy_idle_to_armed():
    d = _make_detector("legacy_v1")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    result = d.update(_make_features(), 0)
    assert result["state"] == "ARMED"
    assert result["fsm_version"] == "legacy_v1"
    assert result["arm_step"] == 0


def test_legacy_idle_stays_idle_low_cp():
    d = _make_detector("legacy_v1")
    _patch_inference(d, "stable_carry", 0.1, 0.001)
    result = d.update(_make_features(), 0)
    assert result["state"] == "IDLE"


def test_legacy_armed_to_emitted():
    d = _make_detector("legacy_v1")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 0)  # arm
    for step in range(1, 5):
        d.update(_make_features(), step)  # guard
    result = d.update(_make_features(), 5)
    assert result["state"] == "EMITTED"
    assert result["emitted"] is True
    assert result["emit_step"] == 5


def test_legacy_no_disarm():
    """ARMED→ARMED even when evidence drops."""
    d = _make_detector("legacy_v1")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 0)  # arm
    _patch_inference(d, "approach", 0.05, 0.8)  # evidence lost
    result = d.update(_make_features(), 1)
    assert result["state"] == "ARMED"  # still armed!


def test_legacy_one_shot():
    """EMITTED stays EMITTED, no re-trigger."""
    d = _make_detector("legacy_v1")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 0)
    for step in range(1, 5):
        d.update(_make_features(), step)
    d.update(_make_features(), 5)  # emit
    result = d.update(_make_features(), 6)
    assert result["state"] == "EMITTED"
    assert result["emitted"] is True


def test_legacy_reset():
    d = _make_detector("legacy_v1")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 0)
    d.update(_make_features(), 5)
    assert d.emitted is True
    d.reset()
    assert d.state == "IDLE"
    assert d.arm_step == -1
    assert d.emit_step == -1
    assert d.emitted is False


# ── R1: Minimal disarm ─────────────────────────────────────────────

def test_r1_idle_to_armed():
    d = _make_detector("v1r_r1")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    result = d.update(_make_features(), 0)
    assert result["state"] == "ARMED"
    assert result["fsm_version"] == "v1r_r1"


def test_r1_armed_to_idle_phase_change():
    d = _make_detector("v1r_r1")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 0)  # arm
    _patch_inference(d, "approach", 0.9, 0.001)  # phase lost
    result = d.update(_make_features(), 1)
    assert result["state"] == "IDLE"
    assert result["disarm_count"] == 1
    assert result["disarm_reason"] == "PHASE_EXIT"


def test_r1_armed_to_idle_cp_drop():
    d = _make_detector("v1r_r1")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 0)
    _patch_inference(d, "stable_carry", 0.1, 0.001)  # cp dropped
    result = d.update(_make_features(), 1)
    assert result["state"] == "IDLE"
    assert result["disarm_reason"] == "CORRIDOR_DROP"


def test_r1_armed_to_idle_rp_rise():
    d = _make_detector("v1r_r1")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 0)
    _patch_inference(d, "stable_carry", 0.9, 0.9)  # release rose
    result = d.update(_make_features(), 1)
    assert result["state"] == "IDLE"
    assert result["disarm_reason"] == "RELEASE_RISE"


def test_r1_rearm_after_disarm():
    """Can re-arm after disarming."""
    d = _make_detector("v1r_r1")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 0)  # arm
    _patch_inference(d, "approach", 0.9, 0.001)  # disarm
    d.update(_make_features(), 1)
    assert d.state == "IDLE"
    assert d.disarm_count == 1
    _patch_inference(d, "stable_carry", 0.9, 0.001)  # re-arm
    result = d.update(_make_features(), 2)
    assert result["state"] == "ARMED"


def test_r1_still_emits_on_sustained():
    d = _make_detector("v1r_r1")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 0)
    for step in range(1, 5):
        d.update(_make_features(), step)
    result = d.update(_make_features(), 5)
    assert result["state"] == "EMITTED"

    # disarms accumulated during guard?
    assert result["disarm_count"] == 0  # evidence never broke
    assert result["disarm_reason"] == ""


def test_r1_reset_includes_telemetry():
    d = _make_detector("v1r_r1")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 0)
    _patch_inference(d, "approach", 0.1, 0.9)
    d.update(_make_features(), 1)
    assert d.disarm_count == 1
    d.reset()
    assert d.disarm_count == 0
    assert d.last_disarm_step == -1
    assert d.disarm_reason == ""


# ── R2: Full candidate machine ─────────────────────────────────────

def test_r2_idle_to_candidate():
    d = _make_detector("v1r_r2")
    _patch_inference(d, "stable_carry", 0.9, 0.001)  # cp > tau_on=0.5
    result = d.update(_make_features(), 0)
    assert result["state"] == "CANDIDATE"
    assert result["candidate_step"] == 0
    assert result["candidate_streak"] == 1


def test_r2_idle_stays_idle_below_tau_on():
    """tau_on=0.5 — cp=0.4 is above tau_c but below tau_on, should stay IDLE."""
    d = _make_detector("v1r_r2")
    _patch_inference(d, "stable_carry", 0.4, 0.001)
    result = d.update(_make_features(), 0)
    assert result["state"] == "IDLE"


def test_r2_candidate_accumulates():
    d = _make_detector("v1r_r2")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 0)
    assert d.candidate_streak == 1
    d.update(_make_features(), 1)
    assert d.candidate_streak == 2


def test_r2_candidate_to_armed():
    d = _make_detector("v1r_r2")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 0)
    d.update(_make_features(), 1)
    result = d.update(_make_features(), 2)  # 3rd consecutive → ARM
    assert result["state"] == "ARMED"
    assert d.arm_step == 2
    assert d.candidate_streak == 0  # reset after arming


def test_r2_candidate_break_to_idle():
    d = _make_detector("v1r_r2")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 0)
    d.update(_make_features(), 1)  # streak=2
    _patch_inference(d, "approach", 0.1, 0.9)  # break
    result = d.update(_make_features(), 2)
    assert result["state"] == "IDLE"
    assert result["disarm_reason"] == "CANDIDATE_BREAK"
    assert d.candidate_streak == 0


def test_r2_armed_disarm_back_to_idle():
    d = _make_detector("v1r_r2")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 0)
    d.update(_make_features(), 1)
    d.update(_make_features(), 2)  # armed
    _patch_inference(d, "approach", 0.1, 0.9)  # disarm
    result = d.update(_make_features(), 3)
    assert result["state"] == "IDLE"


def test_r2_arm_timeout():
    d = _make_detector("v1r_r2", max_arm_age=5)
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 0)
    d.update(_make_features(), 1)
    d.update(_make_features(), 2)  # armed at step 2
    # Sustain evidence but never emit — guard=5 not reached before timeout
    for step in range(3, 7):
        result = d.update(_make_features(), step)
        assert result["state"] == "ARMED"
    # step 7: arm_age = 7-2 = 5 >= max_arm_age=5 → timeout
    result = d.update(_make_features(), 7)
    assert result["state"] == "IDLE"
    assert result["disarm_reason"] == "ARM_TIMEOUT"


def test_r2_emits_after_guard():
    d = _make_detector("v1r_r2")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 0)  # candidate
    d.update(_make_features(), 1)  # candidate
    d.update(_make_features(), 2)  # armed at step 2
    for step in range(3, 7):
        d.update(_make_features(), step)
    result = d.update(_make_features(), 7)  # guard=5: step >= 2+5=7
    assert result["state"] == "EMITTED"
    assert result["emit_step"] == 7


def test_r2_does_not_emit_if_evidence_lost_during_guard():
    d = _make_detector("v1r_r2")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 0)
    d.update(_make_features(), 1)
    d.update(_make_features(), 2)  # armed
    # Guard steps 3,4 — ok. At step 5, evidence drops then recovers.
    d.update(_make_features(), 3)
    d.update(_make_features(), 4)
    _patch_inference(d, "approach", 0.1, 0.9)  # disarm
    d.update(_make_features(), 5)
    assert d.state == "IDLE"  # disarmed
    _patch_inference(d, "stable_carry", 0.9, 0.001)  # re-arm
    d.update(_make_features(), 6)  # candidate streak=1 — not yet armed
    assert d.state == "CANDIDATE"


def test_r2_reset():
    d = _make_detector("v1r_r2")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 0)
    assert d.state == "CANDIDATE"
    d.reset()
    assert d.state == "IDLE"
    assert d.candidate_step == -1
    assert d.candidate_streak == 0
    assert d.disarm_count == 0


# ── Cross-version: legacy v1 unchanged ─────────────────────────────

def test_legacy_v1_output_fields():
    """Original fields must still be present."""
    d = _make_detector("legacy_v1")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    result = d.update(_make_features(), 0)
    for key in ["state", "arm_step", "emit_step", "emitted", "corridor_p",
                "release_p", "pred_phase", "step", "fsm_version"]:
        assert key in result


def test_all_versions_report_fsm_version():
    for ver in ["legacy_v1", "v1r_r1", "v1r_r2"]:
        d = _make_detector(ver)
        _patch_inference(d, "stable_carry", 0.9, 0.001)
        result = d.update(_make_features(), 0)
        assert result["fsm_version"] == ver


def test_legacy_disarm_fields_are_zero():
    """Legacy v1 should report zero disarm fields (not NaN or missing)."""
    d = _make_detector("legacy_v1")
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    result = d.update(_make_features(), 0)
    assert result["disarm_count"] == 0
    assert result["disarm_reason"] == ""


# ── Known case fixtures ────────────────────────────────────────────

def test_butter_s1_scenario_r1_disarms():
    """Simulate butter_s1/B0: arm at step 105, cp drops later → R1 should disarm."""
    d = _make_detector("v1r_r1")
    # Arm at step 105
    _patch_inference(d, "stable_carry", 0.62, 0.0002)
    d.update(_make_features(), 105)
    assert d.state == "ARMED"
    # Simulate evidence break (cp dropped — Phase A shows 4 breaks)
    _patch_inference(d, "stable_carry", 0.25, 0.0002)  # below tau_c
    result = d.update(_make_features(), 106)
    assert result["state"] == "IDLE"
    assert result["disarm_reason"] == "CORRIDOR_DROP"


def test_chocolate_pudding_s1_r1_blocks_emit():
    """Simulate choc_pudding_s1/B0: emit at step 42 but phase was pre_place_unsupported."""
    d = _make_detector("v1r_r1")
    _patch_inference(d, "stable_carry", 0.44, 0.035)
    d.update(_make_features(), 37)  # arm
    for step in range(38, 42):
        _patch_inference(d, "stable_carry", 0.44, 0.035)
        d.update(_make_features(), step)
    # At emit step (42), phase was pre_place_unsupported
    _patch_inference(d, "pre_place_unsupported", 0.81, 0.176)
    result = d.update(_make_features(), 42)
    assert result["state"] == "IDLE"  # R1 disarmed before emit
    assert result["emitted"] is False


def test_orange_juice_s2_r2_disarms_immediately():
    """Simulate orange_juice_s2/B0: arm at 94, evidence breaks immediately.
    R2 should disarm at first broken step — no 303-step stall."""
    d = _make_detector("v1r_r2", max_arm_age=200)
    _patch_inference(d, "stable_carry", 0.9, 0.001)
    d.update(_make_features(), 91)
    d.update(_make_features(), 92)
    d.update(_make_features(), 93)
    assert d.state == "ARMED"  # armed at step 93
    # Evidence breaks immediately (Phase A: 303 evidence breaks)
    _patch_inference(d, "stable_carry", 0.1, 0.9)  # cp low, rp high
    result = d.update(_make_features(), 94)
    assert result["state"] == "IDLE"
    assert result["disarm_reason"] in ("CORRIDOR_DROP", "RELEASE_RISE")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
