"""Unit tests for guard_should_block_open()."""
import numpy as np
from v4_run_eval_openvla import guard_should_block_open


def _env_action(gripper: float) -> np.ndarray:
    a = np.zeros(7, dtype=np.float32)
    a[-1] = float(gripper)
    return a


def _meta(**kwargs) -> dict:
    m = {
        "grasp_close_intent": False,
        "grasp_first_close_step": None,
        "grasp_first_gate_step": None,
        "grasp_bowl_z_delta": 0.0,
        "grasp_bowl_plate_dxy": 999.0,
        "grasp_bowl_plate_dz": 999.0,
    }
    m.update(kwargs)
    return m


# ── block tests ──

def test_blocks_open_during_lift():
    m = _meta(grasp_close_intent=True, grasp_bowl_z_delta=0.05,
              grasp_bowl_plate_dxy=0.20, grasp_bowl_plate_dz=0.10)
    blocked, reason = guard_should_block_open(m, _env_action(-1.0))
    assert blocked
    assert reason == "guard_blocked"


def test_blocks_open_with_partial_close():
    m = _meta(grasp_close_intent=True, grasp_bowl_z_delta=0.03,
              grasp_bowl_plate_dxy=0.15)
    blocked, reason = guard_should_block_open(m, _env_action(-0.6))
    assert blocked


def test_blocks_after_prior_close_step_even_if_current_open():
    m = _meta(grasp_close_intent=False, grasp_first_close_step=38,
              grasp_bowl_z_delta=0.03, grasp_bowl_plate_dxy=0.15)
    blocked, reason = guard_should_block_open(m, _env_action(-1.0))
    assert blocked
    assert reason == "guard_blocked"


def test_strict_blocks_after_prior_close_even_before_lift():
    m = _meta(grasp_close_intent=False, grasp_first_close_step=38,
              grasp_bowl_z_delta=0.0, grasp_bowl_plate_dxy=0.15)
    blocked, reason = guard_should_block_open(m, _env_action(-1.0), mode="strict_after_close")
    assert blocked
    assert reason == "guard_blocked_strict_after_close"


# ── allow tests (should NOT block) ──

def test_allows_closed_command():
    m = _meta(grasp_close_intent=True, grasp_bowl_z_delta=0.05,
              grasp_bowl_plate_dxy=0.20)
    blocked, reason = guard_should_block_open(m, _env_action(1.0))
    assert not blocked
    assert reason == "gripper_not_open"


def test_allows_open_when_no_close_intent():
    m = _meta(grasp_close_intent=False, grasp_bowl_z_delta=0.05,
              grasp_bowl_plate_dxy=0.20)
    blocked, reason = guard_should_block_open(m, _env_action(-1.0))
    assert not blocked
    assert reason == "no_close_intent"


def test_allows_open_when_bowl_not_lifted():
    m = _meta(grasp_close_intent=True, grasp_bowl_z_delta=0.01)
    blocked, reason = guard_should_block_open(m, _env_action(-1.0))
    assert not blocked
    assert reason == "bowl_not_lifted"


def test_strict_still_allows_open_before_close_seen():
    m = _meta(grasp_close_intent=False, grasp_bowl_z_delta=0.05)
    blocked, reason = guard_should_block_open(m, _env_action(-1.0), mode="strict_after_close")
    assert not blocked
    assert reason == "no_close_intent"


def test_allows_open_near_release_target():
    m = _meta(grasp_close_intent=True, grasp_bowl_z_delta=0.05,
              grasp_bowl_plate_dxy=0.02, grasp_bowl_plate_dz=0.01)
    blocked, reason = guard_should_block_open(m, _env_action(-1.0))
    assert not blocked
    assert reason == "near_release_target"


def test_allows_open_at_exact_threshold():
    m = _meta(grasp_close_intent=True, grasp_bowl_z_delta=0.02)
    blocked, reason = guard_should_block_open(m, _env_action(-0.5))
    assert not blocked  # close_threshold is -0.5, >= means not open


# ── clean trajectory should never trigger ──

def test_clean_trajectory_never_blocks():
    """Guard must not interfere with clean rollouts."""
    for z_delta in [0.0, 0.01, 0.05, 0.10]:
        for dxy in [999, 0.20, 0.04]:
            for dz in [999, 0.20, 0.04]:
                m = _meta(grasp_close_intent=False, grasp_bowl_z_delta=z_delta,
                          grasp_bowl_plate_dxy=dxy, grasp_bowl_plate_dz=dz)
                blocked, _ = guard_should_block_open(m, _env_action(-1.0))
                assert not blocked, f"false positive: z_delta={z_delta} dxy={dxy} dz={dz}"
                blocked, _ = guard_should_block_open(m, _env_action(1.0))
                assert not blocked


# ── edge cases ──

def test_missing_metadata_defaults():
    blocked, reason = guard_should_block_open({}, _env_action(-1.0))
    assert not blocked
    assert reason == "no_close_intent"


def test_None_metadata_values():
    m = _meta(grasp_close_intent=True, grasp_bowl_z_delta=None,
              grasp_bowl_plate_dxy=None, grasp_bowl_plate_dz=None)
    blocked, reason = guard_should_block_open(m, _env_action(-1.0))
    assert not blocked  # None treated as 0.0 / 999.0


def test_mixed_true_false_with_zeros():
    """zero bowl_z_delta with close_intent=True should not trigger."""
    m = _meta(grasp_close_intent=True, grasp_bowl_z_delta=0.0)
    blocked, reason = guard_should_block_open(m, _env_action(-1.0))
    assert not blocked
    assert reason == "bowl_not_lifted"
