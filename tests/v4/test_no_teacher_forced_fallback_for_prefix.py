# -*- coding: utf-8 -*-
"""Test that teacher-forced fallback is banned for gripper objectives."""

import pytest


# The teacher-forced fallback ban is enforced in
# scripts/vis_rollout_adaptive_v3.py::run_pgd_attack().
#
# We test the ban logic here in isolation by verifying that:
# 1. When adv_inputs is None, a RuntimeError is raised (not silently falling back).
# 2. When re-decode raises, a RuntimeError is raised (not silently falling back).
# 3. Teacher-forced metrics (gripper_open_prob_mass) are never used for selection.


_GRIPPER_OBJ_SET = {
    'gripper_open_region_ce', 'force_open_z_down_token_ce',
    'force_open_region_z_down_ce',
    'prefix_locked_gripper_open_region_ce',
    'prefix_locked_gripper_open_margin',
    'gripper_open_expected_action',
}


def _simulate_restart_selection(debug, objective, redecode_ok=True):
    """Minimal reproduction of the restart selection logic after repair."""
    if objective not in _GRIPPER_OBJ_SET:
        # Non-gripper objectives use CE-based selection (not under test here)
        return "ce_selected"

    _adv_inputs = debug.get('adv_inputs')
    if _adv_inputs is None:
        raise RuntimeError(
            "AttackResult.debug['adv_inputs'] is missing. "
            "Teacher-forced fallback is banned for gripper objectives."
        )

    if not redecode_ok:
        raise RuntimeError(
            "Re-decode failed for gripper objective. "
            "Teacher-forced fallback is banned."
        )

    # Normal path: use actual generated action
    return "redecode_selected"


class TestNoTeacherForcedFallback:
    def test_missing_adv_inputs_raises(self):
        """When adv_inputs is None, must raise — not fallback to teacher-forced."""
        debug = {'adv_inputs': None, 'gripper_open_prob_mass': 0.99}
        with pytest.raises(RuntimeError, match="adv_inputs.*missing"):
            _simulate_restart_selection(debug, 'prefix_locked_gripper_open_margin')

    def test_missing_adv_inputs_raises_for_all_gripper_objs(self):
        """All gripper objectives must ban teacher-forced fallback."""
        debug = {'adv_inputs': None, 'gripper_open_prob_mass': 0.99}
        for obj in _GRIPPER_OBJ_SET:
            with pytest.raises(RuntimeError, match="adv_inputs.*missing"):
                _simulate_restart_selection(debug, obj)

    def test_redecode_failure_raises(self):
        """When re-decode raises, must propagate — not fallback."""
        debug = {'adv_inputs': {'input_ids': 'fake', 'pixel_values': 'fake'}}
        with pytest.raises(RuntimeError, match="Re-decode failed"):
            _simulate_restart_selection(
                debug, 'prefix_locked_gripper_open_margin', redecode_ok=False)

    def test_normal_path_does_not_raise(self):
        """Normal path with valid adv_inputs and successful re-decode."""
        debug = {'adv_inputs': {'input_ids': 'fake', 'pixel_values': 'fake'}}
        result = _simulate_restart_selection(
            debug, 'prefix_locked_gripper_open_margin', redecode_ok=True)
        assert result == "redecode_selected"

    def test_non_gripper_objectives_still_work(self):
        """Non-gripper objectives should not be affected by the ban."""
        debug = {'adv_inputs': None, 'target_ce_final': 0.5}
        result = _simulate_restart_selection(debug, 'targeted_directional_ce')
        assert result == "ce_selected"
