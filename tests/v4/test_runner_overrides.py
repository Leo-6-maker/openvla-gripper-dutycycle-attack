from types import SimpleNamespace

import numpy as np

from v4_run_eval_openvla import (
    apply_attack_objective_override,
    build_target_action_for_objective,
    effective_attack_objective,
)


def test_apply_attack_objective_override_sets_paper_params():
    args = SimpleNamespace(
        attack_objective="gripper_logit_margin_cw",
        epsilon=0.10,
        step_size=0.020,
        attack_steps=20,
        temporal_init="prev_delta",
        cw_margin=5.0,
    )
    cfg = {"attack_optimizer": {"objective": "targeted_directional_ce"}}
    assert apply_attack_objective_override(args, cfg) == "gripper_logit_margin_cw"
    opt = cfg["attack_optimizer"]
    assert opt["epsilon"] == 0.10
    assert opt["step_size"] == 0.020
    assert opt["num_steps"] == 20
    assert opt["temporal_init"] == "prev_delta"
    assert opt["cw_margin"] == 5.0


def test_constant_delta_pregrasp_changes_gripper_only(monkeypatch):
    monkeypatch.setenv("V4_CONSTANT_DELTA_GRIPPER", "-1.0")
    args = SimpleNamespace(attack_objective="constant_delta_pregrasp")
    clean = np.array([0.1, 0.2, 0.3, 0.4, -0.1, 0.2, 0.5], dtype=np.float32)
    target = build_target_action_for_objective(clean, None, {}, args)
    np.testing.assert_allclose(target[:-1], clean[:-1])
    assert np.isclose(target[-1], -0.5)


def test_effective_attack_objective_reads_env_fallback(monkeypatch):
    monkeypatch.setenv("V4_ATTACK_OBJECTIVE", "untargeted_arm_clean_token_ce")
    args = SimpleNamespace(attack_objective="")
    cfg = {"attack_optimizer": {"objective": "targeted_directional_ce"}}
    assert effective_attack_objective(args, cfg) == "untargeted_arm_clean_token_ce"
