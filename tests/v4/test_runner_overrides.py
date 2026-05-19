from types import SimpleNamespace

import numpy as np

from v4_run_eval_openvla import (
    action_clamp_audit,
    apply_action_clamp,
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


def test_action_clamp_gripper_clean_replaces_only_gripper():
    attacked = np.array([0.2, -0.3, 0.4, 0.1, 0.2, -0.1, -1.0], dtype=np.float32)
    clean = np.array([0.7, 0.8, 0.9, -0.4, -0.5, 0.6, 1.0], dtype=np.float32)
    clamped, before = apply_action_clamp(attacked, clean, "gripper_clean")
    np.testing.assert_allclose(before, attacked)
    np.testing.assert_allclose(clamped[:-1], attacked[:-1])
    assert np.isclose(clamped[-1], clean[-1])
    audit = action_clamp_audit("gripper_clean", before, clamped, clean)
    assert audit["action_clamp_mode"] == "gripper_clean"
    assert np.isclose(audit["gripper_delta_env"], 0.0)
    assert audit["arm_delta_l2"] > 0.0


def test_action_clamp_arm_clean_replaces_only_arm():
    attacked = np.array([0.2, -0.3, 0.4, 0.1, 0.2, -0.1, -1.0], dtype=np.float32)
    clean = np.array([0.7, 0.8, 0.9, -0.4, -0.5, 0.6, 1.0], dtype=np.float32)
    clamped, before = apply_action_clamp(attacked, clean, "arm_clean")
    np.testing.assert_allclose(before, attacked)
    np.testing.assert_allclose(clamped[:-1], clean[:-1])
    assert np.isclose(clamped[-1], attacked[-1])
    audit = action_clamp_audit("arm_clean", before, clamped, clean)
    assert audit["action_clamp_mode"] == "arm_clean"
    assert np.isclose(audit["arm_delta_l2"], 0.0)
    assert np.isclose(audit["gripper_delta_env"], -2.0)
