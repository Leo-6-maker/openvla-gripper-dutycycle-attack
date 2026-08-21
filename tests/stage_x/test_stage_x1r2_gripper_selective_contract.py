from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import torch

from gripper_attack.attack_adapter import RouteContractError, TokenPrefixPGDAttacker
from gripper_attack.execution_target import native_open_logratio_loss_and_stats
from stage_x.run_stage_x1r2_f1b_dev import build_attack
from stage_x.run_stage_x1r_primary_matrix import audit_direct_action_tokens


ROOT = Path(__file__).resolve().parents[2]


def test_direct_action_audit_reports_only_arm_mismatch():
    audit = audit_direct_action_tokens([1, 2, 3, 4, 5, 6, 10], [1, 2, 3, 9, 5, 6, 20])
    assert audit["arm_token_ids_equal"] is False
    assert audit["arm_mismatch_dimensions"] == [3]
    assert audit["gripper_token_changed"] is True


def test_strict_candidate_audit_requires_clean_to_open_transition():
    attacker = object.__new__(TokenPrefixPGDAttacker)

    def fake_generate(_prompt, pixel_values, *, prefix_len):
        candidate_index = int(pixel_values.item())
        if candidate_index == 0:
            return torch.tensor([1, 2, 3, 4, 5, 6, 10])
        if candidate_index == 1:
            return torch.tensor([1, 2, 3, 9, 5, 6, 20])
        return torch.tensor([1, 2, 3, 4, 5, 6, 20])

    attacker._generate_action_prefix_tokens = fake_generate
    candidates = [
        {"candidate_index": index, "candidate_source": "test", "pixel_values": torch.tensor([index])}
        for index in range(4)
    ]
    selected, audit = attacker._select_strict_arm_candidate(
        torch.tensor([[101]]), candidates, torch.tensor([1, 2, 3, 4, 5, 6, 10]), torch.tensor([20])
    )
    assert selected["candidate_index"] == 2
    assert len(audit) == 4
    assert audit[0]["clean_gripper_is_native_open"] is False
    assert audit[0]["direct_generated_gripper_is_native_open"] is False
    assert audit[2]["arm_token_ids_equal"] is True
    assert audit[2]["gripper_token_changed"] is True
    assert audit[3]["candidate_index"] == 3


def test_strict_candidate_audit_fails_closed_without_selective_candidate():
    attacker = object.__new__(TokenPrefixPGDAttacker)
    attacker._generate_action_prefix_tokens = lambda _prompt, _pixel_values, *, prefix_len: torch.tensor([1, 2, 3, 4, 5, 6, 10])
    try:
        attacker._select_strict_arm_candidate(
            torch.tensor([[101]]),
            [{"candidate_index": 0, "candidate_source": "test", "pixel_values": torch.tensor([0])}],
            torch.tensor([1, 2, 3, 4, 5, 6, 10]),
            torch.tensor([20]),
        )
    except RouteContractError as exc:
        assert str(exc) == "STRUCTURAL_INVALID_NO_SELECTIVE_CANDIDATE"
    else:
        raise AssertionError("non-selective candidate was accepted")


def test_strict_candidate_audit_fails_closed_without_native_open_set():
    attacker = object.__new__(TokenPrefixPGDAttacker)
    attacker._generate_action_prefix_tokens = lambda _prompt, _pixel_values, *, prefix_len: torch.tensor([1, 2, 3, 4, 5, 6, 20])
    try:
        attacker._select_strict_arm_candidate(
            torch.tensor([[101]]),
            [{"candidate_index": 0, "candidate_source": "test", "pixel_values": torch.tensor([0])}],
            torch.tensor([1, 2, 3, 4, 5, 6, 10]),
            None,
        )
    except RouteContractError as exc:
        assert str(exc) == "STRICT_CANDIDATE_AUDIT_OPEN_TOKEN_SET_MISSING"
    else:
        raise AssertionError("missing native OPEN set was not rejected")


def test_contract_freezes_direct_generation_and_transition_gates():
    path = ROOT / "configs/STAGE_X_X1R2_GRIPPER_SELECTIVE_ATTACK_CONTRACT_V1.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    assert contract["scientific_authority"] is False
    assert contract["definition"]["direct_actuator_edit"] is False
    assert contract["definition"]["action_token_count"] == 7
    assert "clean direct-generated gripper execution class != NATIVE_OPEN" in contract["required_gates_before_attacked_env_step"]
    assert "adversarial direct-generated gripper execution class = NATIVE_OPEN" in contract["required_gates_before_attacked_env_step"]
    assert contract["repair_candidate_policy"]["name"] == "STRICT_CANDIDATE_AUDIT_V1"


def test_native_open_set_objective_uses_all_open_tokens_and_excludes_them_from_competitors():
    row = torch.tensor([1.0, 5.0, 2.0, 4.0])
    loss, stats = native_open_logratio_loss_and_stats(row, open_token_ids=[1, 3])
    expected = torch.logsumexp(torch.tensor([1.0, 2.0]), 0) - torch.logsumexp(torch.tensor([5.0, 4.0]), 0)
    assert torch.allclose(loss, expected)
    assert stats["native_open_token_count"] == 2
    assert stats["non_open_competitor_count"] == 2
    assert stats["best_native_open_token"] == 1
    assert stats["best_non_open_token"] == 2


def test_build_attack_accepts_f1c_single_method_protocol():
    protocol = json.loads((ROOT / "configs/STAGE_X_X1R2_F1C_METHOD_FREEZE_T5_CANARY_PROTOCOL_V3.json").read_text(encoding="utf-8"))
    with patch("gripper_attack.attack_adapter.OpenVLAVisualAttacker") as constructor:
        build_attack("M1", 10, 7, None, None, "cpu", protocol, temporal_init="prev_delta")
    optimizer = constructor.call_args.args[2]["attack_optimizer"]
    assert optimizer["objective"] == protocol["method"]["objective"]
    assert optimizer["epsilon"] == protocol["method"]["epsilon_processor_pixel_values"]
    assert optimizer["step_size"] == protocol["method"]["step_size"]
    assert optimizer["temporal_init"] == "prev_delta"
