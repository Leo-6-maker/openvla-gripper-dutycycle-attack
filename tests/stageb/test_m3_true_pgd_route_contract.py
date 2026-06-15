from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gripper_attack.attack_adapter import OpenVLAVisualAttacker
from gripper_attack.execution_target import (
    TARGET_31744,
    TARGET_31744_EXECUTION_CLASS,
    classify_execution_token,
    native_action_token_ids,
    target_token_cw_loss_and_stats,
    target_token_logratio_loss_and_stats,
    validate_execution_target,
)
from gripper_attack.m3_controls import (
    project_and_cast_processor_values,
    project_processor_space,
    rand_seed_schedule,
    sample_processor_delta,
    select_best_surrogate_only,
)
from gripper_attack.route_contract import (
    RouteContractError,
    route_config_from_attack_config,
    validate_true_pgd_attack_result,
)
from gripper_attack.types import AttackResult
from stageb.diagnose_m3_true_pgd_fixed_frame import (
    actual_generated_arm_prefix,
    extract_exact_new_tokens,
    require_runner_uses_adv_inputs,
    validate_processed_argmax_matches_emitted,
)


VOCAB_EFF = TARGET_31744
ROW_VOCAB = TARGET_31744 + 1
TARGET_TOKEN = TARGET_31744
N_BINS = 3
ARM_TOKEN = 15
CLOSE_TOKEN = 20
PROMPT_TOKENS = [3, 29871]


class TinyProcessor:
    def __call__(self, text, image, return_tensors="pt"):
        return {
            "input_ids": torch.tensor([[3, 29871]], dtype=torch.long),
            "pixel_values": torch.tensor([[[[0.0]]]], dtype=torch.float32),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }


class TargetTokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(vocab_size=VOCAB_EFF),
            pad_to_multiple_of=0,
        )
        self.bin_centers = np.asarray([1.0, 0.0, -1.0], dtype=np.float32)

    def get_action_dim(self, unnorm_key):
        return 7

    def get_action_stats(self, unnorm_key):
        return {
            "q01": np.zeros(7, dtype=np.float32),
            "q99": np.ones(7, dtype=np.float32),
            "mask": np.ones(7, dtype=bool),
        }

    def forward(self, input_ids, pixel_values=None, labels=None, use_cache=False, return_dict=True, past_key_values=None):
        bsz, seq_len = input_ids.shape
        if pixel_values is None:
            if past_key_values is None:
                pixel_signal = self.anchor * 0.0
            else:
                pixel_signal = past_key_values[0]
        else:
            pixel_signal = pixel_values.float().mean()
        logits = pixel_values.mean() * torch.zeros(
            (bsz, seq_len, ROW_VOCAB), dtype=torch.float32, device=pixel_values.device
        ) if pixel_values is not None else pixel_signal * torch.zeros(
            (bsz, seq_len, ROW_VOCAB), dtype=torch.float32, device=input_ids.device
        )
        signal = pixel_signal * 4.0 + self.anchor * 0.0
        logits[0, -1, TARGET_TOKEN] = signal
        logits[0, -1, CLOSE_TOKEN] = 1.0
        logits[0, -1, 0] = 0.5
        for dim in range(6):
            row = -(7 - dim + 1)
            if abs(row) <= seq_len:
                logits[0, row, ARM_TOKEN] = 1.0
        past = (pixel_signal,) if use_cache else None
        return SimpleNamespace(logits=logits, loss=None, past_key_values=past)

    def generate(self, input_ids, pixel_values, max_new_tokens, do_sample=False, return_dict_in_generate=True, output_scores=False):
        toks = torch.tensor([[ARM_TOKEN] * int(max_new_tokens)], dtype=torch.long, device=input_ids.device)
        return SimpleNamespace(sequences=torch.cat([input_ids, toks], dim=1))


def strict_config(**overrides):
    cfg = {
        "method": "token_prefix_pgd",
        "strict_route": True,
        "allow_fallback": False,
        "objective": "autoregressive_prefix_gripper_target_token_cw_v1",
        "target_token_id": TARGET_TOKEN,
        "target_execution_class": "CLIP_MEDIATED_OPEN",
        "epsilon": 0.25,
        "step_size": 0.25,
        "num_steps": 1,
        "random_start": False,
        "gripper_margin": 0.5,
    }
    cfg.update(overrides)
    return {"attack_optimizer": cfg}


def make_attacker(**overrides):
    return OpenVLAVisualAttacker(
        model=TargetTokenModel(),
        processor=TinyProcessor(),
        config=strict_config(**overrides),
        seed=0,
        preprocess_kwargs={"libero_preprocess_backend": "none"},
        device="cpu",
    )


def _valid_debug(**updates):
    debug = {
        "strict_route": True,
        "allow_fallback": False,
        "fallback_used": False,
        "fallback_reason": None,
        "resolved_adapter_class": "TokenPrefixPGDAttacker",
        "requested_objective": "autoregressive_prefix_gripper_target_token_cw_v1",
        "resolved_objective": "autoregressive_prefix_gripper_target_token_cw_v1",
        "target_token_id": TARGET_31744,
        "target_execution_class": TARGET_31744_EXECUTION_CLASS,
        "adv_inputs": {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "pixel_values": torch.zeros((1, 1, 1, 1)),
        },
        "x_adv_is_none": True,
        "action_adv_is_none": True,
        "num_backwards": 1,
        "num_loss_forwards": 2,
        "pixel_space": "processor_pixel_values",
        "pixel_budget_adv_inputs_linf": 0.0,
    }
    debug.update(updates)
    return debug


def _valid_result(**updates):
    debug_updates = updates.pop("debug_updates", {})
    return AttackResult(
        x_adv=updates.pop("x_adv", None),
        action_adv=updates.pop("action_adv", None),
        attack_method=updates.pop("attack_method", "token_prefix_pgd_pixel_values_target_token_cw_v1"),
        directional_loss_available=updates.pop("directional_loss_available", True),
        epsilon=updates.pop("epsilon", 0.25),
        debug=_valid_debug(**debug_updates),
    )


def clean_generation(tokens=None):
    action = [ARM_TOKEN] * 6 + [CLOSE_TOKEN] if tokens is None else [int(x) for x in tokens]
    return SimpleNamespace(
        sequences=torch.tensor([PROMPT_TOKENS + action], dtype=torch.long)
    )


def test_strict_route_rejects_missing_method():
    with pytest.raises(RouteContractError, match="explicit method"):
        OpenVLAVisualAttacker(config={"attack_optimizer": {"strict_route": True, "allow_fallback": False}})


def test_strict_route_rejects_unknown_method():
    with pytest.raises(RouteContractError, match="unknown attack method"):
        OpenVLAVisualAttacker(config={"attack_optimizer": {"strict_route": True, "allow_fallback": False, "method": "dense"}})


def test_strict_route_resolves_token_prefix_pgd():
    attacker = make_attacker()
    assert attacker.resolved_adapter_class == "TokenPrefixPGDAttacker"
    assert attacker.route.strict_route is True
    assert attacker.route.allow_fallback is False


def test_strict_route_rejects_missing_target_token():
    with pytest.raises(RouteContractError, match="target_token_id"):
        make_attacker(target_token_id=None)


def test_strict_route_rejects_targeted_objective_without_target():
    attacker = make_attacker(objective="targeted_directional_ce")
    with pytest.raises(RouteContractError, match="target_action"):
        attacker.attack(np.zeros((2, 2, 3), dtype=np.uint8), "pick", None, None, clean_generation(), unnorm_key="libero_object")


def test_strict_route_disables_typeerror_retry():
    class TypeErrorAdapter:
        def attack(self, *args, **kwargs):
            raise TypeError("strict path should not retry")

    attacker = make_attacker()
    attacker.adapter = TypeErrorAdapter()
    with pytest.raises(TypeError, match="strict path"):
        attacker.attack(
            np.zeros((2, 2, 3), dtype=np.uint8),
            "pick",
            np.zeros(7, dtype=np.float32),
            np.zeros(7, dtype=np.float32),
            None,
            unnorm_key="libero_object",
        )


def test_strict_route_rejects_existing_dense_adapter():
    cfg = route_config_from_attack_config({"attack_optimizer": {"method": "visual_linf_noise_adapter", "strict_route": True, "allow_fallback": False}})
    assert cfg.strict_route is True
    with pytest.raises(RouteContractError):
        OpenVLAVisualAttacker(config={"attack_optimizer": {"method": "visual_linf_noise_adapter", "strict_route": True, "allow_fallback": False}})


def test_true_pgd_result_contract_rejects_bad_results():
    route = route_config_from_attack_config(strict_config()["attack_optimizer"])
    with pytest.raises(RouteContractError, match="adv_inputs"):
        validate_true_pgd_attack_result(_valid_result(debug_updates={"adv_inputs": None}), route)
    with pytest.raises(RouteContractError, match="x_adv"):
        validate_true_pgd_attack_result(
            _valid_result(x_adv=np.zeros((1,))),
            route,
        )


def test_contract_rejects_wrong_attack_method():
    route = route_config_from_attack_config(strict_config()["attack_optimizer"])
    with pytest.raises(RouteContractError, match="attack_method"):
        validate_true_pgd_attack_result(_valid_result(attack_method="visual_linf_noise_adapter"), route)


def test_contract_rejects_directional_loss_false():
    route = route_config_from_attack_config(strict_config()["attack_optimizer"])
    with pytest.raises(RouteContractError, match="directional_loss"):
        validate_true_pgd_attack_result(_valid_result(directional_loss_available=False), route)


def test_contract_rejects_wrong_objective():
    route = route_config_from_attack_config(strict_config()["attack_optimizer"])
    with pytest.raises(RouteContractError, match="wrong resolved objective|requested_objective"):
        validate_true_pgd_attack_result(
            _valid_result(debug_updates={"resolved_objective": "prefix_locked_gripper_open_margin"}),
            route,
        )


def test_contract_accepts_logratio_v2_target_objective():
    route = route_config_from_attack_config(
        strict_config(objective="autoregressive_prefix_gripper_target_token_logratio_v2")["attack_optimizer"]
    )
    validate_true_pgd_attack_result(
        _valid_result(
            attack_method="token_prefix_pgd_pixel_values_target_token_logratio_v2",
            debug_updates={
                "requested_objective": "autoregressive_prefix_gripper_target_token_logratio_v2",
                "resolved_objective": "autoregressive_prefix_gripper_target_token_logratio_v2",
            },
        ),
        route,
    )


def test_contract_accepts_logratio_arm_v3_target_objective():
    route = route_config_from_attack_config(
        strict_config(objective="autoregressive_prefix_gripper_target_token_logratio_arm_v3")["attack_optimizer"]
    )
    validate_true_pgd_attack_result(
        _valid_result(
            attack_method="token_prefix_pgd_pixel_values_target_token_logratio_arm_v3",
            debug_updates={
                "requested_objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
                "resolved_objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
            },
        ),
        route,
    )


def test_contract_rejects_wrong_target_token():
    route = route_config_from_attack_config(strict_config()["attack_optimizer"])
    with pytest.raises(RouteContractError, match="target_token_id"):
        validate_true_pgd_attack_result(_valid_result(debug_updates={"target_token_id": 123}), route)


def test_contract_rejects_missing_adv_input_keys():
    route = route_config_from_attack_config(strict_config()["attack_optimizer"])
    with pytest.raises(RouteContractError, match="adv_inputs missing keys"):
        validate_true_pgd_attack_result(
            _valid_result(debug_updates={"adv_inputs": {"pixel_values": torch.zeros((1, 1, 1, 1))}}),
            route,
        )


def test_contract_rejects_action_adv():
    route = route_config_from_attack_config(strict_config()["attack_optimizer"])
    with pytest.raises(RouteContractError, match="action_adv"):
        validate_true_pgd_attack_result(_valid_result(action_adv=np.zeros(7)), route)


def test_contract_rejects_insufficient_loss_forwards():
    route = route_config_from_attack_config(strict_config()["attack_optimizer"])
    with pytest.raises(RouteContractError, match="num_loss_forwards"):
        validate_true_pgd_attack_result(_valid_result(debug_updates={"num_loss_forwards": 1}), route)


def test_true_pgd_target_token_objective_improves_mock_margin_and_records_contract():
    attacker = make_attacker(num_steps=1)
    result = attacker.attack(
        np.zeros((2, 2, 3), dtype=np.uint8),
        "pick object",
        np.zeros(7, dtype=np.float32),
        np.zeros(7, dtype=np.float32),
        clean_generation(),
        unnorm_key="libero_object",
    )
    debug = result.debug
    assert result.x_adv is None
    assert debug["fallback_used"] is False
    assert debug["adv_inputs_present"] is True
    assert debug["num_backwards"] == 1
    assert debug["target_token_id"] == TARGET_TOKEN
    assert debug["target_execution_class"] == "CLIP_MEDIATED_OPEN"
    assert debug["target_token_cw_margin_final"] > debug["target_token_cw_margin_initial"]
    assert debug["arm_preservation_as_acceptance_gate"] is True
    assert debug["arm_gate_reference"] == "clean_actual_generation"
    assert debug["clean_generated_arm_prefix_token_ids"] == [ARM_TOKEN] * 6
    assert debug["retokenized_clean_action_arm_token_ids"] != debug["clean_generated_arm_prefix_token_ids"]
    assert debug["generated_adv_arm_prefix_token_ids"] == [ARM_TOKEN] * 6
    assert debug["arm_prefix_match_count"] == 6
    assert debug["delta0_sha256"]
    assert debug["delta_final_sha256"]
    assert debug["delta0_processor_input_sha256"]
    assert debug["processor_input_sha256"]
    assert debug["pixel_budget_delta0_adv_inputs_linf"] <= attacker.adapter.epsilon + 1e-7
    assert len(debug["target_token_cw_loss_trajectory"]) == 1
    assert len(debug["target_token_cw_margin_trajectory"]) == 1
    assert len(debug["gradient_norm_trajectory"]) == 1
    assert debug["gradient_transform"] == "none"
    assert require_runner_uses_adv_inputs(result)["pixel_values"].shape == (1, 1, 1, 1)


def test_shuffled_gradient_control_records_transform_and_preserves_contract():
    attacker = make_attacker(gradient_transform="rademacher", gradient_transform_seed=123)
    result = attacker.attack(
        np.zeros((2, 2, 3), dtype=np.uint8),
        "pick object",
        np.zeros(7, dtype=np.float32),
        np.zeros(7, dtype=np.float32),
        clean_generation(),
        unnorm_key="libero_object",
    )
    debug = result.debug
    assert debug["gradient_transform"] == "rademacher"
    assert debug["gradient_transform_seed"] == 123
    assert debug["fallback_used"] is False
    assert debug["num_backwards"] == 1
    assert len(debug["gradient_norm_trajectory"]) == 1


def test_strict_target_token_requires_clean_generation():
    attacker = make_attacker(num_steps=1)
    with pytest.raises(RouteContractError, match="clean_model_output"):
        attacker.attack(
            np.zeros((2, 2, 3), dtype=np.uint8),
            "pick object",
            np.zeros(7, dtype=np.float32),
            np.zeros(7, dtype=np.float32),
            None,
            unnorm_key="libero_object",
        )


def test_arm_gate_uses_actual_clean_generated_prefix():
    attacker = make_attacker(num_steps=1)
    result = attacker.attack(
        np.zeros((2, 2, 3), dtype=np.uint8),
        "pick object",
        np.zeros(7, dtype=np.float32),
        np.zeros(7, dtype=np.float32),
        clean_generation([ARM_TOKEN, ARM_TOKEN, ARM_TOKEN, ARM_TOKEN, ARM_TOKEN, 99, CLOSE_TOKEN]),
        unnorm_key="libero_object",
    )
    debug = result.debug
    assert debug["clean_generated_arm_prefix_token_ids"] == [ARM_TOKEN, ARM_TOKEN, ARM_TOKEN, ARM_TOKEN, ARM_TOKEN, 99]
    assert debug["generated_adv_arm_prefix_token_ids"] == [ARM_TOKEN] * 6
    assert debug["arm_prefix_match_count"] == 5
    assert debug["arm_prefix_match_denominator"] == 6


def test_retokenized_prefix_mismatch_does_not_mask_arm_change():
    attacker = make_attacker(num_steps=1)
    result = attacker.attack(
        np.zeros((2, 2, 3), dtype=np.uint8),
        "pick object",
        np.zeros(7, dtype=np.float32),
        np.zeros(7, dtype=np.float32),
        clean_generation([99, 99, 99, 99, 99, 99, CLOSE_TOKEN]),
        unnorm_key="libero_object",
    )
    debug = result.debug
    assert debug["retokenized_arm_prefix_match_count"] == 0
    assert debug["arm_prefix_match_count"] == 0
    assert debug["arm_gate_reference"] == "clean_actual_generation"


def test_31744_clip_mediated_semantics_are_explicit():
    stats = {"q01": np.zeros(7, dtype=np.float32), "q99": np.ones(7, dtype=np.float32), "mask": np.ones(7, dtype=bool)}
    execution = validate_execution_target(
        token_id=TARGET_31744,
        expected_execution_class=TARGET_31744_EXECUTION_CLASS,
        vocab_eff=TARGET_31744,
        n_bins=3,
        bin_centers=np.asarray([1.0, 0.0, -1.0], dtype=np.float32),
        action_stats=stats,
    )
    assert execution.execution_class == "CLIP_MEDIATED_OPEN"
    assert TARGET_31744 not in native_action_token_ids(vocab_eff=TARGET_31744, n_bins=3)


def test_native_open_region_cannot_replace_31744():
    native = native_action_token_ids(vocab_eff=TARGET_31744, n_bins=256)
    assert TARGET_31744 not in native
    execution = classify_execution_token(
        TARGET_31744,
        vocab_eff=TARGET_31744,
        n_bins=3,
        bin_centers=np.asarray([1.0, 0.0, -1.0], dtype=np.float32),
        action_stats={"q01": np.zeros(7, dtype=np.float32), "q99": np.ones(7, dtype=np.float32), "mask": np.ones(7, dtype=bool)},
    )
    assert execution.clipped is True


def test_target_token_cw_gradient_improves_mock_margin():
    target_signal = torch.tensor(0.0, requires_grad=True)
    row = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    row = row.clone()
    row[2] = target_signal
    loss, stats = target_token_cw_loss_and_stats(row, target_token_id=2, margin=0.5)
    loss.backward()
    assert stats["best_competitor_token_id"] == 3
    assert target_signal.grad is not None
    assert float(target_signal.grad) < 0.0
    with torch.no_grad():
        improved_signal = target_signal - 0.25 * target_signal.grad.sign()
        improved_row = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
        improved_row = improved_row.clone()
        improved_row[2] = improved_signal
        _, improved_stats = target_token_cw_loss_and_stats(improved_row, target_token_id=2, margin=0.5)
    assert improved_stats["target_minus_best_competitor_margin"] > stats["target_minus_best_competitor_margin"]


def test_target_token_logratio_has_no_zero_loss_plateau():
    target_signal = torch.tensor(10.0, requires_grad=True)
    row = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    row = row.clone()
    row[2] = target_signal
    loss, stats = target_token_logratio_loss_and_stats(row, target_token_id=2)
    loss.backward()
    assert float(loss.detach()) < 0.0
    assert stats["target_objective_margin_name"] == "target_minus_competitor_logsumexp_margin"
    assert target_signal.grad is not None
    assert float(target_signal.grad) == pytest.approx(-1.0)


def test_logratio_v2_requires_cached_surrogate_path():
    attacker = make_attacker(
        objective="autoregressive_prefix_gripper_target_token_logratio_v2",
        surrogate_score_path="uncached_full_context_v1",
    )
    with pytest.raises(RouteContractError, match="requires cached_autoregressive_generate_v1"):
        attacker.attack(
            np.zeros((2, 2, 3), dtype=np.uint8),
            "pick object",
            np.zeros(7, dtype=np.float32),
            np.zeros(7, dtype=np.float32),
            clean_generation(),
            unnorm_key="libero_object",
        )


def test_logratio_arm_v3_requires_cached_surrogate_path():
    attacker = make_attacker(
        objective="autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        surrogate_score_path="uncached_full_context_v1",
    )
    with pytest.raises(RouteContractError, match="requires cached_autoregressive_generate_v1"):
        attacker.attack(
            np.zeros((2, 2, 3), dtype=np.uint8),
            "pick object",
            np.zeros(7, dtype=np.float32),
            np.zeros(7, dtype=np.float32),
            clean_generation(),
            unnorm_key="libero_object",
        )


def test_true_pgd_logratio_v2_mock_records_non_saturating_margin():
    attacker = make_attacker(
        objective="autoregressive_prefix_gripper_target_token_logratio_v2",
        surrogate_score_path="cached_autoregressive_generate_v1",
        num_steps=1,
    )
    result = attacker.attack(
        np.zeros((2, 2, 3), dtype=np.uint8),
        "pick object",
        np.zeros(7, dtype=np.float32),
        np.zeros(7, dtype=np.float32),
        clean_generation(),
        unnorm_key="libero_object",
    )
    debug = result.debug
    assert result.attack_method == "token_prefix_pgd_pixel_values_target_token_logratio_v2"
    assert debug["requested_objective"] == "autoregressive_prefix_gripper_target_token_logratio_v2"
    assert debug["autoregressive_prefix_target_token_logratio_loss"] is True
    assert debug["autoregressive_prefix_target_token_cw_loss"] is False
    assert debug["surrogate_score_path"] == "cached_autoregressive_generate_v1"
    assert debug["target_token_logratio_margin_final"] > debug["target_token_logratio_margin_initial"]
    assert debug["target_token_objective_margin_name"] == "target_minus_competitor_logsumexp_margin"
    assert len(debug["target_token_logratio_margin_trajectory"]) == 1


def test_true_pgd_logratio_arm_v3_records_combined_arm_penalty():
    attacker = make_attacker(
        objective="autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        surrogate_score_path="cached_autoregressive_generate_v1",
        arm_preserve_weight=0.5,
        num_steps=1,
    )
    result = attacker.attack(
        np.zeros((2, 2, 3), dtype=np.uint8),
        "pick object",
        np.zeros(7, dtype=np.float32),
        np.zeros(7, dtype=np.float32),
        clean_generation(),
        unnorm_key="libero_object",
    )
    debug = result.debug
    assert result.attack_method == "token_prefix_pgd_pixel_values_target_token_logratio_arm_v3"
    assert debug["requested_objective"] == "autoregressive_prefix_gripper_target_token_logratio_arm_v3"
    assert debug["autoregressive_prefix_target_token_logratio_loss"] is True
    assert debug["autoregressive_prefix_target_token_logratio_arm_loss"] is True
    assert debug["arm_preservation_role"] == "combined_gradient_penalty"
    assert debug["arm_preserve_weight"] == 0.5
    assert debug["target_token_logratio_margin_final"] > debug["target_token_logratio_margin_initial"]
    assert debug["target_token_arm_preservation_loss_final"] is not None
    assert len(debug["target_token_arm_preservation_loss_trajectory"]) == 1
    assert debug["arm_gate_reference"] == "clean_actual_generation"
    assert debug["trajectory_candidate_count"] == 2
    assert len(debug["trajectory_candidate_inputs"]) == 2
    assert debug["trajectory_candidate_inputs"][0]["candidate_index"] == 0
    assert debug["trajectory_candidate_inputs"][0]["candidate_source"] == "delta0"
    assert debug["trajectory_candidate_inputs"][1]["candidate_index"] == 1
    assert debug["trajectory_candidate_inputs"][1]["candidate_source"] == "pgd_iteration"
    assert debug["trajectory_candidate_inputs"][0]["delta_sha256"]
    assert debug["trajectory_candidate_inputs"][1]["processor_input_sha256"]


def test_rand20_controls_are_reproducible_and_processor_space():
    s1 = rand_seed_schedule(7, count=20)
    s2 = rand_seed_schedule(7, count=20)
    assert s1 == s2
    x = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    d1 = sample_processor_delta(x.shape, epsilon=0.1, seed=s1[0], dtype=x.dtype)
    d2 = sample_processor_delta(x.shape, epsilon=0.1, seed=s1[0], dtype=x.dtype)
    assert torch.equal(d1, d2)
    adv = project_processor_space(x, d1, epsilon=0.1)
    assert float((adv - x).abs().max()) <= 0.1 + 1e-7
    assert select_best_surrogate_only([0, 1, 2], [0.0, 3.0, 2.0]) == 1


def test_rand20_bf16_budget_after_cast():
    x = torch.tensor([[[[0.25, -0.25]]]], dtype=torch.bfloat16)
    delta = torch.tensor([[[[0.015, -0.015]]]], dtype=torch.float32)
    adv, corrections = project_and_cast_processor_values(x, delta, epsilon=0.01, candidate_is_delta=True)
    assert adv.dtype == torch.bfloat16
    assert int(corrections) >= 0
    assert float((adv.float() - x.float()).abs().max()) <= 0.01 + 1e-7


def test_rand20_fp16_budget_after_cast():
    x = torch.tensor([[[[0.25, -0.25]]]], dtype=torch.float16)
    delta = torch.tensor([[[[0.015, -0.015]]]], dtype=torch.float32)
    adv, corrections = project_and_cast_processor_values(x, delta, epsilon=0.01, candidate_is_delta=True)
    assert adv.dtype == torch.float16
    assert int(corrections) >= 0
    assert float((adv.float() - x.float()).abs().max()) <= 0.01 + 1e-7


def test_pgd_and_rand_share_exact_projection_helper():
    attacker = make_attacker(num_steps=1, epsilon=0.01)
    x = torch.tensor([[[[0.25, -0.25]]]], dtype=torch.bfloat16)
    delta = torch.tensor([[[[0.015, -0.015]]]], dtype=torch.float32)
    shared, correction_count = project_and_cast_processor_values(x, x.float() + delta, epsilon=0.01, candidate_is_delta=False)
    adapter_cast = attacker.adapter._cast_projected_pixel_values(x.float() + delta, x)
    adapter_corrections = attacker.adapter._count_quantized_budget_corrections(adapter_cast, x.float() + delta, x)
    assert torch.equal(shared, adapter_cast)
    assert int(correction_count) == int(adapter_corrections)


def test_official_generation_helpers_validate_exact_tokens_and_ties():
    seq = torch.tensor([[3, 29871, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    toks = extract_exact_new_tokens(seq, 2, expected_len=7)
    assert toks == [1, 2, 3, 4, 5, 6, 7]
    assert actual_generated_arm_prefix(toks) == [1, 2, 3, 4, 5, 6]
    row = torch.zeros((8,), dtype=torch.float32)
    row[7] = 1.0
    row[6] = 1.0
    strict = validate_processed_argmax_matches_emitted(row, 7, tolerance=0.0)
    assert strict["tie_aware_pass"] is True
    assert strict["strict_argmax_match"] is False
