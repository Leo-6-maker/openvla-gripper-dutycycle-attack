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
    validate_execution_target,
)
from gripper_attack.m3_controls import (
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


VOCAB_EFF = 21
ROW_VOCAB = 22
TARGET_TOKEN = 21
N_BINS = 3
ARM_TOKEN = 15
CLOSE_TOKEN = 20


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

    def forward(self, input_ids, pixel_values, labels=None, use_cache=False, return_dict=True):
        bsz, seq_len = input_ids.shape
        logits = pixel_values.mean() * torch.zeros(
            (bsz, seq_len, ROW_VOCAB), dtype=torch.float32, device=pixel_values.device
        )
        signal = pixel_values.float().mean() * 4.0 + self.anchor * 0.0
        logits[0, -1, TARGET_TOKEN] = signal
        logits[0, -1, CLOSE_TOKEN] = 1.0
        logits[0, -1, 0] = 0.5
        for dim in range(6):
            row = -(7 - dim + 1)
            if abs(row) <= seq_len:
                logits[0, row, ARM_TOKEN] = 1.0
        return SimpleNamespace(logits=logits, loss=None)

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
        attacker.attack(np.zeros((2, 2, 3), dtype=np.uint8), "pick", None, None, None, unnorm_key="libero_object")


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
        validate_true_pgd_attack_result(AttackResult(debug={"fallback_used": False, "resolved_adapter_class": "TokenPrefixPGDAttacker", "num_backwards": 1}), route)
    with pytest.raises(RouteContractError, match="x_adv"):
        validate_true_pgd_attack_result(
            AttackResult(x_adv=np.zeros((1,)), epsilon=1.0, debug={"fallback_used": False, "resolved_adapter_class": "TokenPrefixPGDAttacker", "num_backwards": 1, "adv_inputs": {"input_ids": torch.ones((1, 1), dtype=torch.long), "pixel_values": torch.zeros((1, 1, 1, 1))}}),
            route,
        )


def test_true_pgd_target_token_objective_improves_mock_margin_and_records_contract():
    attacker = make_attacker(num_steps=1)
    result = attacker.attack(
        np.zeros((2, 2, 3), dtype=np.uint8),
        "pick object",
        np.zeros(7, dtype=np.float32),
        np.zeros(7, dtype=np.float32),
        None,
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
    assert require_runner_uses_adv_inputs(result)["pixel_values"].shape == (1, 1, 1, 1)


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
