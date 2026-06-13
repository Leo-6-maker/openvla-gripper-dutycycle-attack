from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from gripper_attack.attack_adapter import TokenPrefixPGDAttacker, action_token_logit_row_index
from gripper_attack.openvla_libero_exec_spec import (
    raw_gripper_is_open,
    raw_gripper_is_close,
    raw_gripper_is_boundary,
    raw_gripper_to_env_gripper,
)


VOCAB = 20
CLOSE_TOKEN = 19  # disc 0 -> raw 0.0
BOUNDARY_TOKEN = 18  # disc 1 -> raw 0.5
OPEN_TOKEN = 17  # disc 2 -> raw 1.0
CLEAN_ARM_TOKEN = 15
GENERATED_ARM_TOKEN = 14


class TinyProcessor:
    def __call__(self, text, image, return_tensors="pt"):
        return {
            "input_ids": torch.tensor([[3, 29871]], dtype=torch.long),
            "pixel_values": torch.tensor([[[[0.0]]]], dtype=torch.float32),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }


class TinyModel(torch.nn.Module):
    def __init__(self, generated_prefix=None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(vocab_size=VOCAB),
            pad_to_multiple_of=0,
        )
        self.bin_centers = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
        self.generated_prefix = list(generated_prefix or [GENERATED_ARM_TOKEN] * 6)
        self.generate_calls = 0

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
            (bsz, seq_len, VOCAB), dtype=torch.float32, device=pixel_values.device
        )
        image_signal = pixel_values.float().mean() * 4.0 + self.anchor * 0.0

        # Teacher-forced v2 context: prompt + 7 target tokens, row -2 predicts gripper.
        if seq_len >= 9 and int(input_ids[0, -2]) == CLEAN_ARM_TOKEN:
            row = seq_len - 2
            logits[0, row, OPEN_TOKEN] = 5.0 + image_signal
            logits[0, row, CLOSE_TOKEN] = 0.0

        # Generated-prefix v3 context: prompt + generated first-six tokens, row -1 predicts gripper.
        if seq_len >= 8 and int(input_ids[0, -1]) == GENERATED_ARM_TOKEN:
            row = seq_len - 1
            logits[0, row, OPEN_TOKEN] = image_signal
            logits[0, row, CLOSE_TOKEN] = 2.0

        # Arm preservation rows for dims 0..5; nonzero CE if target distributions differ.
        for dim in range(6):
            row = action_token_logit_row_index(dim, 7)
            if abs(row) <= seq_len:
                logits[0, row, CLEAN_ARM_TOKEN] = 1.0 + image_signal * 0.01
                logits[0, row, GENERATED_ARM_TOKEN] = -1.0
        return SimpleNamespace(logits=logits, loss=None)

    def generate(self, input_ids, pixel_values, max_new_tokens, do_sample=False, return_dict_in_generate=True, output_scores=False):
        self.generate_calls += 1
        toks = torch.tensor([self.generated_prefix[: int(max_new_tokens)]], dtype=torch.long, device=input_ids.device)
        return SimpleNamespace(sequences=torch.cat([input_ids, toks], dim=1))


def make_attacker(num_steps=1, epsilon=0.25, step_size=0.25, prefix_refresh_interval=1):
    return TokenPrefixPGDAttacker(
        TinyModel(),
        TinyProcessor(),
        {
            "attack_optimizer": {
                "objective": "autoregressive_prefix_gripper_open_execspec_v3",
                "epsilon": epsilon,
                "step_size": step_size,
                "num_steps": num_steps,
                "random_start": False,
                "arm_preserve_weight": 0.0,
                "gripper_margin": 1.0,
                "prefix_refresh_interval": prefix_refresh_interval,
            }
        },
        seed=0,
        preprocess_kwargs={"libero_preprocess_backend": "none"},
        device="cpu",
    )


def test_action_position_indexing_for_layer3():
    assert action_token_logit_row_index(0, 7) == -8
    assert action_token_logit_row_index(5, 7) == -3
    assert action_token_logit_row_index(6, 7) == -2

    attacker = make_attacker()
    target_ids = torch.tensor([CLEAN_ARM_TOKEN] * 6 + [CLOSE_TOKEN], dtype=torch.long)
    clean_ids, full_ids, labels, _ = attacker._build_inputs_and_labels(
        np.zeros((2, 2, 3), dtype=np.uint8), "pick object", target_ids
    )
    rows = attacker._active_label_rows(
        torch.zeros((1, full_ids.shape[1], VOCAB)),
        labels,
        7,
    )
    assert [r[2] for r in rows] == list(range(7))
    assert rows[-1][3] == -2
    assert full_ids.shape[1] == clean_ids.shape[1] + 7


def test_teacher_forced_and_generated_prefix_conditioning_are_distinct():
    attacker = make_attacker(num_steps=1, epsilon=0.0, step_size=0.0)
    target_ids = torch.tensor([CLEAN_ARM_TOKEN] * 6 + [CLOSE_TOKEN], dtype=torch.long)
    clean_ids, full_ids, labels, x0 = attacker._build_inputs_and_labels(
        np.zeros((2, 2, 3), dtype=np.uint8), "pick object", target_ids
    )
    region = attacker.get_gripper_region_by_decoded_action("libero_object", postprocess_gripper=True)
    teacher = attacker._teacher_forced_gripper_margin_stats(
        full_ids, x0, 7, region["open_token_ids"], region["close_token_ids"]
    )
    generated_prefix = attacker._generate_action_prefix_tokens(clean_ids, x0, prefix_len=6)
    loss, generated = attacker._generated_prefix_gripper_loss_and_stats(
        clean_ids,
        generated_prefix,
        x0,
        region["open_token_ids"],
        region["close_token_ids"],
        margin=1.0,
    )

    assert teacher["conditioning"] == "teacher_forced_target_arm_prefix"
    assert generated["conditioning"] == "generated_arm_prefix_stop_gradient"
    assert teacher["open_minus_close_margin"] > 0
    assert generated["open_minus_close_margin"] < 0
    assert float(loss.detach()) > 0


def test_v3_gradient_direction_projection_and_no_action_replacement():
    attacker = make_attacker(num_steps=1, epsilon=0.25, step_size=0.25)
    result = attacker.attack(
        np.zeros((2, 2, 3), dtype=np.uint8),
        "pick object",
        np.asarray([0.0] * 7, dtype=np.float32),
        np.asarray([0.0] * 7, dtype=np.float32),
        None,
        unnorm_key="libero_object",
    )
    debug = result.debug
    assert result.action_adv is None
    assert result.x_adv is None
    assert result.attack_method == "token_prefix_pgd_pixel_values_autoregressive_prefix_v3"
    assert debug["selected_loss_final"] < debug["selected_loss_initial"]
    assert debug["pixel_budget_master_linf"] <= 0.25 + 1e-7
    assert debug["pixel_budget_adv_inputs_linf"] <= 0.25 + 1e-7
    assert debug["generated_prefix_stop_gradient"] is True
    assert debug["gradient_through_generated_token_ids"] is False


def test_open_close_boundary_regions_and_official_clip_parity():
    assert raw_gripper_is_open(1.0)
    assert raw_gripper_is_close(0.0)
    assert raw_gripper_is_boundary(0.5)

    attacker = make_attacker()
    region = attacker.get_gripper_region_by_decoded_action("libero_object", postprocess_gripper=True)
    assert int(OPEN_TOKEN) in [int(x) for x in region["open_token_ids"].cpu().tolist()]
    assert int(CLOSE_TOKEN) in [int(x) for x in region["close_token_ids"].cpu().tolist()]
    assert int(BOUNDARY_TOKEN) in [int(x) for x in region["boundary_token_ids"]]

    token_ids = np.asarray([VOCAB + 5, CLOSE_TOKEN, OPEN_TOKEN, -7], dtype=np.int64)
    disc_before = VOCAB - token_ids - 1
    disc_after = np.clip(disc_before, 0, 2)
    assert disc_after.tolist() == [0, 0, 2, 2]

    assert raw_gripper_to_env_gripper(1.0) == -1.0


def test_arm_preservation_indexing_and_prefix_refresh_count():
    attacker = make_attacker(num_steps=3, epsilon=0.0, step_size=0.0, prefix_refresh_interval=2)
    target_ids = torch.tensor([CLEAN_ARM_TOKEN] * 6 + [CLOSE_TOKEN], dtype=torch.long)
    _, full_ids, labels, x0 = attacker._build_inputs_and_labels(
        np.zeros((2, 2, 3), dtype=np.uint8), "pick object", target_ids
    )
    loss, stats = attacker._arm_preservation_loss_and_stats(
        full_ids, labels, x0, 7, arm_preserve_weight=1.0
    )
    assert stats["arm_preservation_dims"] == [0, 1, 2, 3, 4, 5]
    assert stats["arm_preservation_dim_count"] == 6
    assert float(loss.detach()) > 0

    result = attacker.attack(
        np.zeros((2, 2, 3), dtype=np.uint8),
        "pick object",
        np.asarray([0.0] * 7, dtype=np.float32),
        np.asarray([0.0] * 7, dtype=np.float32),
        None,
        unnorm_key="libero_object",
    )
    assert result.debug["prefix_refresh_count"] == 2
    assert result.debug["num_generation_forwards"] == 3
    assert result.debug["arm_preservation_debug_final"]["arm_preservation_dims"] == [0, 1, 2, 3, 4, 5]


def test_sequential_gradient_equivalence():
    """∇(L_gripper + L_arm) = ∇L_gripper + ∇L_arm with non-zero arm weight."""
    attacker = make_attacker(num_steps=1, epsilon=0.03, step_size=0.006,
                             prefix_refresh_interval=1)
    target_ids = torch.tensor([CLEAN_ARM_TOKEN] * 6 + [CLOSE_TOKEN], dtype=torch.long)
    clean_ids, full_ids, labels, x0 = attacker._build_inputs_and_labels(
        np.zeros((2, 2, 3), dtype=np.uint8), "pick object", target_ids)

    # Get region tokens
    region = attacker.get_gripper_region_by_decoded_action("libero_object")
    open_ids = region["open_token_ids"]
    close_ids = region["close_token_ids"]

    # Generate prefix on same image
    prefix = attacker._generate_action_prefix_tokens(clean_ids, x0, prefix_len=6)

    adv = x0.detach().clone().requires_grad_(True)
    adv_fp = attacker._cast_projected_pixel_values(adv, x0)

    # Joint gradient: L_gripper + L_arm together
    g_loss, _ = attacker._generated_prefix_gripper_loss_and_stats(
        clean_ids, prefix, adv_fp, open_ids, close_ids, margin=0.5)
    a_loss, _ = attacker._arm_preservation_loss_and_stats(
        full_ids, labels, adv_fp, 7, arm_preserve_weight=1.0)
    loss_joint = g_loss + a_loss
    grad_joint = torch.autograd.grad(loss_joint, adv, retain_graph=True, create_graph=False)[0]

    # Sequential gradients
    grad_g = torch.autograd.grad(g_loss, adv, retain_graph=False, create_graph=False)[0]
    # Rebuild forward for arm (graph was freed)
    adv_fp2 = attacker._cast_projected_pixel_values(adv, x0)
    a_loss2, _ = attacker._arm_preservation_loss_and_stats(
        full_ids, labels, adv_fp2, 7, arm_preserve_weight=1.0)
    grad_a = torch.autograd.grad(a_loss2, adv, retain_graph=False, create_graph=False)[0]
    grad_seq = grad_g + grad_a

    assert torch.allclose(grad_joint, grad_seq, atol=1e-6), \
        f"max diff: {(grad_joint - grad_seq).abs().max().item():.8f}"
    assert (grad_joint.sign() == grad_seq.sign()).all().item(), \
        "sign mismatch in sequential vs joint gradient"
