"""Lightweight TokenPrefixPGD interface tests.

These tests use mocks and do not load OpenVLA weights.
"""
import unittest
from unittest.mock import patch
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gripper_attack.attack_adapter import (
    TokenPrefixPGDAttacker,
    get_adv_inputs_from_attack_result,
)
from gripper_attack.types import AttackResult


class FakeProcessor:
    def __call__(self, prompt, image, return_tensors="pt"):
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "pixel_values": torch.ones((1, 3, 4, 4), dtype=torch.float32) * 0.5,
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }


class FakeModel(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((), dtype=dtype))


class TestTokenPrefixPGDInterface(unittest.TestCase):
    def _attacker_for_dtype(self, dtype):
        return TokenPrefixPGDAttacker(
            FakeModel(dtype),
            FakeProcessor(),
            {"attack_optimizer": {"num_steps": 1, "epsilon": 0.01, "step_size": 0.01}},
            device="cpu",
        )

    def test_build_inputs_uses_bfloat16_model_dtype(self):
        self._assert_pixel_dtype(torch.bfloat16)

    def test_build_inputs_uses_float16_model_dtype(self):
        self._assert_pixel_dtype(torch.float16)

    def test_build_inputs_uses_float32_model_dtype(self):
        self._assert_pixel_dtype(torch.float32)

    def _assert_pixel_dtype(self, dtype):
        attacker = self._attacker_for_dtype(dtype)
        with patch("gripper_attack.attack_adapter.prepare_openvla_image_for_attack", return_value=object()):
            _, _, _, pixel_values = attacker._build_inputs_and_labels(
                observation=object(),
                instruction="pick up the test object",
                target_token_ids=torch.tensor([10, 11], dtype=torch.long),
            )
        self.assertEqual(pixel_values.dtype, dtype)

    def test_attack_returns_adv_inputs_not_action_adv(self):
        attacker = self._attacker_for_dtype(torch.float32)
        attacker.action_to_token_ids = lambda action, unnorm_key: torch.tensor([10, 11], dtype=torch.long)
        attacker._build_inputs_and_labels = lambda obs, instr, toks: (
            torch.tensor([[1, 2, 29871]], dtype=torch.long),
            torch.tensor([[1, 2, 29871, 10, 11]], dtype=torch.long),
            torch.tensor([[-100, -100, -100, 10, 11]], dtype=torch.long),
            torch.full((1, 3, 4, 4), 0.5, dtype=torch.float32),
        )
        attacker._loss = lambda full_ids, labels, pixel_values, **kwargs: pixel_values.sum()
        attacker._audit_logits = lambda *args, **kwargs: {}

        result = attacker.attack(
            observation=object(),
            instruction="pick up the test object",
            target_action=[0.0, 0.0],
            unnorm_key="libero_object",
        )

        self.assertIsNone(result.action_adv)
        adv_inputs = get_adv_inputs_from_attack_result(result)
        self.assertIn("pixel_values", adv_inputs)
        self.assertIn("input_ids", adv_inputs)

    def test_helper_rejects_missing_adv_inputs(self):
        with self.assertRaisesRegex(ValueError, "adv_inputs"):
            get_adv_inputs_from_attack_result(AttackResult(action_adv=None, debug={}))

    def test_helper_rejects_incomplete_adv_inputs(self):
        result = AttackResult(action_adv=None, debug={"adv_inputs": {"pixel_values": torch.zeros(1)}})
        with self.assertRaisesRegex(ValueError, "input_ids"):
            get_adv_inputs_from_attack_result(result)


if __name__ == "__main__":
    unittest.main()
