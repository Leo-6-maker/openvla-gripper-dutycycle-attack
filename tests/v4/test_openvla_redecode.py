"""Lightweight OpenVLA re-decode helper tests.

These tests use fake models only; they do not load OpenVLA weights.
"""
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gripper_attack.openvla_redecode import (  # noqa: E402
    decode_openvla_generation_to_action,
    redecode_openvla_action_from_adv_inputs,
    validate_adv_inputs,
)


class FakeGeneration:
    def __init__(self, sequences):
        self.sequences = sequences


class FakeOpenVLAModel(torch.nn.Module):
    def __init__(self, dtype=torch.float32, action_dim=3):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((), dtype=dtype))
        self.action_dim = action_dim
        self.config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=32000), pad_to_multiple_of=0)
        self.bin_centers = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
        self.generated_tokens = torch.tensor([[31999, 31871, 31744]], dtype=torch.long)
        self.last_pixel_dtype = None

    def get_action_dim(self, unnorm_key):
        return self.action_dim

    def get_action_stats(self, unnorm_key):
        return {
            "q01": np.zeros(self.action_dim, dtype=np.float32),
            "q99": np.ones(self.action_dim, dtype=np.float32),
            "mask": np.ones(self.action_dim, dtype=bool),
        }

    def generate(self, **kwargs):
        self.last_pixel_dtype = kwargs["pixel_values"].dtype
        prefix = kwargs["input_ids"]
        tokens = self.generated_tokens.to(device=prefix.device, dtype=prefix.dtype)
        return FakeGeneration(torch.cat([prefix, tokens], dim=1))


class TestOpenVLARedecode(unittest.TestCase):
    def _adv_inputs(self, dtype=torch.float32):
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "pixel_values": torch.ones((1, 3, 4, 4), dtype=dtype),
        }

    def test_validate_rejects_missing_adv_inputs(self):
        with self.assertRaisesRegex(ValueError, "adv_inputs"):
            validate_adv_inputs(None)

    def test_validate_rejects_missing_pixel_values(self):
        with self.assertRaisesRegex(ValueError, "pixel_values"):
            validate_adv_inputs({"input_ids": torch.ones((1, 2), dtype=torch.long)})

    def test_validate_rejects_missing_input_ids(self):
        with self.assertRaisesRegex(ValueError, "input_ids"):
            validate_adv_inputs({"pixel_values": torch.ones((1, 3, 4, 4))})

    def test_redecode_preserves_pixel_dtype_and_returns_finite_action(self):
        model = FakeOpenVLAModel(dtype=torch.bfloat16)
        result = redecode_openvla_action_from_adv_inputs(
            model=model,
            adv_inputs=self._adv_inputs(dtype=torch.bfloat16),
            instruction="pick up the test object",
            unnorm_key="libero_object",
        )
        self.assertEqual(model.last_pixel_dtype, torch.bfloat16)
        self.assertEqual(result.action.shape, (3,))
        self.assertTrue(np.all(np.isfinite(result.action)))
        self.assertFalse(np.allclose(result.action, 0.0))
        self.assertEqual(result.token_ids.tolist(), [31999, 31871, 31744])

    def test_decode_rejects_dimension_mismatch(self):
        model = FakeOpenVLAModel(action_dim=3)
        model.get_action_stats = lambda key: {
            "q01": np.zeros(2, dtype=np.float32),
            "q99": np.ones(2, dtype=np.float32),
            "mask": np.ones(2, dtype=bool),
        }
        gen = FakeGeneration(torch.tensor([[1, 2, 31999, 31871, 31744]], dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            decode_openvla_generation_to_action(model, gen, "libero_object")


if __name__ == "__main__":
    unittest.main()
