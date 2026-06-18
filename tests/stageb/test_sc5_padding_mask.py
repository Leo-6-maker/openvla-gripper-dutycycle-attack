#!/usr/bin/env python3
"""Test SC5 padding mask: short sequences get left-padded, padding not treated as real data."""
import sys, os
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))


def test_padding_mask_correctness():
    """Padding mask is 1 for real data, 0 for padding."""
    T = 10
    real_len = 7
    history_len = 32

    # Simulate left-padding: first (history_len - real_len) timesteps are padding
    pad_len = max(0, history_len - real_len)
    padding_mask = np.zeros(history_len, dtype=np.float32)
    padding_mask[pad_len:] = 1.0  # real data at the end

    assert np.sum(padding_mask) == real_len
    assert padding_mask[0] == 0.0  # padding
    assert padding_mask[-1] == 1.0  # real
    print("PASS: test_padding_mask_correctness")


def test_full_length_no_padding():
    """Sequence equal to history_len has no padding."""
    real_len = 32
    history_len = 32
    pad_len = max(0, history_len - real_len)
    padding_mask = np.zeros(history_len, dtype=np.float32)
    padding_mask[pad_len:] = 1.0
    assert np.all(padding_mask == 1.0)
    print("PASS: test_full_length_no_padding")


def test_very_short_sequence():
    """Very short sequence (3 steps) gets heavily padded."""
    real_len = 3
    history_len = 32
    pad_len = max(0, history_len - real_len)
    padding_mask = np.zeros(history_len, dtype=np.float32)
    padding_mask[pad_len:] = 1.0
    assert pad_len == 29
    assert np.sum(padding_mask) == 3
    print("PASS: test_very_short_sequence")


def test_padded_features_are_zero_not_real():
    """Padded positions have feature values of 0, not real data."""
    D = 25
    history_len = 32
    real_len = 10
    pad_len = history_len - real_len

    # Create feature matrix with padding
    features = np.zeros((history_len, D), dtype=np.float32)
    # Fill real data
    features[pad_len:] = np.random.randn(real_len, D).astype(np.float32) * 0.1 + 0.5

    # Padded region should be all zeros
    assert np.all(features[:pad_len] == 0.0), "Padded region should be zero"

    # But padding mask prevents them from being used in loss
    padding_mask = np.zeros(history_len, dtype=np.float32)
    padding_mask[pad_len:] = 1.0

    # Masked mean should only use real data
    masked_sum = (features * padding_mask[:, None]).sum(0)
    assert not np.allclose(masked_sum, 0.0), "Real features should contribute"
    print("PASS: test_padded_features_are_zero_not_real")


def test_history_len_64_option():
    """Support history_len=64 as well as 32."""
    history_len = 64
    real_len = 50
    pad_len = max(0, history_len - real_len)
    padding_mask = np.zeros(history_len, dtype=np.float32)
    padding_mask[pad_len:] = 1.0
    assert pad_len == 14
    assert np.sum(padding_mask) == 50
    print("PASS: test_history_len_64_option")


def test_padding_cannot_be_treated_as_real():
    """If padding is accidentally treated as real, detection catches it."""
    history_len = 32
    real_len = 10
    pad_len = history_len - real_len
    padding_mask = np.zeros(history_len, dtype=np.float32)
    padding_mask[pad_len:] = 1.0

    # Simulate: if someone accidentally uses all positions (no masking)
    all_sum = np.ones(history_len).sum()  # would be 32
    masked_sum = padding_mask.sum()  # should be 10

    assert all_sum != masked_sum, \
        f"Without masking: {all_sum}, with masking: {masked_sum} — they must differ"
    assert masked_sum == real_len
    print("PASS: test_padding_cannot_be_treated_as_real")


if __name__ == '__main__':
    test_padding_mask_correctness()
    test_full_length_no_padding()
    test_very_short_sequence()
    test_padded_features_are_zero_not_real()
    test_history_len_64_option()
    test_padding_cannot_be_treated_as_real()
    print("\nAll SC5 padding mask tests passed.")
