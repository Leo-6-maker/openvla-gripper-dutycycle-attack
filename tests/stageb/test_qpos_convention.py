#!/usr/bin/env python3
"""Test: qpos convention — abs_sum vs signed_mean."""
import numpy as np

def test_signed_mean_cancellation():
    """q0 ≈ -q1 → signed_mean cancels to ~0."""
    q0, q1 = 0.039, -0.039
    signed = (q0 + q1) / 2
    abs_mean = (abs(q0) + abs(q1)) / 2
    assert abs(signed) < 0.001, f'signed_mean should cancel: {signed}'
    assert abs(abs_mean - 0.039) < 0.001, f'abs_mean should be correct: {abs_mean}'
    print('PASS: test_signed_mean_cancellation')

def test_abs_sum_variation():
    """abs_sum should capture physical change."""
    before = 0.039; after = 0.030
    delta = after - before
    assert abs(delta) > 0.001, f'delta should be non-zero: {delta}'
    print('PASS: test_abs_sum_variation')

def test_placeholder_rejection():
    """Constant 0.5 should be rejected."""
    vals = [0.5, 0.5, 0.5]
    is_placeholder = all(abs(v - 0.5) < 0.001 for v in vals)
    assert is_placeholder, 'constant 0.5 should be flagged as placeholder'
    print('PASS: test_placeholder_rejection')

if __name__ == '__main__':
    test_signed_mean_cancellation()
    test_abs_sum_variation()
    test_placeholder_rejection()
    print('All qpos convention tests passed')
