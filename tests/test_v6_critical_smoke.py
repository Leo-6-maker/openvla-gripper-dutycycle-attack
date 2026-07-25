"""Smoke tests for V2 Critical Trigger Student scaffold.

Verifies: forward shapes, causality, NaN-free, config serialization,
         bypass/no-bypass parity, episode-balanced loss, trigger score range.
"""
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gripper_attack.v6_critical_student import (
    CriticalTriggerStudentV2, build_v2_recommended, build_v2_minimal)
from gripper_attack.v6_critical_loss import compute_episode_balanced_loss


def test_recommended_forward():
    """Recommended V2: 25D + policy bypass + gripper bypass → forward shapes."""
    model = build_v2_recommended()
    B, T = 4, 200
    x_25d = torch.randn(B, T, 25)
    x_policy = torch.randn(B, T, 9)
    x_gripper = torch.randn(B, T, 9)

    out = model(x_25d, x_policy, x_gripper)

    assert 'critical_prob' in out, f"Missing critical_prob, got {list(out.keys())}"
    assert 'release_safety' in out, f"Missing release_safety, got {list(out.keys())}"
    for name, tensor in out.items():
        assert tensor.shape == (B, T, 1), f"{name}: expected {(B,T,1)}, got {tensor.shape}"
        assert torch.isfinite(tensor).all(), f"{name}: contains NaN/Inf"

    # Trigger score
    trigger = model.compute_trigger_score(out)
    assert trigger.shape == (B, T, 1)
    assert (trigger >= 0).all() and (trigger <= 1).all(), "Trigger score out of [0,1]"

    # Config serializable
    cfg = model.config
    assert isinstance(cfg, dict)
    assert cfg['head_names'] == ['critical_prob', 'release_safety']

    print('  PASS: recommended forward')


def test_minimal_forward():
    """Minimal V2: 25D only, no bypass."""
    model = build_v2_minimal()
    B, T = 2, 100
    x_25d = torch.randn(B, T, 25)

    out = model(x_25d)

    assert 'critical_prob' in out
    assert out['critical_prob'].shape == (B, T, 1)

    # Should work even with missing optional inputs
    out2 = model(x_25d, x_policy=torch.randn(B, T, 9))
    assert out2['critical_prob'].shape == (B, T, 1)

    print('  PASS: minimal forward')


def test_hidden_extraction():
    """Hidden state extraction for probing."""
    model = build_v2_recommended()
    B, T = 2, 64
    x_25d = torch.randn(B, T, 25)
    x_policy = torch.randn(B, T, 9)

    hidden = model.get_hidden(x_25d, x_policy)
    assert hidden.shape == (B, T, 64)
    assert torch.isfinite(hidden).all()

    # Without policy
    hidden2 = model.get_hidden(x_25d)
    assert hidden2.shape == (B, T, 64)

    print('  PASS: hidden extraction')


def test_causality():
    """Verify future-step independence: changing x[t+1] doesn't affect output[t]."""
    model = build_v2_recommended()
    model.eval()
    B, T = 1, 64
    x_25d = torch.randn(B, T, 25)
    x_policy = torch.randn(B, T, 9)

    with torch.no_grad():
        out1 = model(x_25d, x_policy)

    # Perturb step 40+
    x_25d_perturbed = x_25d.clone()
    x_25d_perturbed[:, 40:, :] += 10.0

    with torch.no_grad():
        out2 = model(x_25d_perturbed, x_policy)

    # Steps before 40 should be identical
    diff = (out1['critical_prob'][:, :39, :] - out2['critical_prob'][:, :39, :]).abs().max()
    assert diff < 1e-5, f"Causality violated: max pre-perturb diff = {diff}"

    print('  PASS: causality (future-step independence)')


def test_episode_balanced_loss():
    """Episode-balanced loss computes correctly."""
    B = 3
    T1, T2, T3 = 200, 180, 220
    max_T = max(T1, T2, T3)

    logits = {
        'critical_prob': torch.randn(B, max_T, 1),
        'release_safety': torch.randn(B, max_T, 1),
    }
    targets = {
        'critical_prob': (torch.rand(B, max_T, 1) > 0.3).float(),
        'release_safety': (torch.rand(B, max_T, 1) > 0.8).float(),
    }
    known_masks = {
        'critical_prob': torch.ones(B, max_T, 1, dtype=torch.bool),
        'release_safety': torch.ones(B, max_T, 1, dtype=torch.bool),
    }

    loss = compute_episode_balanced_loss(
        logits, targets, known_masks,
        episode_boundaries=[T1, T2, T3],
        head_names=['critical_prob', 'release_safety'],
    )

    assert 'critical_prob_bce' in loss
    assert 'release_safety_bce' in loss
    assert 'total' in loss
    assert torch.isfinite(loss['total'])
    assert loss['total'] > 0

    # Episode-balanced should be different from naive mean
    # (verify it doesn't crash, at minimum)
    print('  PASS: episode-balanced loss (total={:.4f})'.format(loss['total'].item()))


def test_config_custom_heads():
    """Custom head names work."""
    model = CriticalTriggerStudentV2(
        head_names=['critical_prob', 'release_safety', 'burst_feasible'],
        use_policy_bypass=True,
        use_gripper_bypass=False,
    )
    B, T = 2, 32
    x_25d = torch.randn(B, T, 25)
    x_policy = torch.randn(B, T, 9)

    out = model(x_25d, x_policy)
    assert 'critical_prob' in out
    assert 'release_safety' in out
    assert 'burst_feasible' in out
    assert out['burst_feasible'].shape == (B, T, 1)

    print('  PASS: custom heads')


def test_no_policy_bypass():
    """Model without policy bypass works when policy not provided."""
    model = CriticalTriggerStudentV2(
        use_policy_bypass=False,
        use_gripper_bypass=False,
        head_names=['critical_prob'],
    )
    B, T = 2, 32
    x_25d = torch.randn(B, T, 25)

    out = model(x_25d)  # no policy, no gripper
    assert out['critical_prob'].shape == (B, T, 1)

    out2 = model(x_25d, x_policy=torch.randn(B, T, 9))  # policy provided but ignored
    assert out2['critical_prob'].shape == (B, T, 1)

    # Should get same result whether policy is provided or not (since bypass is off)
    # Need eval mode because dropout introduces randomness
    model.eval()
    with torch.no_grad():
        out_eval = model(x_25d)
        out2_eval = model(x_25d, x_policy=torch.randn(B, T, 9))
    diff = (out_eval['critical_prob'] - out2_eval['critical_prob']).abs().max()
    assert diff < 1e-5, f"Policy bypass off but policy input changed output: diff={diff}"

    print('  PASS: no-policy-bypass invariance')


def test_trigger_score_with_flip():
    """Trigger score with policy flip probability."""
    model = build_v2_recommended()
    B, T = 2, 32
    x_25d = torch.randn(B, T, 25)
    x_policy = torch.randn(B, T, 9)

    out = model(x_25d, x_policy)

    # Without flip
    trigger1 = model.compute_trigger_score(out)
    assert (trigger1 >= 0).all() and (trigger1 <= 1).all()

    # With flip = 1.0 → same
    trigger2 = model.compute_trigger_score(out, policy_flip_prob=torch.ones(B, T, 1))
    assert (trigger1 - trigger2).abs().max() < 1e-5

    # With flip = 0.0 → trigger = 0
    trigger3 = model.compute_trigger_score(out, policy_flip_prob=torch.zeros(B, T, 1))
    assert trigger3.abs().max() < 1e-5

    print('  PASS: trigger score with policy flip')


if __name__ == '__main__':
    print('=== V2 CRITICAL TRIGGER STUDENT SMOKE TESTS ===')
    test_recommended_forward()
    test_minimal_forward()
    test_hidden_extraction()
    test_causality()
    test_episode_balanced_loss()
    test_config_custom_heads()
    test_no_policy_bypass()
    test_trigger_score_with_flip()
    print()
    print('ALL TESTS PASSED')
