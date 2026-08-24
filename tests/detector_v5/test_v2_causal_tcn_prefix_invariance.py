"""Test V2 causal TCN prefix invariance: modifying t+1 must not change <=t outputs."""
import torch
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from gripper_attack.v5_factorized_student_v2 import FactorizedStudentV2


def test_tcn_prefix_invariance():
    """Changing future inputs must not affect past outputs."""
    model = FactorizedStudentV2(hidden_dim=64, receptive_field=32,
                                 encoder_type='tcn', dropout=0.0)
    model.eval()

    B, T = 2, 50
    x1 = torch.randn(B, T, 25)
    mask = torch.ones(B, T, dtype=torch.bool)

    # Baseline: all steps
    with torch.no_grad():
        out1 = model.forward_sequence(x1, None, mask, None, 'single_object_pick_place')

    # Modify step 30 (future relative to steps 0-25)
    x2 = x1.clone()
    x2[:, 30, :] = 999.0
    with torch.no_grad():
        out2 = model.forward_sequence(x2, None, mask, None, 'single_object_pick_place')

    # Steps 0-20 must be identical
    for head in ['grasp', 'manipulation', 'release']:
        diff = (out1[head][:, :21] - out2[head][:, :21]).abs().max().item()
        assert diff < 1e-5, f'{head} prefix changed after future modification: max_diff={diff}'


def test_tcn_finite_receptive_field():
    """Output at step t depends only on inputs [t-RF+1, t]."""
    model = FactorizedStudentV2(hidden_dim=64, receptive_field=16,
                                 encoder_type='tcn', dropout=0.0)
    model.eval()
    B, T = 2, 100
    actual_rf = model.encoder_25d.actual_receptive_field
    # Place perturbation well outside the actual RF
    perturb_step = 20
    target_step = perturb_step + actual_rf + 10  # well beyond RF

    x1 = torch.randn(B, T, 25)
    x2 = x1.clone()
    x2[:, perturb_step, :] = 999.0

    mask = torch.ones(B, T, dtype=torch.bool)
    with torch.no_grad():
        o1 = model.forward_sequence(x1, None, mask, None, 'single_object_pick_place')
        o2 = model.forward_sequence(x2, None, mask, None, 'single_object_pick_place')

    # Output at target_step should be unchanged
    for head in ['grasp', 'manipulation', 'release']:
        diff = (o1[head][:, target_step] - o2[head][:, target_step]).abs().max().item()
        assert diff < 1e-4, f'{head} at step {target_step} changed (RF={actual_rf}): {diff}'


def test_route_abstention():
    """Unsupported routes must output zero probabilities."""
    model = FactorizedStudentV2(hidden_dim=64, encoder_type='tcn')
    model.eval()
    x = torch.randn(1, 10, 25)
    mask = torch.ones(1, 10, dtype=torch.bool)
    with torch.no_grad():
        out = model.forward_sequence(x, None, mask, None, 'unknown_or_ambiguous')
    for head in ['grasp', 'manipulation', 'release']:
        assert out[head].abs().max().item() == 0.0, f'{head} non-zero on unsupported route'
