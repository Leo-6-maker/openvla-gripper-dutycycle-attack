import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n5" / "phase3_student"))

from gripper_attack.v5_r3_student import finite_head_outputs, r3_multihead_loss, shuffle_known_targets
from n5_student_model import N5MultiHeadStudent


def _labels(batch=2, steps=5):
    targets = {head: torch.tensor([[0, 1, 0, 1, 0]] * batch, dtype=torch.float32) for head in (
        "physical_criticality", "k10_feasibility", "safe_release", "instability", "gripper_closing_state"
    )}
    masks = {head: torch.ones(batch, steps, dtype=torch.bool) for head in targets}
    masks["safe_release"][:, -1] = False
    return targets, masks


def test_unknown_mask_is_excluded_from_loss():
    model = N5MultiHeadStudent(input_dim=25, hidden=16, short_rf=4, long_rf=8, dropout=0.0)
    x = torch.randn(2, 5, 25)
    logits = model(x, timestep_mask=torch.ones(2, 5, dtype=torch.bool))
    targets, masks = _labels()
    loss, details = r3_multihead_loss(logits, targets, masks)
    assert torch.isfinite(loss)
    assert details["safe_release"] is not None
    assert loss.requires_grad


def test_label_shuffle_preserves_masks_and_is_deterministic():
    targets, masks = _labels()
    first = shuffle_known_targets(targets, masks, 20260717)
    second = shuffle_known_targets(targets, masks, 20260717)
    for head in targets:
        assert torch.equal(first[head], second[head])
        assert torch.equal(first[head][~masks[head]], targets[head][~masks[head]])


def test_model_outputs_are_finite_and_checkpoint_roundtrip():
    model = N5MultiHeadStudent(input_dim=25, hidden=16, short_rf=4, long_rf=8, dropout=0.0)
    model.eval()
    x = torch.randn(2, 5, 25)
    with torch.no_grad():
        before = model(x, timestep_mask=torch.ones(2, 5, dtype=torch.bool))
    assert finite_head_outputs(before)
    restored = N5MultiHeadStudent(input_dim=25, hidden=16, short_rf=4, long_rf=8, dropout=0.0)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.eval()
    with torch.no_grad():
        after = restored(x, timestep_mask=torch.ones(2, 5, dtype=torch.bool))
    for head in before:
        assert torch.allclose(before[head], after[head])
