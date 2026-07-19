"""CPU unit tests for R7.3 K10 detector training — models, loss, OOF."""
import pytest, sys, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "detector_v4"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train_k10_detector_v3 import (
    R7SLinear25D, R7AGRU25D, K10TrainingEpisode,
    compute_k10_loss, build_oof_folds,
)


def _make_ep(T=100, feasible_starts=None, release_risk=0.0, regrasp_risk=0.0, has_feasible=True):
    if feasible_starts is None:
        feasible_starts = []
    k10_target = torch.full((T,), -1.0)
    k10_known = torch.zeros(T, dtype=torch.bool)
    for t in range(T):
        k10_target[t] = 0.0
        k10_known[t] = True
    for t in feasible_starts:
        k10_target[t] = 1.0

    rankable_mask = torch.ones(T, dtype=torch.bool)
    rankable_mask[:5] = False  # first 5 steps not rankable
    valid = torch.ones(T, dtype=torch.bool)
    candidate = torch.ones(T, dtype=torch.bool)
    candidate[:3] = False

    release_target = torch.full((T,), float(release_risk > 0.5))
    release_known = torch.ones(T, dtype=torch.bool)
    regrasp_target = torch.full((T,), float(regrasp_risk > 0.5))
    regrasp_known = torch.ones(T, dtype=torch.bool)

    return K10TrainingEpisode(
        identity="test/task_00/state_00", features_25d=torch.randn(T, 25),
        valid_mask=valid, candidate_close=candidate,
        k10_target=k10_target, k10_known=k10_known,
        release_target=release_target, release_known=release_known,
        regrasp_target=regrasp_target, regrasp_known=regrasp_known,
        suite="libero_10", task_idx=0, has_feasible=has_feasible,
        n_steps=T, feasible_starts=feasible_starts,
    )


class TestSLinear:
    def test_output_shape(self):
        model = R7SLinear25D()
        x = torch.randn(4, 100, 25)
        outputs = model(x)
        assert outputs["utility_logit"].shape == (4, 100)
        assert outputs["release_logit"].shape == (4, 100)
        assert outputs["regrasp_logit"].shape == (4, 100)

    def test_parameter_count(self):
        model = R7SLinear25D()
        n = sum(p.numel() for p in model.parameters())
        assert n == 25 * 3 + 3  # 3 heads × (25w + 1b)


class TestAGRU:
    def test_output_shape(self):
        model = R7AGRU25D()
        x = torch.randn(4, 100, 25)
        svm = torch.ones(4, 100, dtype=torch.bool)
        bnd = torch.zeros(4, 100, dtype=torch.bool)
        bnd[:, 0] = True
        outputs = model(x, svm, bnd)
        assert outputs["utility_logit"].shape == (4, 100)

    def test_stateful_masking(self):
        model = R7AGRU25D()
        x = torch.randn(2, 20, 25)
        svm = torch.ones(2, 20, dtype=torch.bool)
        svm[:, 10] = False  # invalid step
        bnd = torch.zeros(2, 20, dtype=torch.bool)
        bnd[:, 0] = True
        outputs = model(x, svm, bnd)
        # Invalid step should not crash; hidden should not update
        assert outputs["utility_logit"].shape == (2, 20)

    def test_boundary_reset(self):
        model = R7AGRU25D()
        x = torch.randn(2, 20, 25)
        svm = torch.ones(2, 20, dtype=torch.bool)
        bnd = torch.zeros(2, 20, dtype=torch.bool)
        bnd[:, 0] = True
        bnd[:, 10] = True  # boundary reset
        outputs_a = model(x, svm, bnd)

        # Without boundary reset at step 10, different outputs after step 10
        bnd2 = bnd.clone()
        bnd2[:, 10] = False
        outputs_b = model(x, svm, bnd2)
        diff = (outputs_a["utility_logit"][:, 11:] - outputs_b["utility_logit"][:, 11:]).abs().max()
        assert diff > 0  # boundary reset changes hidden state


class TestLoss:
    def test_loss_finite(self):
        model = R7SLinear25D()
        ep = _make_ep(T=50, feasible_starts=[20, 21, 22])
        x = ep.features_25d.unsqueeze(0)
        outputs = model(x)
        loss = compute_k10_loss(outputs, ep, torch.device("cpu"))
        assert torch.isfinite(loss["total"])
        assert loss["total"].item() >= 0

    def test_loss_positive_episode_balanced(self):
        model = R7SLinear25D()
        ep = _make_ep(T=50, feasible_starts=list(range(20, 30)))
        x = ep.features_25d.unsqueeze(0)
        outputs = model(x)
        loss = compute_k10_loss(outputs, ep, torch.device("cpu"))
        # Should have both utility and aux components
        assert "utility" in loss
        assert "release" in loss
        assert "regrasp" in loss

    def test_loss_no_feasible_episode(self):
        model = R7SLinear25D()
        ep = _make_ep(T=50, feasible_starts=[], has_feasible=False)
        x = ep.features_25d.unsqueeze(0)
        outputs = model(x)
        loss = compute_k10_loss(outputs, ep, torch.device("cpu"))
        assert torch.isfinite(loss["total"])

    def test_loss_zero_on_empty_rankable(self):
        model = R7SLinear25D()
        ep = _make_ep(T=50, feasible_starts=[])
        ep.candidate_close = torch.zeros(50, dtype=torch.bool)
        x = ep.features_25d.unsqueeze(0)
        outputs = model(x)
        loss = compute_k10_loss(outputs, ep, torch.device("cpu"))
        assert loss["total"].item() == 0.0


class TestOOFFolds:
    def test_five_folds(self):
        eps = [_make_ep(T=50, feasible_starts=[20] if i % 4 == 0 else [],
                         has_feasible=(i % 4 == 0))
               for i in range(600)]
        folds = build_oof_folds(eps, seed=20260717)
        assert len(folds) == 5

    def test_no_overlap(self):
        eps = [_make_ep(T=50, feasible_starts=[20] if i % 3 == 0 else [],
                         has_feasible=(i % 3 == 0))
               for i in range(600)]
        folds = build_oof_folds(eps, seed=20260717)
        all_val = set()
        for train_idx, val_idx in folds:
            val_set = set(val_idx)
            train_set = set(train_idx)
            assert not (val_set & train_set)
            assert not (val_set & all_val)
            all_val |= val_set
        assert len(all_val) == 600

    def test_feasible_stratified(self):
        eps = [_make_ep(T=50, feasible_starts=[20] if i < 100 else [],
                         has_feasible=(i < 100))
               for i in range(600)]
        folds = build_oof_folds(eps, seed=20260717)
        for train_idx, val_idx in folds:
            val_feas = sum(1 for i in val_idx if eps[i].has_feasible)
            # Should have roughly 20 feasible per fold (100/5)
            assert 15 <= val_feas <= 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
