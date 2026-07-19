"""CPU unit tests for R7.2.2 closure replay — model forward, scheduler, diagnostics."""
import pytest, sys, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "detector_v4"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from replay_k10_v5_v22_closure import (
    ModelScores, run_scheduler_at_threshold, compute_score_diagnostics,
    aggregate_score_diagnostics, ReplayContext,
)


class DummyV5:
    def __init__(self, T, valid_mask, candidate_close):
        self.canonical_parent_key = "test/suite/task_00/state_00"
        self.features_25d = torch.randn(T, 25)
        self.valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
        self.candidate_close = torch.tensor(candidate_close, dtype=torch.bool)
        self.policy_intent_9d = torch.randn(T, 9)
        self.intent_valid_mask = torch.ones(T, dtype=torch.bool)


def make_ctx(T=200, feasible_starts=None, valid=None, candidate_close=None):
    if valid is None:
        valid = [True] * T
    if candidate_close is None:
        candidate_close = [True] * T
    if feasible_starts is None:
        feasible_starts = set()
    v5 = DummyV5(T, valid, candidate_close)
    has_feas = len(feasible_starts) > 0
    first_feas = min(feasible_starts) if has_feas else -1
    return ReplayContext(v5=v5, feasible_starts=feasible_starts,
                         has_feasible=has_feas, first_feasible=first_feas)


def make_scores(T=200, utility_vals=None):
    if utility_vals is None:
        utility_vals = [0.1] * T
    return ModelScores(
        utility=torch.tensor(utility_vals, dtype=torch.float32),
        release=torch.zeros(T, dtype=torch.float32),
        regrasp=torch.zeros(T, dtype=torch.float32),
    )


class TestScheduler:
    def test_no_emit_below_threshold(self):
        ctx = make_ctx(T=50, feasible_starts={25})
        scores = make_scores(T=50, utility_vals=[0.05] * 50)
        result = run_scheduler_at_threshold(ctx, scores, threshold=0.5)
        assert not result["emitted"]
        assert result["emit_step"] == -1

    def test_single_threshold_crossing_does_not_satisfy_persistence(self):
        ctx = make_ctx(T=50, feasible_starts={25})
        vals = [0.05] * 50
        vals[20] = 0.8
        scores = make_scores(T=50, utility_vals=vals)
        result = run_scheduler_at_threshold(ctx, scores, threshold=0.5)
        assert not result["emitted"]
        assert result["emit_step"] == -1

    def test_emit_with_persistence(self):
        ctx = make_ctx(T=50, feasible_starts={25})
        vals = [0.05] * 50
        for i in range(10, 25):
            vals[i] = 0.8
        scores = make_scores(T=50, utility_vals=vals)
        result = run_scheduler_at_threshold(ctx, scores, threshold=0.5)
        assert result["emitted"]
        assert result["emit_step"] >= 10

    def test_emit_within_k10(self):
        ctx = make_ctx(T=50, feasible_starts=set(range(10, 25)))
        vals = [0.05] * 50
        for i in range(10, 30):
            vals[i] = 0.8
        scores = make_scores(T=50, utility_vals=vals)
        result = run_scheduler_at_threshold(ctx, scores, threshold=0.5)
        assert result["emitted"]
        assert result["within_k10"]

    def test_false_emit(self):
        ctx = make_ctx(T=50, feasible_starts={40, 41, 42, 43, 44, 45, 46, 47, 48, 49})
        vals = [0.05] * 50
        for i in range(10, 25):
            vals[i] = 0.8
        scores = make_scores(T=50, utility_vals=vals)
        result = run_scheduler_at_threshold(ctx, scores, threshold=0.5)
        assert result["emitted"]
        assert result["false_emit"]
        assert not result["within_k10"]

    def test_release_veto_blocks(self):
        ctx = make_ctx(T=50, feasible_starts={25, 26, 27, 28, 29, 30, 31, 32, 33, 34})
        vals = [0.05] * 50
        for i in range(10, 30):
            vals[i] = 0.8
        scores = ModelScores(
            utility=torch.tensor(vals, dtype=torch.float32),
            release=torch.full((50,), 0.9, dtype=torch.float32),
            regrasp=torch.zeros(50, dtype=torch.float32),
        )
        result = run_scheduler_at_threshold(ctx, scores, threshold=0.5)
        assert not result["emitted"]

    def test_invalid_step_resets_dwell(self):
        valid = [True] * 50
        valid[15] = False
        ctx = make_ctx(T=50, feasible_starts={25}, valid=valid)
        vals = [0.05] * 50
        for i in range(10, 18):
            vals[i] = 0.8
        scores = make_scores(T=50, utility_vals=vals)
        result = run_scheduler_at_threshold(ctx, scores, threshold=0.5)
        assert not result["emitted"]


class TestScoreDiagnostics:
    def test_paired_delta_positive(self):
        ctx = make_ctx(T=50, feasible_starts={25, 26, 27})
        vals = [0.05] * 50
        vals[25] = 0.8
        scores = make_scores(T=50, utility_vals=vals)
        diag = compute_score_diagnostics(scores, ctx)
        assert diag["has_feasible"]
        assert diag["delta"] is not None
        assert diag["delta"] > 0
        assert diag["best_in_corridor"]

    def test_paired_delta_negative(self):
        ctx = make_ctx(T=50, feasible_starts={25, 26, 27})
        vals = [0.05] * 50
        vals[30] = 0.8
        scores = make_scores(T=50, utility_vals=vals)
        diag = compute_score_diagnostics(scores, ctx)
        assert diag["has_feasible"]
        assert diag["delta"] is not None
        assert diag["delta"] < 0
        assert not diag["best_in_corridor"]

    def test_no_feasible_episode(self):
        ctx = make_ctx(T=50)
        scores = make_scores(T=50)
        diag = compute_score_diagnostics(scores, ctx)
        assert not diag["has_feasible"]
        assert diag["delta"] is not None or diag["max_inside"] < 0

    def test_feasible_rank(self):
        ctx = make_ctx(T=50, feasible_starts={25, 26, 27})
        vals = [0.05] * 50
        vals[25] = 0.9
        scores = make_scores(T=50, utility_vals=vals)
        diag = compute_score_diagnostics(scores, ctx)
        assert diag["feasible_rank"] > 0
        assert diag["feasible_percentile"] is not None
        assert diag["feasible_percentile"] <= 0.1


class TestAggregateDiagnostics:
    def test_aggregate_positive_deltas(self):
        diags = []
        for i in range(26):
            ctx = make_ctx(T=50, feasible_starts={20 + i % 5})
            vals = [0.05] * 50
            vals[20 + (i % 5)] = 0.6 if i < 13 else 0.01
            vals[30] = 0.8 if i >= 13 else 0.01
            scores = make_scores(T=50, utility_vals=vals)
            diags.append(compute_score_diagnostics(scores, ctx))
            diags[-1]["candidate"] = "V5-A"

        for _ in range(174):
            ctx = make_ctx(T=50)
            scores = make_scores(T=50)
            diags.append(compute_score_diagnostics(scores, ctx))
            diags[-1]["candidate"] = "V5-A"

        agg = aggregate_score_diagnostics(diags)
        assert agg["n_feasible_episodes"] == 26
        assert agg["n_delta_valid"] > 0
        assert agg["mean_delta"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
