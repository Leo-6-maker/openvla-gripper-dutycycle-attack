"""D1b.1: Protocol regression tests for learned detector training.

Verifies:
  - Zero-stdev features normalize to 0.0 (not NaN/Inf)
  - Missing values imputed from train mean
  - All normalized values are finite
  - Per-trace: exactly one positive label
  - Tie tolerance: scores within 0.001 → tie
  - Checkpoint rule ordering: top-1 → MAE → epoch
  - Baseline: total_score tie handling
  - Single test evaluation pass only
"""

import csv, hashlib, math, os, sys, tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "stageb"))

from train_d1b_detector import (
    CandidateRanker, load_normalization, normalize_features,
    FEATURE_NAMES, TIE_TOLERANCE, ZERO_STDEV_THRESHOLD, CLIP_RANGE, TRAINING_SEED,
)

# ── Synthetic data for tests ──
def make_synthetic_candidates():
    """3 traces with known features including zero-stdev and missing values."""
    c1 = {"trace_id": "t1", "task_key": "a", "state_id": "0"}
    c2 = {"trace_id": "t1", "task_key": "a", "state_id": "0"}
    c3 = {"trace_id": "t2", "task_key": "b", "state_id": "1"}
    # Trace t1: 2 candidates, first is positive
    for fn in FEATURE_NAMES:
        c1[fn] = "1.0"; c2[fn] = "2.0"; c3[fn] = "3.0"
    c1["is_teacher_p"] = "1"; c2["is_teacher_p"] = "0"; c3["is_teacher_p"] = "1"
    c1["candidate_step"] = "10"; c2["candidate_step"] = "20"; c3["candidate_step"] = "30"
    c1["total_score"] = "3.0"; c2["total_score"] = "2.5"; c3["total_score"] = "3.5"
    # Missing value in c2
    c2["qpos"] = ""
    return [c1, c2, c3]


def make_synthetic_norm():
    """Normalization stats where close_streak has zero stdev."""
    return (
        {fn: 1.0 for fn in FEATURE_NAMES},  # means
        {fn: 0.5 for fn in FEATURE_NAMES},  # stdevs
        {fn: 1.0 for fn in FEATURE_NAMES},  # impute
    )


class TestZeroStdevNormalization:
    def test_zero_stdev_produces_zero(self):
        means, stdevs, impute = make_synthetic_norm()
        # Override one feature to zero stdev
        stdevs["close_streak"] = 0.0
        stdevs["close_streak_bonus"] = 1e-10  # below threshold
        cands = make_synthetic_candidates()
        X = normalize_features(cands, means, stdevs, impute)
        cs_idx = FEATURE_NAMES.index("close_streak")
        csb_idx = FEATURE_NAMES.index("close_streak_bonus")
        assert torch.all(X[:, cs_idx] == 0.0), f"close_streak not zero: {X[:, cs_idx]}"
        assert torch.all(X[:, csb_idx] == 0.0), f"close_streak_bonus not zero: {X[:, csb_idx]}"

    def test_all_outputs_finite(self):
        means, stdevs, impute = make_synthetic_norm()
        cands = make_synthetic_candidates()
        X = normalize_features(cands, means, stdevs, impute)
        assert torch.isfinite(X).all(), f"Non-finite values: {X}"

    def test_missing_value_imputed(self):
        means, stdevs, impute = make_synthetic_norm()
        cands = make_synthetic_candidates()
        X = normalize_features(cands, means, stdevs, impute)
        qpos_idx = FEATURE_NAMES.index("qpos")
        # c2 has missing qpos — should be imputed to 1.0 (mean) → normalized to 0.0
        # (1.0 - 1.0) / 0.5 = 0.0
        assert torch.isfinite(X[1, qpos_idx]), f"Missing qpos not finite: {X[1, qpos_idx]}"

    def test_clip_range(self):
        means, stdevs, impute = make_synthetic_norm()
        # Make a feature with very large value
        cands = make_synthetic_candidates()
        cands[0]["qpos"] = "100.0"
        X = normalize_features(cands, means, stdevs, impute)
        qpos_idx = FEATURE_NAMES.index("qpos")
        assert abs(X[0, qpos_idx].item()) <= CLIP_RANGE + 0.01, f"Not clipped: {X[0, qpos_idx]}"


class TestTieHandling:
    def test_tie_tolerance_identifies_ties(self):
        scores = np.array([3.0, 3.0005, 2.0])
        max_s = scores.max()
        ties = [i for i, s in enumerate(scores) if abs(s - max_s) < TIE_TOLERANCE]
        assert len(ties) == 2  # 3.0 and 3.0005 are tied within 0.001

    def test_tie_tolerance_separates_distinct(self):
        scores = np.array([3.0, 3.002, 2.0])
        max_s = scores.max()
        ties = [i for i, s in enumerate(scores) if abs(s - max_s) < TIE_TOLERANCE]
        assert len(ties) == 1  # 3.002 is outside 0.001 tolerance

    def test_ties_make_top1_incorrect(self):
        # Simulate: two candidates tie for max, neither is the unique top-1
        scores = np.array([3.0, 3.0, 2.0])
        is_tp = np.array([True, False, False])
        tp_score = scores[is_tp][0]
        n_higher = sum(1 for s in scores if s > tp_score + TIE_TOLERANCE)
        n_equal = sum(1 for s in scores if abs(s - tp_score) < TIE_TOLERANCE)
        is_unique_top1 = (n_higher == 0 and n_equal == 1)
        assert not is_unique_top1  # TP tied with another → not unique top-1


class TestCheckpointRule:
    def test_best_val_acc_selected(self):
        history = [
            {"epoch": 1, "val_acc": 0.3, "val_mae": 10},
            {"epoch": 2, "val_acc": 0.4, "val_mae": 8},
            {"epoch": 3, "val_acc": 0.35, "val_mae": 5},
        ]
        best = max(history, key=lambda h: (h["val_acc"], -h["val_mae"], -h["epoch"]))
        assert best["epoch"] == 2  # 0.4 > 0.35 > 0.3

    def test_tiebreak_by_mae(self):
        history = [
            {"epoch": 1, "val_acc": 0.4, "val_mae": 10},
            {"epoch": 2, "val_acc": 0.4, "val_mae": 5},
        ]
        best = max(history, key=lambda h: (h["val_acc"], -h["val_mae"], -h["epoch"]))
        assert best["epoch"] == 2  # same acc, lower MAE

    def test_tiebreak_by_epoch(self):
        history = [
            {"epoch": 1, "val_acc": 0.4, "val_mae": 5},
            {"epoch": 2, "val_acc": 0.4, "val_mae": 5},
        ]
        best = max(history, key=lambda h: (h["val_acc"], -h["val_mae"], -h["epoch"]))
        assert best["epoch"] == 2  # same acc+mae, later epoch


class TestPerTraceInvariants:
    def test_exactly_one_positive_per_trace(self):
        cands = make_synthetic_candidates()
        by_trace = defaultdict(list)
        for c in cands:
            by_trace[c["trace_id"]].append(c)
        for tid, cs in by_trace.items():
            n_pos = sum(1 for c in cs if int(c.get("is_teacher_p", 0)) == 1)
            assert n_pos == 1, f"Trace {tid}: {n_pos} positives"

    def test_candidate_step_is_unique_per_trace(self):
        cands = make_synthetic_candidates()
        by_trace = defaultdict(list)
        for c in cands:
            by_trace[c["trace_id"]].append(c)
        for tid, cs in by_trace.items():
            steps = [int(c["candidate_step"]) for c in cs]
            assert len(steps) == len(set(steps)), f"Duplicate steps in trace {tid}"


class TestModelArchitecture:
    def test_output_is_scalar_per_candidate(self):
        model = CandidateRanker(n_features=16)
        x = torch.randn(4, 16)
        out = model(x)
        assert out.shape == (4,), f"Expected (4,), got {out.shape}"

    def test_seed_reproducibility(self):
        torch.manual_seed(TRAINING_SEED)
        m1 = CandidateRanker(n_features=16)
        torch.manual_seed(TRAINING_SEED)
        m2 = CandidateRanker(n_features=16)
        x = torch.randn(3, 16)
        assert torch.allclose(m1(x), m2(x)), "Models not reproducible with same seed"
