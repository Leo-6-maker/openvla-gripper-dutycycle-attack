"""D5 Frozen Runtime v1 — immutable MLP + normalization.

Self-contained copy of CandidateRanker and normalize_features from
train_d1b_detector.py at commit 52bdc335. This module must NOT import from
scripts/stageb — it is the frozen source of truth for D5 scoring.

Any change to this file changes its SHA256 and breaks the production bundle.
"""
import torch
import torch.nn as nn

# ── Frozen constants (from train_d1b_detector.py) ──
ZERO_STDEV_THRESHOLD = 1e-8
CLIP_RANGE = 3.0

FEATURE_NAMES_V1 = [
    "total_score", "raw_crossing_bonus", "close_streak_bonus", "close_onset_qpos_bonus",
    "eef_deceleration_bonus", "qpos_ready_bonus", "eef_speed_now", "eef_speed_prev",
    "eef_deceleration_delta", "close_streak", "raw_crossing", "close_onset",
    "qpos", "time_since_prev_close", "time_since_last_open", "candidate_index",
]


class CandidateRankerV1(nn.Module):
    """MLP: 16 → 128 → 64 → 32 → 1 (scalar score per candidate).

    Exact copy from train_d1b_detector.py commit 52bdc335.
    """
    def __init__(self, n_features=16, hidden=(128, 64, 32), dropout=0.1):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def normalize_features_v1(candidates, means, stdevs, impute):
    """Normalize with frozen stats. Zero-stdev → 0.0. Missing → impute. Clip [-3,3].

    Exact copy from train_d1b_detector.py commit 52bdc335.
    """
    X = []
    for c in candidates:
        row = []
        for fn in FEATURE_NAMES_V1:
            v = c.get(fn, "")
            if v == "" or v is None:
                v = impute[fn]
            else:
                try:
                    v = float(v)
                except Exception:
                    v = impute[fn]
            s = stdevs[fn]
            if s < ZERO_STDEV_THRESHOLD:
                nv = 0.0
            else:
                nv = (v - means[fn]) / s
            row.append(max(-CLIP_RANGE, min(CLIP_RANGE, nv)))
        X.append(row)
    X = torch.tensor(X, dtype=torch.float32)
    assert torch.isfinite(X).all(), f"Non-finite values after normalization"
    return X
