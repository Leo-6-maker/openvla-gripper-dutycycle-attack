#!/usr/bin/env python3
"""SC5MLP-v1: Three-head model — no confidence head.

Architecture (frozen):
  input: 25D canonical features
  backbone: Linear(25,64) → ReLU → Linear(64,64) → ReLU
  heads: phase(9), corridor(1), release(1)

This is the canonical training checkpoint format. The 4-head SC5MLP in
sc5_detector_runtime.py is kept for backward compatibility but confidence_head
must NOT appear in new checkpoints.
"""
import torch.nn as nn

SC5_FEATURES = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed",
    "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
]

SC5_PHASES = [
    "approach", "grasp_close", "stable_grasp", "first_lift",
    "stable_carry", "pre_place_unsupported", "release_safe",
    "recovery_or_regrasp", "abstain_unsupported",
]

N_FEATURES = 25
N_PHASES = 9
HIDDEN_DIM = 64


class SC5MLPV1(nn.Module):
    """Three-head SC5MLP for ProprioNoStep detection. No confidence head."""

    def __init__(self, n_feat: int = N_FEATURES, hidden: int = HIDDEN_DIM,
                 n_phases: int = N_PHASES):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.phase_head = nn.Linear(hidden, n_phases)
        self.corridor_head = nn.Linear(hidden, 1)
        self.release_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return {
            "phase_logits": self.phase_head(h),
            "corridor_logit": self.corridor_head(h),
            "release_logit": self.release_head(h),
        }
