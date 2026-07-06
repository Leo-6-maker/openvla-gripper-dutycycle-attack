#!/usr/bin/env python3
"""Multi-suite SC5 detector v2 runtime.

This runtime intentionally stays close to the mature SC5 runtime while adding an
optional event-role/abstain head for LIBERO-10 multi-segment trajectories.

Boundary:
- clean online proprio/action features only;
- no timestep / normalized_step;
- no privileged state;
- no attack outcome;
- one-shot primary emission FSM by default.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

SC5_V2_FEATURES = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed",
    "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
]

SC5_V2_PHASES = [
    "approach",
    "grasp_close",
    "stable_grasp",
    "first_lift",
    "stable_carry",
    "pre_place_unsupported",
    "release_safe",
    "recovery_or_regrasp",
    "abstain_unsupported",
]

SC5_V2_EVENT_ROLES = [
    "primary_attackable",
    "auxiliary_manipulation",
    "distractor_or_setup",
    "unsupported_or_abstain",
]

FORBIDDEN_INPUT_HINTS = [
    "normalized_step",
    "timestep",
    "task_id",
    "state_id",
    "episode_key",
    "run_id",
    "parent_id",
    "object_pose",
    "target_pose",
    "object_to_target",
    "teacher_window",
    "teacher_anchor",
    "attack_outcome",
    "rand_outcome",
    "manual_anchor",
    "oracle_window",
]


class SC5MultiSuiteMLP(nn.Module):
    """SC5 v2 MLP with optional event-role/abstain head."""

    def __init__(self, n_feat: int, hidden: int = 64, n_phase: int = len(SC5_V2_PHASES), n_event_role: int = len(SC5_V2_EVENT_ROLES)):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.phase_head = nn.Linear(hidden, n_phase)
        self.corridor_head = nn.Linear(hidden, 1)
        self.release_head = nn.Linear(hidden, 1)
        self.event_role_head = nn.Linear(hidden, n_event_role)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.shared(x)
        return {
            "phase_logits": self.phase_head(h),
            "corridor_logit": self.corridor_head(h),
            "release_logit": self.release_head(h),
            "event_role_logits": self.event_role_head(h),
        }


def validate_no_forbidden_inputs(feature_names: List[str]) -> None:
    bad = []
    for name in feature_names:
        low = str(name).lower()
        for hint in FORBIDDEN_INPUT_HINTS:
            if hint in low:
                bad.append(name)
                break
    if bad:
        raise ValueError(f"forbidden model-input feature names present: {bad}")


class SC5MultiSuiteDetectorRuntime:
    """Strict checkpoint loader and one-shot primary-emission FSM."""

    def __init__(
        self,
        checkpoint_path: str,
        tau_corridor: Optional[float] = None,
        tau_release: Optional[float] = None,
        tau_primary: Optional[float] = None,
        guard: Optional[int] = None,
        require_primary_event_role: bool = True,
    ):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        feat_names = list(ckpt.get("feature_names", []))
        if feat_names != SC5_V2_FEATURES:
            raise ValueError(f"Feature mismatch: got {len(feat_names)}, expected {len(SC5_V2_FEATURES)}")
        validate_no_forbidden_inputs(feat_names)

        phase_classes = list(ckpt.get("phase_classes", []))
        if phase_classes != SC5_V2_PHASES:
            raise ValueError(f"phase_classes mismatch: got {phase_classes}")
        event_roles = list(ckpt.get("event_role_classes", []))
        if event_roles != SC5_V2_EVENT_ROLES:
            raise ValueError(f"event_role_classes mismatch: got {event_roles}")

        dataset_sha = str(ckpt.get("dataset_sha256", ""))
        if not dataset_sha:
            raise ValueError("Missing dataset_sha256")
        self.dataset_sha256 = dataset_sha
        with open(checkpoint_path, "rb") as f:
            self.checkpoint_sha256 = hashlib.sha256(f.read()).hexdigest()

        mean = np.asarray(ckpt["mean"], dtype=np.float32)
        std = np.asarray(ckpt["std"], dtype=np.float32)
        if mean.shape[0] != len(SC5_V2_FEATURES) or std.shape[0] != len(SC5_V2_FEATURES):
            raise ValueError(f"mean/std shape mismatch: mean={mean.shape}, std={std.shape}")
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std))):
            raise ValueError("NaN/Inf in mean/std")
        if not np.all(std > 0):
            raise ValueError("Zero in std")
        self.mean = mean
        self.std = std

        hidden = int(ckpt.get("hidden", 64))
        self.model = SC5MultiSuiteMLP(n_feat=len(SC5_V2_FEATURES), hidden=hidden)
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.eval()

        thresholds = dict(ckpt.get("thresholds", {}) or {})
        self.tau_c = float(tau_corridor if tau_corridor is not None else thresholds.get("tau_corridor", 0.3))
        self.tau_r = float(tau_release if tau_release is not None else thresholds.get("tau_release", 0.3))
        self.tau_p = float(tau_primary if tau_primary is not None else thresholds.get("tau_primary", 0.5))
        self.guard = int(guard if guard is not None else thresholds.get("guard", 5))
        self.require_primary_event_role = bool(require_primary_event_role)
        self.reset()

    def reset(self) -> None:
        self.state = "IDLE"
        self.arm_step = -1
        self.emit_step = -1
        self.emitted = False

    def update(self, features_25d: Dict[str, float], step: int) -> Dict[str, object]:
        if self.emitted:
            return self._decision(step)
        x = np.array([[features_25d[fn] for fn in SC5_V2_FEATURES]], dtype=np.float32)
        if not np.all(np.isfinite(x)):
            raise ValueError("NaN/Inf in input features")
        x = (x - self.mean) / (self.std + 1e-8)
        with torch.no_grad():
            out = self.model(torch.tensor(x, dtype=torch.float32))
        corridor_p = torch.sigmoid(out["corridor_logit"]).item()
        release_p = torch.sigmoid(out["release_logit"]).item()
        phase = SC5_V2_PHASES[out["phase_logits"][0].argmax().item()]
        event_probs = torch.softmax(out["event_role_logits"][0], dim=0).cpu().numpy()
        event_role_idx = int(np.argmax(event_probs))
        event_role = SC5_V2_EVENT_ROLES[event_role_idx]
        primary_p = float(event_probs[SC5_V2_EVENT_ROLES.index("primary_attackable")])
        primary_ok = (primary_p >= self.tau_p and event_role == "primary_attackable") or (not self.require_primary_event_role)

        if self.state == "IDLE":
            if phase == "stable_carry" and corridor_p > self.tau_c and primary_ok:
                self.state = "ARMED"
                self.arm_step = int(step)
        elif self.state == "ARMED":
            if step >= self.arm_step + self.guard and corridor_p > self.tau_c and release_p < self.tau_r and primary_ok:
                self.state = "EMITTED"
                self.emit_step = int(step)
                self.emitted = True
        return self._decision(step, corridor_p, release_p, phase, event_role, primary_p)

    def _decision(self, step: int, corridor_p=None, release_p=None, phase=None, event_role=None, primary_p=None) -> Dict[str, object]:
        return {
            "state": self.state,
            "arm_step": self.arm_step,
            "emit_step": self.emit_step,
            "emitted": self.emitted,
            "corridor_p": corridor_p,
            "release_p": release_p,
            "pred_phase": phase,
            "pred_event_role": event_role,
            "primary_p": primary_p,
            "step": int(step),
        }
