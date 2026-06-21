#!/usr/bin/env python3
"""SC5 shared detector runtime — proper nn.Module, strict loading, single state machine.

Frozen trigger logic (matches offline replay 6/6 Gate):
  IDLE → (pred_phase==stable_carry AND corridor_p > tau_c) → ARMED
  ARMED → (step >= arm_step + 5 AND corridor_p > tau_c AND release_p < tau_r) → EMITTED
  EMITTED → one-shot latch
"""
import hashlib, numpy as np, torch, torch.nn as nn
from typing import Dict

SC5_FEATURES = [
    "gripper_command","gripper_qpos","gripper_opening_proxy",
    "eef_x","eef_y","eef_z","eef_vx","eef_vy","eef_vz",
    "action_dx","action_dy","action_dz","action_gripper",
    "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
    "close_onset","time_since_close","eef_speed",
    "eef_z_delta_since_close","qpos_delta_1","qpos_delta_3",
    "opening_proxy_delta_3","opening_proxy_variance_5","eef_speed_variance_5",
]
SC5_PHASES = ["approach","grasp_close","stable_grasp","first_lift","stable_carry",
              "pre_place_unsupported","release_safe","recovery_or_regrasp","abstain_unsupported"]


class SC5MLP(nn.Module):
    """Exact match to train_sc5_v4.SC5MLP architecture."""
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.phase_head = nn.Linear(hidden, len(SC5_PHASES))
        self.corridor_head = nn.Linear(hidden, 1)
        self.release_head = nn.Linear(hidden, 1)
        self.confidence_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return {"phase_logits": self.phase_head(h),
                "corridor_logit": self.corridor_head(h),
                "release_logit": self.release_head(h)}


class SC5DetectorRuntime:
    """Shared runtime: strict-loads frozen MLP, runs single state machine."""

    def __init__(self, checkpoint_path: str, tau_corridor: float = 0.3,
                 tau_release: float = 0.3, guard: int = 5,
                 allowed_split_modes=("frozen",)):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Strict validation
        feat_names = list(ckpt.get("feature_names", []))
        if feat_names != SC5_FEATURES:
            raise ValueError(f"Feature mismatch: got {len(feat_names)}, expected {len(SC5_FEATURES)}")
        ds_sha = ckpt.get("dataset_sha256", "")
        if not ds_sha:
            raise ValueError("Missing dataset_sha256")
        self.dataset_sha256 = ds_sha
        with open(checkpoint_path, 'rb') as f:
            self.checkpoint_sha256 = hashlib.sha256(f.read()).hexdigest()

        phase_classes = list(ckpt.get("phase_classes", []))
        if phase_classes != SC5_PHASES:
            raise ValueError(f"phase_classes mismatch: got {len(phase_classes)}, expected {len(SC5_PHASES)}")

        mean = ckpt["mean"]; std = ckpt["std"]
        if mean.shape[0] != 25 or std.shape[0] != 25:
            raise ValueError(f"mean/std shape {mean.shape}")
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std))):
            raise ValueError("NaN/Inf in mean/std")
        if not np.all(std > 0):
            raise ValueError("Zero in std")

        split_mode = ckpt.get("split_mode", "unknown")
        if split_mode not in set(allowed_split_modes):
            raise ValueError(f"Checkpoint split_mode={split_mode}, expected one of {sorted(set(allowed_split_modes))}")

        # Build model + strict load
        self.model = SC5MLP(n_feat=len(SC5_FEATURES))
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.eval()
        self.mean = mean; self.std = std

        # Trigger params
        self.tau_c = tau_corridor; self.tau_r = tau_release; self.guard = guard
        self.reset()

    def reset(self):
        self.state = "IDLE"; self.arm_step = -1; self.emit_step = -1; self.emitted = False

    def update(self, features_25d: Dict[str, float], step: int) -> dict:
        if self.emitted:
            return self._decision(step)
        X = np.array([[features_25d[fn] for fn in SC5_FEATURES]], dtype=np.float32)
        if not np.all(np.isfinite(X)):
            raise ValueError("NaN/Inf in input features")
        X = (X - self.mean) / (self.std + 1e-8)
        with torch.no_grad():
            out = self.model(torch.tensor(X, dtype=torch.float32))
        cp = torch.sigmoid(out["corridor_logit"]).item()
        rp = torch.sigmoid(out["release_logit"]).item()
        pp = SC5_PHASES[out["phase_logits"][0].argmax().item()]

        if self.state == "IDLE":
            if pp == "stable_carry" and cp > self.tau_c:
                self.state = "ARMED"; self.arm_step = step
        elif self.state == "ARMED":
            if step >= self.arm_step + self.guard and cp > self.tau_c and rp < self.tau_r:
                self.state = "EMITTED"; self.emit_step = step; self.emitted = True
        return self._decision(step, cp, rp, pp)

    def _decision(self, step, cp=None, rp=None, pp=None):
        return {"state": self.state, "arm_step": self.arm_step, "emit_step": self.emit_step,
                "emitted": self.emitted, "corridor_p": cp, "release_p": rp,
                "pred_phase": pp, "step": step}
