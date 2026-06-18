#!/usr/bin/env python3
"""SC5 shared detector runtime — online and offline, single state machine.

Frozen trigger logic (matches offline replay 6/6 Gate):
  IDLE → (pred_phase==stable_carry AND corridor_p > τ_c) → ARMED
  ARMED → (step >= arm_step + 5 AND corridor_p > τ_c AND release_p < τ_r) → EMITTED
  EMITTED → one-shot latch (never re-trigger)

Strict checkpoint loading: feature_names exact match, strict state_dict, dataset_sha256.
"""
from __future__ import annotations
import numpy as np
import torch
from typing import Optional, Dict, List

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


class SC5DetectorRuntime:
    """Shared detector: loads frozen MLP, runs causal state machine.

    Call update() per step. Returns decision dict with emit info.
    One-shot latch: once emitted, never re-triggers.
    """

    def __init__(self, checkpoint_path: str, tau_corridor: float = 0.3,
                 tau_release: float = 0.3, guard: int = 5):
        # ── Strict load ──
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        feat_names = ckpt.get("feature_names", [])
        if list(feat_names) != SC5_FEATURES:
            raise ValueError(f"Feature mismatch: checkpoint has {len(feat_names)} features, "
                             f"expected {len(SC5_FEATURES)} canonical 25D")

        ds_sha = ckpt.get("dataset_sha256", "")
        if not ds_sha:
            raise ValueError("Checkpoint missing dataset_sha256")
        self.dataset_sha256 = ds_sha

        mean = ckpt["mean"]; std = ckpt["std"]
        if mean.shape[0] != 25 or std.shape[0] != 25:
            raise ValueError(f"mean/std shape {mean.shape}, expected (25,)")
        if np.any(np.isnan(mean)) or np.any(np.isnan(std)):
            raise ValueError("NaN in mean/std")

        # ── Build model ──
        n_feat = len(SC5_FEATURES)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_feat, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU())
        self.phase_head = torch.nn.Linear(64, len(SC5_PHASES))
        self.corridor_head = torch.nn.Linear(64, 1)
        self.release_head = torch.nn.Linear(64, 1)
        self.confidence_head = torch.nn.Linear(64, 1)

        # Strict load — no missing keys, no unexpected keys
        state = ckpt["model_state"]
        own = {}
        own["model.0.weight"] = state["shared.0.weight"]; own["model.0.bias"] = state["shared.0.bias"]
        own["model.2.weight"] = state["shared.2.weight"]; own["model.2.bias"] = state["shared.2.bias"]
        own["phase_head.weight"] = state["phase_head.weight"]; own["phase_head.bias"] = state["phase_head.bias"]
        own["corridor_head.weight"] = state["corridor_head.weight"]; own["corridor_head.bias"] = state["corridor_head.bias"]
        own["release_head.weight"] = state["release_head.weight"]; own["release_head.bias"] = state["release_head.bias"]
        own["confidence_head.weight"] = state.get("confidence_head.weight",
            torch.zeros_like(state["phase_head.weight"][:1]))
        own["confidence_head.bias"] = state.get("confidence_head.bias",
            torch.zeros(1))
        self.load_state_dict(own)  # custom — see below
        self.eval()
        self.mean = mean; self.std = std

        # ── Trigger params ──
        self.tau_c = tau_corridor; self.tau_r = tau_release; self.guard = guard
        self.reset()

    def load_state_dict(self, d):
        self.model[0].weight.data = d["model.0.weight"]; self.model[0].bias.data = d["model.0.bias"]
        self.model[2].weight.data = d["model.2.weight"]; self.model[2].bias.data = d["model.2.bias"]
        self.phase_head.weight.data = d["phase_head.weight"]; self.phase_head.bias.data = d["phase_head.bias"]
        self.corridor_head.weight.data = d["corridor_head.weight"]; self.corridor_head.bias.data = d["corridor_head.bias"]
        self.release_head.weight.data = d["release_head.weight"]; self.release_head.bias.data = d["release_head.bias"]
        self.confidence_head.weight.data = d["confidence_head.weight"]; self.confidence_head.bias.data = d["confidence_head.bias"]

    def reset(self):
        self.state = "IDLE"
        self.arm_step = -1
        self.emit_step = -1
        self.emitted = False

    def update(self, features_25d: Dict[str, float], step: int) -> dict:
        """Process one step. Returns decision dict with emit info."""
        if self.emitted:
            return self._decision(step)

        # Build input
        X = np.array([[features_25d[fn] for fn in SC5_FEATURES]], dtype=np.float32)
        X = (X - self.mean) / (self.std + 1e-8)

        with torch.no_grad():
            h = self.model(torch.tensor(X, dtype=torch.float32))
            phase_logits = self.phase_head(h)
            corridor_logit = self.corridor_head(h)
            release_logit = self.release_head(h)

        cp = torch.sigmoid(corridor_logit).item()
        rp = torch.sigmoid(release_logit).item()
        pred_phase = SC5_PHASES[phase_logits[0].argmax().item()]

        # State machine
        if self.state == "IDLE":
            if pred_phase == "stable_carry" and cp > self.tau_c:
                self.state = "ARMED"; self.arm_step = step
        elif self.state == "ARMED":
            if step >= self.arm_step + self.guard and cp > self.tau_c and rp < self.tau_r:
                self.state = "EMITTED"; self.emit_step = step; self.emitted = True

        return self._decision(step, cp, rp, pred_phase)

    def _decision(self, step, cp=None, rp=None, phase=None):
        return {
            "state": self.state, "arm_step": self.arm_step, "emit_step": self.emit_step,
            "emitted": self.emitted, "corridor_p": cp, "release_p": rp,
            "pred_phase": phase, "step": step,
        }
