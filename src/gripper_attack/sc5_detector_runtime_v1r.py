#!/usr/bin/env python3
"""SC5 V1R detector runtime — revocable state machine with hysteresis.

Preserves original SC5DetectorRuntime (legacy_v1) semantics by default.
Adds two new FSM versions:

  R1 ("v1r_r1"): Minimal disarm — ARMED→IDLE on evidence loss.
  R2 ("v1r_r2"): Full candidate-machine with hysteresis and arm timeout.

R2 timing: N_min - 1 + guard steps from first on-evidence frame to earliest
possible emit (3 - 1 + 5 = 7 steps with defaults). This is by design — the
candidate phase requires sustained evidence before arming.

Shared with legacy: SC5MLP model, SC5_FEATURES, SC5_PHASES, tau defaults.
Only the state machine transition logic differs.

Usage:
  detector = SC5DetectorRuntimeV1R(ckpt_path, fsm_version="v1r_r1")
  # Online: model inference
  decision = detector.update(features_25d, step)
  # Offline: replay from recorded scores (no model, no GPU)
  decision = detector.update_from_scores(cp, rp, phase, step, feat_valid=True)
"""
import hashlib, numpy as np, torch, torch.nn as nn
from typing import Dict, Optional

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

DISARM_REASONS = [
    "FEATURE_INVALID", "PHASE_EXIT", "CORRIDOR_DROP", "RELEASE_RISE",
    "ARM_TIMEOUT", "CANDIDATE_BREAK",
]


class SC5MLP(nn.Module):
    """Exact copy of SC5DetectorRuntime.SC5MLP for independent module use."""
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


class SC5DetectorRuntimeV1R:
    """Revocable state machine with configurable FSM version.

    fsm_version:
      "legacy_v1" — original IDLE→ARMED→EMITTED (no disarm), for regression
      "v1r_r1"   — minimal disarm on evidence loss
      "v1r_r2"   — full candidate-machine with hysteresis + timeout
    """

    def __init__(self, checkpoint_path: str, tau_corridor: float = 0.3,
                 tau_release: float = 0.3, guard: int = 5,
                 fsm_version: str = "legacy_v1",
                 tau_on: float = 0.5, tau_off: float = 0.3,
                 n_candidate: int = 3, max_arm_age: int = 50):
        if fsm_version not in ("legacy_v1", "v1r_r1", "v1r_r2"):
            raise ValueError(f"Unknown fsm_version: {fsm_version}")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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
            raise ValueError(f"phase_classes mismatch")

        mean = ckpt["mean"]; std = ckpt["std"]
        if mean.shape[0] != 25 or std.shape[0] != 25:
            raise ValueError(f"mean/std shape {mean.shape}")
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std))):
            raise ValueError("NaN/Inf in mean/std")
        if not np.all(std > 0):
            raise ValueError("Zero in std")

        split_mode = ckpt.get("split_mode", "unknown")
        if split_mode != "frozen":
            raise ValueError(f"Checkpoint split_mode={split_mode}, expected 'frozen'")

        self.model = SC5MLP(n_feat=len(SC5_FEATURES))
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.eval()
        self.mean = mean; self.std = std

        # FSM config
        self.fsm_version = fsm_version
        self.tau_c = tau_corridor
        self.tau_r = tau_release
        self.guard = guard
        self.tau_on = tau_on
        self.tau_off = tau_off
        self.n_candidate = n_candidate
        self.max_arm_age = max_arm_age

        self._validate_config()

        self.reset()

    def _validate_config(self):
        """Reject nonsensical parameter combinations."""
        if self.tau_on <= self.tau_off:
            raise ValueError(f"tau_on ({self.tau_on}) must be > tau_off ({self.tau_off})")
        if self.n_candidate < 1:
            raise ValueError(f"n_candidate ({self.n_candidate}) must be >= 1")
        if self.guard < 0:
            raise ValueError(f"guard ({self.guard}) must be >= 0")
        if self.max_arm_age < 1:
            raise ValueError(f"max_arm_age ({self.max_arm_age}) must be >= 1")

    # ── Public API ─────────────────────────────────────────────────

    def reset(self):
        self.state = "IDLE"
        self.arm_step = -1
        self.emit_step = -1
        self.emitted = False
        # History for telemetry cleanliness
        self.last_arm_step = -1
        self.last_candidate_step = -1
        # R1/R2 fields
        self.candidate_step = -1
        self.candidate_streak = 0
        self.arm_age = 0
        self.evidence_valid = False
        self.disarm_count = 0
        self.last_disarm_step = -1
        self.disarm_reason = ""

    def update(self, features_25d: Dict[str, float], step: int) -> dict:
        """Online: compute model predictions, then run FSM."""
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

        return self.update_from_scores(cp, rp, pp, step, feat_valid=True)

    def update_from_scores(self, corridor_p: float, release_p: float,
                           pred_phase: str, step: int,
                           feat_valid: bool = True) -> dict:
        """Offline: run FSM directly from recorded scores. No model, no GPU.

        This is the authoritative FSM entry point — update() delegates to it.
        """
        if self.emitted:
            return self._decision(step, corridor_p, release_p, pred_phase)

        cp = corridor_p; rp = release_p; pp = pred_phase

        # Validate NaN
        if cp is None or (isinstance(cp, float) and np.isnan(cp)):
            cp = float("nan")
        if rp is None or (isinstance(rp, float) and np.isnan(rp)):
            rp = float("nan")

        self.evidence_valid = self._check_arm_evidence(cp, rp, pp)

        # Compute arm_age before FSM update so timeout check uses current step
        if self.state == "ARMED":
            self.arm_age = step - self.arm_step
        elif self.state != "ARMED":
            self.arm_age = 0

        if self.fsm_version == "legacy_v1":
            self._update_legacy(step, cp, rp, pp)
        elif self.fsm_version == "v1r_r1":
            self._update_r1(step, cp, rp, pp, feat_valid)
        elif self.fsm_version == "v1r_r2":
            self._update_r2(step, cp, rp, pp, feat_valid)

        return self._decision(step, cp, rp, pp)

    # ── Evidence helpers ───────────────────────────────────────────

    def _check_arm_evidence(self, cp, rp, pp):
        """Conditions that would satisfy ARM in legacy v1."""
        return (pp == "stable_carry" and cp is not None
                and not np.isnan(cp) and cp > self.tau_c)

    def _check_keep_evidence(self, cp, rp, pp):
        """Conditions required to STAY armed (keep conditions)."""
        if cp is None or np.isnan(cp):
            return False
        if rp is None or np.isnan(rp):
            return False
        return (pp == "stable_carry" and cp > self.tau_off and rp < self.tau_r)

    def _check_on_evidence(self, cp, rp, pp):
        """Stricter conditions for entering candidate state (R2)."""
        if cp is None or np.isnan(cp):
            return False
        if rp is None or np.isnan(rp):
            return False
        return (pp == "stable_carry" and cp > self.tau_on and rp < self.tau_r)

    # ── FSM implementations ────────────────────────────────────────

    def _update_legacy(self, step, cp, rp, pp):
        """Original frozen logic — no disarm path."""
        if self.state == "IDLE":
            if pp == "stable_carry" and not np.isnan(cp) and cp > self.tau_c:
                self.state = "ARMED"; self.arm_step = step; self.disarm_reason = ""
        elif self.state == "ARMED":
            if step >= self.arm_step + self.guard and not np.isnan(cp) and cp > self.tau_c and not np.isnan(rp) and rp < self.tau_r:
                self.state = "EMITTED"; self.emit_step = step; self.emitted = True

    def _update_r1(self, step, cp, rp, pp, feat_valid):
        """R1: Minimal disarm on evidence loss. Same arm conditions as legacy."""
        if self.state == "IDLE":
            if self._check_arm_evidence(cp, rp, pp):
                self.state = "ARMED"; self.arm_step = step; self.disarm_reason = ""

        elif self.state == "ARMED":
            disarm = False
            keep = self._check_keep_evidence(cp, rp, pp)
            if not keep:
                disarm = True
                self.disarm_reason = self._classify_disarm(feat_valid, cp, rp, pp)
            if disarm:
                self.state = "IDLE"
                self.disarm_count += 1
                self.last_disarm_step = step
                self.last_arm_step = self.arm_step
                self.arm_step = -1
                self.arm_age = 0
            elif step >= self.arm_step + self.guard and not np.isnan(cp) and cp > self.tau_c and not np.isnan(rp) and rp < self.tau_r:
                self.state = "EMITTED"; self.emit_step = step; self.emitted = True

    def _update_r2(self, step, cp, rp, pp, feat_valid):
        """R2: Full candidate-machine with hysteresis + timeout."""
        if self.state == "IDLE":
            if self._check_on_evidence(cp, rp, pp):
                self.state = "CANDIDATE"
                self.candidate_step = step
                self.candidate_streak = 1
                self.disarm_reason = ""

        elif self.state == "CANDIDATE":
            if self._check_on_evidence(cp, rp, pp):
                self.candidate_streak += 1
                if self.candidate_streak >= self.n_candidate:
                    self.state = "ARMED"; self.arm_step = step
                    self.candidate_streak = 0; self.disarm_reason = ""
            else:
                self.disarm_reason = "CANDIDATE_BREAK"
                self.last_candidate_step = self.candidate_step
                self.state = "IDLE"
                self.candidate_step = -1
                self.candidate_streak = 0

        elif self.state == "ARMED":
            disarm = False
            keep = self._check_keep_evidence(cp, rp, pp)
            if not keep:
                disarm = True
                self.disarm_reason = self._classify_disarm(feat_valid, cp, rp, pp)
            elif self.arm_age >= self.max_arm_age:
                disarm = True
                self.disarm_reason = "ARM_TIMEOUT"

            if disarm:
                self.state = "IDLE"
                self.disarm_count += 1
                self.last_disarm_step = step
                self.last_arm_step = self.arm_step
                self.arm_step = -1
                self.arm_age = 0
            elif step >= self.arm_step + self.guard:
                self.state = "EMITTED"; self.emit_step = step; self.emitted = True

    # ── Helpers ────────────────────────────────────────────────────

    def _classify_disarm(self, feat_valid, cp, rp, pp):
        """Classify disarm reason for telemetry."""
        if not feat_valid:
            return "FEATURE_INVALID"
        if pp != "stable_carry":
            return "PHASE_EXIT"
        if cp is None or np.isnan(cp) or cp <= self.tau_off:
            return "CORRIDOR_DROP"
        if rp is None or np.isnan(rp) or rp >= self.tau_r:
            return "RELEASE_RISE"
        return "UNKNOWN"

    def _decision(self, step, cp=None, rp=None, pp=None):
        return {
            "state": self.state,
            "arm_step": self.arm_step,
            "emit_step": self.emit_step,
            "emitted": self.emitted,
            "corridor_p": cp,
            "release_p": rp,
            "pred_phase": pp,
            "step": step,
            # Extended telemetry
            "fsm_version": self.fsm_version,
            "candidate_step": self.candidate_step,
            "candidate_streak": self.candidate_streak,
            "arm_age": self.arm_age,
            "evidence_valid": self.evidence_valid,
            "disarm_count": self.disarm_count,
            "last_disarm_step": self.last_disarm_step,
            "disarm_reason": self.disarm_reason,
        }
