"""D4.2c: Production-grade streaming first-trigger detector (fail-closed repair).

Accepts ONLY deployment-safe per-step inputs:
  - step_id (strict sequence 0,1,2,...)
  - raw gripper command (float 0-1)
  - environment gripper (float, usually -1 or +1)
  - gripper qpos (float)
  - EEF x, y, z (float)
  - decoded_open flag (int 0/1)
  - validity flags

Internally computes all derived fields. Never accesses:
  - future steps
  - Teacher-P
  - committed candidate table
  - episode total length
  - attack outcomes

Fail-closed validity: missing/invalid fields → skip or abstain.
Honors predictor abstention: gripper_semantics_invalid, gripper_already_open,
  too_early, low_confidence → no emission.
"""

from __future__ import annotations
import math
import os
import sys
from typing import Optional

import torch

# Ensure scripts/stageb is on path for train_d1b_detector import
_stageb_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "stageb")
if os.path.isdir(_stageb_path):
    sys.path.insert(0, _stageb_path)


def _is_valid_float(val) -> bool:
    """Check if val is a finite float or int (not None, NaN, inf, bool)."""
    if val is None:
        return False
    if isinstance(val, bool):
        return False
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return False
        return True
    return False


def _is_valid_binary(val) -> bool:
    """Check if val is exactly 0 or 1 (int, float, or bool)."""
    if val is None:
        return False
    if isinstance(val, bool):
        return True
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return False
        return val == 0 or val == 1
    return False


class ProductionStreamingDetector:
    """Causal first-trigger detector using only deployment-safe per-step inputs.

    State machine maintains:
      - gripper history (raw, env, OPEN/CLOSE state)
      - EEF velocity (3-step delta)
      - CLOSE candidate list
      - first-trigger lock (at most one emission)

    Fail-closed: invalid inputs, predictor abstain, or validity failures
    prevent candidate generation and emission.
    """

    def __init__(self, model, means, stdevs, impute, threshold=0.236312, device="cpu"):
        self.model = model
        self.means = means
        self.stdevs = stdevs
        self.impute = impute
        self.threshold = threshold
        self.device = device

        # Internal state — reset per episode
        self._reset_episode()

    def _reset_episode(self):
        self._next_expected_step = 0
        self.history = []          # list of per-step record dicts
        self.prev_raw = None
        self.prev_raw_valid = False
        self.prev_gripper_valid = True
        self.close_streak = 0
        self.close_steps = []      # candidate step indices
        self.open_steps = []       # OPEN step indices
        self.emit_step = -1
        self.emit_idx = -1
        self.candidate_features = []  # (step, features, score, abstain_reason)

    def reset(self):
        """Reset state for new episode. Must be called between episodes."""
        self._reset_episode()

    # ── Public read-only accessors ──

    @property
    def next_expected_step(self) -> int:
        return self._next_expected_step

    @property
    def has_emitted(self) -> bool:
        return self.emit_step >= 0

    # ── Main update ──

    def update(self, step_id: int,
               raw_gripper: float, env_gripper: float, gripper_qpos: float,
               eef_x: float, eef_y: float, eef_z: float,
               decoded_open: int,
               raw_valid: bool = True, env_valid: bool = True,
               qpos_valid: bool = True, eef_valid: bool = True,
               gripper_semantics_valid: bool = True) -> Optional[dict]:
        """Process one step. Returns None (no candidate) or result dict.

        Args:
            step_id: Strict monotonic sequence 0, 1, 2, ...
            raw_gripper: Raw gripper command (0-1 range).
            env_gripper: Environment gripper (-1 or +1 typically).
            gripper_qpos: Gripper joint position.
            eef_x, eef_y, eef_z: End-effector position.
            decoded_open: 0 (not open) or 1 (open).
            raw_valid: Whether raw_gripper is valid.
            env_valid: Whether env_gripper is valid.
            qpos_valid: Whether gripper_qpos is valid.
            eef_valid: Whether all EEF coordinates are valid.
            gripper_semantics_valid: Whether gripper semantics are valid.

        Returns:
            None if no candidate, or dict with:
              step, features, normalized_features, score,
              abstain, abstained

        Raises:
            ValueError: on step sequence violation.
        """
        # ── Strict step sequence ──
        if step_id != self._next_expected_step:
            raise ValueError(
                f"Step sequence violation: expected {self._next_expected_step}, got {step_id}"
            )
        self._next_expected_step = step_id + 1

        # ── None / NaN / inf fail-closed ──
        raw_ok = raw_valid and _is_valid_float(raw_gripper)
        env_ok = env_valid and _is_valid_float(env_gripper)
        qpos_ok = qpos_valid and _is_valid_float(gripper_qpos)
        eef_ok = eef_valid and all(_is_valid_float(v) for v in (eef_x, eef_y, eef_z))

        # ── decoded_open validity ──
        decoded_open_ok = _is_valid_binary(decoded_open)

        semantics_ok = gripper_semantics_valid

        # Build record compatible with rule_based_close_predictor
        record = {
            "step": step_id,
            "clean_gripper_env": float(env_gripper) if env_ok else "",
            "clean_gripper_raw": float(raw_gripper) if raw_ok else "",
            "clean_gripper_raw_proxy": float(raw_gripper) if raw_ok else "",
            "gripper_qpos_before": float(gripper_qpos) if qpos_ok else "",
            "eef_x": float(eef_x) if eef_ok else "",
            "eef_y": float(eef_y) if eef_ok else "",
            "eef_z": float(eef_z) if eef_ok else "",
            "decoded_open_bool": int(decoded_open) if decoded_open_ok else "",
            "gripper_semantics_valid": str(int(semantics_ok)),
        }
        self.history.append(record)

        # Track OPEN steps (only when decoded_open is valid)
        if decoded_open_ok and decoded_open == 1:
            self.open_steps.append(step_id)

        # ── CLOSE / onset / streak (fail-closed: requires env + semantics) ──
        gripper_field_valid = env_ok and semantics_ok

        if gripper_field_valid and env_gripper > 0.5:
            clean_close = 1
        else:
            clean_close = 0

        close_onset = 1 if (clean_close and self.close_streak == 0) else 0

        if clean_close:
            self.close_streak += 1
        else:
            self.close_streak = 0

        # Raw crossing (fail-closed: requires raw + env + semantics)
        raw_crossing = False
        if (self.prev_raw is not None and self.prev_raw_valid
                and self.prev_gripper_valid and semantics_ok and raw_ok and env_ok):
            if self.prev_raw > 0.5 and raw_gripper <= 0.5:
                raw_crossing = True

        # Candidate generation — fail-closed on all required fields
        is_candidate = (raw_crossing or bool(close_onset) or self.close_streak == 1)
        is_candidate = is_candidate and decoded_open_ok

        self.prev_raw = float(raw_gripper) if raw_ok else None
        self.prev_raw_valid = raw_ok
        self.prev_gripper_valid = semantics_ok

        # Add derived fields to record for predictor compatibility
        record["clean_close"] = clean_close
        record["close_onset"] = close_onset
        record["close_streak"] = self.close_streak
        record["qpos_abs_before"] = abs(float(gripper_qpos)) if qpos_ok else 0.0
        record["decoded_open_bool"] = int(decoded_open) if decoded_open_ok else ""
        record["gripper_semantics_valid"] = str(int(semantics_ok))

        if not is_candidate:
            return None

        # ── Record candidate ──
        self.close_steps.append(step_id)

        # Compute causally — predictor sees only history up to current step
        from gripper_attack.critical_close_selector import (
            rule_based_close_predictor,
            PREDICTION_HORIZON,
        )
        preds = rule_based_close_predictor(
            self.history, horizon=PREDICTION_HORIZON, teacher_anchor=-1,
        )
        pred = preds[step_id]

        # ── Honor predictor abstention ──
        abstain_reason = str(pred.get("abstain", ""))
        abstained = abstain_reason != ""

        features = self._extract_features(pred, step_id)

        # Normalize
        from train_d1b_detector import normalize_features

        X = normalize_features([features], self.means, self.stdevs, self.impute)
        X = X.to(self.device)
        norm_vec = [round(float(v), 10) for v in X[0].cpu().tolist()]

        # MLP score
        with torch.no_grad():
            score = float(self.model(X).item())

        # First-trigger — only when NOT abstained
        if not abstained and self.emit_step < 0 and score >= self.threshold:
            self.emit_step = step_id
            self.emit_idx = len(self.close_steps) - 1

        self.candidate_features.append(
            (step_id, features, round(score, 6), abstain_reason)
        )
        return {
            "step": step_id,
            "features": features,
            "normalized_features": norm_vec,
            "score": round(score, 6),
            "abstain": abstain_reason,
            "abstained": abstained,
        }

    def _extract_features(self, pred: dict, step: int) -> dict:
        """Extract all 16 features from predictor output (causal, past-only)."""
        from train_d1b_detector import FEATURE_NAMES as FN

        features = {}
        for fn in FN:
            if fn == "total_score":
                features[fn] = round(pred.get("score", 0), 4)
            elif fn == "raw_crossing_bonus":
                features[fn] = pred.get("raw_crossing_bonus", "")
            elif fn == "close_streak_bonus":
                features[fn] = pred.get("close_streak_bonus", "")
            elif fn == "close_onset_qpos_bonus":
                features[fn] = pred.get("close_onset_qpos_bonus", "")
            elif fn == "eef_deceleration_bonus":
                features[fn] = pred.get("eef_deceleration_bonus", "")
            elif fn == "qpos_ready_bonus":
                features[fn] = pred.get("qpos_ready_bonus", "")
            elif fn == "eef_speed_now":
                features[fn] = pred.get("eef_speed_now", "")
            elif fn == "eef_speed_prev":
                features[fn] = pred.get("eef_speed_prev", "")
            elif fn == "eef_deceleration_delta":
                sn = pred.get("eef_speed_now", "")
                sp = pred.get("eef_speed_prev", "")
                if sn != "" and sp != "":
                    features[fn] = round(float(sn) - float(sp), 6)
                else:
                    features[fn] = ""
            elif fn == "close_streak":
                features[fn] = pred.get("close_streak_value", "")
            elif fn == "raw_crossing":
                features[fn] = int(pred.get("raw_open_to_close_crossing", 0))
            elif fn == "close_onset":
                features[fn] = int(pred.get("close_onset", 0))
            elif fn == "qpos":
                features[fn] = pred.get("qpos", "")
            elif fn == "time_since_prev_close":
                prevs = [s for s in self.close_steps[:-1]]
                features[fn] = step - max(prevs) if prevs else ""
            elif fn == "time_since_last_open":
                priors = [s for s in self.open_steps if s < step]
                features[fn] = step - max(priors) if priors else ""
            elif fn == "candidate_index":
                features[fn] = len(self.close_steps) - 1
        return features
