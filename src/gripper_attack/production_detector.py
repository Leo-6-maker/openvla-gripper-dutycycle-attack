"""D4.2b: Production-grade streaming first-trigger detector.

Accepts ONLY deployment-safe per-step inputs:
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

Fail-closed validity: missing/invalid fields → abstain or disabled.
"""

from __future__ import annotations
import os, sys
import numpy as np
import torch
from typing import Optional

# Ensure scripts/stageb is on path for train_d1b_detector import
_stageb_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "stageb")
if os.path.isdir(_stageb_path):
    sys.path.insert(0, _stageb_path)


class ProductionStreamingDetector:
    """Causal first-trigger detector using only deployment-safe per-step inputs.

    State machine maintains:
      - gripper history (raw, env, OPEN/CLOSE state)
      - EEF velocity (3-step delta)
      - CLOSE candidate list
      - first-trigger lock (at most one emission)
    """

    def __init__(self, model, means, stdevs, impute, threshold=0.236312, device="cpu"):
        import torch
        self.model = model
        self.means = means
        self.stdevs = stdevs
        self.impute = impute
        self.threshold = threshold
        self.device = device

        # Internal state — reset per episode
        self._reset_episode()

    def _reset_episode(self):
        self.step = 0
        self.history = []  # list of per-step feature dicts (for rule_based_close_predictor compat)
        self.prev_raw = None
        self.prev_gripper_valid = True
        self.curr_gripper_valid = True
        self.close_streak = 0
        self.close_steps = []  # candidate step indices
        self.open_steps = []   # OPEN step indices
        self.emit_step = -1
        self.emit_idx = -1
        self.candidate_features = []  # for audit

    def reset(self):
        """Reset state for new episode."""
        self._reset_episode()

    def update(self, raw_gripper: float, env_gripper: float, gripper_qpos: float,
               eef_x: float, eef_y: float, eef_z: float,
               decoded_open: int,
               raw_valid: bool = True, env_valid: bool = True,
               qpos_valid: bool = True, eef_valid: bool = True,
               gripper_semantics_valid: bool = True) -> Optional[dict]:
        """Process one step. Returns None or (candidate_step, features_dict, mlp_score)."""
        step = self.step
        self.step += 1

        # ── Fail-closed validity ──
        raw_ok = raw_valid and env_valid
        qpos_ok = qpos_valid
        eef_ok = eef_valid
        semantics_ok = gripper_semantics_valid

        # Build record compatible with rule_based_close_predictor
        record = {
            "step": step,
            "clean_gripper_env": env_gripper if env_valid else "",
            "clean_gripper_raw": raw_gripper if raw_valid else "",
            "clean_gripper_raw_proxy": raw_gripper if raw_valid else "",
            "gripper_qpos_before": gripper_qpos if qpos_ok else "",
            "eef_x": eef_x if eef_ok else "",
            "eef_y": eef_y if eef_ok else "",
            "eef_z": eef_z if eef_ok else "",
            "decoded_open_bool": decoded_open,
            "gripper_semantics_valid": str(int(semantics_ok)),
        }
        self.history.append(record)

        # Track OPEN steps
        if decoded_open == 1:
            self.open_steps.append(step)

        # ── CLOSE/onset/streak from raw env ──
        # CLOSE = env_gripper > 0.5
        clean_close = 1 if (env_valid and env_gripper > 0.5) else 0
        # CLOSE onset: first step of a close sequence
        close_onset = 1 if (clean_close and self.close_streak == 0) else 0

        if clean_close:
            self.close_streak += 1
        else:
            self.close_streak = 0

        # Raw crossing detection (fail-closed)
        raw_crossing = False
        if self.prev_raw is not None and self.prev_gripper_valid and semantics_ok and raw_ok:
            if self.prev_raw > 0.5 and raw_gripper <= 0.5:
                raw_crossing = True

        is_candidate = raw_crossing or bool(close_onset) or self.close_streak == 1

        self.prev_raw = raw_gripper if raw_ok else None
        self.prev_gripper_valid = semantics_ok

        # Add derived fields to record for predictor compatibility
        record["clean_close"] = clean_close
        record["close_onset"] = close_onset
        record["close_streak"] = self.close_streak
        record["qpos_abs_before"] = abs(gripper_qpos) if qpos_ok else 0.0
        record["decoded_open_bool"] = decoded_open
        record["gripper_semantics_valid"] = str(int(semantics_ok))

        if not is_candidate:
            return None

        # ── Record candidate ──
        self.close_steps.append(step)

        # Compute features using rule_based_close_predictor (causal, uses history[:step+1])
        from gripper_attack.critical_close_selector import rule_based_close_predictor, PREDICTION_HORIZON
        preds = rule_based_close_predictor(self.history, horizon=PREDICTION_HORIZON, teacher_anchor=-1)
        pred = preds[step]

        features = self._extract_features(pred, step)

        # MLP score
        from train_d1b_detector import FEATURE_NAMES as FN, normalize_features
        X = normalize_features([features], self.means, self.stdevs, self.impute)
        X = X.to(self.device)
        with torch.no_grad():
            score = float(self.model(X).item())

        # First-trigger check
        if self.emit_step < 0 and score >= self.threshold:
            self.emit_step = step
            self.emit_idx = len(self.close_steps) - 1

        self.candidate_features.append((step, features, round(score, 6)))
        return {"step": step, "features": features, "score": round(score, 6)}

    def _extract_features(self, pred, step):
        """Extract all 16 features from predictor output."""
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
                sn = pred.get("eef_speed_now", ""); sp = pred.get("eef_speed_prev", "")
                features[fn] = round(float(sn) - float(sp), 6) if sn != "" and sp != "" else ""
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
