"""D5 Frozen Feature Adapter v1 — capture-time feature semantics.

Snapshot of ProductionStreamingDetector and critical_close_selector.py from
commit 44bf7b86 (D5 data collection). This module recovers the EXACT feature
computation that produced detector_candidates.csv during D4.4D capture.

Rules:
  1. Only accepts deployment-safe per-step inputs.
  2. Does NOT read Teacher-P, detector_candidates.csv, future steps, or
     episode length.
  3. Output features must match saved detector_candidates.csv features
     exactly (float tolerance ≤ 1e-6).
  4. No dependency on evolved critical_close_selector.py — all feature
     computation code is self-contained.

Usage:
    adapter = D5FrozenFeatureAdapter()
    adapter.reset()
    for row in step_trace_csv:
        result = adapter.update(
            step_id, raw_gripper, env_gripper, gripper_qpos,
            eef_x, eef_y, eef_z, decoded_open,
            raw_valid, env_valid, qpos_valid, eef_valid,
            gripper_semantics_valid,
        )
        if result is not None:
            # result = {"step": int, "features": {16 feature dict}}
            pass
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

# ── Frozen config from 44bf7b86 ──
PREDICTION_HORIZON = 4
FEATURE_NAMES = [
    "total_score", "raw_crossing_bonus", "close_streak_bonus",
    "close_onset_qpos_bonus", "eef_deceleration_bonus", "qpos_ready_bonus",
    "eef_speed_now", "eef_speed_prev", "eef_deceleration_delta",
    "close_streak", "raw_crossing", "close_onset", "qpos",
    "time_since_prev_close", "time_since_last_open", "candidate_index",
]


# ── Helper functions (from 44bf7b86) ──

def _is_valid_float(val) -> bool:
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
    if val is None:
        return False
    if isinstance(val, bool):
        return True
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return False
        return val == 0 or val == 1
    return False


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _field_is_present_and_valid(r: dict, field: str) -> bool:
    val = r.get(field)
    if val is None or val == "" or val == "nan" or val == "NaN":
        return False
    validity_flag = r.get(f"{field}_valid")
    if validity_flag is not None and str(validity_flag).strip() in ("0", "False", "false"):
        return False
    try:
        f = float(val)
        return not math.isnan(f)
    except (ValueError, TypeError):
        return False


def _check_feature_validity(r: dict, field: str) -> bool:
    val = r.get(field)
    if val is None or val == "":
        return True
    return str(val).strip() not in ("0", "False", "false")


def _eef_speed_if_valid(records: list[dict], t: int, window: int = 3):
    if t < window:
        return None
    t0, t1 = t - window, t
    eef0_valid = (_field_is_present_and_valid(records[t0], "eef_x") and
                  _field_is_present_and_valid(records[t0], "eef_y") and
                  _field_is_present_and_valid(records[t0], "eef_z"))
    eef1_valid = (_field_is_present_and_valid(records[t1], "eef_x") and
                  _field_is_present_and_valid(records[t1], "eef_y") and
                  _field_is_present_and_valid(records[t1], "eef_z"))
    if not (eef0_valid and eef1_valid):
        return None
    dx = _safe_float(records[t1].get("eef_x", 0)) - _safe_float(records[t0].get("eef_x", 0))
    dy = _safe_float(records[t1].get("eef_y", 0)) - _safe_float(records[t0].get("eef_y", 0))
    dz = _safe_float(records[t1].get("eef_z", 0)) - _safe_float(records[t0].get("eef_z", 0))
    return float(np.sqrt(dx**2 + dy**2 + dz**2))


# ── Frozen rule_based_close_predictor (from 44bf7b86) ──

def _rule_based_close_predictor(records: list[dict],
                                 horizon: int = PREDICTION_HORIZON,
                                 teacher_anchor: int = -1) -> list[dict]:
    """EXACT copy of rule_based_close_predictor from commit 44bf7b86.

    Must NOT be modified — this is the capture-time feature semantics.
    """
    T = len(records)
    predictions = []

    for t in range(T):
        visible = records[:t + 1]
        r = visible[-1]

        clean_close = int(_safe_float(r.get("clean_close", 0)))
        close_onset = int(_safe_float(r.get("close_onset", 0)))
        close_streak = int(_safe_float(r.get("close_streak", 0)))
        decoded_open = int(_safe_float(r.get("decoded_open_bool", 0)))
        qpos = _safe_float(r.get("gripper_qpos_before", 0))
        raw_now = _safe_float(r.get("clean_gripper_raw",
                                     r.get("clean_gripper_raw_proxy", 0.5)))

        gripper_valid = _check_feature_validity(r, "gripper_semantics_valid")
        qpos_valid = _field_is_present_and_valid(r, "gripper_qpos_before")
        eef_valid = (_field_is_present_and_valid(r, "eef_x") and
                     _field_is_present_and_valid(r, "eef_y") and
                     _field_is_present_and_valid(r, "eef_z"))
        raw_valid = (_field_is_present_and_valid(r, "clean_gripper_raw") or
                     _field_is_present_and_valid(r, "clean_gripper_raw_proxy"))
        disabled_features = []

        score = 0.0
        raw_crossing_bonus = 0.0
        close_streak_bonus = 0.0
        close_onset_qpos_bonus = 0.0
        eef_deceleration_bonus = 0.0
        qpos_ready_bonus = 0.0
        decoded_open_penalty = 0.0
        speed_now_val = None
        speed_prev_val = None

        raw_open_to_close_crossing = False
        if t >= 1:
            prev_raw_valid = (_field_is_present_and_valid(visible[t - 1], "clean_gripper_raw") or
                             _field_is_present_and_valid(visible[t - 1], "clean_gripper_raw_proxy"))
            prev_gripper_valid = _check_feature_validity(visible[t - 1], "gripper_semantics_valid")
            curr_gripper_valid = _check_feature_validity(r, "gripper_semantics_valid")
            crossing_allowed = (raw_valid and prev_raw_valid and
                               prev_gripper_valid and curr_gripper_valid)
            if crossing_allowed:
                raw_prev = _safe_float(visible[t - 1].get("clean_gripper_raw",
                        visible[t - 1].get("clean_gripper_raw_proxy", 0.5)))
                if raw_prev > 0.5 and raw_now <= 0.5:
                    raw_open_to_close_crossing = True
                    raw_crossing_bonus = 1.5
                    score += 1.5
            if not crossing_allowed:
                disabled_features.append("raw_crossing")

        if close_streak == 1:
            close_streak_bonus = 1.0
            score += 1.0

        if close_onset and qpos_valid and qpos < 0.005:
            close_onset_qpos_bonus = 0.5
            score += 0.5
        elif close_onset and not qpos_valid:
            disabled_features.append("qpos_close_response")

        if t >= 4:
            speed_now_val = _eef_speed_if_valid(visible, t, window=3)
            speed_prev_val = _eef_speed_if_valid(visible, t - 1, window=3)
            if (speed_now_val is not None and speed_prev_val is not None and
                speed_prev_val > 0 and speed_now_val < speed_prev_val and speed_now_val < 0.01):
                eef_deceleration_bonus = 0.5
                score += 0.5
            elif speed_now_val is None or speed_prev_val is None:
                disabled_features.append("eef_deceleration")

        if qpos_valid and qpos < 0.01 and not decoded_open:
            qpos_ready_bonus = 0.3
            score += 0.3
        elif not qpos_valid:
            disabled_features.append("qpos_ready")

        if decoded_open:
            decoded_open_penalty = -2.0
            score -= 2.0

        is_close_event_candidate = (
            raw_open_to_close_crossing
            or bool(close_onset)
            or close_streak == 1
        )

        abstain = ""
        if not gripper_valid:
            abstain = "gripper_semantics_invalid"
        elif decoded_open:
            abstain = "gripper_already_open"
        elif t < 3:
            abstain = "too_early"
        elif score < 0.5:
            abstain = "low_confidence"

        will_close = False
        close_at = -1
        if teacher_anchor >= 0:
            if t < teacher_anchor <= t + horizon:
                will_close = True
                close_at = teacher_anchor

        predictions.append({
            "step": t,
            "score": max(0.0, score),
            "abstain": abstain,
            "clean_close": clean_close,
            "close_onset": close_onset,
            "qpos": qpos,
            "raw_open_to_close_crossing": raw_open_to_close_crossing,
            "close_streak_value": close_streak,
            "is_close_event_candidate": is_close_event_candidate,
            "will_critical_close_within_horizon": will_close,
            "predicted_close_horizon": close_at - t if close_at > 0 else -1,
            "horizon": horizon,
            "disabled_features": disabled_features,
            "raw_crossing_bonus": raw_crossing_bonus,
            "close_streak_bonus": close_streak_bonus,
            "close_onset_qpos_bonus": close_onset_qpos_bonus,
            "eef_deceleration_bonus": eef_deceleration_bonus,
            "qpos_ready_bonus": qpos_ready_bonus,
            "decoded_open_penalty": decoded_open_penalty,
            "eef_speed_now": round(speed_now_val, 6) if speed_now_val is not None else "",
            "eef_speed_prev": round(speed_prev_val, 6) if speed_prev_val is not None else "",
        })

    return predictions


# ── Frozen feature extraction (from ProductionStreamingDetector 44bf7b86) ──

def _extract_features(pred: dict, step: int, close_steps: list[int],
                      open_steps: list[int]) -> dict:
    """EXACT copy of _extract_features from commit 44bf7b86."""
    features = {}
    for fn in FEATURE_NAMES:
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
            prevs = [s for s in close_steps[:-1]]
            features[fn] = step - max(prevs) if prevs else ""
        elif fn == "time_since_last_open":
            priors = [s for s in open_steps if s < step]
            features[fn] = step - max(priors) if priors else ""
        elif fn == "candidate_index":
            features[fn] = len(close_steps) - 1
    return features


# ── Frozen adapter class ──

FEATURE_SCHEMA_VERSION = "d5_frozen_v1"
SOURCE_COMMIT = "44bf7b86bafdda79837b4089dd5250901bb3ae75"


class D5FrozenFeatureAdapter:
    """Frozen feature adapter recovering capture-time (44bf7b86) semantics.

    Returns per-candidate dict:
      step, features (16), abstain, abstained, candidate_reason,
      feature_schema_version, source_commit
    """

    def __init__(self):
        self._reset_episode()

    def _reset_episode(self):
        self._next_expected_step = 0
        self.history = []
        self.prev_raw = None
        self.prev_raw_valid = False
        self.prev_gripper_valid = True
        self.close_streak = 0
        self.close_steps = []
        self.open_steps = []
        self.candidate_features = []

    def reset(self):
        self._reset_episode()

    @property
    def next_expected_step(self) -> int:
        return self._next_expected_step

    def update(self, step_id: int,
               raw_gripper: float, env_gripper: float, gripper_qpos: float,
               eef_x: float, eef_y: float, eef_z: float,
               decoded_open: int,
               raw_valid: bool = True, env_valid: bool = True,
               qpos_valid: bool = True, eef_valid: bool = True,
               gripper_semantics_valid: bool = True) -> Optional[dict]:
        """Process one step. Returns None (no candidate) or feature dict.

        Raises ValueError on step sequence violation.
        """
        if step_id != self._next_expected_step:
            raise ValueError(
                f"Step sequence violation: expected {self._next_expected_step}, got {step_id}"
            )
        self._next_expected_step = step_id + 1

        # ── Validity gates (from 44bf7b86 ProductionStreamingDetector.update) ──
        raw_valid_ok = _is_valid_binary(raw_valid) and bool(raw_valid)
        env_valid_ok = _is_valid_binary(env_valid) and bool(env_valid)
        qpos_valid_ok = _is_valid_binary(qpos_valid) and bool(qpos_valid)
        eef_valid_ok = _is_valid_binary(eef_valid) and bool(eef_valid)

        raw_ok = raw_valid_ok and _is_valid_float(raw_gripper)
        env_ok = env_valid_ok and _is_valid_float(env_gripper)
        qpos_ok = qpos_valid_ok and _is_valid_float(gripper_qpos)
        eef_ok = eef_valid_ok and all(_is_valid_float(v) for v in (eef_x, eef_y, eef_z))

        decoded_open_ok = _is_valid_binary(decoded_open)

        semantics_ok = (
            _is_valid_binary(gripper_semantics_valid)
            and bool(gripper_semantics_valid)
        )

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

        if decoded_open_ok and decoded_open == 1:
            self.open_steps.append(step_id)

        # ── CLOSE / onset / streak ──
        gripper_field_valid = env_ok and semantics_ok
        clean_close = 1 if (gripper_field_valid and env_gripper > 0.5) else 0
        close_onset = 1 if (clean_close and self.close_streak == 0) else 0

        if clean_close:
            self.close_streak += 1
        else:
            self.close_streak = 0

        # Raw crossing
        raw_crossing = False
        if (self.prev_raw is not None and self.prev_raw_valid
                and self.prev_gripper_valid and semantics_ok and raw_ok and env_ok):
            if self.prev_raw > 0.5 and raw_gripper <= 0.5:
                raw_crossing = True

        is_candidate = (raw_crossing or bool(close_onset) or self.close_streak == 1)
        is_candidate = is_candidate and decoded_open_ok

        self.prev_raw = float(raw_gripper) if raw_ok else None
        self.prev_raw_valid = raw_ok
        self.prev_gripper_valid = semantics_ok

        # Add derived fields for predictor compatibility
        record["clean_close"] = clean_close
        record["close_onset"] = close_onset
        record["close_streak"] = self.close_streak
        record["qpos_abs_before"] = abs(float(gripper_qpos)) if qpos_ok else 0.0
        record["decoded_open_bool"] = int(decoded_open) if decoded_open_ok else ""
        record["gripper_semantics_valid"] = str(int(semantics_ok))

        if not is_candidate:
            return None

        self.close_steps.append(step_id)

        # Determine candidate trigger reason
        reasons = []
        if raw_crossing:
            reasons.append("raw_crossing")
        if bool(close_onset):
            reasons.append("close_onset")
        if self.close_streak == 1:
            reasons.append("close_streak_first")
        candidate_reason = "+".join(reasons) if reasons else "unknown"

        # Compute features using FROZEN predictor + frozen extractor
        preds = _rule_based_close_predictor(self.history, horizon=PREDICTION_HORIZON, teacher_anchor=-1)
        pred = preds[step_id]
        features = _extract_features(pred, step_id, self.close_steps, self.open_steps)

        self.candidate_features.append((step_id, features))
        return {
            "step": step_id,
            "features": features,
            "abstain": pred.get("abstain", ""),
            "abstained": bool(pred.get("abstain", "")),
            "candidate_reason": candidate_reason,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "source_commit": SOURCE_COMMIT,
        }
