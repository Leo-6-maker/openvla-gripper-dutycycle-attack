"""Layer2: Critical first-close opportunity scorer.

Predicts when a task-critical first CLOSE event will occur, using only
deployment-safe (causal) features. Does not use attack outcomes.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import numpy as np

from .window_contract import WindowProposal

# ── Frozen config ──
WINDOW_LEN = 10
PRE_OFFSET = 2
HISTORY_LEN = 16
PREDICTION_HORIZON = 4         # H: predict critical close within [t+1, t+H]
TIE_TOLERANCE = 0.5            # score tie tolerance for ambiguous close detection
MIN_CLOSE_SEPARATION = 10      # steps: min separation to consider two closes distinct
EVENT_SCORE_FLOOR = 0.5        # min score for ambiguity consideration
SELECTOR_VERSION = "l12_close_event_interceptor_v4"
FEATURE_SCHEMA_VERSION = "l12_v4"


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _field_is_present_and_valid(r: dict, field: str) -> bool:
    """Check a field is present, non-empty, and non-NaN in the record."""
    val = r.get(field)
    if val is None or val == "" or val == "nan" or val == "NaN":
        return False
    # Check explicit validity flags when present
    validity_flag = r.get(f"{field}_valid")
    if validity_flag is not None and str(validity_flag).strip() in ("0", "False", "false"):
        return False
    try:
        f = float(val)
        import numpy as np
        return not np.isnan(f)
    except (ValueError, TypeError):
        return False


def _check_feature_validity(r: dict, field: str) -> bool:
    """Check a boolean validity flag field (e.g. gripper_semantics_valid)."""
    val = r.get(field)
    if val is None or val == "":
        return True  # flag absent → assume valid
    return str(val).strip() not in ("0", "False", "false")


def extract_deployment_features(records: list[dict]) -> tuple:
    """Extract deployment-safe feature matrix from clean trace records.

    Returns (feats, validity) where validity[i,j] is True iff the
    underlying field was present and non-NaN. Validity is NOT derived
    from a different field — each feature's validity comes from its
    own source column(s).

    DEPRECATED: silently fills missing with zero. Callers should prefer
    rule_based_close_predictor which uses per-field validity gates.
    """
    T = len(records)
    F = 13
    feats = np.zeros((T, F), dtype=np.float32)
    validity = np.zeros((T, F), dtype=np.bool_)

    for t in range(T):
        r = records[t]
        # Env: valid when field present AND gripper semantics not invalid
        env_raw = r.get("clean_gripper_env", "")
        env_valid = _field_is_present_and_valid(r, "clean_gripper_env")
        feats[t, 0] = _safe_float(env_raw, 0.0)
        validity[t, 0] = env_valid

        # Raw: valid when native or proxy field present
        raw_valid_flag = (_field_is_present_and_valid(r, "clean_gripper_raw") or
                          _field_is_present_and_valid(r, "clean_gripper_raw_proxy"))
        feats[t, 1] = _safe_float(r.get("clean_gripper_raw",
                                        r.get("clean_gripper_raw_proxy", 0.0)))
        validity[t, 1] = raw_valid_flag

        # Qpos: value and validity from same field
        feats[t, 2] = _safe_float(r.get("gripper_qpos_before", 0.0))
        validity[t, 2] = _field_is_present_and_valid(r, "gripper_qpos_before")

        # Qpos_abs: derived from qpos value using its validity
        feats[t, 3] = abs(feats[t, 2]) if validity[t, 2] else 0.0
        validity[t, 3] = validity[t, 2]

        # EEF: each coordinate from its own field
        feats[t, 4] = _safe_float(r.get("eef_x", 0.0))
        feats[t, 5] = _safe_float(r.get("eef_y", 0.0))
        feats[t, 6] = _safe_float(r.get("eef_z", 0.0))
        validity[t, 4] = _field_is_present_and_valid(r, "eef_x")
        validity[t, 5] = _field_is_present_and_valid(r, "eef_y")
        validity[t, 6] = _field_is_present_and_valid(r, "eef_z")

        feats[t, 7] = _safe_float(r.get("close_streak", 0))
        validity[t, 7] = _field_is_present_and_valid(r, "close_streak")

        feats[t, 8] = _safe_float(r.get("decoded_open_bool", 0))
        validity[t, 8] = _field_is_present_and_valid(r, "decoded_open_bool")

        # EEF velocity (3-step) — only valid when both endpoints valid
        if t >= 3:
            if validity[t, 4] and validity[t - 3, 4]:
                feats[t, 9] = feats[t, 4] - feats[t - 3, 4]
                validity[t, 9] = True
            if validity[t, 5] and validity[t - 3, 5]:
                feats[t, 10] = feats[t, 5] - feats[t - 3, 5]
                validity[t, 10] = True
            if validity[t, 6] and validity[t - 3, 6]:
                feats[t, 11] = feats[t, 6] - feats[t - 3, 6]
                validity[t, 11] = True
        feats[t, 12] = 1.0  # bias
        validity[t, 12] = True

    return feats, validity


def _eef_speed(records: list[dict], t: int, window: int = 3) -> float:
    """Compute EEF speed magnitude at step t using `window`-step deltas."""
    if t < window:
        return 0.0
    dx = _safe_float(records[t].get("eef_x", 0)) - _safe_float(records[t - window].get("eef_x", 0))
    dy = _safe_float(records[t].get("eef_y", 0)) - _safe_float(records[t - window].get("eef_y", 0))
    dz = _safe_float(records[t].get("eef_z", 0)) - _safe_float(records[t - window].get("eef_z", 0))
    return float(np.sqrt(dx**2 + dy**2 + dz**2))


def _eef_speed_if_valid(records: list[dict], t: int, window: int = 3) -> Optional[float]:
    """Compute EEF speed only when both endpoint positions are valid.
    Returns None if either endpoint has invalid/missing coordinates.
    """
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


def rule_based_close_predictor(records: list[dict],
                                horizon: int = PREDICTION_HORIZON,
                                teacher_anchor: int = -1) -> list[dict]:
    """Rule-based causal predictor: at each step t, score the likelihood that
    a critical CLOSE will occur within [t+1, t+horizon].

    Uses only deployment-safe features from records[:t+1] (strictly causal).
    Does NOT use absolute step thresholds (t < 60, t > 200).
    close_onset is a signal, not the dominant +3.0 copy of the teacher rule.

    Args:
        records: full clean trace.
        horizon: prediction horizon H (default 4).
        teacher_anchor: teacher-identified critical-close step.
            Used ONLY to compute horizon ground-truth labels, NOT for scoring.

    Returns:
        List of per-step dicts with prediction scores and horizon labels.
    """
    T = len(records)
    predictions = []

    for t in range(T):
        visible = records[:t + 1]
        r = visible[-1]

        # ── Deployment-safe features ──
        clean_close = int(_safe_float(r.get("clean_close", 0)))
        close_onset = int(_safe_float(r.get("close_onset", 0)))
        close_streak = int(_safe_float(r.get("close_streak", 0)))
        decoded_open = int(_safe_float(r.get("decoded_open_bool", 0)))
        qpos = _safe_float(r.get("gripper_qpos_before", 0))
        # raw: prefer native field, fall back to proxy (V4 remapped traces)
        raw_now = _safe_float(r.get("clean_gripper_raw",
                                     r.get("clean_gripper_raw_proxy", 0.5)))

        # ── Feature validity flags ──
        gripper_valid = _check_feature_validity(r, "gripper_semantics_valid")
        qpos_valid = _field_is_present_and_valid(r, "gripper_qpos_before")
        eef_valid = (_field_is_present_and_valid(r, "eef_x") and
                     _field_is_present_and_valid(r, "eef_y") and
                     _field_is_present_and_valid(r, "eef_z"))
        raw_valid = (_field_is_present_and_valid(r, "clean_gripper_raw") or
                     _field_is_present_and_valid(r, "clean_gripper_raw_proxy"))
        disabled_features = []

        # ── Precursor-based scoring (no absolute step thresholds) ──
        score = 0.0
        # Score decomposition (E4A)
        raw_crossing_bonus = 0.0
        close_streak_bonus = 0.0
        close_onset_qpos_bonus = 0.0
        eef_deceleration_bonus = 0.0
        qpos_ready_bonus = 0.0
        decoded_open_penalty = 0.0
        speed_now_val = None
        speed_prev_val = None

        # Detect explicit close-event signals
        # Raw crossing requires: current AND previous raw valid, AND both
        # gripper semantics valid (cannot bridge invalid gap)
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

        # First close in a streak (potential grasp start, not sustained close)
        if close_streak == 1:
            close_streak_bonus = 1.0
            score += 1.0

        # CLOSE onset with gripper not yet responding (pre-grasp close)
        if close_onset and qpos_valid and qpos < 0.005:
            close_onset_qpos_bonus = 0.5
            score += 0.5
        elif close_onset and not qpos_valid:
            disabled_features.append("qpos_close_response")

        # EEF decelerating (approaching grasp point)
        if t >= 4:
            speed_now_val = _eef_speed_if_valid(visible, t, window=3)
            speed_prev_val = _eef_speed_if_valid(visible, t - 1, window=3)
            if (speed_now_val is not None and speed_prev_val is not None and
                speed_prev_val > 0 and speed_now_val < speed_prev_val and speed_now_val < 0.01):
                eef_deceleration_bonus = 0.5
                score += 0.5
            elif speed_now_val is None or speed_prev_val is None:
                disabled_features.append("eef_deceleration")

        # Gripper qpos low (physically ready for close/grasp)
        if qpos_valid and qpos < 0.01 and not decoded_open:
            qpos_ready_bonus = 0.3
            score += 0.3
        elif not qpos_valid:
            disabled_features.append("qpos_ready")

        # ── Penalties ──
        if decoded_open:
            decoded_open_penalty = -2.0
            score -= 2.0

        # ── Explicit close-event candidate flag ──
        is_close_event_candidate = (
            raw_open_to_close_crossing
            or bool(close_onset)
            or close_streak == 1
        )

        # ── Abstain detection ──
        abstain = ""
        if not gripper_valid:
            abstain = "gripper_semantics_invalid"
        elif decoded_open:
            abstain = "gripper_already_open"
        elif t < 3:
            abstain = "too_early"
        elif score < 0.5:
            abstain = "low_confidence"

        # ── Horizon ground-truth labels (for evaluation only, not scoring) ──
        will_close = False
        close_at = -1
        if teacher_anchor >= 0:
            # Critical close is within [t+1, t+horizon]
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
            "is_close_event_candidate": is_close_event_candidate,
            "will_critical_close_within_horizon": will_close,
            "predicted_close_horizon": close_at - t if close_at > 0 else -1,
            "horizon": horizon,
            "disabled_features": disabled_features,
            # E4A score decomposition
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


def _detect_ambiguous_multiple_closes(predictions: list[dict],
                                       tie_tolerance: float = TIE_TOLERANCE,
                                       min_separation: int = MIN_CLOSE_SEPARATION,
                                       event_score_floor: float = EVENT_SCORE_FLOOR) -> bool:
    """Return True if two or more high-score CLOSE EVENT candidates are far apart
    with scores within tie_tolerance.

    Only considers steps where is_close_event_candidate=True (raw crossing,
    close_onset, or close_streak==1). Non-close-event high-score steps
    (e.g. EEF deceleration alone) do NOT trigger ambiguity.
    """
    close_events = [p for p in predictions
                    if not p["abstain"]
                    and p.get("is_close_event_candidate", False)
                    and p["score"] >= event_score_floor]
    if len(close_events) < 2:
        return False

    sorted_events = sorted(close_events, key=lambda p: p["score"], reverse=True)
    best_score = sorted_events[0]["score"]

    for p in sorted_events[1:]:
        if abs(p["step"] - sorted_events[0]["step"]) >= min_separation:
            if abs(p["score"] - best_score) <= tie_tolerance:
                return True

    return False


def select_best_window(predictions: list[dict],
                       window_len: int = WINDOW_LEN,
                       pre_offset: int = PRE_OFFSET,
                       tie_tolerance: float = TIE_TOLERANCE,
                       min_separation: int = MIN_CLOSE_SEPARATION,
                       event_score_floor: float = EVENT_SCORE_FLOOR) -> Optional[dict]:
    """Offline clean-repeat: select best window from full-trajectory predictions.

    Picks the highest-scoring non-abstaining step as the anchor.
    Allowed to scan the full trajectory (offline mode).

    Abstains (ambiguous_multiple_close_candidates) when two distinct high-score
    closes are far apart with scores within tie_tolerance — refuses to
    silently pick the earliest.

    Args:
        tie_tolerance: max score difference to consider two closes tied.
        min_separation: min steps between closes to consider them distinct.
        event_score_floor: min score for a step to be a close-event candidate.

    Returns dict with window_start, window_end, anchor_step, score, abstain_reason.
    """
    valid = [p for p in predictions if not p["abstain"]]

    if not valid:
        return {
            "window_start": -1, "window_end": -1,
            "anchor_step": -1, "score": 0.0,
            "abstain_reason": "all_abstain",
        }

    # Check for ambiguous multiple closes before selecting
    if _detect_ambiguous_multiple_closes(
        predictions,
        tie_tolerance=tie_tolerance,
        min_separation=min_separation,
        event_score_floor=event_score_floor,
    ):
        return {
            "window_start": -1, "window_end": -1,
            "anchor_step": -1, "score": 0.0,
            "abstain_reason": "ambiguous_multiple_close_candidates",
        }

    best = max(valid, key=lambda p: p["score"])

    anchor = best["step"]
    ws = max(0, anchor - pre_offset)
    we = ws + window_len

    return {
        "window_start": ws,
        "window_end": we,
        "anchor_step": anchor,
        "score": best["score"],
        "abstain_reason": best["abstain"] if best["score"] < 0.5 else "",
    }


def select_online_trigger(predictions: list[dict],
                           score_threshold: float = 1.5,
                           confirmation_steps: int = 1,
                           cooldown_steps: int = 20,
                           window_len: int = WINDOW_LEN,
                           pre_offset: int = PRE_OFFSET,
                           mode: str = "close_interception") -> Optional[dict]:
    """Online streaming: same-step close interception (Mode 1).

    This is NOT a pre-close forecast. It detects and intercepts a CLOSE
    command at the moment it is issued (gripper_raw OPEN→CLOSE crossing).

    Scans causally through predictions. Triggers when:
      1. Score crosses `score_threshold` (first crossing only).
      2. Score stays above threshold for `confirmation_steps` consecutive steps.
      3. Cooldown: no re-trigger within `cooldown_steps` of last trigger.

    Window start >= trigger_step (cannot start in the past).

    Args:
        mode: Must be "close_interception" (the only implemented mode).
              Raises ValueError for "future_close_forecast", "", or unknown
              modes — refuses to silently execute wrong behavior.

    Returns dict with window_start, window_end, trigger_step, score, abstain_reason,
    and prediction_mode.
    If no trigger fires, returns all_abstain sentinel.
    """
    allowed_modes = {"close_interception"}
    if mode not in allowed_modes:
        if mode == "future_close_forecast":
            raise ValueError(
                "future_close_forecast (Mode 2) is not yet implemented. "
                "Only close_interception (Mode 1) is available.")
        raise ValueError(
            f"Unknown online trigger mode: '{mode}'. "
            f"Allowed: {sorted(allowed_modes)}")

    triggered = False
    confirm_count = 0
    last_trigger = -cooldown_steps

    for p in predictions:
        t = p["step"]
        if p["abstain"]:
            confirm_count = 0
            continue

        if p["score"] >= score_threshold and t - last_trigger >= cooldown_steps:
            confirm_count += 1
            if confirm_count >= confirmation_steps and not triggered:
                triggered = True
                trigger_step = t
                last_trigger = t
                ws = max(trigger_step, trigger_step - pre_offset)
                we = ws + window_len
                return {
                    "window_start": ws,
                    "window_end": we,
                    "anchor_step": trigger_step,
                    "trigger_step": trigger_step,
                    "score": p["score"],
                    "abstain_reason": "",
                    "prediction_mode": "observed_close_interception",
                }
        else:
            confirm_count = 0

    return {
        "window_start": -1, "window_end": -1,
        "anchor_step": -1, "trigger_step": -1,
        "score": 0.0,
        "abstain_reason": "no_online_trigger",
        "prediction_mode": "",
    }


def build_clean_proposal(
    task_key: str,
    state_id: int,
    trace_path: str,
    trace_sha256: str,
    commit: str,
    window_info: dict,
    phase_label: str = "",
    selection_mode: str = "offline_clean_repeat",
    is_online: bool = False,
    first_close_horizon: int = 0,
    prediction_mode: str = "",
) -> WindowProposal:
    """Build a frozen WindowProposal from clean-only selector output.

    Provenance is set correctly per mode:
      offline_clean_repeat: features_are_causal=True, selection_is_causal=False
      online_streaming:     features_are_causal=True, selection_is_causal=True

    Args:
        selection_mode: "offline_clean_repeat" or "online_streaming"
        is_online: True if online streaming selection
        prediction_mode: "observed_close_interception" or "future_close_forecast"
    """
    import hashlib as _hl
    _trace_stem = os.path.basename(trace_path).replace(".csv", "")[-20:]
    _mode_short = "on" if is_online else "off"
    pid = f"{task_key}_s{state_id}_l12v3_{_mode_short}_{_trace_stem}"

    # Correct provenance per mode
    features_are_causal = True  # per-step feature extraction is always causal
    selection_is_causal = is_online  # only online mode has causal selection
    pred_mode = prediction_mode or window_info.get("prediction_mode", "")

    # predicted_first_close_step semantics
    trigger = window_info.get("trigger_step", -1)
    if trigger >= 0 and pred_mode == "observed_close_interception":
        predicted_close = trigger  # same-step: detected close IS the predicted close
    else:
        predicted_close = window_info.get("anchor_step", -1)

    return WindowProposal(
        proposal_id=pid,
        selector_version=SELECTOR_VERSION,
        source_commit=commit,
        source_trace_path=trace_path,
        source_trace_sha256=trace_sha256,
        task_key=task_key,
        task_id=task_key,
        state_id=state_id,
        window_start=window_info["window_start"],
        window_end=window_info["window_end"],
        anchor_step=window_info["anchor_step"],
        predicted_first_close_step=predicted_close,
        first_close_horizon=first_close_horizon,
        phase_label=phase_label,
        phase_confidence=min(1.0, window_info["score"] / 5.0),
        closure_criticality=window_info.get("score", 0.0) / 5.0,
        selector_score=window_info["score"],
        eligible=window_info["window_start"] >= 0,
        abstain_reason=window_info.get("abstain_reason", ""),
        prediction_mode=pred_mode,
        uses_clean_only=True,
        uses_attack_outcome=False,
        uses_random_outcome=False,
        features_are_causal=features_are_causal,
        selection_is_causal=selection_is_causal,
        history_length=HISTORY_LEN,
        selector_config_sha256=_sha256_str(
            f"{SELECTOR_VERSION}:{WINDOW_LEN}:{PRE_OFFSET}:{HISTORY_LEN}:{PREDICTION_HORIZON}"),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        selector_role="student",
        selection_mode=selection_mode,
        is_online=is_online,
    )
