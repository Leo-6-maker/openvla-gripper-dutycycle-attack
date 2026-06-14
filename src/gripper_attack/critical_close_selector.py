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


def extract_deployment_features(records: list[dict]) -> np.ndarray:
    """Extract deployment-safe feature matrix from clean trace records."""
    T = len(records)
    F = 13
    feats = np.zeros((T, F), dtype=np.float32)

    for t in range(T):
        r = records[t]
        feats[t, 0] = _safe_float(r.get("clean_gripper_env", 0))
        feats[t, 1] = _safe_float(r.get("clean_gripper_raw", 0.0))
        feats[t, 2] = _safe_float(r.get("gripper_qpos_before", 0.0))
        feats[t, 3] = _safe_float(r.get("qpos_abs_before", 0.0))
        feats[t, 4] = _safe_float(r.get("eef_x", 0.0))
        feats[t, 5] = _safe_float(r.get("eef_y", 0.0))
        feats[t, 6] = _safe_float(r.get("eef_z", 0.0))
        feats[t, 7] = _safe_float(r.get("close_streak", 0))
        feats[t, 8] = _safe_float(r.get("decoded_open_bool", 0))
        # EEF velocity (3-step)
        if t >= 3:
            feats[t, 9] = feats[t, 4] - feats[t - 3, 4]
            feats[t, 10] = feats[t, 5] - feats[t - 3, 5]
            feats[t, 11] = feats[t, 6] - feats[t - 3, 6]
        feats[t, 12] = 1.0  # bias

    return feats


def _eef_speed(records: list[dict], t: int, window: int = 3) -> float:
    """Compute EEF speed magnitude at step t using `window`-step deltas."""
    if t < window:
        return 0.0
    dx = _safe_float(records[t].get("eef_x", 0)) - _safe_float(records[t - window].get("eef_x", 0))
    dy = _safe_float(records[t].get("eef_y", 0)) - _safe_float(records[t - window].get("eef_y", 0))
    dz = _safe_float(records[t].get("eef_z", 0)) - _safe_float(records[t - window].get("eef_z", 0))
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
        raw_now = _safe_float(r.get("clean_gripper_raw", 0.5))

        # ── Precursor-based scoring (no absolute step thresholds) ──
        score = 0.0

        # Detect explicit close-event signals
        raw_open_to_close_crossing = False
        if t >= 1:
            raw_prev = _safe_float(visible[t - 1].get("clean_gripper_raw", 0.5))
            if raw_prev > 0.5 and raw_now <= 0.5:
                raw_open_to_close_crossing = True
                score += 1.5

        # First close in a streak (potential grasp start, not sustained close)
        if close_streak == 1:
            score += 1.0

        # CLOSE onset with gripper not yet responding (pre-grasp close)
        if close_onset and qpos < 0.005:
            score += 0.5

        # EEF decelerating (approaching grasp point)
        if t >= 4:
            speed_now = _eef_speed(visible, t)
            speed_prev = _eef_speed(visible, t - 1)
            if speed_prev > 0 and speed_now < speed_prev and speed_now < 0.01:
                score += 0.5

        # Gripper qpos low (physically ready for close/grasp)
        if qpos < 0.01 and not decoded_open:
            score += 0.3

        # ── Penalties ──
        # Gripper already open (post-release)
        if decoded_open:
            score -= 2.0

        # ── Explicit close-event candidate flag ──
        is_close_event_candidate = (
            raw_open_to_close_crossing
            or bool(close_onset)
            or close_streak == 1
        )

        # ── Abstain detection ──
        abstain = ""
        if decoded_open:
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
                       tie_tolerance: float = TIE_TOLERANCE) -> Optional[dict]:
    """Offline clean-repeat: select best window from full-trajectory predictions.

    Picks the highest-scoring non-abstaining step as the anchor.
    Allowed to scan the full trajectory (offline mode).

    Abstains (ambiguous_multiple_close_candidates) when two distinct high-score
    closes are far apart with scores within tie_tolerance — refuses to
    silently pick the earliest.

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
    if _detect_ambiguous_multiple_closes(predictions, tie_tolerance=tie_tolerance):
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
