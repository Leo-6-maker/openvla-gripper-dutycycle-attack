"""Layer2: Critical first-close opportunity scorer.

Predicts when a task-critical first CLOSE event will occur, using only
deployment-safe (causal) features. Does not use attack outcomes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

import numpy as np

from .window_contract import WindowProposal

# ── Frozen config ──
WINDOW_LEN = 10
PRE_OFFSET = 2
HISTORY_LEN = 16
SELECTOR_VERSION = "l12_critical_close_selector_v1"
FEATURE_SCHEMA_VERSION = "l12_v1"


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def extract_deployment_features(records: list[dict]) -> np.ndarray:
    """Extract deployment-safe feature matrix from clean trace records.

    Returns (T, F) array where F = 13: gripper_env, gripper_raw, gripper_qpos,
    qpos_abs, eef_x/y/z, close_streak, decoded_open, eef_vel_x/y/z (computed).
    """
    T = len(records)
    F = 13
    feats = np.zeros((T, F), dtype=np.float32)

    for t in range(T):
        r = records[t]
        feats[t, 0] = float(r.get("clean_gripper_env", 0))
        feats[t, 1] = float(r.get("clean_gripper_raw", 0.0))
        feats[t, 2] = float(r.get("gripper_qpos_before", 0.0))
        feats[t, 3] = float(r.get("qpos_abs_before", 0.0))
        feats[t, 4] = float(r.get("eef_x", 0.0))
        feats[t, 5] = float(r.get("eef_y", 0.0))
        feats[t, 6] = float(r.get("eef_z", 0.0))
        feats[t, 7] = float(r.get("close_streak", 0))
        feats[t, 8] = float(r.get("decoded_open_bool", 0))
        # EEF velocity (3-step)
        if t >= 3:
            feats[t, 9] = feats[t, 4] - feats[t - 3, 4]
            feats[t, 10] = feats[t, 5] - feats[t - 3, 5]
            feats[t, 11] = feats[t, 6] - feats[t - 3, 6]
        feats[t, 12] = 1.0  # bias

    return feats


def rule_based_close_predictor(records: list[dict]) -> list[dict]:
    """Rule-based causal predictor: score each step for critical-close likelihood.

    Returns list of per-step dicts with prediction scores.
    Uses only deployment-safe features from records[:t] at each step.
    """
    T = len(records)
    predictions = []

    for t in range(T):
        # Only see records[:t] (causal)
        visible = records[:t + 1]
        r = visible[-1]

        # ── Features ──
        clean_close = r.get("clean_close", 0)
        close_onset = r.get("close_onset", 0)
        close_streak = r.get("close_streak", 0)
        decoded_open = r.get("decoded_open_bool", 0)
        qpos = float(r.get("gripper_qpos_before", 0))
        qpos_abs = float(r.get("qpos_abs_before", 0))
        eef_z = float(r.get("eef_z", 0))

        # ── Scoring ──
        score = 0.0

        # Strong signal: first CLOSE onset with gripper not yet responding
        if close_onset and qpos < 0.005:
            score += 3.0

        # CLOSE command with low qpos (gripper truly closed)
        if clean_close and qpos < 0.01:
            score += 1.0

        # Early in trajectory (pre-grasp phase)
        if t < 60 and not decoded_open:
            score += 0.5

        # CLOSE streak > 2 (sustained CLOSE command)
        if close_streak > 2:
            score += 0.3

        # Penalize: gripper already open (post-release)
        if decoded_open or qpos > 0.01:
            score -= 2.0

        # Penalize: very late in trajectory
        if t > 200:
            score -= 1.0

        # ── Abstain detection ──
        abstain = ""
        if decoded_open:
            abstain = "gripper_already_open"
        elif t < 3:
            abstain = "too_early"
        elif score < 0.5:
            abstain = "low_confidence"

        predictions.append({
            "step": t,
            "score": max(0.0, score),
            "abstain": abstain,
            "clean_close": clean_close,
            "close_onset": close_onset,
            "qpos": qpos,
        })

    return predictions


def select_best_window(predictions: list[dict],
                       window_len: int = WINDOW_LEN,
                       pre_offset: int = PRE_OFFSET) -> Optional[dict]:
    """Select the best window proposal from causal predictions.

    Returns dict with window_start, window_end, anchor_step, score, abstain_reason.
    """
    # Filter non-abstaining predictions
    valid = [p for p in predictions if not p["abstain"]]

    if not valid:
        return {
            "window_start": -1, "window_end": -1,
            "anchor_step": -1, "score": 0.0,
            "abstain_reason": "all_abstain",
        }

    # Pick highest score
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


def build_clean_proposal(
    task_key: str,
    state_id: int,
    trace_path: str,
    trace_sha256: str,
    commit: str,
    window_info: dict,
    phase_label: str = "",
) -> WindowProposal:
    """Build a frozen WindowProposal from clean-only selector output."""
    pid = f"{task_key}_s{state_id}_l12v1"
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
        predicted_first_close_step=window_info["anchor_step"],
        phase_label=phase_label,
        phase_confidence=min(1.0, window_info["score"] / 5.0),
        selector_score=window_info["score"],
        eligible=window_info["window_start"] >= 0,
        abstain_reason=window_info.get("abstain_reason", ""),
        uses_clean_only=True,
        uses_attack_outcome=False,
        uses_random_outcome=False,
        is_causal=True,
        history_length=HISTORY_LEN,
        selector_config_sha256=_sha256_str(
            f"{SELECTOR_VERSION}:{WINDOW_LEN}:{PRE_OFFSET}:{HISTORY_LEN}"),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        selector_role="student",
    )
