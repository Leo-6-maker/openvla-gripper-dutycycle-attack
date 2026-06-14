"""L12 baselines: independent implementations, do not pollute formal selector.

Baselines:
  TimeOnly:       predicts based solely on absolute/normalized step
  TaskOnly:       predicts based on task identity (no physical state access)
  CloseEventRule: the current causal close-event detector (for comparison)
  LabelShuffle:   shuffles Teacher-P anchors, keeps features unchanged
  Prevalence:     always predicts most common anchor or abstains
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import numpy as np

from .critical_close_selector import (
    WINDOW_LEN, PRE_OFFSET, TIE_TOLERANCE,
    rule_based_close_predictor, select_best_window,
    _detect_ambiguous_multiple_closes,
)
from .window_contract import WindowProposal
from .phase_detector import _safe_float

# ── Known task median anchors (from prior dev analysis — Teacher-R anchors) ──
TASK_MEDIAN_ANCHORS = {
    "butter": 4,
    "cream_cheese": 59,
    "bbq_sauce": 47,
    "chocolate_pudding": 58,
    "ketchup": 44,
    "alphabet_soup": 50,
    "tomato_sauce": 78,
}


def time_only_window(n_steps: int,
                     window_len: int = WINDOW_LEN,
                     pre_offset: int = PRE_OFFSET) -> dict:
    """TimeOnly baseline: predict critical close at 25% of trajectory.

    Uses ONLY absolute step information. No physical state access.
    This is a pure time heuristic — should not outperform any
    physically-informed selector.
    """
    anchor = max(0, int(n_steps * 0.25))
    ws = max(0, anchor - pre_offset)
    we = ws + window_len
    return {
        "window_start": ws,
        "window_end": we,
        "anchor_step": anchor,
        "score": 0.5,
        "abstain_reason": "",
        "prediction_mode": "time_only_baseline",
    }


def task_only_window(task_key: str,
                     n_steps: int,
                     window_len: int = WINDOW_LEN,
                     pre_offset: int = PRE_OFFSET) -> dict:
    """TaskOnly baseline: predict using only task identity.

    Returns the task's known median close step (from prior dev analysis).
    Does NOT observe this episode's actions or physical state.
    """
    anchor = TASK_MEDIAN_ANCHORS.get(task_key, int(n_steps * 0.25))
    if anchor < 0:
        return {
            "window_start": -1, "window_end": -1,
            "anchor_step": -1, "score": 0.0,
            "abstain_reason": "task_unknown",
            "prediction_mode": "task_only_baseline",
        }
    ws = max(0, anchor - pre_offset)
    we = ws + window_len
    return {
        "window_start": ws,
        "window_end": we,
        "anchor_step": anchor,
        "score": 0.5,
        "abstain_reason": "",
        "prediction_mode": "task_only_baseline",
    }


def close_event_rule_baseline(records: list[dict],
                               window_len: int = WINDOW_LEN,
                               pre_offset: int = PRE_OFFSET) -> dict:
    """CloseEventRuleBaseline: the current causal close-event detector.

    Uses *_detect_ambiguous_multiple_closes* for ambiguity abstain.
    This is the SAME logic as the formal selector — included here
    as a named baseline for comparison in baseline tables.
    """
    preds = rule_based_close_predictor(records)
    win = select_best_window(preds, window_len, pre_offset)
    win["prediction_mode"] = "close_event_rule_baseline"
    return win


def label_shuffle_baseline(records: list[dict],
                            teacher_p_anchor: int,
                            n_shuffles: int = 20,
                            window_len: int = WINDOW_LEN,
                            pre_offset: int = PRE_OFFSET) -> list[dict]:
    """LabelShuffle baseline: shuffle Teacher-P anchors, keep features.

    For each shuffle seed, randomly reassigns the Teacher-P anchor to a
    different step. The student features and scoring are unchanged.
    Returns the list of per-shuffle window results.
    """
    T = len(records)
    results = []
    for seed in range(n_shuffles):
        rng = np.random.RandomState(seed)
        fake_anchor = int(rng.randint(0, T))
        preds = rule_based_close_predictor(records, teacher_anchor=fake_anchor)
        win = select_best_window(preds, window_len, pre_offset)
        win["shuffle_seed"] = seed
        win["fake_anchor"] = fake_anchor
        win["prediction_mode"] = "label_shuffle_baseline"
        results.append(win)
    return results


def prevalence_baseline(records: list[dict],
                         teacher_p_anchor: int,
                         window_len: int = WINDOW_LEN,
                         pre_offset: int = PRE_OFFSET) -> dict:
    """Prevalence baseline: always predict the given anchor or abstain.

    If teacher_p_anchor >= 0: predicts that anchor (cheating baseline)
    If teacher_p_anchor < 0: abstains
    """
    if teacher_p_anchor < 0:
        return {
            "window_start": -1, "window_end": -1,
            "anchor_step": -1, "score": 0.0,
            "abstain_reason": "teacher_abstained",
            "prediction_mode": "prevalence_baseline",
        }
    ws = max(0, teacher_p_anchor - pre_offset)
    we = ws + window_len
    return {
        "window_start": ws,
        "window_end": we,
        "anchor_step": teacher_p_anchor,
        "score": 5.0,  # maximum — knows the answer
        "abstain_reason": "",
        "prediction_mode": "prevalence_baseline",
    }


def always_abstain_baseline() -> dict:
    """Always-abstain baseline: never proposes a window."""
    return {
        "window_start": -1, "window_end": -1,
        "anchor_step": -1, "score": 0.0,
        "abstain_reason": "always_abstain_baseline",
        "prediction_mode": "always_abstain_baseline",
    }
