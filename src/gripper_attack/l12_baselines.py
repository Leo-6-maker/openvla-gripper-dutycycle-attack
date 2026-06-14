"""L12 baselines: independent implementations, do not pollute formal selector.

Baselines (as specified in review):
  offline_time_only_diagnostic: predicts anchor at 25% of full trajectory length
  online_safe_time: predicts based on current absolute step only
  TaskOnly: predicts from train-fold task median (NOT eval trace itself)
  CloseEventRule: the current causal close-event detector (for comparison)
  label_shuffle_null: shuffles Teacher-P target, evaluates selector invariance
  train_fold_prevalence: always predicts train-fold global median anchor
  oracle_anchor_upper_bound: uses eval trace's actual Teacher-P anchor (NOT a baseline)
  AlwaysAbstain: never proposes a window
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .critical_close_selector import (
    WINDOW_LEN, PRE_OFFSET, TIE_TOLERANCE,
    rule_based_close_predictor, select_best_window,
)
from .phase_detector import _safe_float


def offline_time_only_diagnostic(n_steps: int,
                                  window_len: int = WINDOW_LEN,
                                  pre_offset: int = PRE_OFFSET) -> dict:
    """Time-only diagnostic: predict close at 25% of FULL trajectory length.

    Labeled as DIAGNOSTIC because it uses future knowledge (episode length).
    NOT deployment-safe. For comparison only.
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
        "prediction_mode": "offline_time_only_diagnostic",
    }


def online_safe_time_baseline(current_step: int,
                               window_len: int = WINDOW_LEN,
                               pre_offset: int = PRE_OFFSET) -> dict:
    """Online-safe time baseline: uses ONLY current absolute step.

    Does NOT have access to episode length. Uses a fixed threshold
    (step >= 40) as a naive online trigger.
    """
    if current_step < 40:
        return {
            "window_start": -1, "window_end": -1,
            "anchor_step": -1, "score": 0.0,
            "abstain_reason": "too_early_time_heuristic",
            "prediction_mode": "online_time_baseline",
        }
    anchor = current_step
    ws = max(0, anchor - pre_offset)
    we = ws + window_len
    return {
        "window_start": ws,
        "window_end": we,
        "anchor_step": anchor,
        "score": 0.5,
        "abstain_reason": "",
        "prediction_mode": "online_time_baseline",
    }


def task_only_window(task_key: str,
                      train_fold_median_anchors: dict,
                      global_train_median: int = 50,
                      window_len: int = WINDOW_LEN,
                      pre_offset: int = PRE_OFFSET) -> dict:
    """TaskOnly baseline: predicts from train-fold task median anchor.

    Does NOT use this episode's data. Unknown tasks fall back to
    global_train_median. No hardcoded anchor table.
    """
    anchor = train_fold_median_anchors.get(task_key, global_train_median)
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
    """CloseEventRuleBaseline: the current causal close-event detector."""
    preds = rule_based_close_predictor(records)
    win = select_best_window(preds, window_len, pre_offset)
    win["prediction_mode"] = "close_event_rule_baseline"
    return win


def label_shuffle_null(records: list[dict],
                        teacher_p_anchor: int,
                        n_shuffles: int = 20,
                        window_len: int = WINDOW_LEN,
                        pre_offset: int = PRE_OFFSET) -> list[dict]:
    """LabelShuffle evaluation null: shuffles Teacher-P target, re-evaluates
    selector output invariance. The selector itself is unchanged (rule-based,
    not learned), so this measures whether evaluation metrics are sensitive
    to the specific teacher anchor choice.

    Returns list of dicts with shuffled_eval_target, selector window.
    """
    T = len(records)
    results = []
    for seed in range(n_shuffles):
        rng = np.random.RandomState(seed)
        fake_target = int(rng.randint(0, max(1, T)))
        # Run selector (unchanged — does not use teacher)
        preds = rule_based_close_predictor(records)
        win = select_best_window(preds, window_len, pre_offset)
        win["shuffle_seed"] = seed
        win["shuffled_eval_target"] = fake_target
        win["original_teacher_anchor"] = teacher_p_anchor
        win["prediction_mode"] = "label_shuffle_null"
        results.append(win)
    return results


def train_fold_prevalence(train_fold_global_median: int = 50,
                           window_len: int = WINDOW_LEN,
                           pre_offset: int = PRE_OFFSET) -> dict:
    """Prevalence baseline: always predicts train-fold global median anchor.
    Does NOT access eval trace data.
    """
    anchor = train_fold_global_median
    ws = max(0, anchor - pre_offset)
    we = ws + window_len
    return {
        "window_start": ws,
        "window_end": we,
        "anchor_step": anchor,
        "score": 1.0,
        "abstain_reason": "",
        "prediction_mode": "train_fold_prevalence",
    }


def oracle_anchor_upper_bound(teacher_p_anchor: int,
                               window_len: int = WINDOW_LEN,
                               pre_offset: int = PRE_OFFSET) -> dict:
    """Oracle upper bound: uses the eval trace's actual Teacher-P anchor.

    NOT a baseline — this is a theoretical upper bound. Reported separately,
    never compared against baselines.
    """
    if teacher_p_anchor < 0:
        return {
            "window_start": -1, "window_end": -1,
            "anchor_step": -1, "score": 0.0,
            "abstain_reason": "teacher_abstained",
            "prediction_mode": "oracle_upper_bound",
        }
    ws = max(0, teacher_p_anchor - pre_offset)
    we = ws + window_len
    return {
        "window_start": ws,
        "window_end": we,
        "anchor_step": teacher_p_anchor,
        "score": 5.0,
        "abstain_reason": "",
        "prediction_mode": "oracle_upper_bound",
    }


def always_abstain_baseline() -> dict:
    """Always-abstain baseline: never proposes a window."""
    return {
        "window_start": -1, "window_end": -1,
        "anchor_step": -1, "score": 0.0,
        "abstain_reason": "always_abstain_baseline",
        "prediction_mode": "always_abstain_baseline",
    }
