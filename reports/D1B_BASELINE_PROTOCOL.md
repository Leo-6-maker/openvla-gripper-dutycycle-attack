# D1b — Baseline Protocol (Frozen)

**Date**: 2026-06-15
**Stage**: D1b prereg only

## Baseline: Rule-Based Scorer (frozen)

The baseline is the current hand-designed scalar score from
`critical_close_selector.rule_based_close_predictor()`, specifically the
`total_score` field exported in the candidate CSV.

This is the same scorer that achieved:
- 4/10 Teacher-P unique top-1 on the E2 development set (E4B.3)
- 5/10 first-threshold online near-correct (E4B.1)

## Baseline Evaluation Protocol

1. Run on the **identical 20-val / 21-test traces** as the learned detector
2. Use the same per-trace evaluation: highest-scoring candidate vs Teacher-P
3. Same near-correct definition: |predicted_step - teacher_p_step| <= 4
4. Same denominator: trace-level (not candidate-level, not timestep-level)
5. Report identical metrics: top-1 accuracy, top-2 accuracy, MAE, median AE,
   decision coverage, conditional accuracy, per-task breakdown

## Baseline Code

- Source: `src/gripper_attack/critical_close_selector.py`
- SHA: `81b510ec30716df11a89fbfb45194dceb1cfec09c1a2fd5d357a8a7448b4fa34`
- Function: `rule_based_close_predictor()` with `SELECTOR_VERSION = "l12_close_event_interceptor_v4"`
- Feature: `total_score` column in `l12_e4c2b_close_candidates.csv`

## Historical Reference Only

The E2 development set results (4/10, 5/10) are historical development
references. The formal baseline must be recomputed on the D1b validation
and test splits.

## Forbidden

- Adjusting scorer weights or rules based on new split results
- Using a different scorer version than the one frozen in E4C.2b
- Computing test metrics before validation metrics
- Comparing learned detector against a baseline that was tuned on test
