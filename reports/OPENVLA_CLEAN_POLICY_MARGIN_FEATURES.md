# OpenVLA Clean Policy Margin Features

**Date**: 2026-06-06 16:05
**Source**: milestone_7 clean traces (178 traces, 9 tasks)
**Semantics**: raw_gripper < 0.5 = OPEN, >= 0.5 = CLOSE

---

## Feature Description

All features are computed from clean rollout OpenVLA forward passes.
These are **online-safe**: available before any VIS attack is run.

### Per-Step Features (cache)
- `raw_gripper_action`: post-sigmoid action probability from OpenVLA
- `distance_to_boundary`: abs(raw_gripper_action - 0.5)
- `open_close_decision`: 1 if raw_gripper < 0.5 (open), 0 otherwise

### Per-Window Aggregates
- **Policy margin**: min/mean/max/std of distance_to_boundary
- **Low margin**: count, ratio, longest streak of steps where distance < 0.1
- **Gripper dynamics**: action mean/std/delta/min/max
- **Decision dynamics**: flip count/rate, open fraction, margin reversal flag

---

## Coverage

| Metric | Value |
|--------|-------|
| Candidates matched | 31 |
| Candidates missed | 0 |
| Step-level rows | 555 |
| Window-level rows | 31 |
| Train windows | 22 (pos=9, neg=13) |


## Positive vs Negative: Policy Margin Features

| Feature | Pos Mean | Neg Mean | Delta |
|---------|----------|----------|-------|
| distance_to_boundary_min | 0.4961 | 0.4961 | +0.0000 |
| distance_to_boundary_mean | 0.4961 | 0.4961 | +0.0000 |
| low_margin_step_ratio | 0.0000 | 0.0000 | +0.0000 |
| longest_low_margin_streak | 0.0000 | 0.0000 | +0.0000 |
| gripper_action_std | 0.0348 | 0.0000 | +0.0348 |
| open_close_flip_rate | 0.0247 | 0.0000 | +0.0247 |
| open_fraction | 0.0123 | 0.0000 | +0.0123 |
| margin_reversal_flag | 0.0000 | 0.0000 | +0.0000 |

## Interpretation

- **distance_to_boundary_min**: How close the model gets to the decision boundary.
  Low values suggest the model is uncertain about gripper control.
- **low_margin_step_ratio**: Fraction of steps where distance < 0.1.
  High values suggest persistent uncertainty.
- **gripper_action_std**: Variance of gripper action within window.
  High variance suggests unstable control.
- **open_close_flip_rate**: How often the open/close decision flips.
  High flip rates suggest indecisive control.
- **margin_reversal_flag**: Whether the minimum-margin step's decision
  differs from the majority decision. Flags ambiguous states.

These features probe the model's own decision-making confidence
without running a VIS attack. Windows where the model shows low
confidence or unstable control may be more susceptible to perturbation.

## Online-Safe Guarantee

| Check | Status |
|-------|--------|
| VIS outcomes excluded | YES — clean traces only |
| Oracle labels as features | NO — labels are target only |
| Attack outcome leakage | NO — no VIS data used |
| Deployable at inference | YES — only needs clean OpenVLA forward pass |
