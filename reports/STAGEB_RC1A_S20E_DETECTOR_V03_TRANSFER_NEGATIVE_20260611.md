# S20e Detector v0.3 Transfer Negative

**Date:** 2026-06-11

## Finding: v0.3 direct transfer to S20d/V4 failed

## Distribution Shift Audit

| Dimension | K5 Phase-Runner (training) | S20d/V4 (deployment) |
|-----------|---------------------------|---------------------|
| Runner | Phase1-port hand-rolled preprocess | V4 official decode_with_scores |
| clean_open_count in cmd_specific | 0-3 (low) | All windows have variation |
| clean_open_count in random_sensitive | 5-9 (high) | Early approach windows naturally OPEN |
| LR learned rule | high open_count → random_sensitive | Natural OPEN ≠ random_sensitive |
| Early windows (rel<0.40) non-abstain | 5/11 cmd_specific at rel<0.40 | **0/95** |

## Root Cause

v0.3's abstain head (p_random_sensitive) was trained on K5 phase-runner traces where:
- Phase-runner clean trajectories have sustained CLOSED commands during approach
- High clean_open_count in a window correlates with RANDOM_CONFOUNDED

On S20d V4 official trajectories:
- Early approach windows naturally contain OPEN commands (normal grasping behavior)
- clean_open_count is NOT a reliable random_sensitive signal
- v0.3 abstains on ALL early windows (p_rand > 0.1849 for all rel<0.40 windows)

## Result

- 95 candidate windows, 27 non-abstain (28%)
- 0 early (rel<0.40) non-abstain windows
- All selected windows are late-phase (rel>0.66, near success/place)

## Decision

v0.3 cannot directly serve as Layer1 selector on S20d/V4. Needs phase-aware repair → v0.3.1.
