# S20e Detector v0.3 Recovery

**Date:** 2026-06-11

## Identity

Detector v0.3 is a **sklearn LogisticRegression-based window candidate scorer**, not a neural network checkpoint.

## Artifacts Found

| Artifact | Path | Status |
|----------|------|--------|
| Selector v0.3 (OOF) | `scripts/diagnostics/run_selector_v0_3.py` | Found |
| Selector v0.3 P1 | `scripts/diagnostics/run_selector_v0_3_p1.py` | Found |
| Detector v0 framework | `scripts/diagnostics/run_detector_v0_fixed.py` | Found |
| Labeled pairs (72) | `/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/all_labels_rc1a_14cfabe_72pairs.csv` | Found |
| Stable pool (40 parents) | `tables/stageb_v1_1_stable_parent_pool_k5_k5b_k5c_rc1a_ca3a97e.csv` | Found |
| Frozen checkpoint | N/A | **NOT FOUND** — no saved model file |

## Feature Schema (8 features)

1. `clean_open_count` — OPEN count in window during clean rollout
2. `clean_open_frac` — OPEN fraction in window
3. `raw_gripper_mean` — mean clean gripper env action
4. `raw_gripper_max` — max clean gripper env action
5. `qpos_pre` — median qpos before window
6. `qpos_mean` — mean qpos during window
7. `window_center` — (ws + we) / 2
8. `rel_timing` — window_center / actual_max_step

## Targets

- `y_rand` (random_sensitive): 1 if the window is RANDOM_SENSITIVE
- `y_cmd` (cmd_specific): 1 if cmd_any AND NOT random_sensitive

## Abstain Rule

`p_rand > 50th percentile of training set p_rand` → ABSTAIN

## Window Convention

Half-open: `ws <= step < we`, window_len = we - ws

## Materialization Decision

No checkpoint exists. A frozen v0.3_rc1a selector is materialized by:
1. Training LR on full 72-pair labeled set (not OOF)
2. Freezing scaler + model + abstain threshold
3. Applying to S20d candidate universe

Spec frozen in: `configs/stageb_detector_v03_rc1a.yaml`
