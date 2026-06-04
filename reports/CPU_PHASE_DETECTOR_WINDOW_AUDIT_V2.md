# CPU Phase Detector Audit v2 — Strict Split Evaluation

**Status**: window-level runtime descriptor separability smoke.
Do NOT claim online causal detector works.

## Split-Based Evaluation

| Evaluation | Best Row-Random F1 | Mean Episode F1 | Mean Task F1 |
|------------|-------------------|-----------------|-------------|
| Feature Set A (descriptor) | 0.8393 | 0.6069 | 0.6050 |
| A_descriptor | | | 0.7254 |
| B_causal_safe | | | 0.5469 |
| C_no_gripper | | | 0.5429 |

## Feature Sets

### Set A (descriptor_upper_bound)
All 14 available numeric descriptors. May include future-in-window aggregates.
Best row-random macroF1=0.8393.
NOT valid for online causal use.

### Set B (causal_safe_proxy)
Features available at window_start: clean_open_ratio, raw_gripper_mean, qpos_start, qpos_min, eef_speed_mean.
Approximate causal proxy. Separability likely lower than Set A.

### Set C (no_gripper_aggregate)
Excludes clean_open_count/clean_open_ratio. Tests whether model relies on gripper-open labels.

## Leakage Audit

Forbidden features verified absent from model input:
K, T_gform, VIS_OPEN, VIS_done, candidate_source, checkpoint, claim_usable, clean_open_threshold_relaxed, clean_open_threshold_strict, delay...

## Causal Replay

79/79 episodes with causal OPEN onset.
MAE=27.3 steps from oracle T_gform (using first_sustained_OPEN rule).

## Verdict

Pass: window-level runtime descriptor separability.
Fail: online causal detector NOT validated — causal-safe features have lower F1,
and current descriptors include future-in-window fields.

Next: replace coarse phase_bin_proxy with Batch2b VIS-informed vulnerability_ready label.
