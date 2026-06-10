# Stage-B RC1a S8 Layer-3 Final Handoff

**Date**: 2026-06-10
**Branch**: exp/vis-prefix-margin-repair-20260603
**S8 commits**: b640f9b (Phase1), 2cb522d (Phase2), 3d00383 (S8b), 7339348 (S8c)

## Project Status

### Layer 1: VALIDATED

CleanRand abstain-first pipeline produces command-level VIS-specific OPEN.
Fresh + robustness yield = +0.94 across 4 attack seeds on 40-parent stable pool.
This is the only validated pipeline.

### Layer 2: NOT VALIDATED

Action-dynamics / action-logit / action-hidden explored but no fresh-confirmed ranking improvement. HiddenSafe had offline Strong PASS (AUC=0.691, fixed-K yield) but fresh confirmation FAILED (H +0.31 vs B +0.48). Next direction: veto-first (clean-drift / high-baseline / random-trigger), not top-K ranker.

### Layer 3: INCONCLUSIVE (RUNNER MISMATCH)

Physical bridge not resolved. Root cause is Phase1/Phase2 runner parity, not VIS objective weakness.

## S8 Layer-3 Experiment Chain

### Phase 1: ORACLE Upper-Bound (Phase1 runner)

4 windows × 4 durations × 2 conditions = 32 jobs. **All 4 windows physically reachable at L=10.** Opening direction = positive qpos increase. response_delay = 0. Milk is cleanest (clean baseline=0, oracle pos=+0.261).

### Phase 2: Extended VIS/RAND Smoke (Phase2 runner)

Milk only, 8 jobs, short + extended20, VIS+RAND, seeds 9/10. **Command gate PASS** (VIS open_count >> RAND). **Physical bridge FAIL** (pos_area=0 for all jobs). VIS suppresses negative drift by ~85% vs RAND but produces zero positive qpos.

### S8b: Same-Runner ORACLE Calibration

Milk only, 4 jobs, same Phase2 runner. **Runner mismatch confirmed.** Phase2 same-runner baseline = 0.0395 (matches VIS), Phase1 baseline = 0.0019. Phase2 ORACLE pos_area = 0 even with 10/10 forced OPEN. VIS performs similarly to same-runner ORACLE (85% vs 95% drift suppression).

### S8c: Runner Parity Isolation

Milk only, 4 jobs, half_open windows + post_horizon=40. **post_horizon and window convention are NOT the primary mismatch source.** Phase2 ORACLE pos_area = 0 even with half_open 10-step windows and 40-step post horizon. neg_area scales with post horizon (0.010→0.181), confirming persistent negative drift.

## Key Metric Comparison

| Runner | Cond | Baseline | ORACLE pos | ORACLE neg | ORACLE abs |
|--------|------|----------|------------|------------|------------|
| Phase1 | oracle L=10 | **0.0019** | **+0.261** | 0.005 | 0.266 |
| Phase2 S8b | oracle short | 0.0395 | 0 | 0.010 | 0.010 |
| Phase2 S8c | oracle short | 0.0395 | 0 | 0.181 | 0.181 |
| Phase2 S8c | oracle ext20 | 0.0395 | 0.001 | 0.873 | 0.873 |

Phase2 ORACLE cannot reproduce Phase1 positive qpos opening. Depending on duration, it may suppress (short: 0.359→0.181) or amplify (ext20: 0.414→0.873) negative drift relative to clean. **Phase1/Phase2 physical reference mismatch is unresolved and dominates Layer-3 interpretation.**

## Current Verdict

- S8 Phase2 command-level result remains valid: VIS produces OPEN command.
- S8 Phase2 physical bridge is inconclusive due to runner/reference mismatch.
- Physical bridge failure cannot be attributed to VIS objective weakness.
- Full 24-job Phase2 queue remains NOT APPROVED.
- Do not optimize VIS objective until runner parity is restored.

## Next Steps (No GPU)

1. S9: Phase1 vs Phase2 runner root-cause audit (code diff only)
2. Only after diff audit reviewed: minimal 2-job A/B ORACLE parity test
3. Only after runner parity restored: revisit physical bridge evaluation
