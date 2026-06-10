# Stage-B RC1a S7 HiddenSafe Fresh Confirmation — Final

**Date**: 2026-06-10
**Branch**: exp/vis-prefix-margin-repair-20260603
**S7 readout freeze**: de96877
**Queue generation commit**: de31cea
**Shard launch commit**: f671aff
**Final freeze commit**: ef09d45
**GitHub merge HEAD**: 52283bb

## Executive Summary

**Verdict: FAIL** — HiddenSafe offline Strong PASS did not validate under fresh attack_seed 9/10.

HiddenSafe achieved offline AUC=0.691 and fixed-K yield advantage (+0.05 to +0.08) on the 40-parent stable pool. A 32-job fresh confirmation queue (8 windows, attack_seed 9/10, VIS/RAND matched) was launched to test whether this offline advantage translates to improved VIS-specific yield. Group H (HiddenSafeRank, 4 windows) underperformed Group B (RandomRank baseline, 4 windows) in window_yield_score (+0.3125 vs +0.4750, diff = -0.1625), while seed-level cmd_hit and cmd_rand were tied.

Therefore **HiddenSafe remains offline mechanism evidence only and is not a validated Layer-2 ranking improvement**. S6 CleanRand abstain (yield=+0.94) remains the only validated pipeline.

## Queue Provenance

| Item | Value |
|------|-------|
| queue_generation_commit | de31cea |
| shard_launch_commit | f671aff |
| final_freeze_commit | ef09d45 |
| github_merge_head | 52283bb |
| Queue CSV | tables/layer2_hiddensafe_confirmation_queue.csv |
| Queue SHA256 | 8e05d6f3bc3b1bc642662b90946468fcd29ec074f3eb37177c837b479d350d74 |
| Attack seeds | 9, 10 |
| Conditions | VIS (vis_pgd), RAND (random_linf) |
| Eps | 6 raw pixels |
| Total jobs | 32 (8 windows × 2 seeds × 2 conditions) |

## Shard Assignment

| Shard | GPU | Pairs | H/B | Outcome |
|-------|-----|-------|-----|---------|
| shard10 | 1,0 | 6 (12 jobs) | 3H+3B | All OK |
| shard45 | 4,5 | 5 (10 jobs) | 3H+2B | All OK |
| shard26 | 2,6 | 5 (10 jobs) | 2H+3B | 2 tomato infra failures → retry |

## Tomato Sauce Retry Audit

Original shard26 tomato_sauce_s2_w165_175 had infra-abnormal window sizes (n_window_steps=11 for atk=9, n_window_steps=2 for atk=10, expected=10). A 4-job retry was run on GPU 2,6 with identical parameters.

| Seed | Condition | Original | Retry | Verdict |
|------|-----------|----------|-------|---------|
| 9 | VIS | act_ws=11, open=7 | act_ws=11, open=8 | OK |
| 9 | RAND | act_ws=11, open=7 | act_ws=11, open=6 | OK |
| 10 | VIS | act_ws=2, open=1 | act_ws=11, open=8 | **Fixed** |
| 10 | RAND | act_ws=2, open=8 | act_ws=11, open=8 | **Fixed** |

After retry, both seeds are cmd_rand (VIS and RAND both open gripper). Retry confirmed: tomato is not VIS-specific, not an infra mask.

## Seed-Level Results

| Group | cmd_hit | cmd_rand | rand_only | no_effect | Total |
|-------|---------|----------|-----------|-----------|-------|
| H (HiddenSafeRank) | 4 (50%) | 2 (25%) | 2 (25%) | 0 | 8 |
| B (RandomRank) | 4 (50%) | 2 (25%) | 0 | 2 (25%) | 8 |

H has 2 rand_only seeds (cream_cheese both seeds): RAND opens gripper MORE than VIS. B has 2 no_effect seeds but no rand_only.

## Window-Level AND Results

S6 convention: both seeds must be cmd_hit for window to count as cmd_hit. Non-cmd_hit windows contribute 0 to window_yield_score.

**Group H (HiddenSafeRank)**:

| Window | Status | vis_rate atk=9 | vis_rate atk=10 | Yield Score |
|--------|--------|----------------|-----------------|-------------|
| milk_s0_w70_80 | cmd_hit | 0.800 | 0.900 | +0.750 |
| salad_dressing_s1_w50_60 | cmd_hit | 0.500 | 0.600 | +0.500 |
| cream_cheese_s0_w65_75 | **rand_only** | 0.400 | 0.400 | 0 |
| tomato_sauce_s2_w165_175 | cmd_rand | 0.800 | 0.800 | 0 |
| **Total** | | | | **+0.3125** |

**Group B (RandomRank)**:

| Window | Status | vis_rate atk=9 | vis_rate atk=10 | Yield Score |
|--------|--------|----------------|-----------------|-------------|
| milk_s0_w230_240 | cmd_hit | 1.100 | 1.100 | +1.100 |
| bbq_sauce_s2_w100_110 | cmd_hit | 0.800 | 0.800 | +0.800 |
| butter_s0_w90_100 | cmd_rand | 0.800 | 1.000 | 0 |
| cream_cheese_s0_w85_95 | no_effect | 0.000 | 0.300 | 0 |
| **Total** | | | | **+0.4750** |

## H vs B Comparison

| Metric | Group H | Group B | Diff | Gate |
|--------|---------|---------|------|------|
| window_yield_score | +0.3125 | +0.4750 | **-0.1625** | FAIL |
| Seed cmd_hit | 0.500 | 0.500 | 0 | PASS |
| Seed cmd_rand | 0.250 | 0.250 | 0 | PASS |
| Seed rand_only | 0.250 | 0.000 | +0.250 | FAIL |

Primary gate (H yield > B yield): **FAIL**. Strong gate (H yield - B yield >= +0.10): **FAIL**.

## Metric Definition

`window_yield_score` is the mean of `max(0, VIS_decoded_open_count / expected_window_size - RAND_decoded_open_count / expected_window_size)` over seeds where both seeds are cmd_hit. For non-cmd_hit windows, it is 0. Values can exceed 1.0 when decoded_open_count exceeds expected window size (e.g., milk_s0_w230_240: 11 open counts in a 10-step window because the gripper stays open past window end). **This is a count ratio, not the S6 probability-bounded yield_cmd.**

## Root Cause Analysis

**cream_cheese_s0_w65_75** is the decisive failure case for HiddenSafeRank. It was the #2 ranked HiddenSafe window (hs=0.2689) but is a **rand_only** window at both seeds: RAND perturbation opens the gripper MORE than VIS. HiddenSafe's offline "safe" prediction was incorrect — this window is not VIS-specific.

HiddenSafeRank correctly selected milk_s0_w70_80 and salad_dressing_s1_w50_60 (both strong cmd_hit), but these were offset by the cream_cheese false-positive and tomato_sauce cmd_rand.

## Claim Boundary

**Allowed:**
- HiddenSafe offline ranking signal (AUC=0.691) did not fresh-confirm under attack_seed 9/10
- Group H underperformed Group B in this top-4 confirmation
- HiddenSafe remains offline mechanism evidence (FP/FN separation)
- S6 CleanRand abstain remains the only validated pipeline

**Forbidden:**
- Hidden solved Layer-2
- Hidden detector conclusively failed (offline signal is real; fresh transfer failed)
- Action-hidden contains no signal
- Layer-2 is closed
