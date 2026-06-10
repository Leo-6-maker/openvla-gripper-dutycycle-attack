# Layer-2 HiddenSafe Confirmation Queue Audit

**Date**: 2026-06-10
**S7 freeze commit**: de96877
**Queue commit**: TBD

## Queue Design

| Parameter | Value |
|-----------|-------|
| Total jobs | 32 |
| Unique windows | 8 |
| Attack seeds | 9, 10 |
| Conditions | VIS, RAND (matched pairs) |
| Eps | 0.03 |
| Budget | linf |

## Group Composition

### Group H: HiddenSafeRank (4 windows)

| Window | Task | hs_score | cr_score | Label |
|--------|------|----------|----------|-------|
| tomato_sauce_s2_w165_175 | tomato_sauce | 0.1146 | 0.3222 | stable_rand_sensitive |
| cream_cheese_s0_w65_75 | cream_cheese | 0.2689 | 0.4530 | stable_rand_sensitive |
| milk_s0_w70_80 | milk | 0.3007 | 0.1096 | stable_cmd_specific |
| salad_dressing_s1_w50_60 | salad_dressing | 0.4133 | 0.4246 | stable_cmd_specific |

Group H summary: cmd=0.500, rand=0.500, yield=+0.50, tasks=4

### Group B: RandomRank baseline (4 windows)

| Window | Task | hs_score | cr_score | Label |
|--------|------|----------|----------|-------|
| milk_s0_w230_240 | milk | 0.4683 | 0.4074 | stable_cmd_specific |
| bbq_sauce_s2_w100_110 | bbq_sauce | 0.6180 | 0.3204 | stable_negative |
| butter_s0_w90_100 | butter | 0.8233 | 0.4221 | stable_rand_sensitive |
| cream_cheese_s0_w85_95 | cream_cheese | 0.3470 | 0.4246 | stable_rand_sensitive |

Group B summary: cmd=0.250, rand=0.500, yield=+0.45, tasks=4

## Selection Rules (FROZEN)

1. **Candidate pool**: 40-parent stable pool v2, CleanRand pass set (bottom 50% by CleanRand OOF score)
2. **HiddenSafe polarity**: `HiddenSafe = 1 − HiddenRisk` (frozen, not post-hoc)
3. **Group H ranking**: Lowest HiddenSafe score first (ascending), task-diverse greedy
4. **Group B ranking**: Random shuffle of remaining CleanRand pass windows, task-diverse greedy
5. **No overlap** between groups

## Queue Audit Gates

| Gate | Result |
|------|--------|
| Total jobs = 32 | PASS |
| 8 unique windows | PASS |
| Attack seeds = [9, 10] | PASS |
| Each logical_pair has 2 rows | PASS |
| Each logical_pair has 1 VIS + 1 RAND | PASS |
| Group H = 4 windows | PASS |
| Group B = 4 windows | PASS |
| No Group H/B overlap | PASS |
| All windows from CleanRand pass set | PASS |
| online_safe = True | PASS |
| HiddenSafe polarity frozen | PASS |
| No GPU 3,7 in plan | PASS |

**ALL GATES PASS**

## Confirmation Success Gates

Primary:
- Group H yield_cmd > Group B yield_cmd

Secondary:
- Group H cmd_hit >= Group B cmd_hit
- Group H cmd_rand <= Group B cmd_rand

Strong gate:
- Group H yield_cmd − Group B yield_cmd >= +0.10
- Group H cmd_rand <= Group B cmd_rand

Infrastructure:
- 32/32 jobs complete
- 0 unmatched VIS/RAND pairs
- No seed/window mismatch

## GPU Allocation

| GPU Pair | Role | Jobs |
|----------|------|------|
| 1,0 | Group H | 16 jobs (VIS+RAND) |
| 4,5 | Group B | 16 jobs (VIS+RAND) |
| 2,6 | Reserve/retry | — |
| 3,7 | BLACKLIST | NEVER USE |

## Launch Constraints

- Do NOT launch without explicit approval
- Do NOT auto-launch from watcher
- All 32 jobs must use staggered launch (30s between workers)
- Use tmux sessions for persistence
- Per-handoff runbook: model before LIBERO, EGL physical GPU mapping

## Claim Boundary

This queue tests whether HiddenSafeRank's offline fixed-K ranking advantage (from the S7 readout) translates to improved VIS-specific yield under fresh attack_seeds 9 and 10.

Success would support (NOT prove) that hidden features improve Layer-2 ranking. Failure would not invalidate S6 Layer-1 CleanRand abstain.
