# Stage-B RC1a S6 Final Report

**Date**: 2026-06-10
**Commit**: 6bb53db
**Branch**: exp/vis-prefix-margin-repair-20260603

## Executive Summary

S6 established the abstain-first pipeline v0.3 closed loop and conducted Layer-2 feature exploration. The core claim is validated; Layer-2 ranking remains open.

### Core Claim (Validated)

> Layer-1 CleanRand abstain filter improves command-level VIS-specific window selection.
> CleanRand-pass windows achieve yield=+0.94 across 4 independent fresh attack seeds.
> This result is confirmed under fresh matched VIS/RAND confirmation and robustness add-on.

### Layer-2 Status (WIP)

> Action-dynamics and action-logit provide mechanism evidence for FP/FN structure.
> Neither improves selector-level ranking beyond the current best pipeline (CleanRand+RandomRank).
> Action-hidden extraction is the next direction.

### Layer-3 Status (Underpowered)

> Strict vis_phys remains 5/40 (no expansion from K5c).
> Phys differentials <0.1 for all candidates.
> Cmd-first phys mining + extended-window diagnostic is the next step.

---

## S6 Milestone Chain

| Commit | Milestone | Key Result |
|--------|-----------|------------|
| `5107231` | P1 fixes | phys labels, dual metrics, retry provenance |
| `19aaf87` | Pipeline v0.3 fresh confirmation | 5/5 gates PASS, yield=+1.00 seeds 5-6 |
| `3f3eb8f` | Robustness seeds 7-8 | yield=+0.94 across 4 seeds |
| `17538e8` | Threshold sweep + FP/FN audit | root cause: closed gripper + high qpos blind spot |
| `d9ef0df` | Phys candidate audit | top candidates identified, all phys_diff <0.1 |
| `8f2ea07` | P1 online-safe action-logit smoke | pre-window only, official prompt, FP/FN separated |
| `ee50a9e` | P0-fix action-logit smoke | leakage fixed, signal persists |
| `6bb53db` | Action-logit full readout | minimal PASS, mechanism evidence |

---

## Stable Pool v2

40 parents across 9 tasks:
- stable_cmd_specific: 15 (+1 borderline)
- stable_rand_sensitive: 16
- stable_negative: 5
- unstable_or_unknown: 3
- stable_vis_phys: 5

## Pipeline v0.3 Confirmation

| Group | cmd_hit | cmd_rand | yield |
|-------|---------|----------|-------|
| A: CleanRand-pass | 0.94 | 0.06 | +0.94 |
| B: TaskOnly | 0.25 | 0.50 | +0.00 |
| C: High-risk | 0.00 | 0.88 | -0.38 |

## Action-Logit Readout

| Model | RandAUC | FP_tomato | FN_salad | yield |
|-------|---------|-----------|----------|-------|
| TaskOnly | 0.447 | 0.47 | 0.46 | +0.34 |
| CleanProprio | 0.460 | 0.68 | 0.54 | +0.41 |
| ActionLogitOnly | 0.566 | 0.41 | 0.45 | +0.59 |

- ActionLogit improves rand detection (0.566 > 0.460) and fixes FP tomato (0.68→0.41)
- Does NOT exceed CleanRand+RandomRank best pipeline
- FN salad not resolved (static-open ambiguity)
- Decision: MINIMAL PASS — mechanism evidence, not selector improvement

## Allowed Claims

1. Layer-1 CleanRand abstain validated under fresh confirmation
2. Fixed-env K-repeat protocol gives stable probability labels
3. Abstain-first pipeline improves VIS-specific yield over baseline
4. Action-dynamics and action-logit explain FP/FN structure
5. Action-hidden is the next justified direction for Layer-2

## Forbidden Claims

1. Full vulnerable-window detector solved
2. Layer-2 CleanCmd ranking solved
3. Layer-3 strict physical bridge solved
4. Cross-task/suite generalization demonstrated
5. Action-logit alone improves selector ranking

## S7 Plan

1. Action-hidden full extraction + readout
2. If hidden strong pass: Layer-2 fresh confirmation queue
3. Extended-window phys diagnostic
4. Cross-suite clean mechanism eligibility
