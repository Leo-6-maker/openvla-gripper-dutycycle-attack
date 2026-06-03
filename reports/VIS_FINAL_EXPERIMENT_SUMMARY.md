# VIS prefix_margin — Final Experiment Summary

**Date**: 2026-06-03  
**Branch**: `exp/vis-attack-strength-upgrade-20260602`  
**Objective**: `prefix_locked_gripper_open_margin`
**Status**: PRE-REPAIR CANDIDATE EVIDENCE — under audit (see [VIS_PREFIX_MARGIN_REPAIR_STATUS.md](VIS_PREFIX_MARGIN_REPAIR_STATUS.md))

## Main Result

**On ketchup at eps6, prefix_margin is pre-repair candidate evidence; final claim pending canonical gripper semantics, prefix-loss repair, and provenance rerun.** The historical results below were computed with semantically inconsistent OPEN predicates (`adv_grip > 0.5` counts CLOSE, not OPEN) and a prefix-locked loss that masked gripper labels to -100 (zero gripper loss). Provenance must be regenerated from trace CSVs using the canonical helper.

## Ketchup — Core Evidence

### Prefix (ALL fail)

| Window | eps | Seeds | OPEN | armL2 | qpos_delta |
|--------|-----|-------|------|-------|-----------|
| 10-27 | 8 | 3/3 | 17-18/18 | 0.000 | 0.0375-0.0379 |
| 10-27 | 6 | 4/4 | 16-18/18 | 0.000 | 0.0375-0.0378 |
| 10-27 | 4 | 2/2 | 18/18 | 0.000 | 0.0375-0.0378 |
| 20-37 | 8 | 1/1 | 18/18 | 0.000 | 0.0384 |
| 20-37 | 6 | 3/3 | 18/18 | 0.000 | 0.0384 |

**14/14 seeds across 3 budgets, 2 windows. armL2=0.000 on all.**

### Random Controls (overwhelmingly clean)

| Window | eps | Seeds | Success | OPEN |
|--------|-----|-------|---------|------|
| 10-27 | 4 | 3 | 3/3 | 0/18 |
| 10-27 | 6 | 6 | 6/6 | 0/18 |
| 10-27 | 8 | 6 | 5/6 | 0/18 |
| 20-37 | 6 | 6 | 5/6 | 0/18 |
| 20-37 | 8 | 2 | 2/2 | 0/18 |

### Same-Budget eps6 (strongest result)

| Prefix eps6 | Random eps6 |
|-------------|-------------|
| 4/4 fail | 6/6 success |
| 16-18/18 OPEN | 0/18 OPEN |
| qpos 0.039→0.001 | qpos near clean |
| armL2=0.000 | armL2=0 |

## Cream — Physical Bridge, No Task Failure

| Window | eps | OPEN | armL2 | qpos_delta | Done |
|--------|-----|------|-------|-----------|------|
| 12-20 | 8 | 6/9 | 0.000 | 0.0045 | True |
| 26-36 | 8 | 6-7/11 | 0.000 | 0.0173 | True |
| 26-36 | 6 | 7/11 | 0.000 | 0.0177 | True |
| 34-51 | 6 | 12/18 | 0.000 | 0.0325 | True |

Prefix induces OPEN and qpos response but tasks survive. Random clean on all tested windows.

## Salad — Denominator Polluted

| Window | eps | OPEN | armL2 | qpos_delta | Done | Random |
|--------|-----|------|-------|-----------|------|--------|
| 10-25 | 8 | 14-15/16 | 0.000 | 0.035-0.037 | True | 6/6 FAIL |

Physical bridge exists but random eps4 fails 6/6 — denominator polluted. Cannot claim VIS-specific.

## Mechanism: Early Grasp Disruption

Prefix induces generated OPEN at attack-window start (step 10 or 20). Clean natural OPEN occurs at step 63. qpos transitions from closed to fully open within window. Object is never stably grasped. Episode runs to timeout. armL2=0 throughout.

## ProprioNoStep Relation

ProprioNoStep detects release-phase hazards (top windows are natural-release-confounded, clean OPEN >94%). Current VIS windows exploit approach/grasp vulnerability. They are complementary — not aligned. ProprioNoStep-guided VIS is NOT claimed.

## Detector Training Status

NOT ready. Dataset v0 exists (114 windows, 3 tasks) but positives are ketchup-dominated. Cross-task task-level positives missing. Leakage audit pending. Baselines undefined.

## Claim Boundary (pre-repair — under audit)

**Pre-repair allowed** (subject to provenance recomputation): same-budget VIS-specific ketchup early_grasp_disruption at eps6; gripper-channel selectivity (armL2=0); physical bridge; window generalization to 20-37; budget compression to eps4.

**Currently forbidden** (pending repair): all claims above, until provenance is regenerated with canonical semantics and prefix-loss is verified to include gripper term.

**Permanently forbidden**: broad LIBERO generalization; ProprioNoStep-guided attack; pre-release drop; cream/salad task-level claim; trained detector ready; direct `< 0.5` / `> 0.5` comparisons outside `gripper_semantics.py`.

**Repair audit**: see [VIS_PREFIX_MARGIN_REPAIR_STATUS.md](VIS_PREFIX_MARGIN_REPAIR_STATUS.md)
