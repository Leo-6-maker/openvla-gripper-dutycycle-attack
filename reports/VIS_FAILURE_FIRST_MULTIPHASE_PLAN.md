# VIS Failure-First Multi-Phase Plan

**Date**: 2026-06-02

## Objective

Establish whether a reliable VIS task-level failure upper bound exists under high phase coverage and stronger VIS payload. This phase does not optimize budget, selectivity, or controller policy.

Allowed claim after this phase is only: VIS failure upper bound exists or does not exist under multi-phase high-budget settings.

## Boundaries

- Do not continue K-only adaptive controller tuning.
- Do not claim efficient, selective, detector-triggered, budget-optimal, production-controller, or cross-suite attack.
- Do not call this detector-triggered online VIS unless ProprioNoStep supplies the windows. The initial windows are hand-coded/manual phase proposals.
- Do not use simulator success alone; use qpos/CQ/manual audit fields.
- Salad is a vulnerable reference, not a robust control.

## Initial Tasks

1. `cream_cheese`
2. `salad_dressing`
3. `ketchup`
4. `tomato_sauce`

Initial state/seed: state0 seed0.

## Conditions

| Condition | Script Schedule | Attack Type | Total Attack Frames | Purpose |
|-----------|-----------------|-------------|---------------------|---------|
| clean | `clean` | clean | 0 | clean denominator |
| random_linf multi-phase strong | `ultra_three_phase_d20_d20_d20` | random_linf | 60 | matched strongest random control |
| VIS single best phase d20 | `single_best_phase_d20` | vis_pgd | 20 | single-phase baseline |
| VIS two-phase strong | `two_phase_strong_carry_preplace_d20_d20` | vis_pgd | 40 | carry + pre-place coverage |
| VIS three-phase strong | `three_phase_strong_d16_d16_d16` | vis_pgd | 48 | contact + carry + pre-place high coverage |
| VIS ultra diagnostic | `ultra_three_phase_d20_d20_d20` | vis_pgd | 60 | upper-bound diagnostic |

## Payload

Initial payload:

- epsilon: 8/255
- PGD steps: 40
- step size: 1/255
- objective: `gripper_open_region_ce`

If no task-level failure appears, increase to epsilon 12/255 with 40 PGD steps. Lower-payload and objective ablations are deferred until the first reliable failure.

## Success Criteria

Failure-first PASS:

1. VIS multi-phase causes task failure on at least one target task.
2. Matched random does not reproduce VIS-like qpos/open trace.
3. qpos/CQ/manual evidence supports gripper/contact failure.
4. Failure occurs after or during an attacked phase.

Strong PASS:

- cream and at least one of ketchup/tomato fail under multi-phase VIS.
- random matched survives.
- qpos_delta_post increases clearly.

Medium PASS:

- cream/salad fail robustly but ketchup/tomato survive.
- This establishes easy positive but not hard-case transfer.

FAIL:

- even eps=12/255, steps=40, multi-phase d20+d20+d20 does not cause clear VIS-specific failure.
