# VIS Multi-Phase Attack Plan

**Date**: 2026-06-02

## Goal

Test whether distributing VIS perturbation budget across multiple vulnerable task phases improves end-to-end task failure compared with a single best d16 phase.

This is not K-only adaptive controller tuning. It is a fixed phase-budget scheduler. It is not detector-triggered online VIS unless ProprioNoStep actually supplies the phase windows; the initial windows in `tables/vis_multiphase_phase_windows.csv` are hand-coded phase proposals derived from existing single-phase windows and phase intuition.

## Conditions

| Condition | Scheduler | Total VIS Budget | Comparator Role |
|-----------|-----------|------------------|-----------------|
| A | `clean` | 0 | clean denominator |
| B | `random_matched_<schedule>` | matched to schedule | random visual-noise control |
| C | `single_best_phase_d16` | 16 | single-phase baseline |
| D | `two_phase_equal_d8_d8` | 16 | equal total budget vs C |
| E | `three_phase_equal_d6_d6_d6` | 18 | slightly larger distributed budget |
| F | `two_phase_stronger_d12_d12` | 24 | stronger-budget diagnostic only |

Critical comparison order:

1. Compare equal total budget first: C d16 vs D d8+d8.
2. Treat E d6+d6+d6 as slightly larger budget and report it separately.
3. Treat F d12+d12 as stronger-budget diagnostic only; do not claim improvement if only F works.

## Tasks and Seeds

Initial tasks:

- `cream_cheese`
- `ketchup`
- `tomato_sauce`

Optional reference:

- `salad_dressing`, vulnerable reference only, not a robust control.

Initial seed/state:

- state0 seed0 first.
- Expand to seed1/2 only if a signal appears and matched random does not reproduce it.

## Metrics

Per run:

- `official_success`
- CQ placeholders: `cq_success`, `cq_failure`, `manual_audit_needed`
- `failure_phase`
- total attack steps and transparent total budget
- attacked-step token flips, OPEN count, longest OPEN streak
- total qpos/width delta
- mean attacked-step armL2
- matched random result

Per phase:

- phase-wise OPEN count
- phase-wise longest OPEN streak
- phase-wise qpos_delta
- phase-wise width_delta
- phase-wise attack steps
- phase-wise armL2

## Success Gate

A multi-phase claim is allowed only if:

1. Equal-budget D d8+d8 beats C single d16 on at least `cream_cheese` or `tomato_sauce`.
2. Matched random does not reproduce the same failure.
3. qpos/CQ/manual evidence supports gripper/contact failure.
4. Total budget is reported transparently.

If only E or F works, the conclusion is "higher or slightly higher total budget may matter", not "multi-phase improves selective VIS."
