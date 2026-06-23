# M1C R2 Validation Selection Protocol

**Status**: FROZEN_BEFORE_VALIDATION
**Date**: 2026-06-23
**Reference**: M1C Protocol `migration_audit/m1c/M1C_PROTOCOL_DRAFT.md` (FROZEN_AFTER_PHASE_A)

## Purpose

Select optimal R2 candidate-FSM configuration using independent validation data.
R1 is fixed (no tuning). R2 is tuned within a pre-registered grid.

## R1 Fixed Parameters

```json
{
  "fsm_version": "v1r_r1",
  "tau_corridor": 0.3,
  "tau_release": 0.3,
  "guard": 5
}
```

R1 SHALL NOT be modified based on validation results.

## R2 Search Grid

Fixed 72-configuration grid (4 × 2 × 3 × 3):

```text
tau_on        ∈ {0.30, 0.35, 0.40, 0.45}
hysteresis_gap ∈ {0.05, 0.10}
  → tau_off = tau_on - hysteresis_gap
n_candidate   ∈ {1, 2, 3}
max_arm_age   ∈ {20, 50, 100}
guard         = 5  (fixed)
tau_release   = 0.3 (fixed)
```

Grid frozen before any validation results are inspected.
No expansion, contraction, or modification after freeze.

## Six Absolute Gates

| Metric | Threshold |
|---|---|
| Coverage | ≥ 0.80 |
| False-early | ≤ 0.10 |
| Post-release | ≤ 0.05 |
| K10 containment | ≥ 0.85 |
| Median Teacher-anchor error | ≤ 8 |
| No-corridor abstain | ≥ 0.90 |
| Feature-valid rate | ≥ 0.99 |
| Silent ARM stalls | = 0 |

## Selection Rule (Lexicographic)

1. **Filter**: Remove all configs failing any of the 8 gates above.
2. **Maximize**: Among survivors, maximize no-corridor abstain.
3. **Tie-break** (in order):
   a. Higher K10 containment
   b. Lower median Teacher-anchor error
   c. Lower false-early rate
   d. Fewer mean disarms
   e. Lower n_candidate (simpler FSM)
   f. Smaller hysteresis gap (less aggressive separation)
4. If no config survives step 1 → `RUNTIME_ONLY_REPAIR_INSUFFICIENT` → SC5-v2 retrain GO.

## Denominator Requirements

Validation must contain:
- teacher-valid ≥ 30
- no-corridor ≥ 20

If insufficient: stop, report `VALIDATION_DENOMINATOR_INSUFFICIENT`.
No post-hoc sample addition to meet thresholds.

## Prohibited

- Tuning on M1B 60-cell diagnostic data
- Tuning on train corpus
- Tuning on any blind data
- Expanding grid after seeing validation results
- Selecting by composite score rather than lexicographic rule
