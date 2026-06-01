# VIS Final Controlled Rollout Result

**Date**: 2026-06-01 | **Branch**: `exp/vis-payload-upgrade-validation-20260601` | **HEAD**: `8ff150d8`

## 1. Bugs Fixed (3 critical)

| Bug | Before | After |
|-----|--------|-------|
| `get_libero_image` | Single flip `[::-1]` | Double flip `[::-1, ::-1]` |
| `normalize_gripper_action` | Simple `>=0` threshold | Production: `2*val-1 -> sign` |
| `invert_gripper_action` | `1.0 - val` | `-1.0 * val` |

## 2. Evidence Tiers

### Tier 1 — Cream Cheese State0 Task-Level PASS

| Condition | Duration | Success | OPEN Flips | ArmL2 | Verdict |
|-----------|----------|---------|------------|-------|---------|
| clean | d11 | True | 0 | 0 | Baseline valid |
| VIS | d16 | **False** | 10/16 | 0.88 | Minimal threshold |
| VIS | d18 | **False** | 13/18 | 0.87 | Confirmed |
| VIS | d20 | **False** | 11/20 | 0.87 | Stable operating point |
| random | d20 | True | 0/20 | 0.06 | Random negative |
| sparse | d20 | True | 8/20 | 0.88 | Sparse ineffective |

**Claim**: Corrected full-frame VIS PGD causes controlled task-level failure on cream_cheese state0 at d16-d20. Random same-Linf does not reproduce. Sparse strategy is ineffective.

### Tier 2 — Cream Cheese State1 (Denominator Caveat)

| Condition | Duration | Success | Flips | ArmL2 |
|-----------|----------|---------|-------|-------|
| VIS s1 | d20 | False | 11/20 | 0.87 |
| random s0 | d20 | False | 0/20 | 0.11 |
| random s1 | d20 | False | 1/20 | 0.10 |

VIS s1 produces strong gripper-channel effect (11/20 flips, armL2=0.87) matching state0. However matched random runs also fail with 0-1 flips and armL2=0.10 — environmental/natural trajectory failures, not perturbation-driven. **State1 task-level causality is confounded.** Strongly suggest denominator validation before multi-state claim.

### Tier 3 — Tomato Resilience

| Duration | Success | Flips | ArmL2 |
|----------|---------|-------|-------|
| d13 | True | 6/13 | 0.47 |
| d20 | True | 9/20 | 0.32 |
| d30 | True | 14/27 | 0.41 |
| d40 | True | 15/40 | 0.75 |
| d60 | **False** | 32/60 | 0.12 |

Tomato survives through d40, only fails at d60 (3x selective budget). ArmL2 at d60 is surprisingly low (0.12), suggesting clean gripper-channel mechanism at high duration. Tomato is genuinely more resilient than cream_cheese.

### Tier 4 — Robust Controls and Selectivity Window

| Task | d20 | d40 |
|------|-----|-----|
| ketchup | True (6/20) | **False** (22/40) |
| salad | True (5/20) | **False** (17/40) |
| cream_cheese | **False** (11/20) | — |

**d20 is the selective operating point**: cream_cheese fails, ketchup + salad survive. **d40 breaks selectivity**: robust controls also fail.

## 3. GPU Status

| GPU | Status |
|-----|--------|
| 0 | Xid13 — permanent damage |
| 3 | Xid31 — permanent damage |
| 1,2,4,5,6,7 | Healthy |

## 4. Detector-Triggered Readiness

**BLOCKED**. Conditions not met:
- [x] Cream cheese state0 controlled task-level PASS
- [x] Random baseline negative (0 flips)
- [x] Robust controls survive at d20
- [ ] Multi-state repeatability with clean denominator
- [ ] qpos/width physical response audit complete
- [ ] Detector-triggered proposal written and approved

## 5. Allowed Claims

- Corrected full VIS PGD causes controlled task-level failure on cream_cheese state0 at d16-d20
- Random_linf does not reproduce (0/20 flips, armL2=0.06)
- d20 is best selective operating point
- d40 breaks robust controls
- Tomato is resilient through d40
- Sparse strategy is negative ablation

## 6. Forbidden Claims

- Detector-triggered VIS validated
- Multi-state task failure proven (state1 confounded)
- Universal VIS attack
- Clean selectivity at d40
