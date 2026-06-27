# Table 1 — Final Numerical Freeze

**Status**: `TABLE1_CORE_NUMERICAL_FREEZE = PASS`
**Date**: 2026-06-27
**Analysis freeze commit**: `904a26a0a984ec52660143b0b151d3d536c99212`

---

## Panel A: Auditable Reference (metric_refresh_v2, N=27/condition)

9 clean-qualified LIBERO-Object reference cells × 3 perturbation seeds × 4 conditions = 108 runs.
All 108/108 runs emitted (coverage = 100%); ITT and conditional denominators coincide.

| Method | No Lock ITT FR | 95% CI | ArmLock ITT FR | 95% CI |
|--------|:---:|:---:|:---:|:---:|
| TMA | 24/27 = 88.9% | [71.9%, 96.1%] | 22/27 = 81.5% | [63.3%, 91.8%] |
| Prefix | 22/27 = 81.5% | [63.3%, 91.8%] | 27/27 = 100.0% | [87.5%, 100.0%] |

**Paired McNemar (N=27)**:
- TMA: b=5, c=3, Delta = -7.4pp, exact p=0.727
- Prefix: b=0, c=5, Delta = +18.5pp, exact p=0.063

---

## Panel B: New-State Breadth (breadth_120, ITT N=24, Conditional N=21)

8 clean-qualified new state-slots × 3 perturbation seeds × 5 conditions = 120 formal runs.
tomato_sauce_s1 systematic detector non-emission → coverage = 21/24 = 87.5%.

| Condition | ITT FR (N=24) | 95% CI | Conditional FR (N=21) | 95% CI | Coverage |
|-----------|:---:|:---:|:---:|:---:|:---:|
| RAND | 0/24 = 0.0% | [0.0%, 13.8%] | 0/21 = 0.0% | [0.0%, 15.5%] | 87.5% |
| TMA no-lock | 17/24 = 70.8% | [50.8%, 85.1%] | 17/21 = 81.0% | [60.0%, 92.3%] | 87.5% |
| TMA ArmLock | 18/24 = 75.0% | [55.1%, 88.0%] | 18/21 = 85.7% | [65.4%, 95.0%] | 87.5% |
| Prefix no-lock | 19/24 = 79.2% | [59.5%, 90.8%] | 19/21 = 90.5% | [71.1%, 97.3%] | 87.5% |
| Prefix ArmLock | 19/24 = 79.2% | [59.5%, 90.8%] | 19/21 = 90.5% | [71.1%, 97.3%] | 87.5% |

**Paired McNemar (conditional N=21)**:
- TMA: b=0, c=1, Delta = +4.8pp, exact p=1.000
- Prefix: b=1, c=1, Delta = 0.0pp, exact p=1.000

### Per-State Failure Rates (ITT, N=3 seeds per slot)

| State Slot | TMA NL | TMA AL | Prefix NL | Prefix AL | Coverage |
|-----------|:------:|:------:|:---------:|:---------:|:--------:|
| salad_s1 | 2/3 | 3/3 | 3/3 | 3/3 | emit |
| bbq_s4 | 3/3 | 3/3 | 3/3 | 3/3 | emit |
| ketchup_s1 | 3/3 | 3/3 | 3/3 | 3/3 | emit |
| milk_s5 | 3/3 | 3/3 | 3/3 | 3/3 | emit |
| butterA_s5 | 3/3 | 3/3 | 3/3 | 3/3 | emit |
| orange_s2 | 0/3 | 0/3 | 1/3 | 1/3 | emit |
| tomato_s1 | 0/3* | 0/3* | 0/3* | 0/3* | **no-emit** |
| butterB_s6 | 3/3 | 3/3 | 3/3 | 3/3 | emit |

*tomato_s1: 0/3 failure but 0/3 emit — detector never triggered, attack never executed.

### Macro Failure Rates

| Condition | Run-level (N=24) | State-slot macro (8 slots) | Task macro (7 tasks) |
|-----------|:---:|:---:|:---:|
| TMA no-lock | 70.8% | 70.8% | 66.7% |
| TMA ArmLock | 75.0% | 75.0% | 71.4% |
| Prefix no-lock | 79.2% | 79.2% | 76.2% |
| Prefix ArmLock | 79.2% | 79.2% | 76.2% |

---

## Panel C: Timing (Legacy, Provisional)

Historical timing analysis from supplement_7h (329 runs available, formal subset TBD):

| Method | Random-Time | Early-Shift | Student Trigger |
|--------|:---:|:---:|:---:|
| TMA | 0/27 = 0.0% | 12/27 = 44.4% | 22/27 = 81.5% |
| Prefix | 2/27 = 7.4% | 12/27 = 44.4% | 21/27 = 77.8% |

**Status**: Legacy provenance; formal subset reconstruction from 329 supplement_7h runs pending.

---

## Panel D: Baseline

| Method | FR | 95% CI | N |
|--------|:---:|:---:|:--:|
| RAND (matched-budget control) | 0/24 = 0.0% | [0.0%, 13.8%] | 24 |
| Adapted Untargeted Clean-Token CE PGD | 0/27 = 0.0% | [0.0%, 12.5%] | 27 formal (+ 3 canary excluded) |

Untargeted CE: 27/27 emit, attack_frames=10, no fallback, 0 L-inf violations, 0 task failures.

---

## Mechanism Summary

### rNAD (Range-Normalized Action Discrepancy, 108 metric refresh runs)

| Metric | TMA no-lock | TMA ArmLock | Prefix no-lock | Prefix ArmLock |
|--------|:----------:|:----------:|:--------------:|:--------------:|
| rNAD_policy_arm | 0.1622 | 0.1720 | 0.1065 | 0.1063 |
| rNAD_exec_arm | 0.1622 | **0.0000** | 0.1065 | **0.0000** |
| rNAD_exec_grip | 0.1697 | 0.1939 | 0.1622 | 0.1846 |

ArmLock invariant: 540 attack frames (metric refresh), 0 violations.

### Latency (mean, per attacked step)

| Condition | Attack Preparation | Total Step |
|-----------|:---:|:---:|
| TMA no-lock | 12.3 s | 13.7 s |
| TMA ArmLock | 11.1 s | 12.4 s |
| Prefix no-lock | 35.4 s | 36.6 s |
| Prefix ArmLock | 39.0 s | 40.4 s |

Prefix/TMA ratio: ~2.9-3.5x (median), ~3.1-3.5x (mean).

---

## Frozen Claims

1. Student-trigger timing substantially outperforms Random-Time and Early-Shift.
2. TMA and Prefix both produce high task failure rates in LIBERO-Object.
3. Attack transfers to 8 previously unevaluated clean-qualified state slots, with state-dependent effectiveness and coverage.
4. Prefix ArmLock 100% FR is limited to reference cells; new-state breadth shows 79.2% ITT FR.
5. Detector coverage (87.5% in breadth) is a critical bottleneck: tomato_sauce_s1 exhibits state-specific systematic non-emission.
6. ArmLock eliminates executed arm discrepancy (rNAD_exec_arm = 0, 540 frames, 0 violations).
7. High failure rates persist after arm perturbation removal, supporting a causal role for the gripper channel.
8. Prefix implementation incurs ~3x the attack-preparation latency of TMA.
9. Detector emit timing is highly stable across same-key repeats, but binary outcomes vary across state sets.
10. RAND matched-budget perturbation produces zero open-token commands and zero task failures.

## Pending (Not Frozen)

- CQFR human review (55 enriched + 108 full packages ready)
- Timing panel formal subset reconstruction
- Global accepted-run count final adjudication
- Reference N=45 legacy panel audit
- Cross-suite generalization
- True Action-Discrepancy baseline
