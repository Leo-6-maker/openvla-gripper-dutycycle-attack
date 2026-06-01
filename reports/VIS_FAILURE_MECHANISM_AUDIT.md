# VIS Failure Mechanism Audit

**Date**: 2026-06-01

## Core Finding

**VIS task failure is NOT explained by decoded OPEN count alone.** The key mediator is **physical gripper response (qpos/width)**. Decoded OPEN commands must translate to actual gripper opening to cause task failure. This translation is task-dependent.

## Task Vulnerability Ranking

| Task | Fail Rate | Avg Flips | Avg ArmL2 | qpos_delta | Streak | Mechanism |
|------|-----------|-----------|-----------|------------|--------|-----------|
| salad | 71% (5/7) | 0.43 | 0.80 | **0.020-0.024** | 5-7 | High physical transfer |
| cream | 57% (4/7) | 0.58 | 0.87 | 0.004-0.006 | 6 | Moderate physical transfer |
| ketchup | 33% (3/9) | 0.43 | 0.79 | **~0.000** | 2-4 | Low physical transfer |
| tomato | 17% (1/6) | 0.42 | 0.35 | ~0 (d60: 0.038) | 4 (d60:10) | Very resilient |

## Mechanism by Task

### Salad Dressing — Most Vulnerable
- Even d16 produces qpos_delta=0.022 and streak=5-7
- VIS OPEN commands strongly translate to physical gripper opening
- Object geometry/friction likely provides less resistance to opening
- **Not a stable robust control for VIS**

### Cream Cheese — Reliably Vulnerable  
- Requires streak>=6 for failure (streak=4 survives)
- qpos_delta=0.004-0.006 when failing
- Moderate physical transfer: enough flips at enough density → gripper opens → object drops
- **Cleanest controlled VIS positive**

### Ketchup — Action-level Affected, Physical-level Tolerant
- Decoded OPEN flips 6-11/20 but qpos_delta≈0
- VIS commands don't translate to physical gripper opening
- Object/grasp mechanics resist opening despite OPEN commands
- **Explains why ketchup survives high flip counts**

### Tomato — Physically Resilient
- qpos_delta≈0 through d40 despite 15/40 flips
- Only d60 achieves qpos_delta=0.038 and streak=10 → failure
- Object geometry/placement tolerance creates very high resilience
- **d60 is 3x over-budget; d60 failure is not selective**

## Open Count vs Qpos vs Failure

| Metric | Explains Failure? |
|--------|-------------------|
| OPEN count | Weak — ketchup survives 11/20 |
| OPEN streak | Better — cream needs >=6 |
| **qpos_delta** | **Best — strongest mediator** |
| Arm L2 | Weak — similar across tasks |

## Implications

1. **VIS payload is a gripper-channel instantiator, not a universal attack**
2. **Physical transfer is task-dependent**: object geometry, grasp contact, friction, placement tolerance determine whether OPEN commands → qpos opening → failure
3. **Adaptive budget controller** should use qpos/streak feedback, not fixed duration
4. **Salad is not a robust control** for VIS — it's more vulnerable than cream
5. **Ketchup tolerance** is explained by low physical transfer, not task robustness
