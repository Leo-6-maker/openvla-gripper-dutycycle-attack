# Object Detector Matched Attack Pilot Plan

**Status**: BLOCKED (pending shadow validation + user approval)
**Date**: 2026-05-29

## Purpose
Prove that detector-selected windows are physically vulnerable to gripper duty-cycle attacks. All attack conditions use the SAME detector trigger.

## Pilot Design

### Tasks (2 tasks × 5 states = 10 states)
| Task | States |
|------|--------|
| cream_cheese (or ketchup) | s0-s4 |
| milk (or salad_dressing) | s0-s4 |

Optional diagnostic only (not main claim): bbq_sauce (poor clean stability).

### Conditions (4 conditions per state)
| # | Condition | Description |
|---|-----------|-------------|
| 1 | **clean_rerun** | Clean OpenVLA rollout — baseline |
| 2 | **detector_oracle_open** | Trigger → oracle_env_gripper_open (arm action preserved) |
| 3 | **detector_random_control** | Trigger → random gripper control (NOT optimized for open) |
| 4 | **detector_VIS_targeted** | Trigger → VIS_current targeting gripper channel |

**Total**: 2 tasks × 5 states × 4 conditions = **40 rollouts**

### Rules
- One detector trigger for ALL conditions (clean, oracle, random, VIS)
- Attack module CANNOT reselect window
- Random control must NOT optimize gripper-open objective
- Oracle open must ONLY override gripper dimension, preserve arm action
- VIS must target gripper channel only
- No attack outcome feeds back into detector

## Attack Parameters

### Oracle Open
```yaml
attack_objective: oracle_env_gripper_open
rho: 1.0
attack_steps: det_trigger_duration
```

### Random Control
```yaml
attack_objective: ctrl_random_direction_arm_only
rho: 1.0
eps: 0.10
attack_steps: det_trigger_duration
```

### VIS Targeted
```yaml
attack_objective: targeted_directional_ce
target_channel: gripper
rho: 1.0
eps: 0.25
attack_steps: det_trigger_duration
```

## Evaluation Metrics

### Primary
| Metric | Description |
|--------|-------------|
| Official SR | LIBERO success rate |
| CQFR | Contact-quality failure rate |
| CQSR | Contact-quality success rate |
| SR-CQ mismatch | Official SR minus CQSR |

### Gripper-Specific
| Metric | Description |
|--------|-------------|
| Gripper open rate | Fraction of steps with gripper_qpos > threshold |
| Gripper open streak | Max consecutive open steps |
| Gripper qpos change | Pre-trigger vs during-trigger qpos difference |
| Gripper width change | Pre-trigger vs during-trigger width difference |

### Arm Deviation
| Metric | Description |
|--------|-------------|
| Arm deviation / NAD | Normalized arm deviation from clean trajectory (if available) |
| EEF displacement | During-trigger EEF displacement magnitude |

### Detector
| Metric | Description |
|--------|-------------|
| Trigger alignment | Does trigger window match teacher window? |
| False early rate | Triggers before teacher window |
| Miss rate | Failed to trigger on valid windows |
| Failed episode FP | Triggers on clean-fail episodes |

### Failure Audit
| Metric | Description |
|--------|-------------|
| failure_phase_auto | Automatic failure phase classification |
| Manual audit | Stratified random subset (≥20% of failures) |

## Success Criteria (Gates)

### Gate 1: Oracle Gate
- detector_oracle_open CQFR >> clean_rerun CQFR
- Proves selected window is physically gripper-sensitive

### Gate 2: VIS-vs-Random Gate
- detector_VIS CQFR > detector_random CQFR
- detector_VIS gripper-open metrics > random gripper-open metrics
- Random does NOT produce same gripper-open contact failure pattern
- Proves VIS attack is targeted, not just any perturbation

### Gate 3: CQ Gate
- CQFR/CQSR detects failures that official SR misses
- Manual audit agrees on stratified subset (≥80% agreement)
- CQ metrics are not tuned on VIS outcomes

### Gate 4: Detector Gate
- Trigger aligns with teacher/shadow phase
- No excessive false early (tolerant ±2 acceptable)
- Failed episode FP low (≤1)

## Blocking Conditions (Do NOT launch until)
1. [ ] Stage 1 replay/timing audit PASSES ✓ (done)
2. [ ] Stage 2 fusion/trigger policy SELECTED (pending)
3. [ ] Stage 3 online shadow validation PASSES (not launched)
4. [ ] Stage 5 CQ metrics READY (pending)
5. [ ] User explicitly approves attack pilot launch

## Boundary
- Attack pilot uses detector trigger derived from clean-only labels
- No attack outcome is used to tune detector
- No VIS/random/oracle outcome is used for detector training
- Privileged state is teacher-label-only
- Student detector features are deployment-safe and causal
