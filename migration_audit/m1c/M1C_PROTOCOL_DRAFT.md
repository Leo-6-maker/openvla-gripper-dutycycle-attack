# M1C: Clean-Only Abstention Repair — Pre-Registration Draft

**Status**: DRAFT (awaiting Phase A results for freeze)
**Branch**: `feature/sc5-abstention-v2-20260622`
**Parent commit**: `9ab9f26` (M1B formal close)
**Date**: 2026-06-23

## Research Question

> Can we raise `no-corridor abstain` from 0.429 (B0) / 0.500 (D1) to ≥ 0.90
> while maintaining teacher-valid corridor recall (coverage ≥ 0.80, K10 ≥ 0.85)?

## Decision Tree (to be resolved by Phase A)

```
Phase A result = STICKY_ARM dominant
    → M1C-R: Runtime-only state machine repair
    → Implement revocable IDLE↔CANDIDATE↔ARMED hysteresis
    → Re-evaluate on frozen SC5-v1 model

Phase A result = MODEL_SELECTIVITY dominant
    → M1C-M: Train SC5-v2 with abstention head
    → Hard-negative curriculum + temporal consistency loss
    → SC5-v1 frozen as negative baseline

Phase A result = MIXED
    → M1C-RM: State machine repair first
    → Re-evaluate; retrain only for residuals
```

## Six Absolute Gates (unchanged from M1B)

| Metric | Gate |
|--------|------|
| Coverage | ≥ 0.80 |
| False-early | ≤ 0.10 |
| Post-release | ≤ 0.05 |
| K10 containment | ≥ 0.85 |
| Median anchor error | ≤ 8 |
| No-corridor abstain | ≥ 0.90 |

Plus:
- `feature_valid_rate ≥ 0.99`
- `no-corridor blind denominator ≥ 30`
- `teacher-valid blind denominator ≥ 30`

## Data Boundaries

### ALLOWED
- Clean trajectories only
- Privileged Teacher labels (frozen C16 config)
- Causal 25D proprio/action features
- No-corridor clean hard negatives
- Unsupported clean mechanisms (push, articulated)

### FORBIDDEN
- VIS / RAND / attack results of any kind
- Attack success/failure as training signal
- Hand-picked attack windows
- Online object pose as detector input
- Task success as detector input
- Threshold calibration on M1B final 30 episodes

## M1B 30-Episode Set Status

**Diagnostic development set only.** Already observed and analyzed.
Cannot serve as final blind test for M1C evaluation.
Used exclusively for Phase A diagnosis and development iteration.

## Required Blind Test Construction

New episodes with unseen `initial_state_sha256` and `trajectory_content_sha256`:
- Teacher-valid corridor: ≥ 30
- No-corridor negative: ≥ 30
- SHA-group isolation: episodes sharing initial_state_sha cannot be split across train/test
- Must include unsupported mechanisms (push, articulated) where detector should abstain

## M1C-R: State Machine Repair (if Phase A = A or C)

Current state machine:
```
IDLE → (stable_carry AND cp > 0.3) → ARMED
ARMED → (guard=5 AND cp > 0.3 AND rp < 0.3) → EMITTED
```
No path back from ARMED to IDLE.

Proposed replacement:
```
IDLE → (stable_carry AND cp > τ_on) → CANDIDATE
CANDIDATE → (evidence lost) → IDLE
CANDIDATE → (N_consecutive ≥ N_min) → ARMED
ARMED → (corridor drops OR phase≠carry OR release rises) → IDLE
ARMED → (guard=5 AND cp > τ_off AND rp < τ_r) → EMITTED
```

Free parameters to search on frozen train/val:
- `τ_on` (arm threshold, default 0.5)
- `τ_off` (keep threshold, default 0.3)
- `N_min` (consecutive evidence before arm, default 3)
- `guard` (already 5)
- `disarm_on_phase_change` (boolean)
- `disarm_on_release_rise` (boolean)

Constraints during Pareto search:
- Coverage ≥ 0.80
- K10 ≥ 0.85
- False-early ≤ 0.10
- Post-release ≤ 0.05
- Median error ≤ 8
- Maximize no-corridor abstain

**M1B 30 episodes must NOT be used for threshold selection.**

## M1C-M: Model Retraining (if Phase A = B or C)

### SC5-v2 Architecture Candidates

1. **MLP + abstention head** — same 25D→64→64 backbone, add `carry_eligible` and `no_corridor` heads
2. **Causal TCN** — past 16/32 steps of 25D sequence, capture temporal patterns
3. **Two-stage**: eligibility classifier → timing localizer

Ablation plan:
```
SC5-v1 frozen MLP (baseline)
SC5-v1 + new state machine
SC5-v2 MLP + hard negatives + abstention head
SC5-v2 causal TCN
```

### Required Hard Negatives

Training must include trajectories where:
- Gripper closes but no object grasped
- EEF rises but object does not follow
- Brief lift then immediate drop
- Contact then regrasp/recovery
- Gripper stays closed, trajectory looks like carry, but task has no stable carry
- Push / articulated / unsupported tasks

### Modified Loss

```
L = L_phase + λ1·L_corridor + λ2·L_release + λ3·L_eligibility + λ4·L_abstention + λ5·L_temporal_consistency
```

Key changes:
- Remove or reduce corridor BCE `pos_weight` (currently 5.0)
- Add episode-level hard-negative sampling
- Temporal consistency penalty for brief single-frame stable-carry predictions
- Higher penalty for no-corridor false positives

## M1C Success Criteria

```
M1C_ALL_CLEAN_ONLY_GATES_PASS = True
```

Only then may new attack experiments begin:
1. Exact-prefix causal attack matrix (6 frozen parents, VIS/RAND/SHUFFLED)
2. Cross-suite clean transfer (Object→Spatial→Goal→LIBERO-10)
3. Cross-suite matched attack pilot

## References

- M1B final classification: `migration_audit/object_checkpoint_migration/m1_runtime_b0_d1/final_classification.json`
- C16 frozen Teacher: `migration_audit/object_checkpoint_migration/m1_runtime/teacher_config_frozen.json`
- SC5 detector runtime: `src/gripper_attack/sc5_detector_runtime.py`
- Phase A diagnostic script: `scripts/migration/diagnose_false_triggers.py`
