# Layer 3 D5-Triggered VIS-Temporal v1 Audit Addendum

**Date:** 2026-06-18
**Auditor:** v3
**Supersedes:** initial auditor v1 (a7ade62), auditor v2 (1b784c4)
**Freeze document:** `reports/L3_D5_VIS_TEMPORAL_V1_FREEZE_20260618.md` (a05a8a7) — UNCHANGED
**v1 branch:** `exp/l3-d5-vis-temporal-20260617` — FROZEN

## Data Status

**10/10 episodes present.** Both SHUFFLED controls were completed after the initial auditor run.

| Seed | CLEAN_D5 | TRUE_SINGLE | TRUE_T10 | RAND_T10 | SHUFFLED_T10 |
|------|----------|-------------|----------|----------|--------------|
| 81   | 161 steps | 157 steps | 170 steps | 163 steps | 159 steps |
| 82   | 161 steps | 163 steps | 186 steps | 150 steps | 159 steps |

All episodes: task_success=true.

## Gate Results (Auditor v3)

| Gate | seed81 | seed82 | Description |
|------|--------|--------|-------------|
| G0  | PASS | PASS | Data 10/10, D5 emit=60, exact step arrays verified |
| G1  | PASS | PASS | Semantic OPEN token duty = 1.0, selectivity >= 0.50 |
| G2  | **FAIL** | **FAIL** | Arm selectivity: arm_d=0.1, min_arm=0 (seed81); arm_d=0, min_arm=0 (seed82) |
| G3  | PASS | PASS | Env OPEN command duty = 1.0, selectivity >= 0.50 |
| G4  | PASS | PASS | Paired TRUE-CLEAN delta: pd=+0.0051/+0.0062, auc=+0.137/+0.163, lat=1, sus=20 |
| G5A | NOT_AUDITABLE | NOT_AUDITABLE | No object telemetry in runner |
| G5B | FAIL | FAIL | All episodes success=true |
| SPEC | PASS | PASS | RAND/SHUFFLED env_open <= 3 |

## Classification

```
L3_D5_VIS_TEMPORAL_SEMANTIC_COMMAND_DUTY_PASS
PHYSICAL_RESPONSE_OBSERVED
ARM_SELECTIVITY_NOT_ESTABLISHED
```

### What is established

1. **D5 timing proxy**: first emit at step 60 on Butter_s11, verified across all 10 episodes.
2. **Semantic OPEN duty control**: TRUE temporal K=10 produces 10/10 OPEN token 31744, vs RAND 0/10, SHUFFLED 0/10.
3. **Env OPEN command duty control**: 10/10 env=-1 OPEN commands, vs controls 0/10.
4. **Physical response observed**: Paired TRUE-CLEAN peak_delta = +0.0051 (seed81) and +0.0062 (seed82). AUC increase = +0.137/+0.163. Latency from emit+1 = 1 step.
5. **Recovery delay observed**: TRUE_T10 episodes take 170/186 steps vs CLEAN 161 steps. Policy recovers and completes the task.
6. **Prev_delta warm start verified**: First frame flag=False, frames 2-10 flag=True for both seeds.
7. **Temporal AUC > SINGLE**: seed81 TRUE_T10 AUC=0.182 > SINGLE AUC=0.171; seed82 TRUE_T10 AUC=0.208 > SINGLE AUC=0.120. Confirms temporal accumulation of physical opening.

### What is NOT established

1. **Arm selectivity**: TRUE_T10 seed81 arm_d=0.1 (only frame 60 has 5/6 arm match, subsequent frames drop to 0-2/6). seed82 arm_d=0.0 (max match = 4/6 on frame 60, then 0-1/6). The temporal attack with prev_delta warm start preserves the gripper OPEN direction but causes significant arm token drift.
2. **Contact failure**: No object telemetry. G5A is NOT_AUDITABLE.
3. **Task failure**: All episodes success=true. TRUE_T10 episodes are longer but the policy recovers.
4. **Strict paired physical bridge**: While TRUE-CLEAN paired delta is positive and AUC exceeds controls, the arm token drift means the observed qpos response may include arm action contamination — not purely a "selective gripper physical bridge."

### Arm Selectivity Detail

```
seed81 TRUE_T10 adv_arm values: [5, 1, 1, 1, 1, 2, 1, 1, 1, 0]
  Frame 60 (prev_delta=False): arm=5/6  ←  OK
  Frames 61-69 (prev_delta=True): arm=0-2/6  ←  COLLAPSE

seed82 TRUE_T10 adv_arm values: [4, 0, 1, 1, 1, 1, 1, 1, 0, 1]
  Frame 60 (prev_delta=False): arm=4/6  ←  marginal
  Frames 61-69 (prev_delta=True): arm=0-1/6  ←  COLLAPSE
```

For reference, TRUE_SINGLE (single frame, no temporal):
- seed81: adv_arm=5/6 (passes >=5)
- seed82: adv_arm=4/6 (marginal)

The prev_delta warm start successfully maintains the gripper OPEN token direction but the accumulated perturbation causes progressive arm token drift.

### Physical Metric Correction

v2 auditor used pre-action qpos at step 60 as baseline reference, producing latency=0 artifacts. v3 measures from emit+1 (first step where attack action can physically take effect in the environment):

- **latency=1**: qpos responds at emit+1 (step 61) for TRUE conditions
- **sustain=20**: qpos stays below baseline for the full 20-step observation window
- **Paired TRUE-CLEAN delta**: removes natural post-grasp qpos drift from the signal

CLEAN_D5, RAND, and SHUFFLED all show peak_delta=0.007641 — this is the natural gripper opening after grasp establishment, not an attack effect. The TRUE-CLEAN paired delta isolates the VIS-caused additional opening.

## Implications for v2

1. **Arm selectivity must be a v2 gate** alongside gripper control. The prev_delta mechanism preserves gripper OPEN but costs arm preservation. v2 should investigate whether a higher arm_preserve_weight or different temporal_init can maintain both.

2. **D5 emit=60 physical phase must be empirically determined** from clean artifact-rich trajectories. Based on D5 features (raw_crossing at close event), it likely falls near grasp_close/stable_grasp boundary — but this is design inference, not measurement.

3. **v2 teacher must label physical phases from privileged clean state**, then command-hold experiments must verify that later phases (stable_carry, pre_place_unsupported) show greater task consequences than contact-onset phase.

## Versioning

- This addendum does NOT modify the v1 freeze document.
- v1 branch remains immutable at a05a8a7.
- Auditor v3 replaces v1/v2 for gate computation only.
- v1 empirical boundary in the freeze document stands.
