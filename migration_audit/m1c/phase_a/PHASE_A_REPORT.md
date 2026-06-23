# M1C Phase A: Sticky-Arm vs Model-Selectivity Diagnosis

## Gate

```text
M1C_PHASE_A = PASS (17/17 cells, 0 errors, 0 verification warnings)
A800_HOST  = pm-364c0001
EXECUTED   = 2026-06-23
```

## Result

| Group | Count | Summary |
|---|---|---|
| FALSE_TRIGGER | 6 | 2 A_STICKY_ARM, 4 B_SUSTAINED_MODEL_MISCLASSIFICATION |
| TRUE_POSITIVE | 6 | All 0 evidence breaks (teacher-valid, expected) |
| CORRECT_ABSTAIN | 5 | 4 never armed, 1 armed at step 94 but never emitted (303-step stall) |

### False Trigger Detail

| Episode | Profile | Emit | Arm | A→E | Evidence Breaks | Phase@Emit | Verdict |
|---|---|---|---|---|---|---|---|
| butter_s1 | B0 | 112 | 105 | 7 | 4 | stable_carry | **A_STICKY_ARM** |
| chocolate_pudding_s1 | B0 | 42 | 37 | 5 | 4 | pre_place_unsupported | **A_STICKY_ARM** |
| cream_cheese_s0 | B0 | 149 | 144 | 5 | 0 | stable_carry | B_MODEL |
| butter_s2 | B0 | 33 | 28 | 5 | 0 | stable_carry | B_MODEL |
| butter_s1 | D1 | 92 | 87 | 5 | 0 | stable_carry | B_MODEL |
| bbq_sauce_s2 | D1 | 210 | 205 | 5 | 0 | stable_carry | B_MODEL |

### Hidden Finding

`orange_juice_s2/B0`: ARM at step 94, evidence broke on 303 subsequent steps, but
state machine never disarmed. Episode ended at step 400 without emitting. This is
a **silent liveness failure** — the current state machine has no disarm path, so
episodes can remain permanently ARMED after a transient false-positive arm.

## Classification

```text
Strict majority: B_MODEL > n/2 (4 > 3)
Primary failure mode: MODEL_SELECTIVITY_FAILURE
Sticky-arm rate: 0.33 (2/6)
True-positive sticky rate: 0.00 (0/6) — normal, evidence sustained
Correct-abstain ever-armed: 1/5 — silent liveness bug
```

The quantitative classification is MODEL_SELECTIVITY_FAILURE, but the clinical
picture is **mixed**: 2 of 6 false triggers would be eliminated by a revocable
state machine; 1 additional correctly-abstaining episode harbours a silent ARM
stall. The correct route is **M1C-RM**: repair state machine first, re-evaluate,
then retrain SC5-v2 only for residual model-selectivity errors.

## Selected Route

```text
M1C-RM: Runtime repair → Re-evaluate → Model retrain if needed
```

## Reference Artifacts

- Script: `scripts/migration/diagnose_false_triggers.py`
- M1C commit: `8827373` (empty-CSV fix)
- M1B close commit: `9ab9f26`
- Output root (A800): `/mnt/sdc/dty_user/openvla_attack/evidence/m1c/phase_a/`
