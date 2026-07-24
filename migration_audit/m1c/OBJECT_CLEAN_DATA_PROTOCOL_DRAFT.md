# M1C Object Clean Data Collection Protocol — DRAFT

**Status**: FROZEN (replay complete; R1 baseline selected; hard-negative quotas refined)
**Branch**: `feature/sc5-abstention-v2-20260622`
**Date**: 2026-06-23

## Purpose

Build a new, untouched Object clean corpus for:
1. Independent validation of runtime-repair threshold selection
2. Sealed blind evaluation of final SC5-v2 (or v1R) model
3. Hard-negative training data for abstention-aware retraining

## Data Boundaries

### ALLOWED
- LIBERO Object suite only
- Clean trajectories (attack disabled)
- B0 profile primary (BF16 + eager)
- D1 reserved for secondary transfer test (NOT mixed into train)
- Full 25D telemetry with detector scores
- Privileged Teacher labels (frozen C16 config)
- Video optional

### FORBIDDEN
- VIS / RAND / attack results
- Task success as detector input
- Online object pose in detector features
- M1B 30-episode states reused for blind
- Post-hoc state-pool reassignment after seeing results
- D1 trajectories mixed into B0 training

## M1B State Exclusion

The following M1B `(task_idx, state_id)` pairs are **observed** and must NOT appear in blind:

```text
Butter:           0, 1, 2
Cream Cheese:     0, 1, 2
Salad Dressing:   0, 1, 2
BBQ Sauce:        0, 1, 2
Ketchup:          0, 1, 2
Tomato Sauce:     0, 1, 2
Chocolate Pudding:0, 1, 2
Milk:             0, 1, 2
Alphabet Soup:    0, 1, 2
Orange Juice:     0, 1, 2
```

All 10 tasks, all 3 states each = 30 states observed.

## State Pool Allocation

LIBERO Object has 10 tasks. Each task has init states beyond the 3 used in M1B.
Max init states per task varies (typically ~10-20).

### Train Pool
- Use unseen init states for all 10 tasks
- Target: teacher-valid ≥ 120, no-corridor ≥ 80
- Allow multiple rollouts per state with different seeds
- Hard-negative emphasis: states/scenarios known to produce ambiguous carry

### Validation Pool
- Separate init states from train
- Target: teacher-valid ≥ 30, no-corridor ≥ 20
- Used for threshold selection and model selection ONLY
- Can be inspected during development

### Sealed Blind Pool
- Separate init states from train AND validation
- Target: teacher-valid ≥ 30, no-corridor ≥ 30
- Opened exactly ONCE for final evaluation
- SHA-locked: `initial_state_sha256` recorded before rollout
- Membership frozen before any results are seen

## Required Hard Negatives

Training must deliberately include trajectories where the model might falsely trigger:

| Category | Description | Target Count |
|---|---|---|
| Close-no-grasp | Gripper closes but no object grasped | ≥ 20 |
| Lift-no-carry | EEF rises but object does not follow | ≥ 20 |
| Brief-lift-drop | Brief lift then immediate drop | ≥ 15 |
| Recovery/regrasp | Contact then recovery or regrasp | ≥ 15 |
| Pseudo-carry | Gripper closed, looks like carry, but no stable carry | ≥ 20 |
| Unsupported | Push/articulated tasks where corridor concept doesn't apply | TBD after H2 freeze |

## GPU Allocation

```text
GPU2: B0 clean train shard A (unseen states, emphasis on hard negatives)
GPU3: B0 clean train shard B (complementary states)
GPU4: B0 validation shard (separate state pool)
GPU6: B0 sealed blind shard ONLY (state SHA locked before rollout)
```

Each GPU starts with 2-4 cell smoke to verify:
- Checkpoint correct (SHA match)
- `attack_frames = 0`
- Telemetry complete (corridor_p, release_p, pred_phase, feat_valid present)
- Teacher fields extractable
- Disk growth controllable
- `.done` and heartbeat functional

## Protocol Freeze Checklist

- [x] State pool manifest frozen (`object_state_pool_manifest.json`)
- [x] Split membership frozen (`object_frozen_split_manifest.json`)
- [ ] Checkpoint SHA recorded (populate at collection time)
- [ ] Teacher config SHA recorded (populate at collection time)
- [x] Detector config (B0: BF16+Eager) recorded
- [x] VLA model path recorded
- [x] M1B excluded states listed (0,1,2 for all 10 tasks)
- [x] This document status: FROZEN

## Post-Replay Adjustments

Based on R0/R1/R2 replay results (`migration_audit/m1c/runtime_replay/`):

- R1 fixes 2 sticky-arm + 1 silent stall on M1B diagnostic set
- 4/6 false triggers remain as sustained model-selectivity errors
- Hard-negative emphasis: pseudo-carry, lift-no-carry, close-no-grasp categories
- R2 method family preserved for independent validation tuning

## References

- M1C Protocol: `migration_audit/m1c/M1C_PROTOCOL_DRAFT.md`
- Phase A Report: `migration_audit/m1c/phase_a/PHASE_A_REPORT.md`
- M1B Manifest: `migration_audit/object_checkpoint_migration/m1_runtime_b0_d1/artifact_manifest_complete.json`
- C16 Teacher: `migration_audit/object_checkpoint_migration/m1_runtime/teacher_config_frozen.json`
