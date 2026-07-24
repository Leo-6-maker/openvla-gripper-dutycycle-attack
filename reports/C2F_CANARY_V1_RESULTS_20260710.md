# C2f Table1-Candidate Online Canary v1 Results — 2026-07-10

**Run**: `table1_candidate_gpu17_20260709_235106`
**Commits**: `f3c9fc0` (latest fix), `1c181f8` (Goal unnorm_key)
**Matrix**: 48 parents × 3 conditions = 144 episodes
**Detector**: D Full OpenVLA-SigLIP, default gate (τ_emit=0.33, τ_suppress=0.67, τ_abstain=0.5, τ_primary=0.5)

## 1. Results Summary

| Suite/Cond | n | emit | deliver | mean_dc | succ |
|---|---|---|---|---|---|
| **Spatial** CLEAN | 12 | — | — | — | 11/12 (92%) |
| **Spatial** TRUE | 12 | 12 (100%) | 12 | 10.0 | 11/12 (92%) |
| **Spatial** RAND | 12 | 12 (100%) | 12 | 10.0 | 10/12 (83%) |
| **Object** CLEAN | 12 | — | — | — | 10/12 (83%) |
| **Object** TRUE | 12 | 6 (50%) | 6 | 4.3 | **6/12 (50%)** |
| **Object** RAND | 12 | 6 (50%) | 6 | 5.0 | **10/12 (83%)** |
| **L10** CLEAN | 12 | — | — | — | 5/12 (42%) |
| **L10** TRUE | 12 | 5 (42%) | 5 | 4.2 | 4/12 (33%) |
| **L10** RAND | 12 | 5 (42%) | 5 | 4.2 | 4/12 (33%) |
| **Goal** CLEAN | 12 | — | — | — | 0/12 (0%) |
| **Goal** TRUE | 12 | 4 (33%) | 4 | 3.3 | 0/12 (0%) |
| **Goal** RAND | 12 | 4 (33%) | 4 | 3.3 | 0/12 (0%) |

## 2. TRUE vs RAND Gap

| Suite | CLEAN SR | TRUE SR | RAND SR | Gap |
|---|---|---|---|---|
| **Object** | 83% | **50%** | 83% | **-33pp** |
| Spatial | 92% | 92% | 83% | +8pp |
| L10 | 42% | 33% | 33% | 0pp |
| Goal | 0% | 0% | 0% | 0pp |

## 3. Detector Behavior

- **Spatial**: 100% emit rate, full 10-step delivery. Detector fires reliably on all Spatial parents.
- **Object**: 50% emit rate. TRUE_T10 success (50%) significantly lower than RAND_T10 (83%) — **-33pp gap**.
- **L10**: 42% emit rate, no TRUE/RAND gap. Low base SR from multi-object tasks.
- **Goal**: 33% emit rate, 0% success — using libero-10 substitute model (see caveats).

## 4. Bugs Fixed During Canary

| Bug | Symptom | Fix |
|---|---|---|
| RGB float→uint8 | All emit_p ≈ 0 | `c116b01`: dtype conversion |
| Attack injection API | TypeError on `epsilon` param | `7c02187`: direct action manipulation |
| Metadata before env.close | EGL crash lost data | `172b78d`: write outputs first |
| Episode loop EGL crash | env.step() crash skipped metadata | `f3c9fc0`: try/except around loop |
| Goal unnorm_key | `libero_goal` not in norm stats | `1c181f8`: use `libero_10` key |

## 5. Goal Model Status

- **Original model**: `/mnt/sdc/dty_user/table1_dependencies/openvla-7b-finetuned-libero-goal` — **deleted**
- **Local backup**: Missing `model-00001-of-00004.safetensors` (shard 1)
- **Current substitute**: libero-10 model copy (same SigLIP, different policy weights)
- **Result**: Goal episodes complete but 0% success rate with substitute model
- **Downloaded**: Full model from HuggingFace (`openvla/openvla-7b-finetuned-libero-goal`), uploading to server now

## 6. Caveats

- **Goal CLEAN SR = 0%**: Using libero-10 substitute model. Scheduled re-run with real model.
- **Goal model lost**: Need to upload downloaded backup and re-run Goal suite.
- **Action injection**: Simplified TRUE_T10 (force gripper open) vs D7 PGD. Not exact protocol match.
- **L10 low emit**: Multi-object L10 tasks (task_00, task_01, task_06, task_07) have zero primary labels — detector correctly ignores them.
- **Spatial FP**: 100% emit rate with 10-step delivery may indicate over-emission (label primary rate = 74.8%).

## 7. Gate Status

```
C2F_CANARY_V1_144EP             = COMPLETE (3/4 suites healthy)
C2F_OBJECT_TRUE_RAND_GAP        = -33pp (TRUE 50% << RAND 83%)
C2F_SPATIAL_EMIT                = 100% emit, 10-step delivery, no TRUE/RAND gap
C2F_L10_EMIT                    = 42% emit, no TRUE/RAND gap (label-compatible)
C2F_GOAL                        = PENDING_REAL_MODEL_RERUN
C2F_GOAL_MODEL                  = DOWNLOADED_LOCALLY, UPLOADING
D7_TABLE1                       = FROZEN
```

## 8. Next Steps

1. Upload real Goal model → re-run Goal 36 episodes
2. McNemar paired test on Object TRUE vs RAND
3. Spatial FP audit (is 100% emit correct or over-emission?)
4. Write final C2f Table1-candidate report
5. Decision: upgrade to secondary Table1 or hold for fixes
