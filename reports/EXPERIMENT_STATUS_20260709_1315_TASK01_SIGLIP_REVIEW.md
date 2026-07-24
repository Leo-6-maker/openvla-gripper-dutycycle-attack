# Experiment Status — 2026-07-09 13:15 CST — Task01 / SigLIP Review

**Supersedes**: `reports/EXPERIMENT_STATUS_20260709_1130.md` (commit `157fcbe`)
**Superseded sections**: Object label distribution, C2F_V1_1_LABELS gate

## 1. Object label distribution — corrected

### 1.1 Two datasets, different statistics

| Source | Episodes | task_01 primary | Overall Object primary |
|---|---|---|---|
| Object500 v1.1 source (`object500_v1.1_fd3e2db`) | 500 | **0.0%** | **70.8%** |
| Clean2000 v1.1 caveat merged | 2000 (500×4) | **53.8%** | **60.0%** |

### 1.2 Per-task (Clean2000 v1.1 caveat merged, step-level)

| Task | stable_carry | primary | primary/sc_rate |
|---|---|---|---|
| task_00 | 17397 | 4987 | 28.7% |
| task_01 | 12244 | 6584 | **53.8%** |
| task_02 | 30368 | 17777 | 58.5% |
| task_03 | 31974 | 17054 | 53.3% |
| task_04 | 16808 | 11754 | 69.9% |
| task_05 | 31902 | 22835 | 71.6% |
| task_06 | 16834 | 10757 | 63.9% |
| task_07 | 15616 | 10363 | 66.4% |
| task_08 | 29325 | 18167 | 62.0% |
| task_09 | 36640 | 23174 | 63.2% |
| **Overall** | **239108** | **143452** | **60.0%** |

**All 10 Object tasks have non-zero primary in the merged dataset.**

### 1.3 Object500 v1.1 source — task_01 matching bug confirmed

| Task | stable_carry | primary | rate |
|---|---|---|---|
| task_01 | 953 | **0** | **0.0%** |

Root cause: MuJoCo body naming uses `_main` suffix (e.g. `cream_cheese_1_main`).
The v1.1 code's `_identify_grasped_object` sometimes fails to return the correct body,
and the single-direction substring match in `_object_matches_task_target` does not
cover all edge cases.

**Conclusion: task_01=0% is real in the source run, but the merged dataset**
**regenerated labels that bypassed the bug.**

### 1.4 Denominator standardization

The 157fcbe report's `Object primary=12.6%` used an unspecified denominator and
is NOT comparable to the step-level rates above.

All future reports MUST report:

| Metric | Formula |
|---|---|
| primary_all_steps | primary_steps / total_steps |
| primary_given_stable_carry | primary_steps / stable_carry_steps |
| primary_window_rate | windows_with_primary / total_windows |

This report uses `primary_given_stable_carry` (step-level) above.
Window-level rates require NPZ-level audit (see Section 6).

## 2. TeacherLabeler v1.2

**Commit**: `a9b50d4`

Changes:
- Relaxed body filter: only exclude `robot/floor/world/gripper` prefixes
  (previously excluded `link/collision/visual/geom` substrings)
- `_canonical_body_key()`: strip MuJoCo mesh suffixes → canonical object name
- Three-direction matching: object→language, language→object, target_object_name fallback
- `target_object_name` plumbed from episode_cfg through to _TeacherLabeler

Status: **PASS_CODE_REVIEW_PENDING_LABEL_AUDIT**

Risk: relaxed filter may increase false-positive matches on basket/container/fixture
bodies. Needs per-task audit of distractor_or_setup and unsupported rates before
final gate.

## 3. OpenVLA-SigLIP backend

**Commits**: `935bf8c`, `bcff89c`, `7a78f19`, `d0ef917`

### 3.1 Architecture
- Vision: OpenVLA `vision_backbone` (SigLIP ViT-SO400M), pooler output → 2176-dim
- Language: OpenVLA Llama embedding layer, tokenize → embed → mean pool → 4096-dim
- Storage: float16 for visual/language embeddings (reduces NPZ ~10GB→~1GB compressed)
- Path: `rglob("episode_metadata.json")` for deeply nested shard paths
- Boundaries: no attack, no D7B2 outcome, no privileged state, no post-attack hidden state

### 3.2 Smoke test (1 episode)

| Metric | Value |
|---|---|
| n_windows | 285 |
| visual_dim | 2176 |
| language_dim | 4096 |
| runtime | 30s |
| NPZ size | 1.2 MB |
| status | PASS |

### 3.3 Status

**C2F_OPENVLA_SIGLIP_BACKEND = PASS_SMOKE_1EP**
Scaling tests (50/200 episode) pending before full materialization.

## 4. C2f detector training

### 4.1 A (25D + context) on v1.1-caveat stats backend — unchanged

These results from 157fcbe remain valid (stats backend unaffected by visual/language changes):

| Suite | Recall | FP | F1 |
|---|---|---|---|
| L10 | 93.7% | 3.6% | 0.940 |
| Goal | 95.1% | 6.7% | 0.949 |
| Object | 84.7% | 4.5% | 0.751 |
| Spatial | 97.2% | 28.6% | 0.939 |
| **Overall** | **94.9%** | **6.9%** | **0.930** |

### 4.2 Training script — SigLIP dimension compatibility

The detector reads `nv`, `nl` from NPZ shape, so 2176/4096 inputs will work.
Training script will expand float16→float32 in RAM (~10 GB decompressed).
Recommended batch size: 64–128 for initial SigLIP training.

## 5. Gate status (revised)

```
D7_TABLE1                           = FROZEN_MAIN_RESULT
C2F_CLEAN2000_INPUT_CHAIN           = PASS

C2F_V1_1_CAVEAT_REPORT_157FCBE      = SUPERSEDED_FOR_OBJECT_LABEL_DISTRIBUTION
C2F_OBJECT_LABEL_AUDIT              = SOURCE_OBJECT500_V1_1_TASK01_MATCHING_BUG_CONFIRMED
                                      MERGED_CLEAN2000_V1_1_ALL_10_OBJECT_TASKS_HEALTHY
                                      DENOMINATOR_NEEDS_STANDARDIZATION
C2F_TEACHER_V1_2                    = PASS_CODE_REVIEW_PENDING_LABEL_AUDIT
C2F_OPENVLA_SIGLIP_BACKEND          = PASS_SMOKE_1EP
C2F_VISUAL_LANGUAGE_GAIN            = NOT_YET_TESTED
```

## 6. Next steps (ordered)

```
P0: NPZ-level label audit (per-suite, per-task, per-split window rates)
P0: SigLIP 50-episode materialization + RAM profiling
P0: SigLIP 200-episode materialization + training smoke (batch=64)
P1: v1.2 label audit (relaxed matching FP check)
P1: Full SigLIP materialization (363K windows)
P2: Real A/B/C/D ablation with SigLIP features
```

## 7. Boundaries

- D7 Table1 frozen — C2f does not modify
- Student input excludes privileged state
- Object500 v1.1 source: task_01 matching bug confirmed
- Clean2000 v1.1 caveat merged: all 10 Object tasks healthy
- Report 157fcbe superseded for Object label distribution
- OpenVLA-SigLIP backend = victim-aligned (same vision encoder as policy)
- Text encoding = Llama embedding layer only (no full 7B forward pass)
