# Experiment Status Report — 2026-07-09 11:30 CST

**Report commit**: `8c0bfa3` | **Branch**: `plan/codex-gated-experiment-v1-c2e0`

## 1. D7 Table1 — Main Experiment (FROZEN)

| Gate | Status |
|---|---|
| D7B2 Rollout | 716/716 |
| D7C Postrun Audit | PASS (0 missing, 0 violations) |
| D7D Aggregate | Panel A + O/G/S pooled |
| D7E Render | Markdown |
| Paired McNemar | O/G/S TRUE vs RAND p<0.001 |

**O/G/S Pooled (N=129)**: CLEAN 89.1%, TRUE_T10 57.4%, RAND_T10 88.4%, CMD_OPEN 71.3%

Status: `D7_TABLE1_MAIN_RESULT = FROZEN`

## 2. C2f Clean2000 — Collection Complete

| Metric | Value |
|---|---|
| Episodes | 2000 (500 × 4 suites) |
| Steps | 393,513 |
| RGB frames | 393,513 (perfect parity) |
| features_25d | 393,513/393,513 OK |
| task_language | 0 empty |
| Stats windows | 363,513 |

## 3. TeacherLabeler Evolution

| Version | Object primary | Mechanism |
|---|---|---|
| v0 | **0.0%** | `eef_z > 0.85` (absolute) |
| v1.1 | **12.6%** (9/10 tasks) | `rel_lift >= 0.03` (relative) |

Root cause: Object eef_z range 0.01–0.35, absolute threshold never triggers.
v1.1 fix: track `close_start_eef_z`, `closed_streak`, `max_eef_z_since_close`.

Remaining: task_01 (0.0%) — object-language alias matching, not phase threshold.

## 4. Clean2000 v1.1-caveat Label Distribution

| Suite | Primary Rate | Hazard Rate | Label Version |
|---|---|---|---|
| Spatial | 67.4% | 67.4% | v0 |
| Goal | 45.8% | 45.8% | v0 |
| L10 | 31.6% | 31.6% | v0 |
| Object | **12.6%** | 12.6% | **v1.1** |

Object per-task: 9/10 tasks non-zero (0.8%–32.0%), task_01=0% (alias matching caveat).

## 5. C2f Detector Training Results

### A (25D + 108D context) — v1.1-caveat

| Suite | Recall | FP | F1 |
|---|---|---|---|
| L10 | **93.7%** | 3.6% | 0.940 |
| Goal | 95.1% | 6.7% | 0.949 |
| Object | **84.7%** | 4.5% | 0.751 |
| Spatial | 97.2% | 28.6% | 0.939 |
| **Overall** | **94.9%** | **6.9%** | **0.930** |

### A0 (25D only, no context) — v1.1-caveat

Identical to A (context features are zero vectors in stats backend).

### Threshold Sweep (best_f1)

τ_emit=0.25, τ_suppress=0.50:
- Recall=93.4%, FP=5.5%, L10_recall=97.5%, macro_FP=10.2%

### v0 → v1.1 Key Metric Evolution

| Metric | v0 (Object=0) | v1.1-caveat |
|---|---|---|
| Object recall | 0.0% | **84.7%** |
| Overall recall | 97.7% | 94.9% |
| Overall FP | 6.6% | 6.9% |
| L10 recall | 96.7% | 93.7% |
| Macro recall | 73.2% | **92.7%** |
| Macro FP | 12.8% | 10.9% |

Object v1.1 fix resolves macro recall from 73% → 93%. Four-suite detector now viable.

## 6. C2f Pipeline Gates

| Gate | Status |
|---|---|
| C2F_CLEAN2000_INPUT_CHAIN | PASS |
| C2F_V1_1_LABELS | PASS (9/10 Object tasks, task_01 caveat) |
| C2F_V1_1_STATS_ABLATION | PASS |
| C2F_V1_1_THRESHOLD_SWEEP | PASS (best_f1: FP=5.5%, L10=97.5%) |
| C2F_VISUAL_LANGUAGE_GAIN | NOT_TESTED (stats backend, visual/lang=zero) |
| C2F_FINAL_OFFLINE_GATE | PASS (all suites non-zero, FP<30%, L10>C2e3 baseline) |
| CLIP/SigLIP backend | BLOCKED (no internet) |

## 7. Artifacts

| Artifact | Path | SHA256 |
|---|---|---|
| Clean2000 merged v0 | `.../clean2000_merged_199af7b/` | — |
| Clean2000 v1.1-caveat | `.../clean2000_v1.1_caveat/` | — |
| Stats v1.1 NPZ | `.../clean2000_v1.1_emb_stats/c2f_w16_stats_dataset.npz` | `d5651a6a` |
| A checkpoint | `.../c2f_v1.1_runs/A_25d_context/c2f_rgb_lang_temporal_detector_v0.pt` | — |
| A0 checkpoint | `.../c2f_v1.1_runs/A0_25d_no_context/` | — |

## 8. Next Steps

```
P0: task_01 diagnostic + alias fix
P0: OpenVLA-SigLIP backend (vision features)
P1: Clean2000-v1.2 (task_01 fixed)
P1: Real visual-language A/B/C/D ablation
P2: Online C2f canary (D8 protocol, same parents as D7)
```

## 9. Boundaries

- D7 Table1 frozen — C2f does not modify
- Student input excludes privileged state
- CLIP blocked → OpenVLA vision_backbone (SigLIP) as alternative
- Object task_01=0% is alias matching, not phase threshold
