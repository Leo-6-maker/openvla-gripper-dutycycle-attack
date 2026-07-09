# Experiment Status Report — 2026-07-09 10:30 CST

**Commit**: `fd3e2db` | **Branch**: `plan/codex-gated-experiment-v1-c2e0`

## 1. D7 Table1 — Main Experiment (FROZEN)

| Gate | Status |
|---|---|
| D7B2 Rollout | 716/716 |
| D7C Postrun Audit | PASS (0 missing, 0 violations) |
| D7D Aggregate | Panel A + O/G/S pooled |
| D7E Render | Markdown |
| Paired McNemar | O/G/S TRUE vs RAND p<0.001 |

**O/G/S Pooled**: CLEAN 89.1%, TRUE_T10 57.4%, RAND_T10 88.4%

## 2. C2f Clean2000 — Collection Complete

| Metric | Value |
|---|---|
| Episodes | 2000 (500 × 4 suites) |
| Steps | 393,513 |
| RGB frames | 393,513 (perfect parity) |
| features_25d | 393,513/393,513 OK |
| task_language | 0 empty |
| Stats materialization | 363,513 windows (37s, 30 MB) |

### Clean2000 Label Distribution (v0)

| Suite | Primary Rate | Hazard Rate |
|---|---|---|
| Spatial | 67.4% | 67.4% |
| Goal | 45.8% | 45.8% |
| L10 | 31.6% | 31.6% |
| **Object** | **0.0%** | **0.0%** |

Object primary=0 — root cause: `eef_z > 0.85` absolute threshold.
Object eef_z range: 0.01–0.35 (P99=0.332). Threshold never triggers.

## 3. TeacherLabeler v1.1 Fix

| Version | stable_carry rule | Object primary |
|---|---|---|
| v0 | `eef_z > 0.85` (absolute) | 0% |
| v1.1 | `rel_lift >= 0.03 OR eef_z > 0.85` (relative-lift + fallback) | **26.6%** (smoke validated) |

Commit: `fd3e2db`. Track `close_start_eef_z`, `closed_streak`, `max_eef_z_since_close`.

## 4. C2f v0 Ablation (Stats Backend, Zero Visual/Language)

| Variant | Recall | FP | F1 | Macro Rec |
|---|---|---|---|---|
| A (25D only) | 97.7% | 6.6% | 0.942 | 73.2% |
| B (25D+lang) | 97.7% | 6.6% | 0.942 | 73.2% |
| C (25D+RGB) | 97.7% | 6.6% | 0.942 | 73.2% |
| D (full) | 97.7% | 6.6% | 0.942 | 73.2% |

A/B/C/D identical — visual/language features are zero vectors (stats backend).
**Key finding**: 25D + Clean2000 teacher labels → L10 recall 96.7% (C2e3: 45.6%).

### Per-Suite (A variant):

| Suite | Recall | FP | F1 |
|---|---|---|---|
| L10 | 96.7% | 4.4% | 0.950 |
| Goal | 97.0% | 13.9% | 0.933 |
| Spatial | 99.0% | 32.9% | 0.942 |
| Object | 0.0% | 0.0% | 0.000 |

## 5. Object 500 v1.1 Rerun (IN PROGRESS)

- 8 workers across GPUs 0-3,5,7 (GPUs 4,6: other training)
- ~62 episodes/worker, 500 total
- ETA: ~2 hours

## 6. CLIP — Blocked

Server has no internet. No cached CLIP model.
OpenVLA's `vision_backbone` (SigLIP) available as alternative.

## 7. Next Steps

```
Object 500 v1.1 complete →
  merge into Clean2000-v1.1 (Spatial/Goal/L10 v0 + Object v1.1) →
  re-materialize (stats) →
  CLIP/OpenVLA vision materialization →
  real A/B/C/D ablation with visual-language features
```

## 8. Boundaries

- D7 Table1 frozen — C2f does not modify
- Student input excludes privileged state
- CLIP blocked → need offline model download or OpenVLA vision encoder
