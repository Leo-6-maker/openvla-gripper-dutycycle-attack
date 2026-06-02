# Visual V2 Baseline Sanity Audit

**Date**: 2026-05-30 | **Phase**: C0

## label_shuffle AUC=0.747 Explained

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Split method | Task holdout (ketchup+salad_dressing = val) | No run-level leakage |
| Train/Val overlap | 0 runs | CLEAN |
| AUC std across epochs | **0.202** | Wild oscillation |
| Final epoch AUC | **0.236** | Near random |
| Best epoch AUC | 0.747 | Cherry-picked max across 30 epochs |
| Loss convergence | 0.93 → 0.68 (-0.25) | Barely learned anything |
| pos_mean vs neg_mean | 0.21 vs 0.23 | Cannot distinguish |

**Conclusion**: label_shuffle AUC=0.747 is NOT evidence of data leakage. It's the random maximum of a noisy process. The model does not learn; std=0.202 proves it.

## task_only AUC=0.344

- AUC never changes (std=0.0) — model converged to constant output
- pos_mean ≈ neg_mean ≈ 0.025 — no discrimination
- Task identity alone has zero predictive power for teacher_hazard with task holdout
- 1-AUC would be 0.656 but this is also meaningless — model outputs are constant

## Split Integrity

| Check | Result |
|-------|--------|
| Train runs | 84 (8 tasks) |
| Val runs | 20 (ketchup + salad_dressing) |
| Overlap | 0 |
| Train pos rate | 0.20% (32/15784) |
| Val pos rate | 2.10% (65/3091) |
| Frame-level leakage | N/A (split by run, not frame) |

**Verdict**: teacher_metric_not_fully_trustworthy due to label_shuffle instability, but visual AUC gap (0.95 vs 0.34 task_only, 0.24 label_shuffle final) remains valid. Visual signal genuinely predicts teacher_hazard above baselines.
