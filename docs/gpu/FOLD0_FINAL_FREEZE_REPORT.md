# Strict Fold 0 Final Freeze Report

**Trainer commit**: `b0168a2f31723f9f4cf054bfc82e3ff9249104b0`
**Source commit**: `0280c8564773a5e6ca0482c740891d8f9eddad84`

## Fold 0 Split

| Split | Tasks | Episodes |
|-------|-------|----------|
| Train | [0,1,2,3,4,5,7,9] | 400 |
| Val (Butter) | [6] | 50 |
| Test (Chocolate Pudding) | [8] | 50 |

## V2 Checkpoints (corridor label fixed)

| Seed | SHA256 | Best Epoch | Val Phase Loss |
|------|--------|-----------|---------------|
| 1 | `98aac325...` | 0 | 1.2631 |
| 2 | `1bc9865f...` | 0 | 1.2993 |
| 3 | `066f3ce3...` | 0 | 1.1178 |

## V1 Invalidation

V1 trainer (`train_sc5_strict_fold0_v1.py`) used `lab.get("is_corridor", False)`.
Teacher labels contained no `is_corridor` field → all 86,191 corridor labels defaulted to False.
Corridor head learned ~0, producing 0 emissions on all splits.

## Functional Replay (Train + Butter)

| Seed | Train Emit/400 | Butter Emit/50 | cp P99 |
|------|---------------|---------------|--------|
| 1 | 319 | 45 | 0.932 |
| 2 | 320 | 44 | 0.906 |
| 3 | 315 | 45 | 0.926 |

## Corrected Chocolate V2 — Full Confusion Matrix

| Seed | TP | FN | FP(early) | TN | FP(no-corr) | Total Emit |
|------|----|----|-----------|-----|-------------|------------|
| 1 | 28 | 1 | 1 | 1 | 19 | 48 |
| 2 | 28 | 1 | 1 | 2 | 18 | 47 |
| 3 | 28 | 1 | 1 | 6 | 14 | 43 |

30 teacher-positive, 20 teacher-negative episodes per seed.

## Primary Metrics

| Metric | Seed 1 | Seed 2 | Seed 3 | Mean ± Std |
|--------|--------|--------|--------|------------|
| Coverage (TP/Pos) | 0.933 | 0.933 | 0.933 | 0.933 ± 0.000 |
| K10 Containment | 0.933 | 0.933 | 0.933 | 0.933 ± 0.000 |
| False-early rate | 0.033 | 0.033 | 0.033 | 0.033 ± 0.000 |
| Missed-positive rate | 0.033 | 0.033 | 0.033 | 0.033 ± 0.000 |
| No-corridor FPR | 0.950 | 0.900 | 0.700 | 0.850 ± 0.108 |
| Correct abstention | 0.050 | 0.100 | 0.300 | 0.150 ± 0.108 |

## Timing Metrics

**Definition**: `signed_error = emit_step − teacher_anchor`
(`teacher_anchor = first stable_carry step + guard=5`)

| Metric | Seed 1 | Seed 2 | Seed 3 |
|--------|--------|--------|--------|
| Median signed error | +11.0 | +11.0 | +11.0 |
| Mean signed error | +5.3 | +5.2 | +5.9 |
| Median absolute error | 11.0 | 11.0 | 11.0 |
| Mean absolute error | 15.8 | 15.9 | 15.9 |
| IQR (absolute) | 2.0 | 2.0 | 1.0 |
| Min signed | −153 | −154 | −145 |
| Max signed | +13 | +13 | +14 |

Detector emits consistently ~11 steps AFTER teacher anchor (positive delay).
IQR of only 1-2 steps indicates tight timing precision.
One outlier episode per seed (~−150) has early detection relative to teacher.

## Scientific Conclusions

### Positive-corridor timing transfer: PASS
93.3% coverage, σ=0 across 3 seeds. Detector reliably finds valid K10 corridor
windows in unseen Chocolate Pudding task. 11-step median positive delay, IQR=1-2 steps.

### No-corridor selectivity: UNRESOLVED
FPR 70-95% (mean 85%). Detector triggers on most teacher-negative episodes.
Seed 3 shows 70% FPR vs 90-95% for seeds 1-2, suggesting seed-level variance
but no seed achieves acceptable selectivity.

### Overall Fold 0 verdict: PARTIAL PASS
Timing recall is strong and stable. Selectivity requires improvement before
the detector can be used for selective attack triggering.

## Thresholds (unchanged from protocol freeze)

| Parameter | Value |
|-----------|-------|
| tau_corridor | 0.3 |
| tau_release | 0.3 |
| guard | 5 |
| K | 10 |
