# D1b — DeepSeek Learned Detector Training Preregistration

**Date**: 2026-06-15
**Stage**: D1b (prereg only)
**Branch**: `exp/l12-critical-close-window-selector-20260615`
**Parent commit**: `2450031` (E4C.2b accepted)
**TRAINING_STARTED: NO**
**GPU_TRAINING: NOT AUTHORIZED UNTIL D1b AUDIT PASSES**

---

## Data Source

131 ELIGIBLE_MULTI_CANDIDATE traces from E4C.2b frozen label pool.
Each trace has exactly 1 Teacher-P-qualifying candidate and >=1 other
CLOSE candidates.

| Split | Traces | Candidates | Positives | Negatives |
|-------|--------|-----------|-----------|-----------|
| Train | 90 | 561 | 90 | 471 |
| Validation | 20 | 98 | 20 | 78 |
| Test | 21 | 101 | 21 | 80 |
| **Total** | **131** | **760** | **131** | **629** |

All 10 tasks represented in each split. Minimum per-task: train >=4,
val >=1, test >=1.

## Label Rules

- **Positive**: unique `is_teacher_p=1` candidate per trace
- **Negative**: all other CLOSE candidates in the same trace
- **Excluded from training**: ambiguous (131), unavailable (79),
  no-candidate (8), single-candidate (53)
- **Forbidden**: attack outcomes, random-sensitive, Teacher-P unavailable
  as negative, Bronze labels

## Leakage Audit

| Check | Result |
|-------|--------|
| Same trace_id across splits | 0 violations |
| Same logical_group_id across splits | 0 violations |
| Same task+state across splits | 0 violations |
| SHA duplicate across splits | 0 violations |

## Model

- Architecture: MLP [128, 64, 32] with ReLU, dropout 0.1
- Input: 16 frozen E4B.3 features
- Output: scalar score per candidate
- Loss: pairwise margin ranking (per-trace: score(pos) > score(neg) + 0.5)
- Optimizer: AdamW (lr=0.001, weight_decay=1e-4)
- Batch: per-trace grouped (all candidates from one trace)
- Max epochs: 200, early stop patience: 30
- Seed: 42

## Normalization

- Method: z-score (mean, stdev from train split only)
- Missing values: impute train mean
- Outliers: clip at 3 stdev
- Manifest: `tables/deepseek_detector/d1b_feature_normalization.csv`

## Checkpoint Selection (preregistered)

1. Highest validation per-trace Teacher-P top-1 accuracy
2. Tiebreak: lower validation mean absolute timing error
3. Tiebreak: earlier epoch

## Threshold Selection

On validation set only. Maximize per-trace Teacher-P top-1 accuracy.

## Test Evaluation

Single evaluation after checkpoint and threshold frozen. No retraining.

## Baseline

Rule-based scorer (`total_score` from `rule_based_close_predictor()`)
recomputed on the identical 20-val / 21-test splits.

## Primary Metrics (trace-level denominator)

- Teacher-P top-1 accuracy
- Competition top-2 accuracy
- Mean absolute timing error
- Median absolute timing error
- Decision coverage
- Conditional accuracy (emitted only)
- Per-task metrics

## Frozen Artifacts

| File | SHA256 |
|------|--------|
| `tables/deepseek_detector/d1b_training_manifest.csv` | `acc3ecb9...` |
| `tables/deepseek_detector/d1b_split_manifest.csv` | `7a61ffa7...` |
| `tables/deepseek_detector/d1b_split_summary.csv` | `bce20a91...` |
| `tables/deepseek_detector/d1b_leakage_audit.csv` | `9ef7d8ee...` |
| `tables/deepseek_detector/d1b_feature_normalization.csv` | `1ea34404...` |
| `configs/d1b_detector_training.yaml` | (to be computed) |
| `reports/D1B_CHECKPOINT_SELECTION_RULE.md` | (committed) |
| `reports/D1B_BASELINE_PROTOCOL.md` | (committed) |

## Stop Rule

```
TRAINING_STARTED: NO
GPU_TRAINING: NOT AUTHORIZED
NEXT: AWAIT D1b AUDIT
```
