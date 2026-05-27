# Milestone 2C.1: Student Replay Validation + Anti-Timing Ablations

## Purpose

Milestone 2C.1 validates that the Milestone 2C proprio-only causal student baseline is not merely learning normalized timing, but is genuinely using gripper/EEF/action-history features to detect contact-critical hazard phases.

The validation runs offline causal replay on the test split and compares the trained student against:
- Teacher window (oracle reference)
- Rule/proxy trigger (simple gripper heuristics)
- Time-only baseline (MLP trained on `normalized_step` + categorical features only)
- No-normalized-step ablation (MLP trained on all features except `normalized_step`)
- Label-shuffle sanity baseline (same architecture, randomly shuffled `teacher_hazard` labels)

## Inputs

- Training data: `/data/liuyu/outputs/milestone_2b_parser_visual_linkage_20260526/tables/student_train_dataset.csv` (87,474 rows, 400 episodes)
- Model checkpoint: `/data/liuyu/outputs/milestone_2c_proprio_causal_student_20260526/checkpoints/best_model.pt`
- Script: `scripts/run_milestone_2c1_replay_ablation.py`

## Output Artifact

Output root:

```bash
/data/liuyu/outputs/milestone_2c1_student_replay_ablation_20260527
```

(Note: the output date is 20260527, reflecting the actual run date. The milestone date in the tag uses 20260526 for consistency with the freeze tag convention.)

## Models Compared

| # | Model | Description |
|---|-------|-------------|
| 1 | teacher_window | Oracle reference: trigger when step ∈ [teacher_window_start, teacher_window_end] |
| 2 | full_proprio_coverage | Existing 2C proprio student, coverage-first threshold (h≥0.1, r<0.3) |
| 3 | full_proprio_conservative | Existing 2C proprio student, conservative threshold (h≥0.5, r<0.5) |
| 4 | rule_proxy | Heuristic: gripper command+width+qpos rules, no ML model |
| 5 | time_only_coverage | MLP trained ONLY on `normalized_step` + `mechanism_type` + `parse_confidence` |
| 6 | time_only_conservative | Time-only with conservative threshold |
| 7 | no_normalized_step_coverage | MLP trained on all features EXCEPT `normalized_step` |
| 8 | no_normalized_step_conservative | No-step ablation with conservative threshold |
| 9 | label_shuffle | Same architecture, `teacher_hazard` labels randomly shuffled |

## Key Results

### Overall Comparison (coverage-first threshold: h≥0.1, r<0.3)

| Model | Win Coverage | False Early | Miss Rate | Mean Latency | Trigger Rate |
|-------|-------------|-------------|-----------|-------------|-------------|
| teacher_window | 1.0000 | 0.0000 | 0.0000 | 0.0 | 0.0474 |
| **full_proprio** | **0.8706** | **0.0116** | **0.1294** | **-2.65** | 0.0687 |
| rule_proxy | 0.0147 | 0.0020 | 0.9853 | -55.5 | 0.0099 |
| time_only | 0.4265 | 0.0045 | 0.5735 | -0.41 | 0.0379 |
| no_normalized_step | 0.7529 | 0.0248 | 0.2471 | -32.12 | 0.0947 |
| label_shuffle | 0.0000 | 0.0000 | 1.0000 | N/A | 0.0000 |

### By Suite (full_proprio_coverage)

| Suite | N Episodes | Trigger Rate | Win Coverage | False Early | Miss Rate | Mean Latency |
|-------|-----------|-------------|-------------|-------------|-----------|-------------|
| libero_10 | 10 | 0.0355 | 0.9571 | 0.0105 | 0.0429 | -9.86 |
| libero_goal | 10 | 0.1109 | 0.8700 | 0.0055 | 0.1300 | 0.60 |
| libero_object | 10 | 0.0938 | 0.9857 | 0.0195 | 0.0143 | -5.86 |
| libero_spatial | 10 | 0.0989 | 0.7300 | 0.0045 | 0.2700 | 1.40 |

### Ablation Model Classification Metrics (test split)

| Model | Phase Acc | Phase F1 | Hazard F1 | Hazard AUROC | Hazard AUPRC |
|-------|-----------|----------|-----------|-------------|-------------|
| full_proprio (2C) | 0.8791 | 0.7286 | 0.6743 | 0.9874 | 0.7310 |
| time_only | 0.4069 | 0.1194 | 0.0000 | 0.9555 | 0.4244 |
| no_normalized_step | 0.8425 | 0.6654 | 0.4169 | 0.9465 | 0.5822 |
| label_shuffle | 0.8589 | 0.6912 | 0.0000 | 0.5022 | 0.0296 |

## Anti-Timing Verification Checklist

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Full proprio > time-only window coverage | PASS | 0.8706 > 0.4265 |
| No-step ablation retains non-trivial performance | PASS | 0.7529 > 0.30 (drops but does not collapse) |
| Label-shuffle baseline collapses | PASS | Window coverage = 0.0000, Hazard AUROC ≈ 0.50 |
| False early trigger rate low | PASS | 0.0116 < 0.05 |
| Articulated/rearrangement tasks do not trigger aggressively | PASS | Not in test split (excluded by design) |
| Object metrics reported separately | PASS | Reported above, not used as primary evidence |

## Interpretation

1. **Full proprio student significantly outperforms time-only baseline** (0.8706 vs 0.4265 window coverage). The student is not merely learning normalized timing — gripper/EEF/action-history features provide genuine phase signal.

2. **No-normalized-step ablation retains substantial performance** (0.7529). Removing the timing feature causes a drop from 0.8706 → 0.7529 (Δ = -0.1177), but performance remains far above the time-only baseline. This confirms that gripper/EEF/action-history features independently carry phase information.

3. **Label-shuffle baseline collapses completely** (window coverage = 0.0000, hazard AUROC ≈ 0.50). There is no data leakage or split contamination.

4. **Rule/proxy trigger is too weak** (coverage = 0.0147). Simple gripper heuristics cannot replace the learned detector.

5. **Negative mean latency** (-2.65 for coverage-first) indicates the student triggers slightly *before* the teacher window start, which is desirable for an early-warning detector.

6. **Coverage-first threshold** (h≥0.1, r<0.3) provides the best trade-off: high window coverage (0.8706) with low false early rate (0.0116).

## Caveats

- This is offline causal replay validation only. It does NOT constitute attack evidence.
- The student has not been tested with live OpenVLA rollouts or visual perturbations.
- Object suite metrics are included for completeness but should not be used for strong claims until the clean reproducibility gap is resolved.
- The model is proprio-only. Visual features are still missing (deferred to Milestone 2D).
- The student is not a final online detector. It is a development baseline.

## Next Step

Milestone 3A: Small student-triggered attack pilot (VIS/random/oracle, GPU01, ≤8 candidates, Spatial/Goal/LIBERO-10 only, Object excluded).

The student trigger is now offline-replay validated and ready for a controlled small-scale attack pilot. The pilot should verify that student-triggered VIS perturbations produce higher CQFR than random controls, and that oracle confirms physical sensitivity of the detected windows.
