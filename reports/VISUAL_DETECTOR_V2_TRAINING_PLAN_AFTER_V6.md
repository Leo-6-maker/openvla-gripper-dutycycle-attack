# Visual Detector V2 Training Plan (After V6 Non-Production Freeze)

**Date**: 2026-05-30 | **Status**: DRAFT — awaiting approval before execution

## Background

VisualNoStep V6 @ th=0.05 triggers online but is non-selective (breaks ketchup robust control 0/3). The current `VisualNoStep_frozen.pt` was trained on Object-100 full dataset without proper holdout evaluation, and threshold was chosen ad-hoc.

## Non-Negotiable Constraints

1. **Training labels**: Clean teacher labels ONLY. Never oracle/sus30 outcomes.
2. **Architecture**: MUST use `CausalTCNDetector` (same as online runner).
3. **Validation**: Task holdout required to prevent task-identity memorization.
4. **Threshold**: Calibrated on held-out validation set, not ad-hoc 0.05.
5. **Evaluation**: Offline attack-relevance metrics only. No online rollout without approval.

## Training Data

- **Labeled dataset**: `/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527/tables/no_timestep_visual_proprio_student_dataset_labeled.csv`
- **Feature dims**:
  - Visual: 2176 (DINOv2+SigLIP fused)
  - Proprio: 13 (gripper_command, gripper_qpos, gripper_width, eef_xyz, eef_vxyz, action_dxyz, action_gripper)
- **Frozen visual features**: `/data/liuyu/outputs/milestone_2f_full10_frozen_visual_features_20260530/features/full10_frozen_visual_features.npy`
- **Join map**: `/data/liuyu/outputs/milestone_2f_full10_frozen_visual_features_20260530/tables/full10_feature_join_map.csv`

## Model Variants (Priority Order)

### A. VisualNoStep_v2 (visual-only)
- Input dim: 2176
- Tests whether visual signal alone can learn a vulnerability prior
- Task-holdout split required

### B. VisualProprioNoStep_v2 (fusion)
- Input dim: 2176 + 13 = 2189
- Tests whether adding proprio improves over visual-only
- Same task-holdout split

### C. Proprio trigger + Visual re-ranker **(RECOMMENDED)**
- ProprioNoStep produces candidate windows
- Visual model evaluates each window: "is this a real vulnerability window?"
- Prevents visual false positives on clean rollouts

## Training Protocol

1. **Split**: 80% train / 20% val, stratified by task and teacher_hazard
2. **Architecture**: `CausalTCNDetector(in_dim, h_dim=64, n_ph=8, n_l=3, dropout=0.1)`
3. **Loss**: BCEWithLogitsLoss on hazard head output
4. **Optimizer**: Adam, lr=0.001
5. **Epochs**: 30 with early stopping on val AUC
6. **Batch size**: 128
7. **GPU**: GPU7 (lightweight training, no OpenVLA rollout)
8. **Output**: `/data/liuyu/outputs/milestone_2k_visual_detector_v2_training_20260530/`

## Metrics

### Training Metrics
- Val AUC, val accuracy at optimal threshold
- Per-task AUC to detect task-identity shortcut

### Offline Attack-Relevance Metrics (on Full10 held-out tasks)
- Trigger rate on clean rollouts (should be near zero)
- Trigger rate on oracle rollouts (should be higher)
- High-vs-robust separation
- Comparison to ProprioNoStep baseline

## Threshold Calibration

- Use validation set to find threshold that gives:
  - Clean FPR < 5%
  - Optimal F1 on teacher_hazard=1
- Report precision/recall curve
- Do NOT use oracle/sus30 for threshold selection

## Task Holdout Design

Option 1: Hold out 2 tasks (e.g., ketchup + salad_dressing) for validation
Option 2: Hold out 2 states per task for validation
Option 3: Both (task holdout + state holdout) — most rigorous

Recommendation: Option 1 (task holdout) as primary validation, Option 3 for final audit.

## Decision Gate

Do NOT proceed to training without:
1. Leon approval on this plan
2. Verified training script that passes smoke test (2 epochs, verify loss decreases)
3. Confirmed no oracle/sus30 label leakage in training data

## Timeline Estimate

- Training: ~2 hours (GPU7, 30 epochs, ~150k samples)
- Offline evaluation: ~1 hour (replay on Full10 features)
- Report writing: ~1 hour
- Total: ~4 hours wall clock
