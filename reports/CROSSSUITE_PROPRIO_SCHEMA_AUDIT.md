# Cross-Suite Proprio Schema Audit

**Date**: 2026-05-31 | **Status**: Distribution shift confirmed — not an engineering bug

## Root Cause: EEF Position Distribution Shift

| Feature | Object (train) | Spatial | Goal | Shift Factor |
|---------|---------------|---------|------|--------------|
| eef_z (height) | **0.19** ± 0.08 | **1.09** ± 0.10 | **1.08** ± 0.10 | **5.7x** |
| eef_x | -0.04 ± 0.08 | -0.12 ± 0.10 | -0.11 ± 0.08 | 3x |
| eef_y | -0.04 ± 0.17 | 0.10 ± 0.12 | 0.02 ± 0.08 | sign flip |
| action_gripper | 0.01 ± 0.97 | -0.18 ± 0.95 | 0.09 ± 0.94 | distribution shift |

## Feature Validity Check

| Check | Object | Spatial | Goal |
|-------|--------|---------|------|
| NaN count | 0 | 0 | 0 |
| Missing keys | 0 | 0 | 0 |
| Keys with >90% zeros | None | None | None |
| raw_action[-1] mean | 0.996 | 0.996 | 0.996 |
| env_action[-1] mean | -1.000 | -1.000 | -1.000 |

**Schema is correct and consistent across suites.** No engineering bug.

## Why ProprioNoStep is Silent on Goal/Spatial

The detector was trained on Object-100 proprio data where eef_z ≈ 0.19 (table-level workspace). In Spatial/Goal, the robot operates at eef_z ≈ 1.08 (higher workspace). The 5.7x distribution shift means the model's projection layer receives OOD inputs, producing near-zero hazard scores across all steps.

## Implications

1. **Object-ProprioNoStep is Object-production only** — not a schema bug
2. Cross-suite transfer requires suite-specific detector training with appropriate normalizers
3. Lowering threshold won't help — the scores are genuinely near-zero (Goal max=0.15, p99=0.05)
4. Spatial has slightly higher scores (max=0.45) suggesting partial overlap in workspace

## Recommendation

- Object-ProprioNoStep: keep as frozen production baseline
- CrossSuite-ProprioNoStep-v2: train with multi-suite data + suite-aware normalization
- Do NOT lower threshold to force triggers — would produce random noise triggers
