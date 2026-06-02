# Visual/Fusion Ablation Data Readiness — Verified

## Status: ProprioNoStep READY, Visual/Fusion NEEDS EXTRACTION

| Detector | Checkpoint | Training Data | Eval Data (step_records) | Eval Data (visual) | Status |
|----------|-----------|---------------|------------------------|-------------------|--------|
| ProprioNoStep | ✅ SHA: 4b3f3d47 | ✅ Object-100 labeled | ✅ Full10 has 13 proprio features | N/A | **READY** |
| VisualNoStep | ✅ SHA: 2d6defaa | ✅ Object-100 visual | ❌ No visual features in step_records | ❌ Need extraction | **BLOCKED** |
| VisualProprioNoStep | ✅ SHA: e496a4bf | ✅ Object-100 fused | ❌ No visual features in step_records | ❌ Need extraction | **BLOCKED** |
| RuleBaseline | N/A | N/A | ✅ gripper_command in step_records | N/A | **READY** |
| LabelShuffle | N/A | N/A | N/A | N/A | **READY** |

## ProprioNoStep Analysis

Already evaluated from Full10 data:
- Coverage: 100% (all 10 tasks trigger)
- High-sensitive tasks: avg burst 95 (prolonged feedback)
- Robust tasks: avg burst 47 (self-limiting)
- Triggers at step ~115-150 (late grasp/lift phase)
- Attack-relevance: burst length separates high-sensitive from robust

## Visual Feature Extraction Needed

To evaluate VisualNoStep and VisualProprioNoStep:
- Extract from Full10 RGB frames (~150 episodes × ~200 steps = ~30,000 frames)
- Feature dim: 2176 (DINOv2+SigLIP fused)
- GPU7 recommended
- Estimated time: ~2-3 hours on GPU7

## Recommendation
1. Run ProprioNoStep analysis now (data ready)
2. Start GPU7 visual feature extraction in background
3. After extraction: run VisualNoStep + VisualProprioNoStep evaluation
4. Compare all 3 models on attack-relevance metrics
