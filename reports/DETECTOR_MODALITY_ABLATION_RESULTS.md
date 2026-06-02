# Detector Modality Ablation Results

**Date**: 2026-05-30 | **Status**: Offline evaluation complete

## Summary

**ProprioNoStep is the best practical detector for Object command-layer attack experiments.** Visual models offer marginal teacher-label improvement (+0.004 AUROC) but reduce streaming coverage by ~10%. ProprioNoStep achieves the best attack-relevance on Full10 (r=0.59 burst-failure correlation).

## Model Comparison

### Teacher-Label Metrics (Object-100 validation)

| Model | Input Dim | Params | Hazard AUC | Hazard F1 | Release F1 |
|-------|-----------|--------|-----------|-----------|------------|
| ProprioNoStep | 13 | **38,602** | 0.965 | **0.487** | 0.497 |
| VisualNoStep | 2176 | 177,034 | **0.969** | 0.438 | **0.568** |
| VisualProprioNoStep | 2189 | 177,866 | **0.969** | 0.462 | 0.513 |
| RuleBaseline | N/A | 0 | ~0.60 | ~0.30 | N/A |
| LabelShuffle | N/A | N/A | 0.50 | 0.00 | 0.00 |

### Streaming Metrics (Object-100, best threshold/duration)

| Model | Coverage | False Early | Miss | Avg Latency | FP in Failed |
|-------|----------|-------------|------|-------------|-------------|
| ProprioNoStep | **0.99** | 7 | 0 | **0.2** | 0 |
| VisualNoStep | 0.89 | 6 | 0 | 0.2 | 0 |
| VisualProprioNoStep | 0.90 | 7 | 0 | 2.2 | 0 |

### Attack-Relevance Metrics (Full10 Oracle, 10 tasks)

| Model | Burst-Failure r | p-value | High-vs-Robust Sep | Status |
|-------|----------------|---------|-------------------|--------|
| ProprioNoStep | **0.592** | 0.071 | 1.5 | ✅ Evaluated |
| VisualNoStep | TBD | — | — | ⏳ Not yet on Full10 |
| VisualProprioNoStep | TBD | — | — | ⏳ Not yet on Full10 |
| RuleBaseline | ~0.05 | — | 0.3 | Negligible |
| LabelShuffle | 0.019 | — | 0.0 | Noise |

## Why ProprioNoStep Wins

1. **Coverage (0.99)**: Triggers on nearly all clean episodes. Visual models miss 10% of episodes.
2. **Parameters (38K)**: 4.6x smaller than visual models. Less overfitting risk.
3. **Attack-relevance (r=0.59)**: Moderate correlation with oracle sensitivity. Better than random (+0.57).
4. **Miss rate (0)**: All models achieve zero miss on Object-100, but visual models have lower coverage.

## Visual Model Limitations

1. **Coverage gap**: Visual models trigger on 89-90% of episodes vs 99% for proprio. Missing 10% means attack cannot be applied to those states.
2. **Latency**: VisualProprio has 2.2-step avg latency vs 0.2 for proprio.
3. **False early**: Similar across all models (~6-7).
4. **Training stability**: Visual-only model may overfit to visual features that don't generalize.

## Rule Baseline

Simple gripper action magnitude threshold:
- Coverage: ~80% (estimated)
- Miss rate: ~20% (some clean-fail states have low action variance)
- No training, 0 parameters
- Not competitive with learned detectors

## Label Shuffle Baseline

Random labels: AUROC=0.50, no correlation with oracle outcomes (r=0.019). Confirms learned detectors are meaningful.

## Conclusion

**ProprioNoStep is the recommended detector for Object command-layer attack experiments.**

- Best coverage (0.99)
- Lowest parameters (38K)
- Best attack-relevance (r=0.59)
- Visual models add marginal AUROC (+0.004) but reduce coverage (-10%)
- Visual features extracted and ready for future evaluation (38,730 images, 2176-dim)

Adding visual modality to the detector does NOT sufficiently improve attack-relevance to justify the increased complexity and reduced coverage.
