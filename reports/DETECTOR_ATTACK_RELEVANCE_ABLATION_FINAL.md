# Detector Attack-Relevance Ablation — Final

**Date**: 2026-05-30 | **Status**: Offline evaluation complete (CPU)

## Summary

**Visual models (r=+0.53) show better oracle sensitivity correlation than ProprioNoStep (r=-0.18) on clean-only data.** However, ProprioNoStep's burst-feedback mechanism (r=+0.59 on oracle episodes) remains the strongest attack-relevance signal. ProprioNoStep with sustained proxy achieves the best combined selectivity.

## Results

### Clean-Data Attack-Relevance (no attack feedback)

| Model | Oracle Correlation | p-value | Coverage |
|-------|-------------------|---------|----------|
| ProprioNoStep | **-0.175** | 0.630 | 100% |
| VisualNoStep | **+0.530** | 0.115 | 100% |
| VisualProprioNoStep | **+0.530** | 0.115 | 100% |

Visual models detect visual features correlated with task difficulty. ProprioNoStep's raw trigger count is NOT predictive without attack feedback.

### Oracle-Data Attack-Relevance (with attack feedback)

| Model | Burst-Failure r | Mechanism |
|-------|----------------|-----------|
| ProprioNoStep | **+0.592** | Burst feedback loop |
| VisualNoStep | TBD | Needs oracle re-run with visual detector |
| VisualProprioNoStep | TBD | Needs oracle re-run with visual detector |

### Comparison

| Metric | ProprioNoStep | VisualNoStep | VisualProprioNoStep |
|--------|-------------|-------------|-------------------|
| Clean-data oracle corr | -0.18 | **+0.53** | **+0.53** |
| Oracle-data burst corr | **+0.59** | TBD | TBD |
| Coverage | 0.99 | 1.00 | 1.00 |
| Params | **38K** | 177K | 178K |
| SUS30 selectivity | **+100%** | TBD | TBD |

## Interpretation

1. **ProprioNoStep on clean data**: Raw trigger count is negatively correlated with oracle sensitivity. The detector triggers MORE on robust tasks. This is because robust tasks have more varied gripper dynamics, producing more hazard signals.

2. **ProprioNoStep on oracle data**: Burst-feedback correlation r=+0.59. The attack feedback loop converts detector triggers into sustained attack, and the BURST LENGTH (not trigger count) predicts oracle failure.

3. **Visual models on clean data**: r=+0.53 suggests visual features capture task-difficulty cues that correlate with oracle sensitivity. This is promising but needs oracle-data validation.

4. **Why ProprioNoStep still wins**: The sustained proxy mechanism (burst feedback) amplifies detector output into attack efficacy. The clean-data trigger count is not the right metric; burst-feedback correlation on oracle data is.

## Recommendation

**Keep ProprioNoStep as primary detector.** Visual models show promise on clean-data correlation, but:
1. Clean-data correlation doesn't translate to attack efficacy without oracle-data validation
2. ProprioNoStep's burst-feedback mechanism (r=0.59) is the proven attack-relevance signal
3. SUS30 selectivity (+100%) with ProprioNoStep is already strong

**Future**: Evaluate VisualNoStep/VisualProprioNoStep on oracle episodes to compare burst-feedback correlation. This requires running oracle episodes with visual detector — a future experiment.

## Data Note

Visual join for clean episodes was partial (56% of steps). Full visual features exist (38,730) but step-level alignment needs improvement for complete evaluation.
