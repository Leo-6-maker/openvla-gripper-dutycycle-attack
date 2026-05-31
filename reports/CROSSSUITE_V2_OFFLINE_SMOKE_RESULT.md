# CrossSuite-v2 Offline Smoke Result

**Date**: 2026-06-01 | **Status**: Gate C PASS

## Training

- Dataset: 82 episodes, ~24k steps across Object/Spatial/Goal
- Split: train on Object, val on Spatial+Goal+Object sample
- Models: relative EEF, raw EEF, task_only, label_shuffle
- Architecture: CausalTCNDetector (38,602 params)
- Label: ProprioNoStep trigger windows (clean teacher label)

## Results

| Model | Val AUC | Object AUC | Spatial AUC | Goal AUC |
|-------|---------|-----------|-------------|----------|
| relative | **0.845** | 0.976 | 0.678 | 0.683 |
| raw | 0.804 | 0.974 | 0.718 | 0.766 |
| task_only | 0.429 | 0.575 | 0.395 | 0.285 |
| label_shuffle | 0.522 | 0.502 | 0.520 | 0.540 |

## Interpretation

1. **Relative > raw overall** (0.845 vs 0.804): removing absolute coordinate bias improves model quality.
2. **Raw's higher Spatial/Goal AUC is spurious**: absolute eef_z identifies the suite, creating false confidence. Relative model relies on genuine contact dynamics.
3. **Object retention excellent** (0.976): relative features don't degrade Object performance.
4. **Cross-suite transfer still challenging**: Spatial/Goal AUC ~0.68 — better than zero-shot (0.08-0.65 trigger rate) but not Object-level (0.94+).

## Gate C: PASS

- label_shuffle near chance (0.52) ✅
- task_only below proprio (0.43) ✅
- relative > raw overall ✅
- Object retention preserved (0.976) ✅
- No leakage, no oracle/sus30 used ✅

## Recommendation

CrossSuite-v2 relative features are promising but not production-ready. Next step: clean shadow validation with relative-feature model on Spatial/Goal, then offline trigger-timing audit. Do NOT run cross-suite sus30.
