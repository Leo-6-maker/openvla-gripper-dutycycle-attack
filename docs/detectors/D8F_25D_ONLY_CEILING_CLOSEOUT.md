# D8F — 25D-Only Detector Route Closeout

**Status**: CLOSED — ceiling confirmed across 7 model variants.

## Summary

D8F1 and D8F2 tested selective abstention as a potential path beyond C2e3's
FP=31.8% limitation. Both confirmed that the 25D proprio/action feature space
is sufficient for Object/Goal/Spatial but fundamentally insufficient for L10
primary-event disambiguation.

## Evidence

### D8F1: Selective Abstention (best F1)

| Suite | Recall | FP |
|---|---|---|
| Object | 96.1% | 0.0% |
| Goal | 94.1% | 0.0% |
| Spatial | 97.7% | 8.1% |
| **L10** | **1.1%** | 0.6% |

Model learned to abstain on 99% of L10 windows to maintain low overall FP.

### D8F2: Suite-Balanced + L10 Positive Weighting

| Suite | Recall | FP |
|---|---|---|
| Object | 96.1% | 0.0% |
| Goal | 94.1% | 0.0% |
| Spatial | 97.7% | 8.1% |
| **L10** | **1.1%** | 0.6% |

Epoch 0 briefly hit L10 recall=66.3% (FP=59%), then collapsed to abstain.
Suite-balanced sampling + L10 positive weighting did not prevent collapse.

### Prior 25D-Only Results (C2e3/D8A)

| Model | Overall Recall | Overall FP | L10 Recall |
|---|---|---|---|
| GRU (C2e3 baseline) | 75.6% | 31.8% | 45.6% |
| Causal TCN | ~54% | 38.6% | 54.4% |
| FP-aware GRU | — | — | <55% |
| Multi-window GRU | — | — | <55% |
| GRU+TCN Ensemble | — | — | <55% |

All 25D-only models converge to FP floor 31-39%. L10 recall <55% across all variants.

## Root Cause

L10 tasks are long-horizon, language-conditioned, multi-object manipulation.
Multiple setup/auxiliary/distractor/primary events produce similar 25D
proprio/action patterns (gripper close → lift → carry → approach → release),
but with different semantic roles. 25D features cannot distinguish:

- "Is the currently grasped object the task-language-specified primary object?"
- "Is this manipulation a setup action or the primary transfer?"
- "Which of several similar carry phases is the attackable one?"

## Conclusion

```
25D proprio/action is sufficient for simple contact-critical phase detection
(Object/Goal/Spatial), but insufficient for L10 language-conditioned
primary-event disambiguation.

Next route: C2f observation/language-enhanced detector.
(RGB embedding + task language embedding + 25D temporal features)
```

## What Is Now Permanently Closed

- Larger GRU/TCN hidden sizes
- Per-suite/per-task threshold calibration
- FP-aware loss variants
- Multi-window GRU variants
- GRU+TCN ensemble/score fusion
- Suite-balanced 25D-only training
- Any architectural change within 25D-only input space
