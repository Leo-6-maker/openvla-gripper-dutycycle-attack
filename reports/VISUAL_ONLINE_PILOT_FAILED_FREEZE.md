# VisualNoStep Online Pilot — Failed Freeze

**Date**: 2026-05-30 | **Status**: FAILED | **Rollouts**: 24/24 (0 triggers)

## Result

All 24 episodes (4 tasks × 3 states × 2 conditions) completed with 0 detector triggers and 0 attack steps. All clean and sus30 episodes failed at step 10-11 due to visual model producing zero hazard scores.

## Root Cause

`VisualNoStep_frozen.pt` outputs `hs=0.000000` on 100/100 sampled Full10 visual features. The model was trained on Object-100 data (18,875 images) but does NOT generalize to Full10 frames. Feature distributions are similar (norm ~84.4 for both), so the issue is likely training-data overfitting or threshold calibration.

Additional finding: `CausalTCNDetector` (online runner architecture) and the original training architecture differ. The checkpoint loads cleanly (`strict=False`) but produces zero outputs.

## Valid Conclusions

- "Current VisualNoStep checkpoint is silent on Full10 visual features."
- "Visual online evaluation is not possible with current checkpoint."
- "ProprioNoStep remains the only viable online detector."

## Forbidden Conclusions

- "Visual information is useless."
- "Visual attack failed / VIS attack failed."
- "Visual detector evaluated attack-relevance online."

## Actions

1. Do NOT launch more visual online rollouts.
2. Keep ProprioNoStep as production detector.
3. Visual retraining required before any online visual evaluation.
4. Flexible detector code (75a75a1) is correct infrastructure.
