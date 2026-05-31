# VIS Future Work Decision Tree

**Date**: 2026-06-01 | **Status**: All experiments BLOCKED

## Current Result

On verified contact/carry frames, token-prefix PGD at ε≤8/255: CE drops 34→0 but decoded gripper token unchanged (0/32 flips).

## Decision Tree

### A. Larger Epsilon
- Test ε = 16/255, 32/255
- Gate: no-rollout token flip first
- Only if threat model allows visible perturbation

### B. Alternative Objectives
- decoded-token margin (CW-style)
- multi-token open-region CE
- temporal accumulation (multi-frame)
- Gate: same as A

### C. Alternative Perturbation
- patch/localized gripper-object crop
- image trigger / overlay
- object-centric perturbation
- Gate: same as A

### D. Model Interface
- attack pre-decode logits
- inspect greedy token boundary
- compare beam/sampling decode
- Gate: same as A

## Recommendation

Do NOT continue VIS experiments without:
1. New approved hypothesis
2. No-rollout token flip gate first
3. Strict budget + arm drift + random control checks

Current VIS frozen as negative diagnostic.
