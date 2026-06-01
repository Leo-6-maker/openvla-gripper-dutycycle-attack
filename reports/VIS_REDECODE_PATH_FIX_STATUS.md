# VIS Re-decode Path Fix Status

**Date**: 2026-05-31 | **Status**: Token-level attack insufficient for gripper steering

## Phase 1: Dtype Fix — RESOLVED

`TokenPrefixPGDAttacker` line 146 patched from `dtype=torch.float16` to `model_dtype`.

## Phase 2: Re-decode Path — RESOLVED

- `action_adv` is None by design — not a bug
- Adversarial inputs stored in `result.debug['adv_inputs']` (pixel_values + input_ids)
- Re-decode via `model.generate(pixel_values=adv_pv, input_ids=adv_ids)` works correctly

## Phase 3: Target Sign Sweep — INCONCLUSIVE

| Target | CE Initial | CE Final | Gripper | Arm L2 |
|--------|-----------|----------|---------|--------|
| +1.0 | 48 | 4.9 | 0.996→0.996 | 1.52 |
| -1.0 | 47 | 18.1 | 0.996→0.996 | 1.52 |

CE drops significantly (PGD works), but argmax doesn't switch tokens. The model's logits change but the discrete token decision remains unchanged. Arm dimensions drift significantly (1.5 L2) despite gripper-only loss masking.

## Root Cause

Token-level discrete optimization: PGD reduces the loss for the target token, but the cross-entropy minimization doesn't guarantee the argmax switches. At eps=4/255, the pixel perturbation is strong enough to shift logits but not strong enough to flip the discrete token decision from bin 254 to a different bin.

For gripper to change from 0.996 to open direction, the token needs to shift from bin ~254 to bin ~191 (action≈0.5). This requires a much larger logit shift than what 4/255 pixel perturbation can achieve.

## Conclusion

**White-box PGD works, re-decode works, but token-level gripper steering at ε≤4/255 is insufficient.** Higher epsilon may work but risks visible artifacts and arm drift.

## Path Forward

1. Test higher eps (8/255, 16/255) to see if token flips
2. Consider continuous-action attack (attack the decoded action directly, not tokens)
3. Or use PGD as scene-level perturbation rather than precise gripper control
