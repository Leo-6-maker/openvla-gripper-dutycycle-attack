# M3 True-PGD Execution Contract

Date: 2026-06-14

Scope: CPU-only implementation contract for the next Layer3 line. This report
does not contain GPU results, LIBERO rollouts, or scientific outcome claims.

## Contract

M3 scientific runs must use:

```text
strict_route = true
allow_fallback = false
method = token_prefix_pgd
objective = autoregressive_prefix_gripper_target_token_cw_v1
target_token_id = 31744
target_execution_class = CLIP_MEDIATED_OPEN
```

Strict mode hard-fails if:

```text
method is missing
method is unknown
resolved adapter is not TokenPrefixPGDAttacker
targeted objective lacks target_action
target-token objective lacks target_token_id
target-token objective lacks target_execution_class
fallback_reason is present
adv_inputs is missing
x_adv is not None
num_backwards differs from expected
processor-space Linf exceeds epsilon
```

Compatibility mode remains available for historical code paths, but no new
scientific result may enter a denominator unless strict route metadata proves:

```text
strict_route = true
fallback_used = false
resolved_adapter_class = TokenPrefixPGDAttacker
adv_inputs_present = true
x_adv_is_none = true
```

## Objective

The first M3 objective targets the exact clip-mediated execution token observed
in Tomato:

```text
autoregressive_prefix_gripper_target_token_cw_v1
target_token_id = 31744
target_execution_class = CLIP_MEDIATED_OPEN
```

The target-token CW loss is:

```text
max(0, max_j!=31744 z_j - z_31744 + margin)
```

The competition set is the actual seventh-token score row length, not only the
native action-bin OPEN set. Target semantics are validated separately from
native OPEN-region helpers so `31744` cannot be silently replaced by a native
OPEN token set.

## Prefix And Arm Handling

Each PGD step refreshes the generated arm prefix from the current adversarial
processor inputs:

```text
prefix_refresh_interval = 1
generated arm token ids are stop-gradient context
```

The target-token loss is the optimization objective. Arm preservation is an
acceptance gate:

```text
generated arm prefix match >= 5/6
continuous arm action L2 recorded but not thresholded in M3-0
```

## Fixed-Frame Harness

The CPU harness validates:

```text
exact 7 generated tokens
processed-score argmax or tie-aware emitted-token status
actual generated arm prefix extraction
runner use of debug["adv_inputs"] rather than x_adv
```

The first GPU canary remains unauthorized until M3-0 is reviewed and merged.

## Current Validation

Implemented CPU tests cover:

```text
strict route rejects missing/unknown method
strict route resolves TokenPrefixPGDAttacker
strict route rejects missing target metadata
strict route disables TypeError retry
true-PGD result requires adv_inputs and x_adv=None
31744 classifies as CLIP_MEDIATED_OPEN and not native OPEN
target-token CW improves mock surrogate margin
RAND20 seed schedule and processor-space projection are reproducible
official generation exact-token and tie-aware invariants are checked
```

No GPU or LIBERO rollout was launched for this contract.
