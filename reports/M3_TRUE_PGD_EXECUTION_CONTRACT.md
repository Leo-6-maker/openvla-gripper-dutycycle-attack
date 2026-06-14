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
strict target-token objective lacks exact clean 7-token generation
fallback_reason is present
adv_inputs is missing
x_adv is not None
action_adv is not None
attack_method does not start with token_prefix_pgd
directional_loss_available is not true
resolved objective is not autoregressive_prefix_gripper_target_token_cw_v1
target_token_id is not 31744
target_execution_class is not CLIP_MEDIATED_OPEN
num_backwards differs from expected
num_loss_forwards < num_backwards + 1
pixel_space is not processor_pixel_values
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
action_adv_is_none = true
pixel_space = processor_pixel_values
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

For the strict target-token objective, the acceptance gate is computed against
the clean rollout's actual autoregressive 7-token generation, not against a
continuous-action retokenization. The telemetry keeps both references separate:

```text
clean_generated_action_token_ids
clean_generated_arm_prefix_token_ids
retokenized_clean_action_token_ids
retokenized_clean_action_arm_token_ids
generated_adv_arm_prefix_token_ids
arm_gate_reference = clean_actual_generation
```

If clean generation is missing or does not provide exactly 7 new action tokens,
strict target-token execution hard-fails.

## Projection And Random Controls

PGD, RAND20, shuffled-gradient controls, and delta0 controls must share the
same processor-space projection and dtype-cast helper:

```text
project_and_cast_processor_values(...)
```

The helper projects in fp32 and then casts to the model dtype while resetting
bf16/fp16 rounded elements that would otherwise exceed `epsilon`. This prevents
RAND20 from silently using a looser budget path than PGD.

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
true-PGD result rejects wrong attack_method, wrong objective, wrong target token,
missing adv_input keys, action_adv, and insufficient loss forwards
31744 classifies as CLIP_MEDIATED_OPEN and not native OPEN
strict target-token objective requires actual clean exact-7 generation
arm gate uses actual clean generated prefix, not retokenized action prefix
target-token CW improves mock surrogate margin
RAND20 seed schedule and processor-space projection are reproducible
RAND20 bf16/fp16 post-cast budget is checked
PGD and RAND share the same projection helper
official generation exact-token and tie-aware invariants are checked
```

No GPU or LIBERO rollout was launched for this contract.
