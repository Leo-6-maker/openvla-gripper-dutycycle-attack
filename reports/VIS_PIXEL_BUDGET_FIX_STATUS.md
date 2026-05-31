# VIS Pixel Budget Fix Status

Date: 2026-05-31

## Status

Implemented a minimal TokenPrefixPGD perturbation-space fix in:

```text
src/gripper_attack/attack_adapter.py
```

## Root Cause

TokenPrefixPGD optimizes OpenVLA processor-normalized `pixel_values`, but the PGD loop clamped adversarial tensors to raw image range `[0, 1]`.

That was incorrect for normalized processor tensors. If `x_orig` contains negative normalized values, `[0, 1]` clamping can create perturbations far larger than the requested epsilon.

Previous one-frame smoke:

```text
requested eps = 4/255 = 0.015686
observed perturbation_linf = 2.125
```

## Fix

The PGD loop now treats epsilon as:

```text
processor_pixel_values_linf
```

and only projects to:

```text
x_orig +/- epsilon
```

It no longer clamps normalized `pixel_values` to `[0, 1]`.

Debug metadata now records:

```text
pixel_space = processor_pixel_values
pixel_epsilon_space = processor_pixel_values_linf
pixel_value_clamp = project_to_x_orig_plusminus_epsilon_only
```

## Test Coverage

Added a lightweight mock test in:

```text
tests/v4/test_token_prefix_pgd_interface.py
```

The test uses negative normalized `pixel_values` and asserts the resulting adversarial tensor remains within epsilon instead of being clamped to raw image range.

## One-frame Smoke After Fix

Command used physical GPUs `2,6` only and did not run rollout.

Result:

| Metric | Value |
| --- | ---: |
| requested epsilon | 0.015686 |
| observed perturbation Linf | 0.0078125 |
| clean gripper token | 31872 |
| adversarial gripper token | 31872 |
| token flip | false |
| clean gripper action | 0.0 |
| adversarial gripper action | 0.0 |
| gripper delta | 0.0 |
| arm L2 | 0.184442 |
| target CE before | 32.0000 |
| target CE after | 15.9500 |

## Gate Decision

Budget validity: PASS for processor-pixel Linf semantics.

VIS-1: FAIL.

Reason:

- decoded gripper token did not flip
- decoded gripper action did not change
- arm drift is nontrivial for no gripper effect

No VIS rollout, forced-window VIS micro, or heavy arm-drift sweep is allowed from this result.
