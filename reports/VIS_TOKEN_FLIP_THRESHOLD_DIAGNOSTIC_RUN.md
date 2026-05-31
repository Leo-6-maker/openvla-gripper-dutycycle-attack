# VIS Token-Flip Threshold Diagnostic Run

Date: 2026-05-31

## Status

One real one-frame VIS loader smoke ran successfully after fixing TokenPrefixPGD processor-pixel budget semantics. The full 3-objective x 4-epsilon x 3-step threshold sweep did not run because the valid-budget smoke still failed VIS-1.

## Output

```text
tables/vis_token_flip_threshold_diagnostic.csv
tables/vis_one_frame_loader_smoke.csv
```

## Smoke Configuration

- frame: Object ketchup clean frame, step 0
- model: `openvla-7b-finetuned-libero-object`
- objective: `target_action_ce`
- epsilon: `4/255`
- attack steps: `1`
- physical GPUs: `2,6`
- attention backend: `eager`

## Result

| Metric | Value |
| --- | ---: |
| clean gripper token | 31872 |
| adversarial gripper token | 31872 |
| token flip | false |
| clean gripper action | 0.0 |
| adversarial gripper action | 0.0 |
| gripper delta | 0.0 |
| arm L2 | 0.184442 |
| target CE before | 32.0000 |
| target CE after | 15.9500 |
| open-bin prob mass before | 5.87e-13 |
| open-bin prob mass after | 1.52e-07 |
| close-bin prob mass before | 0.999996 |
| close-bin prob mass after | 0.987568 |
| perturbation Linf | 0.0078125 |

## Gate VIS-1

Result: FAIL.

Reasons:

- No decoded gripper token flip.
- No decoded gripper action change.
- Target CE improves, but decoded action remains unchanged.
- Perturbation Linf is now inside the requested budget, so this is a real valid-budget no-flip result under `processor_pixel_values_linf` semantics.
- Arm drift is nontrivial despite no gripper action effect.

## Decision

Do not run:

- full token-flip sweep
- arm-drift sweep
- forced-window VIS micro
- detector-triggered VIS rollout

Next VIS work should improve the VIS objective/optimization path under valid budget before any rollout work.
