# VIS Token-Flip Threshold Diagnostic Run

Date: 2026-05-31

## Status

One real one-frame VIS loader smoke ran successfully. The full 3-objective x 4-epsilon x 3-step threshold sweep did not run because the first real smoke exposed a budget-validity blocker.

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
| arm L2 | 0.054859 |
| target CE before | 32.0000 |
| target CE after | 30.9197 |
| open-bin prob mass before | 5.87e-13 |
| open-bin prob mass after | 1.76e-11 |
| close-bin prob mass before | 0.999996 |
| close-bin prob mass after | 0.562177 |
| perturbation Linf | 2.125 |

## Gate VIS-1

Result: FAIL.

Reasons:

- No decoded gripper token flip.
- No decoded gripper action change.
- Target CE improves, but decoded action remains unchanged.
- Perturbation Linf is far above the requested small epsilon budget, indicating the current TokenPrefixPGD normalized `pixel_values` perturbation accounting/clamp is not yet valid for a small-epsilon VIS claim.

## Decision

Do not run:

- full token-flip sweep
- arm-drift sweep
- forced-window VIS micro
- detector-triggered VIS rollout

Next VIS work should fix/define the valid pixel-space budget semantics before any additional VIS rollout work.
