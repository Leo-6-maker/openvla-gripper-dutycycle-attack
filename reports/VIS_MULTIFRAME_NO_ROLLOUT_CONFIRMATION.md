# VIS Multiframe No-Rollout Confirmation

Date: 2026-05-31

## Scope

No rollout and no training were run.

This diagnostic extends the one-frame VIS arm-drift check to four saved Object
frames:

- `ketchup_s0`
- `cream_cheese_s0`
- `tomato_sauce_s0`
- `cream_cheese_s1`

All runs used physical GPUs `4,5` only through:

```text
CUDA_VISIBLE_DEVICES=4,5
```

## Outputs

Raw per-frame tables:

```text
tables/vis_arm_drift_sweep.csv
tables/vis_arm_drift_cream_s0.csv
tables/vis_arm_drift_tomato_s0.csv
tables/vis_arm_drift_cream_s1.csv
```

Aggregate:

```text
tables/vis_multiframe_no_rollout_confirmation.csv
tables/vis_multiframe_no_rollout_confirmation_summary.csv
```

## Configuration

| Field | Value |
| --- | --- |
| objective | `target_action_ce` |
| eps | `4/255` |
| steps | `4` |
| step_size | `1/255` |
| random baseline | same processor-pixel Linf |

## Aggregate Result

| Metric | Value |
| --- | ---: |
| frames total | 4 |
| frames pass | 2 |
| frames fail | 2 |
| targeted token-flip frames | 2 |
| random token-flip frames | 0 |

## Per-frame Result

| Frame | Object | Targeted Flip | Targeted Gripper Delta | Targeted Arm L2 | Ratio | Random Flip | Gate |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `ketchup_s0` | ketchup | true | 0.996078 | 0.069530 | 14.3259 | false | pass |
| `cream_s0` | cream_cheese | false | 0.0 | 0.142711 | 0.0 | false | fail |
| `tomato_s0` | tomato_sauce | true | 0.996078 | 0.364101 | 2.7357 | false | pass |
| `cream_s1` | cream_cheese | false | 0.0 | 0.105036 | 0.0 | false | fail |

## Interpretation

The no-rollout VIS evidence is positive but not yet robust enough for rollout.

Positive evidence:

- targeted VIS flipped decoded gripper token/action on ketchup and tomato
- random same-Linf did not flip gripper token/action on any tested frame

Limitation:

- both cream_cheese frames failed to produce a gripper token/action flip
- tomato_s0 has a larger arm L2 than ketchup_s0, although the gripper-to-arm
  ratio remains above 1

## Gate Decision

Multiframe no-rollout confirmation: PARTIAL PASS.

Do not run forced-window VIS micro yet.

## Next Recommended Action

Before any rollout proposal:

1. Add a contact-frame selector or verify that the saved `step_0000` frames are
   actually the intended contact/pre-place frames.
2. Run a no-rollout confirmation on selected contact frames for ketchup,
   tomato_sauce, and cream_cheese.
3. If contact-frame evidence passes on at least two objects and cream_cheese is
   understood or excluded with a clear reason, write a forced-window VIS micro
   proposal with matched clean and random controls.
