# M3 log-ratio arm-v3 seed81 canary results

## Result

`FULL_TOKEN_FLIP_FIXED_FRAME`

The arm-constrained v3 objective passed the single fixed-frame Tomato step78 gate for attack seed `81`.

## Primary Metrics

| Metric | Value |
| --- | --- |
| Clean token | `31872` |
| TRUE_PGD token | `31744` |
| Delta0 official margin | `-0.25` |
| TRUE_PGD official margin | `21.499736785888672` |
| RAND20 selected official margin | `6.0` |
| SHUFFLED_GRAD_PGD20 margin | `-0.25` |
| TRUE_PGD arm match | `6/6` |
| Arm preserve weight | `0.5` |

## Gate

- Infra gate: `PASS`
- Official transfer gate: `PASS`
- RAND20 gate: `PASS`
- Shuffled-gradient gate: `PASS`
- Arm selectivity gate: `PASS`

## Output

Server output directory:

`/data/liuyu/outputs/m3_logratio_arm_v3_step78_290fb13_seed81`

## Allowed Claim

On the Tomato development fixed frame step78, `autoregressive_prefix_gripper_target_token_logratio_arm_v3` produced an official token flip from `31872` to `31744`, preserved the actual clean generated arm prefix at `6/6`, and beat the selected `RAND20` and `SHUFFLED_GRAD_PGD20` controls on official target margin.

## Forbidden Claim

Do not claim closed-loop Layer3, task failure, critical-close disruption, held-out transfer, or defense validity from this result. No LIBERO rollout or multi-frame panel was launched.

## Next Gate

The next permitted stage is the preregistered fixed-frame panel. A full-window Tomato rollout remains forbidden until fixed-frame panel and current-runner transfer gates pass.
