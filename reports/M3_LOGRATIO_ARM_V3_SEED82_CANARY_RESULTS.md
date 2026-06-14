# M3 log-ratio arm-v3 seed82 canary results

## Result

`TARGET_ONLY_ARM_FAIL`

The fresh seed82 fixed-frame replication did not pass the full selective gate.
TRUE_PGD emitted the target token `31744` and beat the selected RAND20 and
shuffled-gradient controls on official target margin, but it changed the actual
clean generated arm prefix from `6/6` to `2/6`.

This is a scientific gate failure for seed82. No multi-frame panel, LIBERO
closed-loop rollout, rescue, task-effect experiment, or held-out transfer was
launched.

## Frozen Run

| Field | Value |
| --- | --- |
| Stage | `M3_ARM_V3_FRESH_SEED82_FIXED_FRAME_REPLICATION` |
| Branch | `exp/m3-arm-v3-fresh-seed82-canary-20260615` |
| Commit | `480c95f5d02d2674e7165827300aadf901c9c3d9` |
| Parent implementation commit | `638a97962a038992410c426941291f8a90360d0c` |
| Config | `configs/m3_step78_true_pgd_31744_logratio_arm_v3.yaml` |
| Config SHA256 | `e026f0a98b3b0b7ecea4fe55e29d7406c8fd8d6bb6875e46e70a37c7c49f4e39` |
| Input directory | `/data/liuyu/outputs/m3_step78_true_pgd_20260614/capture_step78_f18537d_r2` |
| Preflight output | `/data/liuyu/outputs/m3_logratio_arm_v3_step78_seed82_preflight_480c95f` |
| Canary output | `/data/liuyu/outputs/m3_logratio_arm_v3_step78_seed82_480c95f` |
| Python env | `/data/aviary/envs/openvla_official_libero_20260525` |
| GPU mapping | `CUDA_VISIBLE_DEVICES=2,6` |
| Model | `/data/aviary/models/openvla/openvla-7b-finetuned-libero-object` |

## Preflight Gate

`PASS`

| Check | Value |
| --- | --- |
| Clean score path | `SURROGATE_OFFICIAL_SCORE_PATH_MATCH` |
| Delta0 score path | `SURROGATE_OFFICIAL_SCORE_PATH_MATCH` |
| Clean exact tokens | `[31900, 31870, 31915, 31882, 31862, 31913, 31872]` |
| Clean gripper token | `31872` |
| Clean margin 31744 vs competitor | `-0.25` |
| Delta0 gripper token | `31744` |
| Delta0 margin | `5.75` |

## Primary Metrics

| Condition | Token | Official target margin | Arm match |
| --- | ---: | ---: | ---: |
| CLEAN | `31872` | `-0.25` | `6/6` |
| PGD_DELTA0 | `31744` | `5.75` | `4/6` |
| TRUE_PGD_FINAL | `31744` | `29.937013626098633` | `2/6` |
| RAND20 | `31744` | `0.25` | `6/6` |
| SHUFFLED_GRAD_PGD20 | `31872` | `-0.5` | `6/6` |

The arm-prefix reference is the actual clean generated arm prefix:

`[31900, 31870, 31915, 31882, 31862, 31913]`

## Route Gate

`PASS`

| Field | Value |
| --- | --- |
| Requested method | `token_prefix_pgd` |
| Resolved adapter | `TokenPrefixPGDAttacker` |
| Strict route | `True` |
| Allow fallback | `False` |
| Fallback used | `False` |
| Objective | `autoregressive_prefix_gripper_target_token_logratio_arm_v3` |
| Target token | `31744` |
| Target class | `CLIP_MEDIATED_OPEN` |
| Num backwards | `20` |
| Num loss forwards | `21` |
| Adv inputs present | `True` |
| x_adv is none | `True` |
| action_adv is none | `True` |
| Processor Linf | `0.023529052734375` |

## Gate Classification

| Gate | Status | Evidence |
| --- | --- | --- |
| Infra | `PASS` | strict route, no fallback, exact 7 tokens, score invariant, budget valid |
| Target | `PASS` | TRUE_PGD emitted `31744` |
| RAND20 margin | `PASS` | `29.937013626098633 > 0.25` |
| Shuffled-gradient margin | `PASS` | `29.937013626098633 > -0.5` |
| Arm selectivity | `FAIL` | TRUE_PGD arm match `2/6`, below `>=5/6` gate |

Final class: `TARGET_ONLY_ARM_FAIL`.

## Artifact Hashes

| File | SHA256 |
| --- | --- |
| `m3_step78_condition_results.csv` | `e279a315d6eb3319cee2a9253c0d4f1db18e1c14b460b422a76055efca9875e5` |
| `m3_step78_route_audit.csv` | `1b3780820f9babc27e02530b6627c7c7013fbe5373b4608ecce70008322670af` |
| `m3_step78_candidate_controls.csv` | `f2f7266fc3842fb187f9747606d69f660a15f1a24e7513da2dcd26b1b5e9bd1d` |
| `m3_step78_manifest.csv` | `0150b24be57ffe3b602e00623d87f80f7145afe816f99de2d1375701ed52bee6` |
| `m3_step78_zero_step_preflight.json` | `de753baea3c12399afaf433636a662f5eb9a2a5868c57974c5520dfff3af32bc` |
| `m3_step78_canary_debug.json` | `08c4b08aa6ad3b0fff122f33691bb0ed7c6d5cd164488445fb2b5a7f9306c764` |

## Local Tables

- `tables/m3_logratio_arm_v3_seed82_condition_results.csv`
- `tables/m3_logratio_arm_v3_seed82_route_audit.csv`
- `tables/m3_logratio_arm_v3_seed82_candidate_controls.csv`
- `tables/m3_logratio_arm_v3_seed82_manifest.csv`
- `tables/m3_logratio_arm_v3_seed82_gate_summary.csv`
- `tables/m3_logratio_arm_v3_seed82_claim_matrix.csv`

## Allowed Claim

On the Tomato state0 step78 development frame, seed82 TRUE_PGD with the
arm-constrained log-ratio v3 objective emitted `31744` and produced a larger
official target margin than the selected RAND20 and shuffled-gradient controls.

## Forbidden Claim

Do not claim seed82 full selective replication, gripper-selective control,
multi-frame robustness, closed-loop Layer3, physical gripper disruption, task
failure, held-out transfer, or general Layer3 success. RAND20 also emitted
`31744`, so do not claim random had no target effect.

## Stop Decision

Stop before fixed-frame panel and before LIBERO rollout. The fresh seed82
replication failed the arm gate. Any further attempt to change objective,
weight, target, epsilon, frame, arm gate, or controls must be preregistered as a
new method version rather than treated as a continuation of v3.
