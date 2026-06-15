# M3 arm-v4 seed83/84 fixed-frame results

## Result

`FULL_SELECTIVE_V4_REPLICATION`

Seeds `83` and `84` both passed the preregistered arm-v4 hard feasible
trajectory-selection gate on the Tomato state0 step78 development frame.

This remains a fixed-frame result only. No multi-frame panel, LIBERO closed-loop
rollout, critical-close rescue, task-effect experiment, held-out transfer, or
Layer1/2 selector attack was launched.

## Fixed Protocol

| Field | Value |
| --- | --- |
| Commit | `b2c1dc0abe20dde03b308301060e7ba3637658e4` |
| Config | `configs/m3_step78_true_pgd_31744_logratio_arm_v4.yaml` |
| Config SHA256 | `2dcef93bf2decf742e0c98f321267ae665b57890f3ab03088dfda3686ae8a2a8` |
| Task/state/frame | `tomato_sauce / state0 / absolute step78` |
| Frozen input | `/data/liuyu/outputs/m3_step78_true_pgd_20260614/capture_step78_f18537d_r2` |
| Target token | `31744` |
| Target class | `CLIP_MEDIATED_OPEN` |
| Epsilon | `6/255` |
| PGD steps | `20` |
| Candidate count | `21` per condition |
| Arm gate | actual clean generated arm-prefix match `>=5/6` |
| GPU mapping | `CUDA_VISIBLE_DEVICES=1,0` |
| Python env | `/data/aviary/envs/openvla_official_libero_20260525` |

## Preflight

Both seeds passed independent zero-step preflight:

| Seed | Clean status | Delta0 status | Output |
| ---: | --- | --- | --- |
| 83 | `SURROGATE_OFFICIAL_SCORE_PATH_MATCH` | `SURROGATE_OFFICIAL_SCORE_PATH_MATCH` | `/data/liuyu/outputs/m3_arm_v4_step78_seed83_preflight_b2c1dc0` |
| 84 | `SURROGATE_OFFICIAL_SCORE_PATH_MATCH` | `SURROGATE_OFFICIAL_SCORE_PATH_MATCH` | `/data/liuyu/outputs/m3_arm_v4_step78_seed84_preflight_b2c1dc0` |

## Selected Candidates

| Seed | Condition | Candidate | Token | Margin | Arm | Linf |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 83 | TRUE_PGD | 15 | 31744 | 20.249820709228516 | 6/6 | 0.023529052734375 |
| 83 | RAND21 | 0 | 31744 | 0.5 | 6/6 | 0.02350616455078125 |
| 83 | SHUFFLED21 | 4 | 31744 | 0.0 | 6/6 | 0.023529052734375 |
| 84 | TRUE_PGD | 16 | 31744 | 17.999996185302734 | 6/6 | 0.023529052734375 |
| 84 | RAND21 | 4 | 31744 | 0.25 | 6/6 | 0.023529052734375 |
| 84 | SHUFFLED21 | 0 | 31744 | 0.5 | 6/6 | 0.023529052734375 |

## Candidate Audit

| Seed | Condition | Candidates | Feasible | Selected | Score invariant failures | Max Linf |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 83 | TRUE_PGD | 21 | 7 | 1 | 0 | 0.023529052734375 |
| 83 | RAND21 | 21 | 15 | 1 | 0 | 0.023529052734375 |
| 83 | SHUFFLED21 | 21 | 6 | 1 | 0 | 0.023529052734375 |
| 84 | TRUE_PGD | 21 | 4 | 1 | 0 | 0.023529052734375 |
| 84 | RAND21 | 21 | 10 | 1 | 0 | 0.023529052734375 |
| 84 | SHUFFLED21 | 21 | 12 | 1 | 0 | 0.023529052734375 |

Control status is explicit: RAND21 and shuffled-gradient both had feasible
target+arm candidates for both seeds. The v4 pass is therefore due to TRUE_PGD
having a larger selected official margin, not because the controls lacked
feasible candidates.

## Route Gate

| Seed | Condition | Fallback | Backwards | Loss forwards | Generation forwards | Trajectory candidates | Linf |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 83 | TRUE_PGD | `False` | 20 | 21 | 21 | 21 | 0.023529052734375 |
| 83 | SHUFFLED21 | `False` | 20 | 21 | 21 | 21 | 0.023529052734375 |
| 84 | TRUE_PGD | `False` | 20 | 21 | 21 | 21 | 0.023529052734375 |
| 84 | SHUFFLED21 | `False` | 20 | 21 | 21 | 21 | 0.023529052734375 |

## Output Directories

- Seed83 preflight: `/data/liuyu/outputs/m3_arm_v4_step78_seed83_preflight_b2c1dc0`
- Seed83 canary: `/data/liuyu/outputs/m3_arm_v4_step78_seed83_b2c1dc0`
- Seed84 preflight: `/data/liuyu/outputs/m3_arm_v4_step78_seed84_preflight_b2c1dc0`
- Seed84 canary: `/data/liuyu/outputs/m3_arm_v4_step78_seed84_b2c1dc0`

## Artifact Hashes

| File | SHA256 |
| --- | --- |
| seed83 selected results | `929c035916e9b974c6c9e15d85b6242dfb543b7811e028957631ae280f48fc4f` |
| seed83 candidate audit | `1093161ccf8b241d44f3be0857f0506e46dcefef0c2756d7f42a15ef3131d078` |
| seed83 route audit | `e72ef896f2d7a16372797df7aca94d1d597c1062fd88fb9db58cbd902337f769` |
| seed83 preflight JSON | `a71881adfaa409a4d442b667b832474baabd19e05bc1dcfb7a4610c9b98a7fdd` |
| seed84 selected results | `ffad41e9ada7854105d41e05bcf50c6b73dc7a2dce9b75983b4b5835a02d4c81` |
| seed84 candidate audit | `7d2799c1e6666aed209f54f8914c0a4ed448c6eda728534e07ec0e319a7397a4` |
| seed84 route audit | `08952d639c4cd6ef1c617543b217809ca470a02417cb0147cdd1dbe9ff8571b0` |
| seed84 preflight JSON | `3d63d5792d7d71be8b0d7b5c47a625fcb1d3ce5af4c3801ef7a1fb9b96c8159c` |

## Local Tables

- `tables/m3_arm_v4_seed83_selected_results.csv`
- `tables/m3_arm_v4_seed83_candidate_audit.csv`
- `tables/m3_arm_v4_seed83_route_audit.csv`
- `tables/m3_arm_v4_seed83_manifest.csv`
- `tables/m3_arm_v4_seed84_selected_results.csv`
- `tables/m3_arm_v4_seed84_candidate_audit.csv`
- `tables/m3_arm_v4_seed84_route_audit.csv`
- `tables/m3_arm_v4_seed84_manifest.csv`
- `tables/m3_arm_v4_seed83_84_summary.csv`
- `tables/m3_arm_v4_seed83_84_claim_matrix.csv`

## Allowed Claim

M3 arm-v4 hard feasible selection achieved fixed-frame target-token control with
actual clean arm-prefix preservation and beat matched RAND21 and shuffled
trajectory controls on the Tomato state0 step78 development frame for fresh
seeds `83` and `84`.

## Forbidden Claim

Do not claim multi-frame robustness, LIBERO closed-loop effect, physical gripper
disruption, task failure, held-out transfer, detector/selector success, or
general Layer3 success from this result.

## Stop Decision

Stop for audit. The next stage, if approved, must be a separately preregistered
fixed-frame panel or production-runner transfer gate. No panel or rollout was
launched in this stage.
