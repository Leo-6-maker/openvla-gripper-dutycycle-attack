# M3 log-ratio arm-v3 seed82 replication preregistration

## Decision

`CONDITIONAL_GO`

This preregistration freezes a single fresh-seed fixed-frame replication of the
M3 arm-constrained log-ratio objective. It is a protocol addendum after the
seed81 development canary and before any fixed-frame panel, LIBERO closed-loop
rollout, rescue, or held-out transfer.

## Frozen Parent

| Field | Value |
| --- | --- |
| Parent head | `638a97962a038992410c426941291f8a90360d0c` |
| Branch | `exp/m3-arm-v3-fresh-seed82-canary-20260615` |
| Config | `configs/m3_step78_true_pgd_31744_logratio_arm_v3.yaml` |
| Config SHA256 | `e026f0a98b3b0b7ecea4fe55e29d7406c8fd8d6bb6875e46e70a37c7c49f4e39` |
| Frozen input dir | `/data/liuyu/outputs/m3_step78_true_pgd_20260614/capture_step78_f18537d_r2` |
| Python env | `/data/aviary/envs/openvla_official_libero_20260525` |
| Preferred GPU topology | `CUDA_VISIBLE_DEVICES=2,6` |

## Authorized Run

| Field | Value |
| --- | --- |
| Stage | `M3_ARM_V3_FRESH_SEED82_FIXED_FRAME_REPLICATION` |
| Suite | `libero_object` |
| Task | `tomato_sauce` |
| State | `0` |
| Absolute frame | `78` |
| Attack seed | `82` |
| Conditions | `CLEAN`, `PGD_DELTA0`, `TRUE_PGD`, `RAND20`, `SHUFFLED_GRAD_PGD20` |
| Target token | `31744` |
| Target execution class | `CLIP_MEDIATED_OPEN` |
| Epsilon | `6/255` |
| PGD steps | `20` |
| Step size | `1.5 * epsilon / 20` |
| Arm preserve weight | `0.5` |
| Arm gate | `>= 5/6` actual clean generated arm-prefix match |
| RAND count | `20` |
| Shuffled-gradient mode | `permute` |

No hyperparameter, target, frame, seed, control-count, objective, epsilon, step
size, arm weight, model, or preprocessing change is authorized by this
preregistration.

## Required Preflight

Before the canary, run `preflight_zero_step` as an independent job and inspect
the result. The canary is authorized only if:

- clean status is not `SURROGATE_OFFICIAL_SCORE_PATH_MISMATCH`;
- delta0 status is not `SURROGATE_OFFICIAL_SCORE_PATH_MISMATCH`;
- clean exact tokens match the frozen step78 input;
- clean gripper token is `31872`;
- clean arm prefix matches the frozen input;
- score invariant passes or is explicitly tie-aware.

## Gate

`FULL_SELECTIVE_REPLICATION` requires all of:

- strict route passes;
- `fallback_used=false`;
- resolved adapter is `TokenPrefixPGDAttacker`;
- attack method starts with `token_prefix_pgd`;
- directional loss is available;
- target token is `31744`;
- target class is `CLIP_MEDIATED_OPEN`;
- `adv_inputs` is present;
- `x_adv is None`;
- `action_adv is None`;
- `num_backwards=20`;
- exact official generation has 7 new tokens;
- official score invariant passes or is explicitly tie-aware;
- processor-space Linf after cast is `<= 6/255`;
- TRUE_PGD gripper token is `31744`;
- TRUE_PGD actual clean arm-prefix match is `>= 5/6`;
- TRUE_PGD official target margin is greater than selected RAND20 margin;
- TRUE_PGD official target margin is greater than SHUFFLED_GRAD_PGD20 margin.

## Result Classes

- `FULL_SELECTIVE_REPLICATION`: target, arm, RAND20, shuffled-gradient, and
  infrastructure gates all pass.
- `TARGET_ONLY_ARM_FAIL`: TRUE_PGD emits `31744`, but arm match is `< 5/6`.
- `TARGET_AND_ARM_BUT_RANDOM_NOT_BEATEN`: target and arm pass, but TRUE_PGD
  margin is `<=` selected RAND20 margin.
- `SHUFFLED_NOT_BEATEN`: TRUE_PGD margin is `<=` shuffled-gradient margin.
- `OFFICIAL_TRANSFER_FAIL`: surrogate improves but official generation does
  not.
- `NO_TARGET_FLIP`: TRUE_PGD final gripper token is not `31744`.
- `INFRA_INVALID`: route, hash, budget, exact-token, score-invariant, or input
  provenance checks fail.

If RAND20 also emits `31744`, that is not automatically a failure. The allowed
comparison is margin and arm-prefix selectivity, not uniqueness of token
flipping.

## Allowed Claim If Gate Passes

The M3 arm-constrained log-ratio v3 fixed-frame feasibility result replicated
on a fresh PGD random start for the Tomato state0 step78 development frame:
TRUE_PGD emitted `31744`, preserved the actual clean generated arm prefix, and
beat the selected RAND20 and shuffled-gradient controls on official target
margin.

## Forbidden Claims

Do not claim:

- multi-frame robustness;
- LIBERO closed-loop effect;
- physical gripper effect;
- task failure;
- held-out transfer;
- detector/selector success;
- general Layer3 success;
- that only TRUE_PGD induced `31744`;
- that random had no target effect;
- that historical Tomato legacy results were true-PGD evidence.

## Stop Rule

After exactly one seed82 canary, archive immutable artifacts and stop for audit.
An infrastructure failure may be rerun with the same configuration only after
the failed artifacts are retained and the root cause is recorded. A scientific
failure must not be retried by changing seed, GPU topology, target, frame,
objective, epsilon, arm gate, or controls.
