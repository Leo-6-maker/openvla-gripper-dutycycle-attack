# M3 arm-v4 hard feasible trajectory selection preregistration

## Decision

`PREREG_ONLY`

This preregistration closes arm-v3 as non-robust and freezes the next method
attempt: M3 arm-v4 hard feasible trajectory selection. No GPU run is authorized
until this preregistration is committed, pushed, and CPU gates pass.

## Prior State

| Item | Status |
| --- | --- |
| arm-v3 seed81 | `FULL_TOKEN_FLIP_FIXED_FRAME` with arm `6/6` |
| arm-v3 seed82 | `TARGET_ONLY_ARM_FAIL` with arm `2/6` |
| arm-v3 conclusion | `CLOSED_AS_NONROBUST` |
| Multi-frame panel | `NOT_AUTHORIZED` |
| LIBERO closed-loop rollout | `NOT_AUTHORIZED` |

Seed82 showed that the weighted final-iterate objective can drive token `31744`
and beat RAND20/shuffled controls on official target margin while still
destroying the actual clean generated arm prefix. Therefore v4 changes only the
selection protocol: from weighted final iterate to hard arm-feasible trajectory
selection.

## Frozen Inputs

| Field | Value |
| --- | --- |
| Task | `tomato_sauce` |
| State | `0` |
| Frame | absolute step `78` |
| Frozen input dir | `/data/liuyu/outputs/m3_step78_true_pgd_20260614/capture_step78_f18537d_r2` |
| Model | `/data/aviary/models/openvla/openvla-7b-finetuned-libero-object` |
| Python env | `/data/aviary/envs/openvla_official_libero_20260525` |
| Target token | `31744` |
| Target class | `CLIP_MEDIATED_OPEN` |
| Epsilon | `6/255` |
| PGD steps | `20` |
| Candidate count per condition | `21` (`delta0` plus iterations `1..20`) |
| Arm gate | actual clean generated arm-prefix match `>= 5/6` |
| Token gate | official generated gripper token is `31744` |
| RAND candidates | `21` total candidates in the same processor space |
| Shuffled candidates | `21` total shuffled-gradient trajectory candidates |
| Fresh attack seeds | `83`, then `84` only if seed83 passes |

The frame, target, epsilon, PGD step count, arm gate, RAND count, shuffled
control, model, and preprocessing are fixed. Any change requires a new method
version and a new preregistration.

## Method

`M3_ARM_V4_HARD_FEASIBLE_SELECTION`

For TRUE_PGD, RAND, and shuffled-gradient controls, evaluate exactly 21
candidates:

1. initial candidate (`delta0` for PGD/shuffled; candidate 0 for RAND);
2. candidates after each of 20 update or random-search steps.

Each candidate must be official-decoded under the same fixed-frame runner and
strict route contract. For each condition, apply the same selection rule:

1. filter to candidates with actual clean generated arm-prefix match `>= 5/6`;
2. filter to candidates whose official gripper token is exactly `31744`;
3. among feasible candidates, select the one with maximum official target
   margin;
4. break ties by smaller processor-space Linf;
5. break remaining ties by earlier candidate index.

If a condition has no feasible candidate, it is recorded as
`NO_FEASIBLE_CANDIDATE`. It is forbidden to fall back to an arm-breaking
candidate.

## Required Conditions

For each fresh seed:

- `PGD_DELTA0`;
- `TRUE_PGD_TRAJECTORY21_SELECTIVE`;
- `RAND21_SELECTIVE`;
- `SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE`.

All conditions must record:

- route metadata;
- candidate index;
- candidate delta hash;
- processor input hash;
- official exact 7 tokens;
- official gripper token;
- official target margin;
- actual clean arm-prefix match;
- score invariant status;
- processor-space Linf;
- budget correction count;
- selection reason.

## Seed Order

Seed83 is the first fresh v4 canary.

- If seed83 fails any target, arm, RAND21, shuffled21, route, budget, exact-token,
  or score-invariant gate, stop and report.
- Only if seed83 passes all gates may seed84 run with the identical frozen
  configuration.
- If both seed83 and seed84 pass all gates, stop for audit. Do not start a
  panel or rollout.

## Success Class

`FULL_SELECTIVE_V4_REPLICATION` requires both seed83 and seed84 to pass:

- strict route and no fallback;
- exact 7 official tokens for selected candidates;
- official score invariant pass or explicit tie-aware pass;
- processor-space Linf `<= 6/255`;
- TRUE_PGD selected candidate emits `31744`;
- TRUE_PGD selected candidate arm match `>= 5/6`;
- TRUE_PGD selected official target margin is greater than selected RAND21
  official target margin;
- TRUE_PGD selected official target margin is greater than selected
  shuffled-gradient official target margin.

## Failure Classes

- `SEED83_FAIL_STOP`: seed83 fails any required gate.
- `SEED84_FAIL_STOP`: seed83 passes but seed84 fails any required gate.
- `NO_FEASIBLE_PGD_CANDIDATE`: no TRUE_PGD candidate satisfies both arm and
  token gates.
- `RANDOM_NOT_BEATEN`: TRUE_PGD feasible candidate exists but does not beat
  RAND21 selected margin.
- `SHUFFLED_NOT_BEATEN`: TRUE_PGD feasible candidate exists but does not beat
  shuffled-gradient selected margin.
- `INFRA_INVALID`: route, budget, exact-token, score-invariant, or provenance
  checks fail.

## Read-Only v3 Trajectory Audit

Before running v4, perform a read-only audit of existing seed81 and seed82
artifacts to check whether any intermediate iterate appeared to satisfy the arm
gate before the final iterate broke the arm prefix.

This audit is explanatory only. It cannot reclassify seed82 as success and
cannot alter v3 outcomes:

- seed81 remains the development success;
- seed82 remains `TARGET_ONLY_ARM_FAIL`;
- arm-v3 remains `CLOSED_AS_NONROBUST`.

## Allowed Claims

If both fresh seeds pass, the only allowed claim is:

`M3 arm-v4 hard feasible selection achieved fixed-frame target-token control
with actual arm-prefix preservation and beat matched RAND21 and shuffled
trajectory controls on the Tomato step78 development frame for seeds 83 and
84.`

## Forbidden Claims

Do not claim:

- multi-frame robustness;
- LIBERO closed-loop effect;
- physical gripper disruption;
- task failure;
- held-out transfer;
- detector/selector success;
- general Layer3 success;
- that historical Tomato legacy results were true-PGD evidence;
- that v3 seed82 succeeded.

## Stop Rule

After the read-only v3 trajectory audit and any authorized v4 seed83/84 runs,
stop for review. Panel, rollout, rescue, held-out, and Layer1/2 proposal attack
experiments remain prohibited.
