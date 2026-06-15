# M3 arm-v4 fixed-frame panel preregistration

## Decision

`PREREG_ONLY`

This commit preregisters the next fixed-frame panel after arm-v4 passed the
Tomato state0 step78 development frame for fresh seeds `83` and `84`.

No panel GPU run, LIBERO closed-loop rollout, production-runner transfer,
critical-close rescue, held-out transfer, or Layer1/2 selector attack is
authorized by this preregistration.

## Current State

| Item | Status |
| --- | --- |
| arm-v3 | `CLOSED_AS_NONROBUST` |
| arm-v4 single development frame | `FULL_SELECTIVE_V4_REPLICATION` |
| arm-v4 multi-frame robustness | `NOT_TESTED` |
| closed-loop Layer3 | `NOT_TESTED` |

The accepted arm-v4 result is a best-of-21 official-decode hard-feasible
fixed-frame search result. It is not a single final-iterate online attack and
not a closed-loop task result.

## Frozen Method

The panel must use the exact arm-v4 method and selection rule:

1. construct 21 candidates per condition;
2. official-decode every candidate;
3. filter to actual clean generated arm-prefix match `>=5/6`;
4. filter to official gripper token `31744`;
5. select maximum official target margin;
6. tie-break by lower processor-space Linf;
7. tie-break by earlier candidate index;
8. do not fall back to an arm-breaking candidate.

Frozen method fields:

| Field | Value |
| --- | --- |
| Target token | `31744` |
| Target class | `CLIP_MEDIATED_OPEN` |
| Epsilon | `6/255` |
| PGD steps | `20` |
| Candidate count | `21` per condition |
| Arm gate | actual clean generated arm-prefix match `>=5/6` |
| RAND control | `RAND21_SELECTIVE` |
| Shuffled control | `SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE` |
| Selection metric | official target margin after hard feasibility filters |
| Existing implementation commit | `98debf2aad97097a14861db97bd34d94042776a2` |

The objective, target token, epsilon, step count, candidate count, arm gate, and
selection rule must not change for the panel.

## Panel Frames

Task/state:

```text
tomato_sauce / state0
```

Panel main denominator uses the following non-development frames:

```text
70, 72, 74, 76, 80, 82, 84, 86
```

Development positive control:

```text
78
```

Step78 must be reported separately as a development positive control and must
not enter the panel main denominator.

## Frame Eligibility

Before running any attack on a frame, capture or load the clean fixed-frame
input and verify:

- raw observation comes from runner input, not video or overlay;
- exact clean official generation has 7 action tokens;
- score invariant passes or has an explicit tie-aware status;
- clean generated arm prefix is available;
- target score row is available;
- processor input and raw image hashes are recorded.

If a frame fails clean eligibility, mark it:

`CLEAN_CONTEXT_INELIGIBLE`

Do not replace it with another frame. Report it as an ineligible panel cell.

## Conditions Per Eligible Frame

Each eligible frame must run:

- `PGD_DELTA0`;
- `TRUE_PGD_TRAJECTORY21_SELECTIVE`;
- `RAND21_SELECTIVE`;
- `SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE`.

Each frame must report, per condition:

- number of candidates (`21` expected);
- feasible candidate count;
- selected candidate index;
- selected official gripper token;
- selected arm match;
- selected official target margin;
- selected Linf;
- score-invariant failure count;
- strict route status for TRUE_PGD and shuffled-gradient;
- route fallback status;
- backward count;
- generation forward count;
- artifact hashes.

Both selected-margin and feasible-candidate rate must be reported. Reporting
only the best selected margin is insufficient.

## Seeds

Panel seed policy is not authorized for execution by this preregistration.

Recommended first reviewable panel run, if later authorized:

```text
attack_seed = 85
```

Recommended replication run, if seed85 panel passes and a separate review
authorizes it:

```text
attack_seed = 86
```

Do not run seed85 or seed86 without explicit post-prereg authorization.

## Panel Aggregate Gate

For the main denominator of 8 non-development frames, a single-seed panel pass
requires all of:

- no `INFRA_INVALID` eligible frame;
- no more than 1 `CLEAN_CONTEXT_INELIGIBLE` frame;
- among eligible frames, TRUE_PGD selected candidate emits `31744` in at least
  6 frames;
- among eligible frames, TRUE_PGD selected candidate arm match is `>=5/6` in at
  least 6 frames;
- among eligible frames, TRUE_PGD selected official margin exceeds selected
  RAND21 margin in at least 6 frames;
- among eligible frames, TRUE_PGD selected official margin exceeds selected
  shuffled-gradient margin in at least 6 frames;
- median paired TRUE_PGD minus RAND21 official margin is positive;
- median paired TRUE_PGD minus shuffled-gradient official margin is positive.

Step78 may be shown as a positive-control row but must not affect the above
gate.

## Result Classes

- `PANEL_PREREG_ONLY`: this commit.
- `PANEL_CLEAN_CONTEXT_INELIGIBLE`: one or more preregistered frames fail clean
  input eligibility.
- `PANEL_SINGLE_SEED_PASS`: one authorized panel seed passes the aggregate gate.
- `PANEL_SINGLE_SEED_FAIL`: one authorized panel seed fails the aggregate gate.
- `PANEL_REPLICATION_PASS`: two authorized panel seeds pass after separate
  review.
- `PANEL_REPLICATION_FAIL`: second authorized panel seed fails.
- `INFRA_INVALID`: route, budget, exact-token, score invariant, provenance, or
  candidate-count checks fail.

## P1 Provenance Fix Required Before Panel GPU

Before any panel GPU execution, fix or otherwise fail-closed around the current
manifest gaps:

- `dirty_status` must record the actual worktree status;
- `model_fingerprint` must be populated;
- `gpu_query` must include GPU UUID or an equivalent `nvidia-smi` snapshot
  reference;
- preflight artifacts must have either committed summaries or hash manifests.

This preregistration records the requirement but does not modify the manifest
writer. A separate code commit is required before panel GPU execution.

## Allowed Claim If A Later Panel Passes

Only after an authorized panel run passes:

`arm-v4 hard feasible selection shows fixed-frame robustness across the
preregistered Tomato state0 non-development frame panel.`

## Forbidden Claims

Even if a future panel passes, do not claim:

- LIBERO closed-loop effect;
- physical gripper disruption;
- task failure;
- production-runner transfer;
- held-out transfer;
- detector/selector success;
- general Layer3 success.

## Stop Rule

After this preregistration, stop for review. Do not launch panel GPU jobs until
the manifest P1 issue is fixed and explicit panel execution authorization is
given.
