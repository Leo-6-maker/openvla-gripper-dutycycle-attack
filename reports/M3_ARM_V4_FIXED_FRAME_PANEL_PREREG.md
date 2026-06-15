# M3 arm-v4 fixed-frame panel preregistration

## Decision

`PREREG_REPAIRED_EXECUTABLE_GATE_ONLY`

This commit repairs the panel preregistration protocol and adds CPU-tested
executable panel gate wiring. It does not authorize panel GPU execution, LIBERO
closed-loop rollout, production-runner transfer, critical-close rescue,
held-out transfer, or Layer1/2 selector attack.

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

## Capture Protocol

Before any panel attack run, capture all panel inputs through one deterministic
Tomato state0 clean replay:

```text
70, 72, 74, 76, 78, 80, 82, 84, 86
```

For every captured frame, save or manifest:

- raw runner input image hash;
- processed tensor hash;
- prompt and prompt-token hash;
- clean exact 7 generated tokens;
- clean generated arm prefix;
- clean gripper token;
- score row / score invariant;
- runner, model, preprocessing, worktree, and GPU provenance.

Frame 78 is a positive-control input-chain parity check against the previously
frozen step78 input. If raw hash, processed tensor hash, prompt-token hash,
clean exact tokens, clean arm prefix, or clean gripper token differs, mark:

```text
POSITIVE_CONTROL_INPUT_MISMATCH
INFRA_INVALID
STOP
```

Do not run the main panel after a positive-control input mismatch.

The executable panel entry point must perform this capture as one trajectory.
Per-frame replays are not valid panel input capture.

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

## Clean Eligibility

Before running any attack on a frame, capture or load the clean fixed-frame
input and verify:

- raw observation comes from runner input, not video or overlay;
- exact clean official generation has 7 action tokens;
- score invariant passes or has an explicit tie-aware status;
- clean generated arm prefix is available;
- target score row is available;
- processor input and raw image hashes are recorded;
- clean official gripper token is exactly `31872`.

Clean-context labels:

| Condition | Label | Main denominator treatment |
| --- | --- | --- |
| clean gripper token `31872` | `CLEAN_ELIGIBLE` | attack may run |
| clean gripper token `31744` | `CLEAN_ALREADY_TARGET` | ineligible; no replacement |
| clean gripper token other than `31872` or `31744` | `CLEAN_NOT_CLOSE` | ineligible; no replacement |
| exact-token, score, prefix, or provenance failure | `CLEAN_CONTEXT_INELIGIBLE` | ineligible or infra invalid as appropriate |

If a frame is ineligible, do not replace it with another frame. Report it as an
ineligible panel cell.

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

## Control Infeasibility Semantics

Frame comparison semantics are frozen as follows:

| TRUE candidate | control candidate | Result |
| --- | --- | --- |
| infeasible | any | frame fails |
| feasible | infeasible | TRUE wins that control, but paired margin is blank |
| feasible | feasible | compare finite official target margins |
| infeasible | infeasible | frame fails |

Automatic wins from an infeasible control must be counted and reported
separately. They do not contribute finite paired margins.

Median paired margin uses only frames where TRUE and the corresponding control
both have feasible selected candidates. Each control requires at least 4 finite
paired frames. The finite median paired margin must be positive for both RAND21
and shuffled-gradient controls.

## Seeds

The first panel seed is frozen to exactly:

```text
attack_seed = 85
```

There is exactly one seed85 run. Do not rerun seed85. Seed86 may only be run
after seed85 artifacts are reviewed and a separate authorization is given.

This preregistration still does not authorize panel GPU execution.

The executable panel entry point must reject any seed other than `85`.

## Frame Full Selective Pass

A main-denominator frame passes only if all conditions are jointly true on that
same frame:

```text
FRAME_FULL_SELECTIVE_PASS =
    clean status is CLEAN_ELIGIBLE
    AND TRUE candidate exists
    AND TRUE token == 31744
    AND TRUE arm match >= 5/6
    AND TRUE budget / route / score invariant is valid
    AND TRUE margin > RAND margin, or RAND has no feasible selected candidate
    AND TRUE margin > shuffled margin, or shuffled has no feasible selected candidate
```

Separate sub-metric counts across different frames cannot satisfy the panel
gate.

## Panel Aggregate Gate

For the main denominator of 8 non-development frames, a single-seed panel pass
requires all of:

- exactly 8 main-denominator frames are reported;
- no `INFRA_INVALID` main-denominator frame;
- no more than 1 clean-ineligible main-denominator frame;
- `FRAME_FULL_SELECTIVE_PASS >= 6/8`;
- RAND finite paired margin count `>=4`;
- shuffled finite paired margin count `>=4`;
- median finite TRUE minus RAND official target margin `>0`;
- median finite TRUE minus shuffled official target margin `>0`.

Step78 may be shown as a positive-control row but must not affect the aggregate
gate.

## Result Classes

- `PANEL_PREREG_REPAIRED_EXECUTABLE_GATE_ONLY`: this commit.
- `POSITIVE_CONTROL_INPUT_MISMATCH`: captured step78 does not match frozen
  step78 input.
- `CLEAN_ALREADY_TARGET`: clean token is already `31744`.
- `CLEAN_NOT_CLOSE`: clean token is not `31872`.
- `CLEAN_CONTEXT_INELIGIBLE`: clean input or clean decode is invalid.
- `FRAME_FULL_SELECTIVE_PASS`: one main-denominator frame satisfies the joint
  frame gate.
- `PANEL_SINGLE_SEED_PASS`: one authorized panel seed passes the aggregate gate.
- `PANEL_SINGLE_SEED_FAIL`: one authorized panel seed fails the aggregate gate.
- `INFRA_INVALID`: route, budget, exact-token, score invariant, provenance, or
  candidate-count checks fail.

## Provenance Fixes Required Before Panel GPU

The panel runner must write:

- `dirty_status` as `CLEAN`, `DIRTY:...`, or `GIT_STATUS_UNAVAILABLE`;
- populated `model_fingerprint`;
- GPU UUID or an equivalent `nvidia-smi` snapshot reference;
- capture/preflight artifact hash manifests.

If these cannot be written, the run must be treated as provenance-invalid.

The executable panel entry point must fail closed on dirty worktrees,
unavailable GPU snapshots, missing model fingerprints, missing artifact hash
manifests, or a pre-existing one-shot sentinel.

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

After this protocol repair commit, stop for review. Do not launch panel GPU
jobs until the repaired preregistration and CPU tests are reviewed and explicit
panel execution authorization is given.
