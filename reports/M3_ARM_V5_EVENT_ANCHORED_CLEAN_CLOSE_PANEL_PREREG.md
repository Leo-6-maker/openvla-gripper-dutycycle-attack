# M3 arm-v5 event-anchored clean-CLOSE panel preregistration

## Decision

```text
PREREG_ONLY
```

V5 replaces the failed absolute-frame denominator with clean-CLOSE event
anchors. This commit freezes the protocol only. It does not authorize GPU clean
capture, PGD, RAND, shuffled-gradient, seed85, seed86, LIBERO rollout, or any
Layer2-triggered attack.

## Base Finding

PR #24 closed the Tomato state0 absolute-frame panel because capture-only
preflight found only `2/8` main frames with clean CLOSE token `31872`; the other
`6/8` were already target token `31744`.

## State Pool

V5 freezes 20 task-state candidates:

```text
10 LIBERO Object tasks x 2 states per task
```

For each task, candidate states are `0..49` except known Layer3 development
states:

```text
butter: 2
cream_cheese: 2
tomato_sauce: 0
```

Each task takes the two smallest hashes:

```text
SHA256("M3_ARM_V5_CLOSE_PANEL|task|state")
```

The frozen table is:

```text
tables/m3_arm_v5_preregistered_state_pool.csv
```

No attack result, margin, detector output, teacher probability, task success,
or event count was used to choose the states.

## Clean-Only Event Selector

For each frozen task-state, V5.1 may run one clean trajectory and search for
the earliest clean CLOSE onset satisfying:

```text
min_step <= step <= max_step
exact official action tokens == 7
score invariant == PASS
current gripper token == 31872
previous step gripper token != 31872
```

Frozen limits:

```text
min_step = 0
max_step = 279
```

Each state contributes at most one event: the earliest qualifying event.

Forbidden during event selection:

- choosing the largest or easiest margin;
- choosing based on arm prefix convenience;
- choosing based on attack outcome;
- choosing multiple adjacent frames as independent samples;
- replacing an event after seeing attack results.

## Final Eight Inputs

After clean capture only, sort the 20 frozen task-state candidates by their
precomputed state hash and take the first 8 states that have a qualifying clean
CLOSE event.

If fewer than 8 states have a qualifying event:

```text
V5_CAPTURE_POOL_INSUFFICIENT
STOP
```

Once the eight exact inputs are frozen, no replacement is allowed.

## Phase Separation

```text
V5.1:
  clean capture only
  freeze exact 8 inputs
  no PGD/RAND/shuffled

review

V5.2:
  exact 8 frozen inputs
  one attack seed
  TRUE/RAND/shuffled
  no LIBERO rollout

review

V5.3:
  second seed only after explicit authorization
```

## Attack Method Frozen For Future V5.2

V5.2, if later authorized, must keep arm-v4 unchanged:

| Field | Value |
| --- | --- |
| Target token | `31744` |
| Target class | `CLIP_MEDIATED_OPEN` |
| Epsilon | `6/255` |
| PGD steps | `20` |
| Candidates | `21` per condition |
| Arm gate | actual clean generated arm-prefix match `>=5/6` |
| Controls | `RAND21_SELECTIVE`, `SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE` |
| Selection | official-decode hard-feasible maximum target margin |

No objective, epsilon, candidate budget, arm gate, frame count, or aggregate
threshold tuning is allowed in V5.0.

## Aggregate Gate Frozen For Future V5.2

For the eight frozen clean-CLOSE event inputs:

```text
infra invalid = 0
FRAME_FULL_SELECTIVE_PASS >= 6/8
RAND finite paired frames >= 4
shuffled finite paired frames >= 4
median TRUE-RAND official target margin > 0
median TRUE-shuffled official target margin > 0
```

## Allowed Claim If V5.0 Is Accepted

V5.0 defines an event-anchored clean-CLOSE panel protocol that avoids the
absolute-frame denominator failure found in PR #24.

## Forbidden Claims

Do not claim:

- V5 has eligible events before V5.1 clean capture;
- arm-v4 generalizes across states;
- TRUE_PGD beats random;
- closed-loop Layer3 is established;
- detector-selected Layer3 is established.

## Stop Rule

After this preregistration, stop for review. No V5 capture is authorized by
this commit.
