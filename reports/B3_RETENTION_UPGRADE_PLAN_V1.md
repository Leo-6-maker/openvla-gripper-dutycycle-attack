# B3-Retention upgrade plan — review draft

Status: `PREPARATION_ONLY / FORMAL_TRAINING_HOLD / ATTACK_HOLD`

This plan is intentionally separate from PR #74. It does not modify the active
Official CLEAN collector, rewrite CLEAN_2000 artifacts, launch GPU work, or
promote a detector checkpoint.

## Decision

The detector target is narrowed from task/object/target semantics to
robot-centric gripper-retention continuation:

> Given causal robot/action history, predict whether the current retention
> event can continue for an exact ten-step interval without a clean-policy
> release transition.

This is not an attack-vulnerability label. A downstream canary must establish
whether an intervention at the predicted interval changes task outcomes.

## Immutable source boundary

The active CLEAN artifacts remain the source of record. `OFFICIAL_25D_V1` and
the recorded 9D clean policy-intent stream are not modified in place. The
offline rebuilder consumes raw fields already persisted in the artifacts and
emits `B3_RETENTION_DERIVED_FEATURES_V1` with source-artifact and rebuilder
hashes.

Student inputs remain only:

```text
features_25d
clean_policy_intent_9d
```

Event IDs, future labels, contact evidence, anonymous object state, and attack
outcomes remain Teacher/audit-only. If derived event features are ever added
to the Student, that requires a new input schema and a new parity audit.

## Event-level Teacher

The unit is a retention event, not a LIBERO task stage:

```text
OPEN -> CLOSE onset -> supported retention -> release -> next event
```

Event boundaries use hysteresis. Default preparation values are three
consecutive CLOSE steps and three consecutive OPEN steps; the final values may
only be selected from FIT-TRAIN/FIT-DEV and must be frozen before CAL/CHECK.

Each event receives:

- event ordinal and start/end steps;
- close and release onset;
- event-local EEF displacement and path length;
- qpos/opening stability;
- `retention_continuation_t10`;
- `retention_unknown_mask` and evidence provenance.

Incomplete future windows, invalid contact capture, and missing required raw
fields are masked/unknown, never silently converted to negative labels.

## Detector and scheduler

The proposed four heads are:

- `grasp_support`: auxiliary training head;
- `retention_active`: runtime primary head;
- `retention_continuation_t10`: runtime primary head;
- `release_imminent`: runtime primary negative guard.

`window_start` is derived by the scheduler, not learned as a fifth head.

The event tracker and attack scheduler are independent. The tracker can follow
multiple L10 events. The scheduler waits for the first event satisfying the
three primary probabilities and 2-of-3 persistence, emits exactly one T10
attack, then latches for the remainder of the episode.

The GRU hidden state resets at episode boundaries only. It does not reset on an
event boundary. Any training TBPTT implementation may detach hidden state, but
must not introduce rolling-window semantics or event resets at inference.

## L10-specific validation

All metrics are stratified by `event_ordinal = 0`, `1`, and `2+` where data
support the bucket. Report event counts, valid T10 counts, containment,
false-window rate, and timing error. L10 cannot pass because event ordinal 0
passes while later events are absent or collapsed.

Sampling is hierarchical:

```text
suite -> task -> episode -> event -> positive/negative anchor
```

Negative strata include pre-close, static close, unstable support,
retention-adjacent, release-imminent, and post-release. `T10_INCOMPLETE` is
masked rather than treated as negative.

## Evidence gates

### R0 — artifact reconstruction

Verify checksum closure, contiguous steps, finite raw fields, official
identity, and sufficient EEF/qpos/action data. Any missing source remains
`REBUILD_HOLD`.

### R1 — CPU multi-event tests

Cover single event, two events, OPEN jitter, release, regrasp, incomplete T10,
invalid contact, and non-contiguous steps. The preparation branch includes
these tests without importing MuJoCo or OpenVLA.

### R2 — old-label agreement

Use the old privileged labels as a reference, not gold. Freeze thresholds
before reading the comparison results. Report event recall, T10 containment,
timing error, release overlap, false-window rate, and per-suite/L10 ordinal
breakdown.

### R3 — trajectory audit

Review a task-balanced event sample using actions, EEF, qpos, and valid contact
evidence. This checks whether the weak Teacher has the intended robot-centric
meaning; it does not grant semantic object/target claims.

### R4 — stateful parity and CHECK

Compare offline full-sequence and online stepwise outputs for logits,
probabilities, event IDs, FSM state, persistence, trigger step, and one-shot
behavior. Include L10 episodes with at least two detected events.

### R5 — 48-cell downstream canary

Only after R0–R4 pass. The canary decides whether predicted retention windows
are useful for the attack mechanism. Agreement with an old Teacher alone cannot
authorize formal attack.

## Explicit holds

- No change to the active CLEAN collector or its 25D semantics.
- No silent mixing of old and derived feature schemas.
- No RGB/language/object identity input to the Student.
- No CAL/CHECK tuning from attack outcomes.
- No formal B3 training, attack canary, or main-table attack from this plan.

## Expected review questions

1. Does the rebuilder use only persisted causal source fields and preserve
   source artifact hashes?
2. Are unknown/masked windows separated from true negatives?
3. Are event tracker and scheduler independently testable?
4. Does L10 retain later-event coverage rather than only first-event coverage?
5. Is `retention_continuation_t10` kept separate from causal attack claims?

The branch is ready for CPU review only. A passing PR does not mean that B3
training or formal attack execution is authorized.
