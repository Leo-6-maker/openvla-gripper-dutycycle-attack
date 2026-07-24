# C2g Counterfactual Manifest Specification

Date: 2026-07-10

Status: `PASS_STATIC_SCHEMA`; no simulator replay was performed.

## Comparison tiers

`TIER_A_MATCHED_ACTION_SHORT_HORIZON` restores a clean snapshot and compares recorded clean actions with the same actions after replacing only the gripper command for the frozen attack horizon.

`TIER_B_CLOSED_LOOP_CONTINUATION` restores the same snapshot and compares independent closed-loop clean and force-open continuations. Tier B must explicitly enable closed-loop continuation; Tier A must not.

Both tiers bind episode/task/state/step identity, candidate reason, snapshot and restore hashes, parity metrics, action source/alignment, exact command values, continuation hashes, effect thresholds, code/simulator/model provenance, deterministic seed, and creation time.

## Known and unknown rules

A known label requires:

- complete snapshot fields;
- restore parity pass;
- matched-action alignment pass;
- exact attack delivery (`delivered_attack_steps == attack_horizon`);
- complete continuation hashes;
- clean full git commit and model/processor manifest SHA256 values.

Any failure is masked with an explicit unknown reason. Supported reasons include `SNAPSHOT_INCOMPLETE`, `RESTORE_MISMATCH`, `ACTION_ALIGNMENT_FAILED`, `INCOMPLETE_ATTACK_DELIVERY`, `TARGET_GROUNDING_FAILED`, `AMBIGUOUS_EFFECT`, and `NOT_REPLAYED`. Unknown rows are never negative labels.

## Causal labels

From the same restored state, compare clean and T10 command-open continuations:

- `y_contact_loss`: clean preserves target-object grasp/contact while command-open loses it;
- `y_object_drop`: command-open produces object-height/contact degradation beyond the frozen tolerance;
- `y_progress_regression`: target-relative progress degrades beyond the frozen margin;
- `y_success_flip`: clean succeeds and command-open fails;
- `y_release_safe`: target-relative evidence says opening is benign or required;
- `y_cmdopen_vulnerable`: at least one frozen causal harm rule is positive and release-safe is false.

Known negatives require valid replay and all frozen benign tolerances. Missing clean success, unreplayed windows, ambiguity, or partial evidence is unknown.

## Future matched-payload online control

The future primary comparison is:

```text
TRUE_CMDOPEN_T10_C2G
vs
CTRL_RANDOM_TIME_CMDOPEN_T10
```

Both conditions must use the identical force-open command and T10 horizon. Only deterministic trigger timing may differ. CLEAN and privileged `ORACLE_CMDOPEN_T10` are additional conditions; `RAND_ACTION_NOISE_T10_C2F` is optional placebo evidence, not the matched causal control and not D7 image-space PGD.

Required analysis includes identical parent/init state, pre-trigger trace parity, executed action evidence, exact delivery, paired contingency, exact McNemar/binomial analysis, and separate pilot/replication cohorts.

```text
COUNTERFACTUAL_MANIFEST_SCHEMA = PASS_STATIC
UNKNOWN_MASKING = PASS_STATIC
MATCHED_PAYLOAD_CONTROL_SPEC = PASS_SPEC
REPLAY_ENGINE = NOT_IMPLEMENTED
REPLAYS_LAUNCHED = 0
```
