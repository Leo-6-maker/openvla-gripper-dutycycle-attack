# T1-D1M0 manual contact-validity rubric

This rubric is frozen before any candidate video pixels are opened. It labels
only the already-frozen clean trajectory; it does not estimate attack success.

Question:

> At the frozen Student emit time, is the clean trajectory visibly in a
> task-relevant, gripper-dependent physical interaction for which a five-step
> gripper-opening intervention would be a meaningful manipulation disturbance
> rather than normal approach or release?

## PASS — `CONTACT_VALID`

All six conditions must be visually supportable from the fixed review clip and
frame strip:

1. The gripper is coupled to the task-relevant object or movable part.
2. Gripper-dependent physical coupling is established at `t_emit`; approach or
   pre-contact positioning is insufficient.
3. Coupling persists through `[t_emit, t_emit+4]` without normal release already
   beginning.
4. The object/part is not already release-safe or independently supported at
   `t_emit`.
5. No slip, detachment, or loss of relevant coupling occurred before `t_emit`.
6. The sealed camera view is sufficient to judge the interaction.

For handles, doors, and fixtures, sustained gripper-dependent manipulation of
the task-relevant movable part is sufficient; free-space lifting is not
required.

## FAIL — `CONTACT_INVALID`

Use exactly one primary reason code:

```text
PRECONTACT_OR_APPROACH
WRONG_OR_IRRELEVANT_OBJECT_PART
RELEASE_ALREADY_STARTED
RELEASE_SAFE_OR_INDEPENDENTLY_SUPPORTED
CONTACT_ALREADY_LOST_OR_SLIPPING
OTHER_CLEARLY_NON_GRIPPER_DEPENDENT
```

## ABSTAIN — `CONTACT_AMBIGUOUS`

Use `ABSTAIN` when occlusion, framing, object identity ambiguity, or
insufficient visual evidence prevents a defensible PASS/FAIL judgment.
`ABSTAIN` is not attack-eligible. Do not change the rubric to resolve an
ambiguous case.

## Blinding and prohibited evidence

Review order is SHA256-ranked with the frozen salt
`STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_ORDER_V1_20260818`. The human sheet shows
only anonymous ID, task instruction, fixed review material, and blank labels.
It must not expose suite/task/state ordinal, Student probabilities or margins,
physical scores, selection rank, attack outcomes, or V_phys.

Do not use CQ-v2 as an automatic classifier: its current calibration is
Black-Bowl-specific and is not LIBERO-wide validated.

The review form remains blank until the owner supplies labels. The next gate is
`OWNER_MANUAL_CONTACT_LABELS_REQUIRED`.
