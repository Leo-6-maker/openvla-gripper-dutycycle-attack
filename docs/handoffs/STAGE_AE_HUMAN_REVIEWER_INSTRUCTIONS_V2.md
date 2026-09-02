# Stage AE human-review instructions V2

Status: `PENDING_REAL_HUMAN_REVIEW`

This package contains the exact 91 present clips from the frozen AC4 neutral
presentation population. The five fixed-missing AC4 slots are not replaced
and are not part of the label sheet. Review only endpoint observability from
visible video evidence.

## Review boundary

1. Work only from the clips and V2 rubric in your assigned package.
2. Use only the assigned reviewer ID (`HR1`, `HR2`, or `HR3`). Do not add a
   real name, contact information, signature, or other personal information.
3. Review independently. Do not look at another review, repository reports,
   AI labels, automatic labels, telemetry, model, suite, parent, condition,
   dose, hidden mapping, or experiment outcomes.
4. Use ordinary playback of the supplied video. Do not extract frames, change
   playback speed, enhance, crop, re-encode, or use an external vision tool.
5. Apply the V2 observability gate before the event-label priority hierarchy.
   Do not infer a stable grasp or a failure from the absence of visible
   failure.
6. Put exactly one legal primary label on every one of the 91 present clips.
   The fixed-missing slots are not labelable and must not be replaced.
7. Set `review_complete` to `true` only after all 91 rows are reviewed.

## Return-file contract

The returned CSV must preserve exactly these five columns, in this order:

```text
reviewer_id,review_order_index,reviewer_clip_id,label,review_complete
```

There is no optional technical-note column in V2. Do not add columns, rows,
source IDs, metadata, or outcome judgments. The labels are visual
observability evidence only; they do not confirm or rewrite the automatic
physical endpoint.

The review is complete only after all three reviewer files have been returned
and structurally sealed. Hidden mapping is opened only in the separately
authorized post-seal phase.
