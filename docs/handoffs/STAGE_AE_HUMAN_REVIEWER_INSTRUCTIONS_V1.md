# Stage AE human blinded endpoint-observability review

Status: `PENDING_REAL_HUMAN_REVIEW`

This package contains the exact 91 present clips from the frozen AC4 neutral
presentation population. The five fixed-missing AC4 slots are not replaced and
are not part of the label sheet. The package is for endpoint observability only;
it does not ask the reviewer to decide whether an automatic physical endpoint
was correct.

## Reviewer rules

1. Work only from the clips in your package and the supplied rubric.
2. Use only your assigned reviewer ID (`HR1`, `HR2`, or `HR3`). Do not add a
   real name, contact information, signature, or other personal information to
   the returned file.
3. Review clips independently. Do not look at another review, repository
   reports, AI labels, automatic labels, telemetry, hidden mapping, model,
   suite, parent, condition, or dose.
4. Use ordinary playback of the supplied video. Do not extract frames, change
   playback speed, enhance, crop, re-encode, or use an external vision tool.
5. Put exactly one legal primary label on every present clip. If visible
   evidence does not establish an event, use `AMBIGUOUS_OR_OCCLUDED` or
   `NOT_IDENTIFIABLE` according to the rubric.
6. Set `review_complete` to `true` only after all 91 rows are reviewed. Do not
   leave a blank label, add a second label, or add a replacement row.
7. An optional technical note may describe a visible ambiguity, but it must not
   contain hidden metadata or a physical-endpoint conclusion.

The returned CSV must preserve the exact columns:

```text
reviewer_id,review_order_index,reviewer_clip_id,label,review_complete
```

The review is complete only after all three reviewers have returned their
files. The reconciliation tool will seal those files before any hidden mapping
can be read. No human labels exist in this repository at pre-registration.
