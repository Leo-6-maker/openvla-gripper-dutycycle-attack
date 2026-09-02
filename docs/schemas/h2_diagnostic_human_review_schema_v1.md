# H2 Diagnostic Human Review Schema v1

This schema is for the diagnostic human review authorized after Train300 final
acceptance. It is not a final blind validation schema.

```text
H2_DIAGNOSTIC_HUMAN_REVIEW = GO
H2_FINAL_BLIND_SELECTION = NO_GO
APPROVE_GATE_H2 = NOT_GRANTED
```

## Inputs

Reviewer-facing package:

```text
tables/layer1_h2_20260620/blind_review_queue.csv
tables/layer1_h2_20260620/h2_diagnostic_human_review_form_template.csv
reports/layer1_h2_20260620/h2_diagnostic_human_review_form_summary.json
```

Server-side review media:

```text
/data/liuyu/layer1_outputs/h2_layer1_resolver_6eb8863_20260620/diagnostic_review_package/videos
/data/liuyu/layer1_outputs/h2_layer1_resolver_6eb8863_20260620/diagnostic_review_package/teacher_timelines
/data/liuyu/layer1_outputs/h2_layer1_resolver_6eb8863_20260620/diagnostic_review_package/teacher_overlays
```

The review package is Teacher-only. It must not include detector telemetry,
Layer2 outputs, VIS/RAND/shuffled/oracle outputs, task-success labels in the
reviewer queue, or attack artifacts.

## Reviewer Fields

Accepted event rows require review of:

```text
object_identity_valid
target_identity_valid
event_exists
close_onset_valid
grasp_established_valid
lift_onset_valid
stable_carry_valid
window_start_valid
anchor_valid
window_end_valid
release_separation_valid
false_positive_carry
```

Abstain, ambiguous, and fail-closed rows require:

```text
abstain_or_fail_closed_correct
```

All judgment fields use:

```text
YES
NO
UNCERTAIN
NA
```

Empty values are allowed only before human review completion.

## Codex Boundary

Codex may:

```text
prepare blank review forms
validate completed form schema
aggregate completed reviews
report disagreements
```

Codex must not:

```text
populate reviewer_id
populate reviewer judgments
adjudicate disagreements
select final blind set
run full CLEAN300 resolver
inspect Layer2 detector telemetry for Teacher decisions
run GPU/LIBERO/VIS/RAND/shuffled/oracle/attack
```

## Next Gate

Fresh final blind selection remains blocked until:

```text
diagnostic human review complete
all disagreements adjudicated
no further resolver/ontology/physics/timing change planned
timing contract = FROZEN_AND_HUMAN_VERIFIED
component SHAs frozen
```
