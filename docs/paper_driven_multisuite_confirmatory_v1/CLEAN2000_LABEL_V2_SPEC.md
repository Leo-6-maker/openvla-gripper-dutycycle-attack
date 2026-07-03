# CLEAN2000 Teacher Labels V2 Spec

Status: PLANNING_ONLY

V2 labels must be built from frozen source records. V1 canonical timing labels
remain historical and must not be overwritten.

## Required Row Fields

```text
episode_key
parent_key
suite
task_id
clean_success
mechanism_eligible
event_present
anchor_absolute_step
window_start
window_end
event_source
source_path
source_sha256
builder_git_sha
builder_sha256
invalid_reason
abstain_reason
mechanism_type
event_id
segment_id
event_rank
coordinate_semantics
trace_length
source_schema_version
teacher_confidence
window_valid
label_validity_status
manual_audit_status
manual_audit_reason
```

## Audited Source Cohorts

| Cohort | Positive | No-event |
|---|---:|---:|
| Primary success eligible | 772 | 271 |
| Eligible clean failure | 31 | 276 |
| Mechanism-ineligible abstention | 0 | 650 |

Total: 803 positive, 1197 no-event.

## Use By Cohort

- Primary detector training uses mechanism-eligible rows only.
- The 650 mechanism-ineligible abstentions go to abstention and boundary evaluation.
- The 31 clean-failure positives are auxiliary robustness rows, not the main attack population.

## Minimum Checks

- exact row count and cohort crosstab;
- source SHA for every row;
- parent-level and state-hash split;
- no parent or initial state crossing train/val/test;
- normalization computed from train only;
- 160-row task-stratified manual spot check;
- all schema anomalies manually audited;
- 25% second-reviewer overlap;
- reproducible builder SHA and git SHA recorded.

## Gate B

```text
total rows = 2000
positive = 803
no-event = 1197
cohort crosstab exact
source SHA coverage = 100%
parent/state leakage = 0
event-presence manual agreement >= 95%
positive anchor within +/-5 steps >= 90%
unexplained label rows = 0
```
