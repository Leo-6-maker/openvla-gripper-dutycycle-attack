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
```

## Audited Source Cohorts

| Cohort | Positive | No-event |
|---|---:|---:|
| Primary success eligible | 772 | 271 |
| Eligible clean failure | 31 | 276 |
| Mechanism-ineligible abstention | 0 | 650 |

Total: 803 positive, 1197 no-event.

## Minimum Checks

- exact row count and cohort crosstab;
- source SHA for every row;
- parent-level split;
- no parent crossing train/val/test;
- normalization computed from train only;
- 100-row stratified manual spot check;
- reproducible builder SHA and git SHA recorded.

