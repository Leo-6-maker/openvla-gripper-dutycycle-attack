# CLEAN2000 Teacher Labels V2 Spec

Status: PLANNING_ONLY

V2 labels must be built from frozen source records. V1 canonical timing labels
remain historical and must not be overwritten.

## Gate A1 Binding

| Field | Value |
|---|---|
| source_manifest_path | `tables/server_freeze/clean2000_teacher_source_availability.csv` |
| source_manifest_sha256 | `09cefaa3f50d552adde6f3040fe25e11e295b8f97f71f05745d8cec710b5d962` |
| supporting_census_path | `tables/server_freeze/clean2000_episode_census.csv` |
| supporting_census_sha256 | `06a4baf446a7f2600425630a0acf35e344423647ea54e9f097675fec1753ef38` |
| supporting_crosstab_path | `tables/server_freeze/clean2000_source_event_crosstab.csv` |
| supporting_crosstab_sha256 | `31b7377aa81e1a28b347ba38e92ee9be312e133edc5e6b288c82b440302b27a0` |
| allowed_source_roots | `/mnt/sdc/dty_user/openvla_attack/evidence/CLEAN2000_CANONICAL_V1`; verified backup `/data/liuyu/openvla_gripper_freeze/20260702_codex_verified_v3` |
| builder_output_root | `outputs/clean2000_teacher_labels_v2/{authorized_commit_sha}_{source_manifest_sha256_12}` |
| builder_command_contract | `python tools/multisuite_detector/build_clean2000_label_v2.py --source-manifest <path> --episode-census <path> --output-root <empty-new-dir> --dry-run` |

The builder path is a contract for the next CPU-only implementation step; this
PR does not add or run that builder.

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

## Coordinate Semantics

- `step_index` is zero-based.
- `anchor_absolute_step` denotes the observation-before-action index whose
  following action belongs to the teacher event.
- `window_start` is inclusive.
- `window_end` is exclusive.
- Coordinates refer to the original full clean trajectory, not a trimmed segment.
- A window is invalid if `window_start < 0`, `window_end <= window_start`,
  `window_end > trace_length`, or the source trajectory is truncated.
- Invalid windows remain in the 2000-row output with `window_valid=false` and
  `label_validity_status=INVALID_WINDOW`; they are not silently dropped.

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
