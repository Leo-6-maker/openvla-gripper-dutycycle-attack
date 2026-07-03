# CLEAN2000 Teacher Labels V2 Spec

Status: PLANNING_ONLY

V2 labels must be built from frozen source records. V1 canonical timing labels
remain historical and must not be overwritten.

## Gate A1 Binding

| Field | Value |
|---|---|
| source_manifest_path | `tables/server_freeze/clean2000_teacher_source_availability.csv` |
| source_manifest_git_blob_sha1 | `22d54409bb01db489d5b2edc0640efafcb6a6408` |
| source_manifest_repo_content_sha256_lf | `268ec095aae19a5aca62141b162c0719706b885c96c84122174fe425493426e4` |
| supporting_census_path | `tables/server_freeze/clean2000_episode_census.csv` |
| supporting_census_git_blob_sha1 | `bb8fe57c14816477d5844e611c86017397a6111c` |
| supporting_census_repo_content_sha256_lf | `6d3696465f3e09cd736677f25ac57d83135774229bd75c5a17b38801c7e956ba` |
| supporting_crosstab_path | `tables/server_freeze/clean2000_source_event_crosstab.csv` |
| supporting_crosstab_git_blob_sha1 | `c4ac870079ff5f84b1da7f38412e8028203247de` |
| supporting_crosstab_repo_content_sha256_lf | `0b78c0749cdf4a17c93ce28859094c0733741f9892f0b9493894bece26cb25a1` |
| primary_source_root | `vla:/data/liuyu/openvla_gripper_freeze/20260702_codex_verified_v3/clean2000/CLEAN2000_CANONICAL_V1` |
| live_source_root | `dty-server:/mnt/sdc/dty_user/openvla_attack/evidence/CLEAN2000_CANONICAL_V1` read-only parity check only |
| future_builder_output_root | `vla:/data/liuyu/openvla_gripper_freeze/20260702_codex_verified_v3/derived/clean2000_teacher_labels_v2/{authorized_commit_sha}_{source_manifest_sha256_12}` |
| builder_command_contract | `python tools/multisuite_detector/build_clean2000_label_v2.py --source-manifest <path> --episode-census <path> --source-crosstab <path> --synthetic-fixture-root <fixture-root> --synthetic-output-root <temp-output-root> --output-root <empty-new-dir> --expected-source-sha256 <sha256> --expected-census-sha256 <sha256> --expected-crosstab-sha256 <sha256> --synthetic --dry-run` |

The current builder implementation is synthetic-only. Formal CLEAN2000 source
reads and the 2000-row build require a later build-execution authorization.

For the current implementation review, the committed source-availability ledger
is the semantic authority. Source JSONL files are checked only for byte-bound
provenance presence, not for independent event semantics. The V2 episode table
is primary-event-only; multi-event event-level labels require a separate
artifact before they can support MULTI_EVENT analysis.

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
confidence_available
confidence_provenance
event_id_provenance
source_semantics_authority
source_jsonl_check_mode
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
- `trace_length` is the full trajectory length (`n_steps`), not the count of
  valid feature rows.
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

## Manual Audit Sampling

```text
manual_audit_sample_n = 160
sampling_seed = 20260703
quota = 40 tasks x 4 rows/task
```

Per task, sample in this priority order:

1. one positive clean-success row;
2. one eligible no-event row;
3. one clean-failure or boundary row;
4. one abstention/ineligible row, else a second positive row.

If a cohort is absent for a task, fill from the next available priority without
crossing task boundaries. Emit `manual_audit_sample_manifest.csv` and bind its
SHA256 before review.

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
