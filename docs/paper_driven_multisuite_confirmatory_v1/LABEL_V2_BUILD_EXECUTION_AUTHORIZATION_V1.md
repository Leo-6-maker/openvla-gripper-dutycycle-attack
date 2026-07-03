# Label V2 Build Execution Authorization V1

Status: NOT_AUTHORIZED_DRAFT

This file defines the required authorization record for a future formal
CLEAN2000 Label V2 ledger build. It does not authorize execution.

## Implementation Review Binding

| Field | Value |
|---|---|
| formal_ledger_mode_review | `PASS_IMPLEMENTATION_ONLY` |
| reviewed_builder_commit_sha | `2366d47d545e21b6f7aac8a702b3de900b6d20b7` |
| immutable_core_commit_sha | `35e17855c57277f866142c34129425e0259ece5b` |
| immutable_core_path | `tools/multisuite_detector/build_clean2000_label_v2.py` |
| full_git_history_required | `true`; immutable core commit must resolve locally |
| builder_file_sha256 | `TBD_FINAL_IDENTITY_FREEZE` |
| expected_git_commit_sha | `TBD_FINAL_IDENTITY_FREEZE` |

## Required Execution Binding

| Field | Value |
|---|---|
| authorization_status | `NOT_AUTHORIZED` |
| clean_worktree_required | `true`; `git status --porcelain=v1 --untracked-files=all` must be empty |
| source_manifest_path | `tables/server_freeze/clean2000_teacher_source_availability.csv` |
| source_manifest_sha256 | `268ec095aae19a5aca62141b162c0719706b885c96c84122174fe425493426e4` |
| episode_census_path | `tables/server_freeze/clean2000_episode_census.csv` |
| episode_census_sha256 | `6d3696465f3e09cd736677f25ac57d83135774229bd75c5a17b38801c7e956ba` |
| source_crosstab_path | `tables/server_freeze/clean2000_source_event_crosstab.csv` |
| source_crosstab_sha256 | `0b78c0749cdf4a17c93ce28859094c0733741f9892f0b9493894bece26cb25a1` |
| output_host_absolute_path | `TBD_SERVER_NEW_NONEXISTENT_DIRECTORY` |
| output_must_be_outside_repo | `true` |
| atomic_publish_required | `true`; stage as hidden sibling and rename only after closure |
| cpu_limit | `TBD_SERVER` |
| gpu_allowed | `false` |
| maximum_runtime | `TBD_SERVER` |
| maximum_storage | `TBD_SERVER` |
| retry_rule | `new nonexistent output path only` |
| abort_rule | `any input SHA, row-count, crosstab, manual-sample, disposition-closure, producer-identity, immutable-core, worktree, staging, or validator failure` |
| independent_validator_command | `defined below` |
| authorization_record_sha256 | `TBD_AFTER_FINAL_RECORD_FREEZE` |
| authorization_expiry | `TBD_SERVER` |

## Formal Build Command Shape

```text
python tools/multisuite_detector/build_clean2000_label_v2.py \
  --mode formal-ledger-build \
  --source-manifest tables/server_freeze/clean2000_teacher_source_availability.csv \
  --episode-census tables/server_freeze/clean2000_episode_census.csv \
  --source-crosstab tables/server_freeze/clean2000_source_event_crosstab.csv \
  --output-root <host:absolute_new_nonexistent_output_dir> \
  --expected-source-sha256 268ec095aae19a5aca62141b162c0719706b885c96c84122174fe425493426e4 \
  --expected-census-sha256 6d3696465f3e09cd736677f25ac57d83135774229bd75c5a17b38801c7e956ba \
  --expected-crosstab-sha256 0b78c0749cdf4a17c93ce28859094c0733741f9892f0b9493894bece26cb25a1 \
  --expected-git-commit-sha <expected_git_commit_sha> \
  --expected-builder-sha256 <builder_file_sha256> \
  --require-clean-worktree
```

The formal output is first written to a hidden sibling staging directory. The
requested output path is created only by a final directory rename after all
builder-side checks and metadata generation succeed.

## Independent Validator Command Shape

Run this only after the formal build command exits successfully:

```text
python tools/multisuite_detector/build_clean2000_label_v2.py \
  --mode validate-formal-output \
  --source-manifest tables/server_freeze/clean2000_teacher_source_availability.csv \
  --episode-census tables/server_freeze/clean2000_episode_census.csv \
  --source-crosstab tables/server_freeze/clean2000_source_event_crosstab.csv \
  --output-root <same_host:absolute_output_dir> \
  --expected-source-sha256 268ec095aae19a5aca62141b162c0719706b885c96c84122174fe425493426e4 \
  --expected-census-sha256 6d3696465f3e09cd736677f25ac57d83135774229bd75c5a17b38801c7e956ba \
  --expected-crosstab-sha256 0b78c0749cdf4a17c93ce28859094c0733741f9892f0b9493894bece26cb25a1 \
  --expected-git-commit-sha <expected_git_commit_sha> \
  --expected-builder-sha256 <builder_file_sha256>
```

The validator reads the three frozen inputs and all five output files, verifies
`SHA256SUMS`, rechecks episode joins and source-to-output semantics, verifies the
manual sample and fallback policy, and prints a JSON PASS report to standard
output. Redirect that report to a separate authorization evidence directory;
do not place it inside the immutable five-file build output.

## Required Closure

```text
joined rows = 2000
positive = 803
no-event = 1197
PRIMARY_SUCCESS_ELIGIBLE = 772 / 271
ELIGIBLE_CLEAN_FAILURE = 31 / 276
MECHANISM_INELIGIBLE_ABSTENTION = 0 / 650
suite-task units = 40
manual audit sample = 160
manual fallback policy = total deterministic matrix
duplicate episode keys = 0
missing joins = 0
unexplained disposition rows = 0
output file set = exactly five files
SHA256SUMS verification = PASS
independent validator = PASS
```

Source JSONL path and SHA fields remain frozen-ledger provenance claims only.
The formal ledger build and validator must not open live server JSONL paths or
claim independent JSONL byte re-verification.

## Still Prohibited

```text
Real CLEAN2000 build execution before final authorization = PROHIBITED
Detector training = PROHIBITED
OpenVLA inference = PROHIBITED
Rollout / attack = PROHIBITED
GPU jobs = PROHIBITED
Live scientific artifact mutation = PROHIBITED
```
