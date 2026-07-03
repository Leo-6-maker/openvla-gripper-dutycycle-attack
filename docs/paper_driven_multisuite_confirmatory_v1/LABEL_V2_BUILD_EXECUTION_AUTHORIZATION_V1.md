# Label V2 Build Execution Authorization V1

Status: NOT_AUTHORIZED_DRAFT

This file defines the required authorization record for a future formal
CLEAN2000 Label V2 ledger build. It does not authorize execution.

## Required Binding

| Field | Value |
|---|---|
| authorization_status | `NOT_AUTHORIZED` |
| reviewed_builder_commit_sha | `TBD_AFTER_FORMAL_MODE_REVIEW` |
| builder_file_sha256 | `TBD_AFTER_FORMAL_MODE_REVIEW` |
| expected_git_commit_sha | `TBD_AFTER_FORMAL_MODE_REVIEW` |
| clean_worktree_required | `true` |
| source_manifest_path | `tables/server_freeze/clean2000_teacher_source_availability.csv` |
| source_manifest_sha256 | `268ec095aae19a5aca62141b162c0719706b885c96c84122174fe425493426e4` |
| episode_census_path | `tables/server_freeze/clean2000_episode_census.csv` |
| episode_census_sha256 | `6d3696465f3e09cd736677f25ac57d83135774229bd75c5a17b38801c7e956ba` |
| source_crosstab_path | `tables/server_freeze/clean2000_source_event_crosstab.csv` |
| source_crosstab_sha256 | `0b78c0749cdf4a17c93ce28859094c0733741f9892f0b9493894bece26cb25a1` |
| output_host_absolute_path | `TBD_FORMAL_EMPTY_DIRECTORY` |
| cpu_limit | `TBD` |
| gpu_allowed | `false` |
| maximum_runtime | `TBD` |
| maximum_storage | `TBD` |
| retry_rule | `new empty output directory only` |
| abort_rule | `any SHA, row-count, crosstab, manual-sample, producer-identity, or worktree-cleanliness failure` |
| authorization_expiry | `TBD` |

## Command Shape

```text
python tools/multisuite_detector/build_clean2000_label_v2.py \
  --mode formal-ledger-build \
  --source-manifest tables/server_freeze/clean2000_teacher_source_availability.csv \
  --episode-census tables/server_freeze/clean2000_episode_census.csv \
  --source-crosstab tables/server_freeze/clean2000_source_event_crosstab.csv \
  --output-root <host:absolute_empty_output_dir> \
  --expected-source-sha256 268ec095aae19a5aca62141b162c0719706b885c96c84122174fe425493426e4 \
  --expected-census-sha256 6d3696465f3e09cd736677f25ac57d83135774229bd75c5a17b38801c7e956ba \
  --expected-crosstab-sha256 0b78c0749cdf4a17c93ce28859094c0733741f9892f0b9493894bece26cb25a1 \
  --expected-git-commit-sha <reviewed_builder_commit_sha> \
  --expected-builder-sha256 <builder_file_sha256> \
  --require-clean-worktree
```

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
duplicate episode keys = 0
missing joins = 0
unexplained disposition rows = 0
```

Source JSONL path and SHA fields remain frozen-ledger provenance claims only.
The formal ledger build must not open live server JSONL paths or claim
independent JSONL byte re-verification.

## Still Prohibited

```text
Real CLEAN2000 build execution before authorization = PROHIBITED
Detector training = PROHIBITED
OpenVLA inference = PROHIBITED
Rollout / attack = PROHIBITED
GPU jobs = PROHIBITED
Live scientific artifact mutation = PROHIBITED
```
