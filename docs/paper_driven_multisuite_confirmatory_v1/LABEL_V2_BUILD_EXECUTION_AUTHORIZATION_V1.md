# Label V2 Build Execution Authorization V1

Status: NOT_AUTHORIZED_DRAFT_READY_FOR_SERVER_BINDING

This record prepares a future formal CLEAN2000 Label V2 ledger build. It does
not authorize execution. The reviewed implementation is complete; the remaining
fields require an exact server checkout, output-path allocation, and final
execution authorization.

## Implementation Review Binding

| Field | Value |
|---|---|
| formal_ledger_mode_review | `PASS_IMPLEMENTATION_ONLY` |
| semantic_adapter_review | `PASS` |
| build_input_closure | `PASS_LEDGER_ONLY` |
| formal_builder_implementation_commit_sha | `22a68974015f5c3ee1d2e8e49eda82789d1efa59` |
| ci_closeout_commit_sha | `c91d6e340021c83e9fbce40201e5c457142c841f` |
| builder_path | `tools/multisuite_detector/build_clean2000_label_v2.py` |
| builder_design | `SELF_CONTAINED_NO_RUNTIME_GIT_SOURCE_LOADING` |
| builder_file_sha256 | `TBD_SERVER_IDENTITY_CAPTURE` |
| expected_git_commit_sha | `TBD_SERVER_CHECKOUT_HEAD` |
| ci_status | `cpu-stageb PASS`; targeted tests, closeout self-test, and identity-report step completed |
| independent_validator | `IMPLEMENTED_AS_validate-formal-output_MODE` |

The server identity capture must run only after checking out the exact approved
commit and before editing this record into an authorized state.

## Frozen Input Binding

| Field | Value |
|---|---|
| source_manifest_path | `tables/server_freeze/clean2000_teacher_source_availability.csv` |
| source_manifest_sha256 | `268ec095aae19a5aca62141b162c0719706b885c96c84122174fe425493426e4` |
| episode_census_path | `tables/server_freeze/clean2000_episode_census.csv` |
| episode_census_sha256 | `6d3696465f3e09cd736677f25ac57d83135774229bd75c5a17b38801c7e956ba` |
| source_crosstab_path | `tables/server_freeze/clean2000_source_event_crosstab.csv` |
| source_crosstab_sha256 | `0b78c0749cdf4a17c93ce28859094c0733741f9892f0b9493894bece26cb25a1` |
| source_semantics_authority | `SOURCE_AVAILABILITY_LEDGER` |
| source_jsonl_runtime_read | `PROHIBITED` |

## Required Server Binding

| Field | Value |
|---|---|
| authorization_status | `NOT_AUTHORIZED` |
| server_host | `TBD_SERVER` |
| repository_absolute_path | `TBD_SERVER` |
| output_host_absolute_path | `TBD_SERVER_NEW_NONEXISTENT_DIRECTORY` |
| output_must_be_outside_repo | `true` |
| output_must_not_exist | `true` |
| clean_worktree_required | `true`; `git status --porcelain=v1 --untracked-files=all` must be empty |
| atomic_publish_required | `true`; hidden sibling staging followed by final rename |
| cpu_limit | `TBD_SERVER` |
| gpu_allowed | `false` |
| maximum_runtime | `TBD_SERVER` |
| maximum_storage | `TBD_SERVER` |
| retry_rule | `new nonexistent output path only` |
| abort_rule | `any input SHA, producer identity, worktree, row-count, crosstab, manual-sample, source-disposition, staging, output-SHA, or validator failure` |
| validator_report_path | `TBD_SERVER_OUTSIDE_IMMUTABLE_OUTPUT` |
| authorization_record_sha256 | `TBD_AFTER_FINAL_AUTHORIZATION_RECORD_FREEZE` |
| authorization_expiry | `TBD_SERVER` |

## Server Identity Capture — Read Only

These commands do not build Label V2 and do not read live source JSONL:

```text
git rev-parse HEAD
git status --porcelain=v1 --untracked-files=all
sha256sum tools/multisuite_detector/build_clean2000_label_v2.py
sha256sum tables/server_freeze/clean2000_teacher_source_availability.csv
sha256sum tables/server_freeze/clean2000_episode_census.csv
sha256sum tables/server_freeze/clean2000_source_event_crosstab.csv
python tools/multisuite_detector/build_clean2000_label_v2.py --mode self-test-closeout
```

The captured Git SHA and builder SHA must be copied into this record in a new
reviewed authorization-only commit. That later commit must still state
`NOT_AUTHORIZED` until host/path/resources/expiry are complete and an explicit
execution authorization is issued.

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
  --expected-git-commit-sha <server_captured_git_sha> \
  --expected-builder-sha256 <server_captured_builder_sha256> \
  --require-clean-worktree
```

## Independent Validator Command Shape

Run only after an authorized build exits successfully:

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
  --expected-git-commit-sha <server_captured_git_sha> \
  --expected-builder-sha256 <server_captured_builder_sha256>
```

The validator independently re-reads all three frozen inputs and all five
output files, verifies `SHA256SUMS`, deterministically reconstructs the 2000
label rows and 160-row manual sample, checks the manifest/summary bindings, and
prints a JSON PASS report. Redirect the report outside the immutable output
directory.

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

## Still Prohibited

```text
Server identity capture may be performed read-only after explicit server access.
Real CLEAN2000 build execution before final authorization = PROHIBITED
Formal Label V2 artifact generation = PROHIBITED
Detector training = PROHIBITED
OpenVLA inference = PROHIBITED
Rollout / attack = PROHIBITED
GPU jobs = PROHIBITED
Live scientific artifact mutation = PROHIBITED
```
