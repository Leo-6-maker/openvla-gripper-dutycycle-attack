# Label V2 Build Execution Authorization V1

Status: NOT_AUTHORIZED_BOUND_READY_FOR_REVIEW

This is an authorization-only server-binding record for a future formal
CLEAN2000 Label V2 ledger build. It does **not** authorize execution. The
scientific producer remains the immutable PR #48 checkout at
`af8217c934e5894c87d3db73b031a93f2536624d`; this authorization record lives on
a separate stacked branch so that reviewing the record does not change the
producer checkout identity.

## Authorization State

| Field | Value |
|---|---|
| authorization_status | `NOT_AUTHORIZED` |
| authorization_scope | `ONE_SHOT_CPU_ONLY_LABEL_V2_BUILD_AND_VALIDATOR` |
| authorization_activation | `REQUIRES_EXPLICIT_SEPARATE_REVIEW_ACTION` |
| detector_training | `PROHIBITED` |
| OpenVLA_inference | `PROHIBITED` |
| simulator_rollout | `PROHIBITED` |
| attack_execution | `PROHIBITED` |
| GPU_jobs | `PROHIBITED` |
| live_scientific_artifact_mutation | `PROHIBITED` |

## Reviewed Implementation Binding

| Field | Value |
|---|---|
| formal_ledger_mode_review | `PASS_IMPLEMENTATION_ONLY` |
| semantic_adapter_review | `PASS` |
| build_input_closure | `PASS_LEDGER_ONLY` |
| producer_git_commit_sha | `af8217c934e5894c87d3db73b031a93f2536624d` |
| formal_builder_implementation_commit_sha | `22a68974015f5c3ee1d2e8e49eda82789d1efa59` |
| ci_closeout_commit_sha | `c91d6e340021c83e9fbce40201e5c457142c841f` |
| audit_freeze_base_sha | `f972041a5ec710876e1dbfc567e1ee8c3d283010` |
| builder_path | `tools/multisuite_detector/build_clean2000_label_v2.py` |
| builder_design | `SELF_CONTAINED_NO_RUNTIME_GIT_SOURCE_LOADING` |
| builder_file_sha256 | `04d83a2f9469a3f45f8ffe54e9c3d993b493d78fe2d8acffd6351da5d3aa317b` |
| independent_validator | `IMPLEMENTED_AS_validate-formal-output_MODE` |
| server_self_test | `PASS` via `/usr/bin/python3` |

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
| source_JSONL_runtime_read | `PROHIBITED` |

## Server Checkout Binding

| Field | Value |
|---|---|
| server_host | `pm-364c0001` |
| repository_absolute_path | `/mnt/sdc/dty_user/openvla_attack_pr48_af8217c` |
| expected_git_commit_sha | `af8217c934e5894c87d3db73b031a93f2536624d` |
| clean_worktree_required | `true` |
| captured_dirty_status_count | `0` |
| required_commit_objects_present | `af8217c`, `22a6897`, `c91d6e`, `f972041` |
| checkout_transport | `LOCALLY_VERIFIED_COMPLETE_HISTORY_GIT_BUNDLE` |
| direct_GitHub_HTTPS_fetch | `TIMED_OUT_NOT_USED_FOR_SCIENTIFIC_BINDING` |
| legacy_checkout_path | `/mnt/sdc/dty_user/openvla_attack` |
| legacy_checkout_mutation | `NONE` |

The server checkout must remain pinned at the producer commit above. The
separate authorization branch/PR must not be checked out in the scientific
producer directory.

## Output and Evidence Binding

| Field | Value |
|---|---|
| output_parent_absolute_path | `/mnt/sdc/dty_user/openvla_attack_outputs` |
| output_parent_owner_mode | `dty_user:dty_user 775` |
| output_host_absolute_path | `/mnt/sdc/dty_user/openvla_attack_outputs/clean2000_label_v2_af8217c` |
| output_must_be_outside_repo | `true` |
| output_must_not_exist_before_build | `true` |
| captured_output_child_state | `NONEXISTENT` |
| evidence_parent_absolute_path | `/mnt/sdc/dty_user/openvla_attack_evidence/label_v2` |
| evidence_parent_owner_mode | `dty_user:dty_user 775` |
| validator_report_path | `/mnt/sdc/dty_user/openvla_attack_evidence/label_v2/validator_report_af8217c934e5894c87d3db73b031a93f2536624d.json` |
| identity_capture_path | `/mnt/sdc/dty_user/openvla_attack_evidence/label_v2/identity_capture_af8217c934e5894c87d3db73b031a93f2536624d.txt` |
| output_file_contract | `EXACTLY_FIVE_FILES` |
| atomic_publish_required | `true`; hidden sibling staging followed by final rename |

## Resource and Safety Envelope

| Field | Value |
|---|---|
| Python | `/usr/bin/python3` |
| CPU_limit | `4 threads maximum; one builder/validator process at a time` |
| thread_environment | `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4` |
| GPU_allowed | `false`; no CUDA process may be started |
| memory_limit | `16 GiB maximum resident set; abort if exceeded` |
| build_runtime_limit | `30 minutes` |
| validator_runtime_limit | `30 minutes` |
| maximum_new_storage | `1 GiB across formal output and validator evidence` |
| observed_filesystem | `/mnt/sdc: 2.9T total, 2.6T used, 125G available, 96% used` |
| minimum_free_space_before_build | `100 GiB` |
| retry_rule | `new nonexistent output child only; never overwrite or reuse a failed final path` |
| abort_rule | `any input SHA, producer identity, worktree, free-space, row-count, crosstab, manual-sample, source-disposition, staging, output-SHA, resource-limit, or validator failure` |
| proposed_authorization_expiry | `2026-07-10T00:00:00Z` |

Because `/mnt/sdc` is already 96% used, the preflight must abort when available
space is below 100 GiB even though the formal artifact itself is expected to be
small.

## Mandatory Preflight — Read Only

Run from the bound repository directory immediately before an authorized build:

```bash
set -euo pipefail
cd /mnt/sdc/dty_user/openvla_attack_pr48_af8217c

test "$(hostname)" = "pm-364c0001"
test "$(git rev-parse HEAD)" = "af8217c934e5894c87d3db73b031a93f2536624d"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

test "$(sha256sum tools/multisuite_detector/build_clean2000_label_v2.py | awk '{print $1}')" = \
  "04d83a2f9469a3f45f8ffe54e9c3d993b493d78fe2d8acffd6351da5d3aa317b"
test "$(sha256sum tables/server_freeze/clean2000_teacher_source_availability.csv | awk '{print $1}')" = \
  "268ec095aae19a5aca62141b162c0719706b885c96c84122174fe425493426e4"
test "$(sha256sum tables/server_freeze/clean2000_episode_census.csv | awk '{print $1}')" = \
  "6d3696465f3e09cd736677f25ac57d83135774229bd75c5a17b38801c7e956ba"
test "$(sha256sum tables/server_freeze/clean2000_source_event_crosstab.csv | awk '{print $1}')" = \
  "0b78c0749cdf4a17c93ce28859094c0733741f9892f0b9493894bece26cb25a1"

test -d /mnt/sdc/dty_user/openvla_attack_outputs
test ! -e /mnt/sdc/dty_user/openvla_attack_outputs/clean2000_label_v2_af8217c
test -d /mnt/sdc/dty_user/openvla_attack_evidence/label_v2

test "$(df --output=avail -B1 /mnt/sdc | tail -1)" -ge 107374182400

/usr/bin/python3 tools/multisuite_detector/build_clean2000_label_v2.py \
  --mode self-test-closeout
```

A preflight failure does not consume the one-shot authorization; it returns the
record to review without attempting a build.

## Formal Build Command — NOT YET AUTHORIZED

```bash
cd /mnt/sdc/dty_user/openvla_attack_pr48_af8217c

env \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 \
  timeout --signal=TERM --kill-after=30s 30m \
  /usr/bin/python3 tools/multisuite_detector/build_clean2000_label_v2.py \
    --mode formal-ledger-build \
    --source-manifest tables/server_freeze/clean2000_teacher_source_availability.csv \
    --episode-census tables/server_freeze/clean2000_episode_census.csv \
    --source-crosstab tables/server_freeze/clean2000_source_event_crosstab.csv \
    --output-root /mnt/sdc/dty_user/openvla_attack_outputs/clean2000_label_v2_af8217c \
    --expected-source-sha256 268ec095aae19a5aca62141b162c0719706b885c96c84122174fe425493426e4 \
    --expected-census-sha256 6d3696465f3e09cd736677f25ac57d83135774229bd75c5a17b38801c7e956ba \
    --expected-crosstab-sha256 0b78c0749cdf4a17c93ce28859094c0733741f9892f0b9493894bece26cb25a1 \
    --expected-git-commit-sha af8217c934e5894c87d3db73b031a93f2536624d \
    --expected-builder-sha256 04d83a2f9469a3f45f8ffe54e9c3d993b493d78fe2d8acffd6351da5d3aa317b \
    --require-clean-worktree
```

## Independent Validator Command — NOT YET AUTHORIZED

Run only if an explicitly authorized build exits successfully:

```bash
cd /mnt/sdc/dty_user/openvla_attack_pr48_af8217c

env \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 \
  timeout --signal=TERM --kill-after=30s 30m \
  /usr/bin/python3 tools/multisuite_detector/build_clean2000_label_v2.py \
    --mode validate-formal-output \
    --source-manifest tables/server_freeze/clean2000_teacher_source_availability.csv \
    --episode-census tables/server_freeze/clean2000_episode_census.csv \
    --source-crosstab tables/server_freeze/clean2000_source_event_crosstab.csv \
    --output-root /mnt/sdc/dty_user/openvla_attack_outputs/clean2000_label_v2_af8217c \
    --expected-source-sha256 268ec095aae19a5aca62141b162c0719706b885c96c84122174fe425493426e4 \
    --expected-census-sha256 6d3696465f3e09cd736677f25ac57d83135774229bd75c5a17b38801c7e956ba \
    --expected-crosstab-sha256 0b78c0749cdf4a17c93ce28859094c0733741f9892f0b9493894bece26cb25a1 \
    --expected-git-commit-sha af8217c934e5894c87d3db73b031a93f2536624d \
    --expected-builder-sha256 04d83a2f9469a3f45f8ffe54e9c3d993b493d78fe2d8acffd6351da5d3aa317b \
  > /mnt/sdc/dty_user/openvla_attack_evidence/label_v2/validator_report_af8217c934e5894c87d3db73b031a93f2536624d.json
```

The validator report must remain outside the immutable five-file build output.
A validator failure freezes the output as failed evidence; it must not be
silently deleted, edited, or reused.

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

## Authorization Record Identity

The authorization record is identified by both:

1. the Git commit of this authorization-only branch; and
2. the SHA256 of this exact Markdown file captured after commit and posted in
   the review record.

The file does not embed its own SHA256 because that would be self-referential.
Neither identity value activates execution by itself.

## Explicit Non-Authorization

```text
LABEL_V2_SERVER_CHECKOUT_IDENTITY = PASS
LABEL_V2_SERVER_BINDING_HANDOFF = PASS_BOUND_READY_FOR_REVIEW
LABEL_V2_BUILD_EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
GATE_A2_DETECTOR = HOLD
GATE_A3_ATTACK = HOLD
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```
