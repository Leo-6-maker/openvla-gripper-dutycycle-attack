# Label V2 Build Execution Authorization V1

Status: NOT_AUTHORIZED_BOUND_READY_FOR_REVIEW

This authorization-only record binds a possible future CLEAN2000 Label V2
ledger build. It does **not** authorize execution. The scientific producer is
the immutable PR #48 checkout at
`af8217c934e5894c87d3db73b031a93f2536624d`. This record is reviewed on a
separate stacked branch so that the producer checkout identity does not change.

## Authorization State

| Field | Value |
|---|---|
| authorization_status | `NOT_AUTHORIZED` |
| proposed_scope | `ONE_SHOT_CPU_ONLY_LABEL_V2_BUILD_AND_VALIDATOR` |
| activation | `REQUIRES_EXPLICIT_SEPARATE_REVIEW_ACTION` |
| detector_training | `PROHIBITED` |
| OpenVLA_inference | `PROHIBITED` |
| simulator_rollout | `PROHIBITED` |
| attack_execution | `PROHIBITED` |
| GPU_jobs | `PROHIBITED` |
| live_scientific_artifact_mutation | `PROHIBITED` |

## Reviewed Producer Binding

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

## Frozen Inputs

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

## Server Checkout

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

The producer directory must remain pinned to `af8217c...`. The authorization
branch must not be checked out there.

## Output and Evidence

| Field | Value |
|---|---|
| output_parent | `/mnt/sdc/dty_user/openvla_attack_outputs` |
| output_parent_owner_mode | `dty_user:dty_user 775` |
| formal_output | `/mnt/sdc/dty_user/openvla_attack_outputs/clean2000_label_v2_af8217c` |
| formal_output_precondition | `NONEXISTENT_AND_OUTSIDE_REPOSITORY` |
| evidence_parent | `/mnt/sdc/dty_user/openvla_attack_evidence/label_v2` |
| evidence_parent_owner_mode | `dty_user:dty_user 775` |
| validator_report | `/mnt/sdc/dty_user/openvla_attack_evidence/label_v2/validator_report_af8217c934e5894c87d3db73b031a93f2536624d.json` |
| validator_report_precondition | `NONEXISTENT` |
| identity_capture | `/mnt/sdc/dty_user/openvla_attack_evidence/label_v2/identity_capture_af8217c934e5894c87d3db73b031a93f2536624d.txt` |
| formal_output_contract | `EXACTLY_FIVE_FILES` |
| atomic_publish | `HIDDEN_SIBLING_STAGING_THEN_FINAL_RENAME` |

## Resource and Safety Envelope

| Field | Value |
|---|---|
| Python | `/usr/bin/python3` |
| CPU_limit | `4 threads; one builder or validator process at a time` |
| thread_environment | `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4` |
| GPU_allowed | `false` |
| memory_limit | `16 GiB virtual memory, enforced with ulimit -v 16777216` |
| build_runtime_limit | `30 minutes, enforced with timeout` |
| validator_runtime_limit | `30 minutes, enforced with timeout` |
| maximum_new_storage | `1 GiB total for formal output plus validator report` |
| observed_filesystem | `/mnt/sdc: 2.9T total, 2.6T used, 125G available, 96% used` |
| minimum_free_space_before_build | `100 GiB` |
| retry_rule | `a new nonexistent output child and validator-report path are required` |
| failure_evidence_rule | `a published output that fails validation is preserved and never edited or reused` |
| abort_rule | `any SHA, identity, worktree, free-space, row-count, crosstab, manual-sample, disposition, staging, output-SHA, resource-limit, or validator failure` |
| proposed_expiry | `2026-07-10T00:00:00Z` |

## Mandatory Read-only Preflight

```bash
set -euo pipefail
cd /mnt/sdc/dty_user/openvla_attack_pr48_af8217c

command -v timeout >/dev/null
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
test ! -e /mnt/sdc/dty_user/openvla_attack_evidence/label_v2/validator_report_af8217c934e5894c87d3db73b031a93f2536624d.json

test "$(df --output=avail -B1 /mnt/sdc | tail -1)" -ge 107374182400

/usr/bin/python3 tools/multisuite_detector/build_clean2000_label_v2.py \
  --mode self-test-closeout
```

A preflight failure performs no build and returns the record to review.

## Formal Build Command — NOT AUTHORIZED

```bash
set -euo pipefail
cd /mnt/sdc/dty_user/openvla_attack_pr48_af8217c

(
  ulimit -v 16777216
  exec env \
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
)

test "$(du -sb /mnt/sdc/dty_user/openvla_attack_outputs/clean2000_label_v2_af8217c | awk '{print $1}')" \
  -le 1073741824
```

## Independent Validator Command — NOT AUTHORIZED

Run only after an explicitly authorized build exits successfully:

```bash
set -euo pipefail
cd /mnt/sdc/dty_user/openvla_attack_pr48_af8217c

test ! -e /mnt/sdc/dty_user/openvla_attack_evidence/label_v2/validator_report_af8217c934e5894c87d3db73b031a93f2536624d.json

(
  ulimit -v 16777216
  exec env \
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
      --expected-builder-sha256 04d83a2f9469a3f45f8ffe54e9c3d993b493d78fe2d8acffd6351da5d3aa317b
) > /mnt/sdc/dty_user/openvla_attack_evidence/label_v2/validator_report_af8217c934e5894c87d3db73b031a93f2536624d.json

test -s /mnt/sdc/dty_user/openvla_attack_evidence/label_v2/validator_report_af8217c934e5894c87d3db73b031a93f2536624d.json

test "$((
  $(du -sb /mnt/sdc/dty_user/openvla_attack_outputs/clean2000_label_v2_af8217c | awk '{print $1}') +
  $(stat -c '%s' /mnt/sdc/dty_user/openvla_attack_evidence/label_v2/validator_report_af8217c934e5894c87d3db73b031a93f2536624d.json)
))" -le 1073741824
```

The validator report remains outside the immutable five-file output. A
validator failure freezes the published output and report as failed evidence.

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

## Record Identity

The record is identified by the authorization-only Git commit and by the SHA256
of this Markdown file captured after commit and posted in the review. The file
does not embed its own hash because that would be self-referential. Neither
identity activates execution.

## Explicit Non-Authorization

```text
LABEL_V2_SERVER_CHECKOUT_IDENTITY = PASS
LABEL_V2_SERVER_BINDING_HANDOFF = PASS_BOUND_READY_FOR_REVIEW
LABEL_V2_BUILD_EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
GATE_A2_DETECTOR = HOLD
GATE_A3_ATTACK = HOLD
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```
