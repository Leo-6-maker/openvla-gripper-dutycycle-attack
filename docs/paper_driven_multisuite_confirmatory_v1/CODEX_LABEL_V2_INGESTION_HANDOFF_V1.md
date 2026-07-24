# Codex Label V2 Ingestion Handoff V1

Status: AUTHORIZED_CPU_CI_ONLY

## Task IDs

```text
C1_01 Label V2 downstream ingestion schema
C1_02 Label V2 ingestion validator
```

Stop after these two tasks and request review.

## Reviewed Base

```text
planning branch = plan/codex-gated-experiment-v1
C0 audit commit = 59ba119901a1019e37c69cde7ae68a9fa2f530ad
C0 review id = 4627119421
```

The implementation branch may advance beyond the audit commit, but the formal
scientific producer remains `af8217c934e5894c87d3db73b031a93f2536624d` and
must not be edited.

## Read First

```text
CODEX_IMPLEMENTATION_AUTHORIZATION_V1.md
CODEX_REPOSITORY_AUDIT_V1.md
CODEX_IMPLEMENTATION_GAP_MATRIX_V1.csv
CLEAN2000_LABEL_V2_SPEC.md
LABEL_V2_BUILD_EXECUTION_AUTHORIZATION_V1.md
tools/multisuite_detector/build_clean2000_label_v2.py
tests/test_build_clean2000_label_v2.py
```

## Scientific Boundary

This batch verifies the internal closure of an already-built five-file artifact.
It does not reconstruct labels from the three frozen ledgers and does not read
source JSONL.

```text
source-based independent validation = existing builder validator
five-file internal-closure validation = this batch
feature joins / detector populations = later C2 batch
```

## Required API

Implement a read-only module at:

```text
tools/multisuite_detector/load_label_v2_artifact.py
```

Recommended public surface:

```python
class LabelV2ArtifactError(ValueError):
    pass


def load_label_v2_artifact(
    artifact_root,
    *,
    expected_mode,
    expected_builder_git_sha=None,
    expected_builder_sha256=None,
):
    """Validate and return structured five-file Label V2 artifact contents."""


def validate_label_v2_artifact(...):
    """Return a JSON-serializable validation report or raise fail-closed."""
```

Exact names may differ, but downstream callers must not need to import the
builder CLI or execute Git-history source.

## Exact File Contract

```text
label_v2.csv
build_manifest.json
validation_summary.json
manual_audit_sample_manifest.csv
SHA256SUMS
```

Reject any missing or extra entry, including hidden files. Reject symlinked root
or child entries.

## Formal Internal Closure

For `expected_mode=formal-ledger-build`, require all of:

```text
rows = 2000
positive = 803
no-event = 1197
PRIMARY_SUCCESS_ELIGIBLE = 772 / 271 / 1043
ELIGIBLE_CLEAN_FAILURE = 31 / 276 / 307
MECHANISM_INELIGIBLE_ABSTENTION = 0 / 650 / 650
suite-task units = 40
manual sample rows = 160
four distinct manual rows per suite-task unit
unexplained disposition rows = 0
status = PASS
synthetic_only = false
atomic_publish = true
```

Validate the manifest and summary against recomputed CSV facts rather than only
cross-comparing the two JSON files.

## Label Row Invariants

At minimum enforce:

```text
unique episode_key
strict lowercase true/false encoding
known cohort classes only
clean outcome and mechanism eligibility match cohort
source semantics authority is ledger-only
source JSONL mode forbids runtime reads
builder identities are uniform and match manifest/expected arguments
trace_length is positive integer
event rows have 0 <= start <= anchor < end <= trace_length
window_end is exclusive
no-event rows use anchor/start/end = -1
event/no-event identifiers are disposition-consistent
window_valid and label_validity_status are internally consistent
source SHA is lowercase 64-hex
```

Do not infer or invent multi-event labels from the episode-primary table.

## Manual Audit Invariants

Every manual row must:

- reference exactly one Label V2 episode;
- match suite, task, cohort, clean outcome, mechanism eligibility, event status,
  and validity from that episode;
- use a known requested/actual category;
- have fallback fields that agree with requested versus actual category;
- not duplicate an episode within one suite-task unit;
- satisfy formal 40-by-4 closure in formal mode.

## Manifest Input Semantics

Validate that the manifest contains three well-formed ledger input records with
path and SHA256 fields. Do not open those paths. The five-file consumer must not
claim that source ledger bytes or source JSONL were independently reverified.

Recommended report language:

```text
five_file_internal_closure = PASS
source_ledger_reverification = NOT_PERFORMED_BY_THIS_LOADER
source_jsonl_runtime_read = NOT_PERFORMED
```

## CLI

Provide a CLI that accepts:

```text
--artifact-root
--expected-mode
--expected-builder-git-sha       optional
--expected-builder-sha256        optional
```

It must print one JSON report to stdout on success and return nonzero with a
concise stderr error on failure. It must not write into the artifact root.

## Tests

Create:

```text
tests/test_load_label_v2_artifact.py
```

Use temporary directories only. Small synthetic artifact factories may be used.
Do not call the bound server path and do not invoke a real formal build.

Minimum negative tests:

```text
missing and extra files
root/child symlink
malformed, duplicate, missing, and mismatched SHA entries
header mismatch
bad bool/int/SHA encodings
duplicate episode
mode mismatch
builder identity mismatch
summary versus recomputed crosstab mismatch
formal exact-count mismatch
inclusive/exclusive window error
no-event coordinate error
manual episode missing
manual context mismatch
manual duplicate within unit
formal manual unit/quota mismatch
wrong semantics authority or JSONL mode
```

Add the test to `cpu-stageb` and compile the loader.

## Documentation Maintenance

The implementation commit may update:

```text
CODEX_REPOSITORY_AUDIT_V1.md
```

only to clarify:

```text
audited_source_head = 4d1a646100738ca4b5bc86076a080cfd1b895465
audit_commit = 59ba119901a1019e37c69cde7ae68a9fa2f530ad
```

It may update G001 in the gap matrix only after the implementation and tests
justify the new status. Do not mark G002 or any later gap closed.

## Prohibited

```text
editing build_clean2000_label_v2.py
formal-ledger-build
server independent validator invocation
real Label V2 artifact access
real feature artifact access
C2 joins/populations/splits
C3 detector code changes
C7 exact-prefix or attack work
C9 CQ implementation
C10 statistics implementation
server shell or SSH
OpenVLA / LIBERO / rollout / attack
A800 query or use
```

## Commit Template

```text
Implement read-only Label V2 artifact ingestion

Tasks: C1_01, C1_02
Scientific settings changed: NONE
Formal builder modified: NO
Real scientific artifacts read: NONE
Server execution: NONE
GPU execution: NONE
Experiment authorization: NOT_AUTHORIZED
Tests: <exact commands and results>
```

## Completion State

```text
CODEX_C1_LABEL_V2_INGESTION = READY_FOR_REVIEW
CODEX_C2_AND_LATER_IMPLEMENTATION = NOT_AUTHORIZED
LABEL_V2_BUILD_EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
DETECTOR_TRAINING_EXECUTION = NOT_AUTHORIZED
A800_GPU_EXPERIMENT_EXECUTION = NOT_AUTHORIZED
GATE_A2_DETECTOR = HOLD
GATE_A3_ATTACK = HOLD
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```
