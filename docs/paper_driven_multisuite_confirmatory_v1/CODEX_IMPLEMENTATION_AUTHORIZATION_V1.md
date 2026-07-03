# Codex Implementation Authorization V1

Status: AUTHORIZED_C1_LABEL_V2_INGESTION_CPU_CI_ONLY

This authorization follows review of commit
`59ba119901a1019e37c69cde7ae68a9fa2f530ad`, which completed `C0_01`,
`C0_02`, and `C0_03`. It authorizes the next smallest repository-only batch:
a read-only downstream parser and internal-closure validator for the Label V2
five-file artifact. It is not a scientific execution authorization.

## Reviewed Audit Binding

```text
C0_01_REPOSITORY_PATH_INVENTORY = PASS
C0_02_IMPLEMENTATION_GAP_MATRIX = PASS
C0_03_ARTIFACT_DEPENDENCY_GRAPH = PASS
CODEX_INITIAL_REPOSITORY_AUDIT = PASS
reviewed_audit_commit = 59ba119901a1019e37c69cde7ae68a9fa2f530ad
review_id = 4627119421
```

## Authorized Task IDs

```text
C1_01 Label V2 downstream ingestion schema
C1_02 Label V2 ingestion validator
```

No other task-matrix row is authorized by this record.

## Authorized Deliverables

Codex may create or modify only the smallest coherent file set needed for the
following deliverables:

```text
tools/multisuite_detector/load_label_v2_artifact.py
tests/test_load_label_v2_artifact.py
.github/workflows/cpu-stageb.yml

docs/paper_driven_multisuite_confirmatory_v1/
  CODEX_LABEL_V2_INGESTION_HANDOFF_V1.md
  CODEX_REPOSITORY_AUDIT_V1.md        # identity clarification only
  CODEX_IMPLEMENTATION_GAP_MATRIX_V1.csv  # G001 status/evidence only
```

A small schema helper module or synthetic fixture factory may be added only when
it is clearly necessary and remains within `tools/multisuite_detector/` or
`tests/`. Generated scientific artifacts must not be committed.

The frozen producer must not be modified:

```text
tools/multisuite_detector/build_clean2000_label_v2.py = IMMUTABLE_IN_THIS_BATCH
```

## Required Loader Contract

The implementation must be read-only and fail closed. At minimum it must:

1. Require an artifact root containing exactly:

```text
label_v2.csv
build_manifest.json
validation_summary.json
manual_audit_sample_manifest.csv
SHA256SUMS
```

2. Reject symlinks, missing files, extra files, malformed hash lines, duplicate
   hash entries, and SHA256 mismatches.
3. Validate exact CSV headers and strict field encodings.
4. Validate manifest/summary mode agreement and require explicit caller choice
   between `synthetic-dry-run` and `formal-ledger-build`.
5. For formal mode, require:

```text
schema_version = clean2000_label_v2_episode_primary_event_v1
synthetic_only = false
atomic_publish = true
source_semantics_authority = SOURCE_AVAILABILITY_LEDGER
source_jsonl_check_mode = LEDGER_PROVENANCE_ONLY_NO_RUNTIME_READ
row_count = 2000
manual_audit_sample_n = 160
PRIMARY_SUCCESS_ELIGIBLE = 772 positive / 271 no-event
ELIGIBLE_CLEAN_FAILURE = 31 positive / 276 no-event
MECHANISM_INELIGIBLE_ABSTENTION = 0 positive / 650 no-event
```

6. Recompute internal row counts and cohort/event crosstabs from
   `label_v2.csv`; do not trust summary counts alone.
7. Require unique `episode_key` values.
8. Enforce event/no-event coordinate semantics, including exclusive
   `window_end`, full-trajectory `trace_length`, and no-event coordinates of
   `-1`.
9. Enforce cohort, clean outcome, mechanism eligibility, event disposition,
   window-validity, and builder-identity consistency that can be checked from
   the five files alone.
10. Verify every manual-audit row references a Label V2 row and matches its
    suite, task, cohort, outcome, mechanism, event, and validity fields.
11. In formal mode, require 40 suite-task units and four distinct manual rows per
    unit with the four requested priority categories.
12. Validate expected builder Git SHA and builder file SHA256 when supplied by
    the caller.
13. Validate that manifest input entries contain well-formed path/SHA ledger
    bindings, but do not read source JSONL or re-read the three source ledgers.
14. Return a typed or structured read-only result suitable for a later detector
    dataset builder.
15. Provide a CLI that prints a JSON validation report to stdout and does not
    mutate the artifact directory.

The loader is an internal-closure consumer, not a replacement for the already
implemented independent source-based validator.

## Required Tests

Tests must use only temporary synthetic artifacts or repository synthetic
fixtures. They must not read the bound server artifact path.

Required positive coverage:

```text
valid synthetic five-file artifact
explicit synthetic mode
formal contract fixture generated entirely inside a temporary test directory
expected builder identity match
manual sample reference closure
```

Required negative coverage includes at least:

```text
missing file
extra file
symlink entry
malformed or duplicate SHA256SUMS entry
hash mismatch
wrong CSV header
duplicate episode_key
manifest/summary mode mismatch
unexpected builder Git SHA or builder SHA256
wrong row count or cohort crosstab
invalid exclusive-end window
no-event coordinate not -1
manual row missing from Label V2
manual context mismatch
wrong source semantics authority
formal manual quota/unit failure
```

The CI workflow must compile the loader and run its test file.

## Explicitly Prohibited

Codex must not in this batch:

- modify the Label V2 builder or its frozen semantics;
- run `formal-ledger-build` or the independent validator against server paths;
- read the real five-file Label V2 output, because it does not yet exist;
- read real frozen clean feature artifacts;
- implement feature ingestion, Label V2-feature joins, populations, splits, or
  normalization (`C2_*`);
- modify detector train/eval/FSM code (`C3_*`);
- implement exact-prefix, attack, CQ, statistics, tables, or figures;
- access a server checkout;
- train a detector;
- load OpenVLA or launch LIBERO;
- run a rollout or attack;
- query, reserve, or use A800 GPUs;
- change any frozen denominator, label semantics, threshold, split, metric,
  attack parameter, or paper claim;
- mark Gate A1, Gate A2, Gate A3, or experiment execution as authorized.

## Commit Requirements

The implementation commit must state:

```text
Task IDs = C1_01, C1_02
Scientific settings changed = NONE
Formal builder modified = NO
Real scientific artifacts read = NONE
Server execution = NONE
GPU execution = NONE
Experiment authorization status = NOT_AUTHORIZED
Tests = exact commands and results
```

## Stop Rules

Codex must stop and request review when:

- five-file fields are insufficient to verify a proposed invariant;
- a requirement would duplicate source-ledger reconstruction rather than
  internal artifact closure;
- the builder contract and frozen documentation disagree;
- supporting formal mode would require reading real server artifacts;
- implementation would cross into any `C2_*` or later task;
- `C1_01` and `C1_02` are complete.

## Current State

```text
CODEX_EXPERIMENT_PLAN_REVIEW = PASS
CODEX_INITIAL_REPOSITORY_AUDIT = PASS
CODEX_C1_LABEL_V2_INGESTION = AUTHORIZED_CPU_CI_ONLY
CODEX_C2_AND_LATER_IMPLEMENTATION = NOT_AUTHORIZED
CODEX_SERVER_EXECUTION = NOT_AUTHORIZED
LABEL_V2_BUILD_EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
DETECTOR_TRAINING_EXECUTION = NOT_AUTHORIZED
A800_GPU_EXPERIMENT_EXECUTION = NOT_AUTHORIZED
GATE_A2_DETECTOR = HOLD
GATE_A3_ATTACK = HOLD
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```
