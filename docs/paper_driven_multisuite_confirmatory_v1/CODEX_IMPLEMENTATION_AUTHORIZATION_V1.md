# Codex Implementation Authorization V1

Status: AUTHORIZED_C2_DETECTOR_DATASET_CPU_CI_ONLY

This authorization follows final review of C1 commit
`d884a247f7a21caf6d088c31356124c3475989ea`. It authorizes the next
repository-only implementation batch for detector dataset, population, split,
normalization, and leakage-closure tooling. It does not authorize construction
from real artifacts or any server/GPU scientific execution.

## Reviewed Binding

```text
C0_01_REPOSITORY_PATH_INVENTORY = PASS
C0_02_IMPLEMENTATION_GAP_MATRIX = PASS
C0_03_ARTIFACT_DEPENDENCY_GRAPH = PASS
C1_01_LABEL_V2_DOWNSTREAM_INGESTION_SCHEMA = PASS
C1_02_LABEL_V2_INGESTION_VALIDATOR = PASS
CODEX_C1_LABEL_V2_INGESTION = PASS_CPU_CI
reviewed_c1_commit = d884a247f7a21caf6d088c31356124c3475989ea
review_id = 4627438633
cpu_stageb_run = 181 PASS
```

## Authorized Task IDs

```text
C2_01 Detector dataset join builder
C2_02 Parent/state split builders
C2_03 Train-only normalization builder
C2_04 Detector dataset closure validator
```

No C2 formal server build and no C3-or-later task is authorized by this record.

## Authorized Scope

Codex may implement and test, using temporary synthetic fixtures only:

- a strict read-only frozen clean-feature artifact reader/contract;
- exact-set `episode_key` joins between the validated Label V2 consumer and
  synthetic feature artifacts;
- exact canonical 25D feature-order binding to `SC5_FEATURES`;
- deterministic population derivation for:

```text
DETECTOR_ELIGIBLE
DETECTOR_SAFETY
DETECTOR_MULTI_EVENT
```

- deterministic schemas/builders for:

```text
parent_random_split_v1
object_leave_task_out_v1
suite_loso_split_v1
```

- parent-key and initial-state-hash leakage validation;
- train-partition-only normalization statistics;
- manifest and SHA256 provenance generation;
- a fail-closed closure validator and JSON-reporting CLI;
- unit tests, synthetic integration tests, `py_compile`, and CPU CI.

The implementation must consume `load_label_v2_artifact()` rather than bypassing
or duplicating its five-file internal-closure checks.

## Expected Deliverables

Codex may create or harden the smallest coherent file set under:

```text
tools/multisuite_detector/
  load_frozen_clean_features.py
  build_detector_dataset_manifest_v1.py
  build_detector_splits_v1.py
  build_detector_normalization_v1.py
  validate_detector_dataset_manifest_v1.py

tests/multisuite_detector/
  test_frozen_feature_artifact_contract.py
  test_detector_dataset_manifest_v1.py
  test_detector_splits_v1.py
  test_detector_normalization_v1.py
  test_detector_dataset_closure_v1.py

.github/workflows/cpu-stageb.yml

docs/paper_driven_multisuite_confirmatory_v1/
  CODEX_DETECTOR_DATASET_CLOSURE_HANDOFF_V1.md
  CODEX_IMPLEMENTATION_GAP_MATRIX_V1.csv   # only C2 gap evidence/status
  CODEX_REPOSITORY_AUDIT_V1.md             # path/status maintenance only
```

Equivalent names are allowed only when they preserve the same separated
contracts. Existing legacy loaders/split tools may be hardened instead of
creating duplicates, but legacy behavior must remain clearly isolated from the
formal V1 path.

## Frozen Feature Contract

The primary detector feature order is exactly the 25-element `SC5_FEATURES`
constant from the reviewed runtime/model source. The builder must record and
validate:

```text
feature_names ordered list
feature_count = 25
feature_schema_sha256
episode_key
parent_key
suite
task_id
initial_state_hash
trace_length
feature artifact path and SHA256
```

If the repository does not contain a frozen, unambiguous definition for the
real feature artifact file set or `initial_state_hash`, Codex must define only a
schema/validator interface and stop for scientific review; it must not invent a
real-data convention.

All feature vectors and required metadata in synthetic tests must be finite and
strictly typed. Unknown/extra feature columns, reordered features, duplicate
steps, malformed lengths, or inconsistent episode metadata must fail closed.

## Join and Population Contract

The formal join key is `episode_key` with exact-set closure:

```text
Label V2 episode set == feature episode set
missing Label V2 rows = 0
missing feature rows = 0
duplicate episode keys = 0
suite/task/parent identity mismatch = 0
trace-length mismatch = 0
```

Population derivation must follow `POPULATION_DEFINITION_V1.md` and must not use
attack outcomes or future attack telemetry:

```text
DETECTOR_ELIGIBLE = mechanism-eligible positive and no-event rows
DETECTOR_SAFETY = mechanism-ineligible / unsupported rows
DETECTOR_MULTI_EVENT = separate event-level artifact only
```

The current Label V2 episode-primary table must not be relabeled or expanded
into multi-event examples. If a separate multi-event artifact is absent, the
formal `DETECTOR_MULTI_EVENT` population is represented as unavailable, not
fabricated from episode-primary rows.

## Split Contract

The unit of separation is the connected component induced by both:

```text
parent_key
initial_state_hash
```

No parent or initial-state hash may cross train/validation/test. Split builders
must be deterministic from an explicit seed and input manifest SHA.

Required schemas:

1. `parent_random_split_v1`
   - pooled parent/state groups;
   - train/validation/test partitions;
   - deterministic assignment and documented ratios supplied by the caller;
   - no implicit default ratio in the formal CLI.

2. `object_leave_task_out_v1`
   - one Object task held out per fold;
   - every parent/state group for that task remains outside training;
   - fold identities and held-out task are explicit.

3. `suite_loso_split_v1`
   - one entire suite held out;
   - held-out suite contributes no normalization, weight, checkpoint-selection,
     or threshold-selection signal.

The implementation must validate exact manifest coverage and reject unknown
split labels, duplicate assignments, missing groups, or target leakage.

## Normalization Contract

Normalization is computed only from training rows/steps for the selected split
and population. Record:

```text
feature_names
count per feature
mean
std
finite status
zero-variance disposition
source dataset manifest SHA256
source split manifest SHA256
population_id
fold/regime identifier
```

Zero variance must fail closed or use a separately reviewed explicit policy; it
must not silently add epsilon and continue. Validation/test rows may be
transformed but must never contribute statistics.

## Required Validators and Reports

The closure validator must independently re-read the generated synthetic
manifests and recompute at least:

```text
exact episode-set join
feature-order and schema SHA
population counts
parent leakage
initial-state-hash leakage
split coverage
held-out task/suite exclusion
normalization source membership
finite mean/std
artifact file SHA256 bindings
```

The success report must distinguish:

```text
synthetic_contract_validation = PASS
real_artifact_validation = NOT_PERFORMED
formal_detector_dataset_build = NOT_PERFORMED
server_execution = NOT_PERFORMED
```

## Required Negative Tests

At minimum cover:

```text
missing/extra/duplicate episode
suite/task/parent mismatch
trace-length mismatch
feature reorder or wrong count
unknown/extra feature
NaN/Inf feature
malformed or missing initial_state_hash
parent leakage
state-hash leakage across different parents
missing or duplicate split assignment
Object held-out-task leakage
LOSO held-out-suite leakage
normalization using validation/test rows
normalization manifest SHA mismatch
zero variance
attack/future telemetry field rejection
attempt to fabricate multi-event rows
```

The workflow must compile every formal C2 module and run the complete C2 test
set in addition to the existing C1 and builder tests.

## Explicitly Prohibited

Codex must not in this batch:

- read the real Label V2 output or real frozen clean-feature artifact;
- run the formal Label V2 build or source-based validator;
- construct formal detector manifests on the server;
- modify Label V2 producer semantics;
- modify detector model, training, evaluation, checkpoint selection, thresholds,
  or FSM (`C3_*`);
- implement exact-prefix, attack, CQ, statistics, tables, or figures;
- SSH to or execute in a server checkout;
- train a detector or load OpenVLA;
- launch LIBERO, rollout, or attack;
- query, reserve, allocate, or use A800 GPUs;
- choose a real split ratio, checkpoint rule, threshold, attack parameter, or
  paper claim from observed outcomes;
- mark Gate A1, Gate A2, Gate A3, or experiment execution as authorized.

## Commit Requirements

Every C2 implementation commit must state:

```text
Task IDs = C2_01, C2_02, C2_03, C2_04
Scientific settings changed = NONE
Real scientific artifacts read = NONE
Formal dataset build = NONE
Server execution = NONE
GPU execution = NONE
Experiment authorization status = NOT_AUTHORIZED
Tests = exact commands and results
```

## Stop Rules

Stop and request review when:

- the real feature artifact contract or initial-state-hash source is ambiguous;
- the formal population definition cannot be derived without inventing fields;
- a proposed split ratio or zero-variance policy is not frozen;
- supporting C2 requires detector-model or C3 changes;
- any test demonstrates parent/state/normalization leakage;
- C2_01 through C2_04 are complete.

## Current State

```text
CODEX_EXPERIMENT_PLAN_REVIEW = PASS
CODEX_INITIAL_REPOSITORY_AUDIT = PASS
CODEX_C1_LABEL_V2_INGESTION = PASS_CPU_CI
CODEX_C2_DETECTOR_DATASET_CLOSURE = AUTHORIZED_CPU_CI_ONLY
CODEX_C3_AND_LATER_IMPLEMENTATION = NOT_AUTHORIZED
CODEX_SERVER_EXECUTION = NOT_AUTHORIZED
LABEL_V2_BUILD_EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
A800_HARDWARE_ONLY_QUALIFICATION = PLANNED_NOT_AUTHORIZED
DETECTOR_TRAINING_EXECUTION = NOT_AUTHORIZED
A800_GPU_EXPERIMENT_EXECUTION = NOT_AUTHORIZED
GATE_A1_LABEL_ARTIFACT = HOLD_PENDING_FORMAL_BUILD_AND_MANUAL_AUDIT
GATE_A2_DETECTOR = HOLD
GATE_A3_ATTACK = HOLD
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```
