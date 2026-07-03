# Codex Detector Dataset Closure Handoff V1

Status: AUTHORIZED_CPU_CI_ONLY

## Task IDs

```text
C2_01 Detector dataset join builder
C2_02 Parent/state split builders
C2_03 Train-only normalization builder
C2_04 Detector dataset closure validator
```

Stop after these four tasks and request review.

## Reviewed Base

```text
planning branch = plan/codex-gated-experiment-v1
C1 closeout commit = d884a247f7a21caf6d088c31356124c3475989ea
C1 review id = 4627438633
C1 cpu-stageb run = 181 PASS
```

## Read First

```text
CODEX_IMPLEMENTATION_AUTHORIZATION_V1.md
CODEX_REPOSITORY_AUDIT_V1.md
CODEX_IMPLEMENTATION_GAP_MATRIX_V1.csv
CODEX_EXPERIMENT_PLAN_V1.md
POPULATION_DEFINITION_V1.md
SPLIT_AND_LEAKAGE_SPEC.md
DETECTOR_PROTOCOL_V1.md
CLEAN2000_LABEL_V2_SPEC.md

tools/multisuite_detector/load_label_v2_artifact.py
src/gripper_attack/sc5_detector_runtime.py
src/gripper_attack/sc5mlp_v1.py
tools/multisuite_detector/strict_loader.py
tools/multisuite_detector/build_detector_splits.py
tools/multisuite_detector/validate_detector_splits.py
```

## Scientific Boundary

This batch is repository-only and synthetic-only. It defines and tests the
formal C2 contracts but does not build the real detector dataset.

```text
real Label V2 read = prohibited
real clean feature read = prohibited
formal server dataset build = prohibited
detector model/training changes = prohibited
GPU/server execution = prohibited
```

## Implementation Sequence

### Step 1 — feature artifact contract

Audit the existing feature writers/readers and define a strict read-only schema
that binds:

```text
episode_key
parent_key
suite
task_id
initial_state_hash
trace_length
ordered SC5_FEATURES
feature_schema_sha256
artifact path and SHA256
```

Do not invent the real `initial_state_hash` source. If it is absent or
ambiguous, implement only the interface and validator, record the blocker, and
stop before claiming C2 completion.

### Step 2 — exact-set dataset join

Consume the structured return from `load_label_v2_artifact()`. Join only on
`episode_key`, then independently require matching parent/suite/task/trace
identity. Reject any missing, extra, duplicate, or mismatched episode.

The output manifest must distinguish Label V2 identity from feature-artifact
identity and include both SHA bindings.

### Step 3 — population derivation

Implement only the frozen populations:

```text
DETECTOR_ELIGIBLE
DETECTOR_SAFETY
DETECTOR_MULTI_EVENT
```

The episode-primary Label V2 table cannot create multi-event rows. When no
separate event-level artifact is supplied, mark `DETECTOR_MULTI_EVENT` as
`UNAVAILABLE_SEPARATE_ARTIFACT_REQUIRED`.

### Step 4 — split builders

Implement deterministic schemas/builders for:

```text
parent_random_split_v1
object_leave_task_out_v1
suite_loso_split_v1
```

The connected split unit is induced by both `parent_key` and
`initial_state_hash`. A shared state hash across otherwise different parents
must keep those parents in one partition.

Formal CLIs require caller-supplied seed and, where relevant, explicit ratios.
Do not choose a default formal split ratio.

### Step 5 — train-only normalization

Compute mean/std only from training rows or steps selected by a bound split and
population. Reject zero variance pending a separately reviewed policy. Bind all
source manifest SHA values and feature order.

### Step 6 — independent closure validator

Re-read synthetic generated outputs and recompute joins, populations, split
coverage, leakage, held-out exclusions, and normalization membership.

## Required Output Schemas

Repository code may define schemas for these future external artifacts:

```text
detector_dataset_manifest_v1.csv
parent_random_split_v1.csv
object_leave_task_out_v1.csv
suite_loso_split_v1.csv
detector_normalization_v1.json
detector_dataset_validation_v1.json
SHA256SUMS or equivalent manifest binding
```

Synthetic tests may materialize these only under temporary directories.
Generated formal artifacts must not enter Git.

## Required Test Families

```text
feature contract and exact 25D order
exact-set episode join
identity and trace mismatch
population derivation
multi-event unavailability
parent leakage
cross-parent state-hash leakage
random split determinism
Object leave-task-out exclusion
suite LOSO exclusion
normalization train-only membership
zero-variance rejection
manifest/hash tamper
future/attack telemetry rejection
CLI JSON success and concise failure
```

Add every C2 module and test to `cpu-stageb` while preserving all existing C0/C1
and builder tests.

## Allowed Legacy Reuse

The following may be imported or hardened after audit:

```text
strict_loader.py
build_detector_splits.py
validate_detector_splits.py
SC5_FEATURES / SC5MLPV1 definitions
```

Do not make a legacy path appear formal merely by renaming its output. Formal
paths require exact schemas, provenance, fail-closed validation, and tests.

## Explicitly Prohibited

```text
formal Label V2 build/validator
real Label V2 or feature artifact access
server checkout access
formal detector dataset generation
C3 detector train/eval/FSM changes
checkpoint or threshold selection
exact-prefix/attack/CQ/statistics implementation
OpenVLA or LIBERO
rollout or attack
A800 query/allocation/use
scientific parameter changes
```

## Commit Template

```text
Implement synthetic detector dataset closure contracts

Tasks: C2_01, C2_02, C2_03, C2_04
Scientific settings changed: NONE
Real scientific artifacts read: NONE
Formal dataset build: NONE
Server execution: NONE
GPU execution: NONE
Experiment authorization: NOT_AUTHORIZED
Tests: <exact commands and results>
```

## Completion State

```text
CODEX_C2_DETECTOR_DATASET_CLOSURE = READY_FOR_REVIEW
CODEX_C3_AND_LATER_IMPLEMENTATION = NOT_AUTHORIZED
LABEL_V2_BUILD_EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
A800_HARDWARE_ONLY_QUALIFICATION = PLANNED_NOT_AUTHORIZED
DETECTOR_TRAINING_EXECUTION = NOT_AUTHORIZED
A800_GPU_EXPERIMENT_EXECUTION = NOT_AUTHORIZED
GATE_A1_LABEL_ARTIFACT = HOLD_PENDING_FORMAL_BUILD_AND_MANUAL_AUDIT
GATE_A2_DETECTOR = HOLD
GATE_A3_ATTACK = HOLD
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```
