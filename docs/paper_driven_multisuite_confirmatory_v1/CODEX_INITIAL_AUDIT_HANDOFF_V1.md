# Codex Initial Audit Handoff V1

Status: AUTHORIZED_REPOSITORY_AUDIT_ONLY

## Objective

Complete task IDs `C0_01`, `C0_02`, and `C0_03` from
`CODEX_TASK_MATRIX_V1.csv`. Produce a rigorous repository audit and dependency
graph before any implementation batch is authorized.

## Checkout

Work only on the planning branch descended from:

```text
producer planning base = af8217c934e5894c87d3db73b031a93f2536624d
planning branch = plan/codex-gated-experiment-v1
```

Do not use or modify the server producer checkout. Do not run server commands.

## Read First

```text
docs/paper_driven_multisuite_confirmatory_v1/CODEX_EXPERIMENT_PLAN_V1.md
docs/paper_driven_multisuite_confirmatory_v1/CODEX_TASK_MATRIX_V1.csv
docs/paper_driven_multisuite_confirmatory_v1/CODEX_IMPLEMENTATION_AUTHORIZATION_V1.md

docs/paper_driven_multisuite_confirmatory_v1/CLEAN2000_LABEL_V2_SPEC.md
docs/paper_driven_multisuite_confirmatory_v1/POPULATION_DEFINITION_V1.md
docs/paper_driven_multisuite_confirmatory_v1/SPLIT_AND_LEAKAGE_SPEC.md
docs/paper_driven_multisuite_confirmatory_v1/DETECTOR_PROTOCOL_V1.md
docs/paper_driven_multisuite_confirmatory_v1/EXACT_PREFIX_BRANCHING_SPEC_V1.md
docs/paper_driven_multisuite_confirmatory_v1/ATTACK_PROTOCOL_V1.md
docs/paper_driven_multisuite_confirmatory_v1/BASELINE_PROTOCOL_V1.md
docs/paper_driven_multisuite_confirmatory_v1/CONTACT_QUALITY_PROTOCOL_V1.md
docs/paper_driven_multisuite_confirmatory_v1/METRIC_DEFINITIONS_V1.md
docs/paper_driven_multisuite_confirmatory_v1/STATISTICAL_ANALYSIS_PLAN_V1.md
docs/paper_driven_multisuite_confirmatory_v1/EXPERIMENT_MATRIX_V2.csv
```

## Required Audit Coverage

Audit the repository for the exact implementation path of:

1. Label V2 five-file ingestion and downstream schema validation.
2. Clean feature artifact ingestion and canonical 25D `SC5_FEATURES` order.
3. Exact-set episode joins and population construction.
4. Parent/state split generation and leakage validation.
5. Detector training, evaluation, checkpoint selection, thresholds, and FSM.
6. Detector event/timing metrics and no-emit handling.
7. Exact-prefix snapshot, serialization, restoration, and parity checks.
8. Matched branch queue generation and same-parent worker assignment.
9. Attack implementations for:

```text
OURS_STUDENT_GRIPPER_TARGET
RAND_DIRECTION
RANDOM_TIME
ADAPTED_TMA_OPEN
EARLY_SHIFT
ARM_TARGETED
COMMAND_OPEN_ORACLE
SHUFFLED_GRADIENT
UNTARGETED_PGD
```

10. Runtime telemetry and actual attack-budget validation.
11. Contact-quality automatic evaluation and blind manual-audit manifests.
12. Paired statistics, ITT/emitted-only reporting, table builders, and figure
    data builders.
13. Artifact manifests, SHA bindings, validators, workflows, and authorization
    boundaries.

## Classification

Every audited component must be classified as exactly one of:

```text
EXISTS_AND_REVIEWED
EXISTS_NEEDS_HARDENING
MISSING_IMPLEMENTATION
LEGACY_NOT_FORMAL
```

`EXISTS_AND_REVIEWED` requires:

- exact repository path;
- relevant function/class/CLI;
- current commit/blob identity when available;
- at least one validating test or workflow path;
- no known conflict with the frozen protocol.

Do not classify a component as reviewed merely because a similarly named legacy
script exists.

## Deliverable 1

Create:

```text
docs/paper_driven_multisuite_confirmatory_v1/CODEX_REPOSITORY_AUDIT_V1.md
```

Required sections:

```text
Executive verdict
Repository and branch identity
Frozen protocol cross-reference
Component-by-component audit
Source path and test evidence
Scientific-semantic conflicts
Authorization-boundary audit
Dependency graph from inputs to paper tables
P0 / P1 / P2 implementation gaps
Recommended implementation batches
Exact commands/searches/tests run
Final gate state
```

The dependency graph must cover:

```text
frozen clean evidence
-> Label V2
-> split/normalization
-> detector checkpoint and predictions
-> exact-prefix branch family
-> attack telemetry
-> CQ labels
-> statistics
-> paper tables/figures
```

## Deliverable 2

Create:

```text
docs/paper_driven_multisuite_confirmatory_v1/CODEX_IMPLEMENTATION_GAP_MATRIX_V1.csv
```

Required columns:

```text
gap_id
component
phase
task_ids
status
current_paths
current_tests
frozen_requirement
observed_behavior
gap_or_risk
scientific_impact
priority
recommended_change
new_files_expected
existing_files_expected
server_needed
gpu_needed
execution_authorized
review_gate
```

## Allowed Verification

Allowed:

```text
repository search
git diff/log/show/status on the local planning checkout
python -m py_compile
existing unit tests
existing synthetic fixture tests
GitHub CPU CI
```

Do not add or modify implementation code in this batch.

## Prohibited

```text
formal-ledger-build
validate-formal-output against server paths
server SSH or shell commands
real Label V2 artifact reads
real feature artifact reads
detector training on real data
OpenVLA loading or inference
LIBERO launch
rollout or attack execution
A800 query/reservation/use
scientific setting changes
```

## Commit Message Template

```text
Audit formal experiment implementation paths

Tasks: C0_01, C0_02, C0_03
Scientific settings changed: NONE
Server execution: NONE
GPU execution: NONE
Experiment authorization: NOT_AUTHORIZED
Tests/searches: <exact commands and results>
```

## Completion State

Finish with:

```text
CODEX_INITIAL_REPOSITORY_AUDIT = READY_FOR_REVIEW
CODEX_IMPLEMENTATION_AFTER_AUDIT = NOT_AUTHORIZED
LABEL_V2_BUILD_EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
A800_GPU_EXPERIMENT_EXECUTION = NOT_AUTHORIZED
GATE_A2_DETECTOR = HOLD
GATE_A3_ATTACK = HOLD
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```

Stop after committing the two audit deliverables and request review.
