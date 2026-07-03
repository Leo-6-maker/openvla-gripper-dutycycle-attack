# Codex Implementation Authorization V1

Status: AUTHORIZED_REPOSITORY_ONLY_CPU_CI

This authorization allows Codex to implement and test repository-side
scaffolding defined in `CODEX_EXPERIMENT_PLAN_V1.md` and
`CODEX_TASK_MATRIX_V1.csv`. It is not a scientific execution authorization.

## Authorized Scope

Codex may perform tasks whose task-matrix row satisfies all of:

```text
execution_authorized = true
server_required = false
gpu_required = false
```

Authorized activities are limited to:

- inspect repository code and planning documents;
- create or harden schemas, parsers, manifest builders, validators, queue
  generators, metric implementations, analysis builders, and CLI contracts;
- add synthetic fixtures;
- run `py_compile`, unit tests, synthetic integration tests, and repository CI;
- update planning, handoff, and authorization documents;
- open commits and pull requests for review.

## Authorized Initial Batch

The first Codex batch is restricted to:

```text
C0_01 Repository path inventory
C0_02 Implementation gap matrix
C0_03 Artifact dependency graph
C1_01 Label V2 downstream ingestion schema
C1_02 Label V2 ingestion validator
C3_01 Train CLI identity hardening audit/implementation
C3_02 Eval CLI identity hardening audit/implementation
C3_03 Event and timing metrics
C7_01 Exact-prefix snapshot schema
C9_01 CQ automatic evaluator audit/implementation
C10_01 Statistical analysis implementation audit
```

Codex must finish and request review of this batch before starting later
repository-only rows.

## Prohibited Scope

Codex must not:

- run `formal-ledger-build` or `validate-formal-output` against the bound server
  artifact paths;
- access, edit, delete, copy, or normalize live scientific artifacts;
- SSH to a server or run commands in a server checkout;
- create the formal Label V2 artifact;
- train a detector on real data;
- load OpenVLA weights for inference;
- launch LIBERO;
- execute any attack or rollout;
- reserve, query, or use A800 GPUs;
- change frozen scientific settings without a separate reviewed planning
  change;
- mark Gate A1, Gate A2, Gate A3, or experiment execution as authorized.

## Commit Requirements

Every Codex implementation commit must include:

```text
Task IDs
Files changed
Scientific settings changed = NONE, or separate reviewed change reference
Tests run with exact command and result
Synthetic-only or repository-only evidence
Server execution = NONE
GPU execution = NONE
Experiment authorization status = NOT_AUTHORIZED
```

Generated scientific artifacts must not be committed to Git.

## Stop Rules

Codex must stop and request review when:

- a frozen document conflicts with implementation behavior;
- a task requires real Label V2 output or real feature artifacts;
- a task requires server paths, GPU identity, OpenVLA, LIBERO, or attack
  execution;
- a test reveals a scientific-semantic ambiguity rather than an engineering
  bug;
- implementing the task would change a denominator, threshold, split, metric,
  attack parameter, or primary claim;
- the initial authorized batch is complete.

## Current State

```text
CODEX_EXPERIMENT_PLAN_REVIEW = PASS
CODEX_REPOSITORY_IMPLEMENTATION = AUTHORIZED_CPU_CI_ONLY
CODEX_SERVER_EXECUTION = NOT_AUTHORIZED
LABEL_V2_BUILD_EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
DETECTOR_TRAINING_EXECUTION = NOT_AUTHORIZED
A800_GPU_EXPERIMENT_EXECUTION = NOT_AUTHORIZED
GATE_A2_DETECTOR = HOLD
GATE_A3_ATTACK = HOLD
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```
