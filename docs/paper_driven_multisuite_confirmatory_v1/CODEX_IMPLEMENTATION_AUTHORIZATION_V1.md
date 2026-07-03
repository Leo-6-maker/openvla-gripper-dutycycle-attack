# Codex Implementation Authorization V1

Status: AUTHORIZED_REPOSITORY_AUDIT_ONLY

This authorization allows Codex to perform the first repository-only audit
batch defined in `CODEX_EXPERIMENT_PLAN_V1.md` and
`CODEX_TASK_MATRIX_V1.csv`. It is not a scientific execution authorization and
does not yet authorize implementation changes outside the audit deliverables.

## Authorized Scope

The first Codex batch is restricted to:

```text
C0_01 Repository path inventory
C0_02 Implementation gap matrix
C0_03 Artifact dependency graph
```

Authorized activities are limited to:

- inspect repository code, tests, workflows, and planning documents;
- classify formal paths as `EXISTS_AND_REVIEWED`,
  `EXISTS_NEEDS_HARDENING`, `MISSING_IMPLEMENTATION`, or
  `LEGACY_NOT_FORMAL`;
- trace frozen input artifacts to detector, attack, CQ, analysis, and paper-table
  outputs;
- create or update only:

```text
docs/paper_driven_multisuite_confirmatory_v1/CODEX_REPOSITORY_AUDIT_V1.md
docs/paper_driven_multisuite_confirmatory_v1/CODEX_IMPLEMENTATION_GAP_MATRIX_V1.csv
docs/paper_driven_multisuite_confirmatory_v1/CODEX_INITIAL_AUDIT_HANDOFF_V1.md
```

- run read-only repository searches, `py_compile`, existing unit tests, and CPU
  CI only when needed to verify current behavior;
- commit the audit deliverables and request review.

Codex must honor task dependencies and stop after C0_01-C0_03. Later
repository-only task rows require a second implementation authorization after
the audit is reviewed.

## Prohibited Scope

Codex must not:

- implement or modify detector, attack, CQ, exact-prefix, split, training,
  evaluation, or analysis code in this first batch;
- run `formal-ledger-build` or `validate-formal-output` against bound server
  artifact paths;
- access, edit, delete, copy, or normalize live scientific artifacts;
- SSH to a server or run commands in a server checkout;
- create the formal Label V2 artifact;
- train a detector on real data;
- load OpenVLA weights for inference;
- launch LIBERO;
- execute any attack or rollout;
- reserve, query, or use A800 GPUs;
- change frozen scientific settings;
- mark Gate A1, Gate A2, Gate A3, or experiment execution as authorized.

## Audit Requirements

The audit must cover at least:

```text
Label V2 ingestion and downstream schema
feature artifact and 25D SC5 feature order
parent/state split builders and leakage checks
detector train/eval CLIs and checkpoint provenance
validation-only threshold selection
exact-prefix snapshot and restore identity
matched branch queue generation
OURS / RAND_DIRECTION / RANDOM_TIME / Adapted TMA-OPEN implementations
runtime attack telemetry and actual-budget validation
CQ evaluator and blind-audit manifest generation
paired statistics, table builders, and figure-data builders
server/GPU authorization boundaries
```

Every formal path must cite its repository path, source commit/blob identity
when available, tests, current limitations, and the exact downstream paper
cell or gate it supports.

## Commit Requirements

The Codex audit commit must include:

```text
Task IDs = C0_01, C0_02, C0_03
Files changed = audit deliverables only
Scientific settings changed = NONE
Tests or searches run with exact command/result
Server execution = NONE
GPU execution = NONE
Experiment authorization status = NOT_AUTHORIZED
```

Generated scientific artifacts must not be committed to Git.

## Stop Rules

Codex must stop and request review when:

- a frozen document conflicts with implementation behavior;
- a relevant path cannot be verified from the repository;
- a task requires real Label V2 output, real clean features, server paths, GPU
  identity, OpenVLA, LIBERO, or attack execution;
- the audit reveals a scientific-semantic ambiguity;
- C0_01-C0_03 are complete.

## Current State

```text
CODEX_EXPERIMENT_PLAN_REVIEW = PASS
CODEX_INITIAL_REPOSITORY_AUDIT = AUTHORIZED_CPU_CI_ONLY
CODEX_IMPLEMENTATION_AFTER_AUDIT = NOT_AUTHORIZED
CODEX_SERVER_EXECUTION = NOT_AUTHORIZED
LABEL_V2_BUILD_EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
DETECTOR_TRAINING_EXECUTION = NOT_AUTHORIZED
A800_GPU_EXPERIMENT_EXECUTION = NOT_AUTHORIZED
GATE_A2_DETECTOR = HOLD
GATE_A3_ATTACK = HOLD
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```
