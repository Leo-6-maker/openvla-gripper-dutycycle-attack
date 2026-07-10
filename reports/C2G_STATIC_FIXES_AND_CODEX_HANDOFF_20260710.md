# C2g Static Fixes and Codex Handoff — 2026-07-10

## Scope and boundary

This branch contains only repository-side, CPU/static corrections derived from the review of `3d06fa550b15362452dd93b876ceed01cc30eafe`.

- Base: `3d06fa550b15362452dd93b876ceed01cc30eafe`
- Branch: `assistant/c2g-static-fixes-20260710`
- No LIBERO rollout was launched.
- No OpenVLA model was loaded.
- No counterfactual replay, embedding materialization, detector training, Goal smoke, Object replication, Spatial expansion, or D7-parity job was launched.
- D7 Table1 remains frozen.

## Corrections completed

### 1. Track A full-matrix provenance binding

`scripts/stageb/run_c2f_table1_candidate_gpu17.sh` now:

- refuses a dirty worktree before launching;
- uses the joint metadata + non-empty step-record completion predicate;
- binds skip, success, retry, archive, and final postrun audit to the exact full commit, parent key, condition, and frozen protocol;
- removes stale retry queue files before rebuilding them;
- passes the frozen full commit to the final audit.

This closes the remaining gap where a runtime-valid episode from a different 40-character commit could be accepted during resume.

### 2. Teacher-v1 audit semantics

`tools/multisuite_detector/audit_c2f_teacher_v1_labels.py` now:

- treats clean success as `true / false / unknown`;
- never converts missing clean-success metadata to failure;
- reports `teacher_primary_attackable` field positives separately from `teacher_event_role == primary_attackable`;
- reports field/role disagreements and a consistency rate;
- uses the union only as the canonical compatibility label while preserving both source counts.

Existing committed Teacher-v1 CSV/JSON artifacts were not regenerated because the source evidence is server-mounted. Codex must rerun the audit on the server and commit regenerated artifacts in a separate provenance-preserving commit.

### 3. C2g loss correctness

`src/gripper_attack/c2g_causal_vulnerability_detector.py` now:

- applies negative-episode loss only to explicitly fully-known negative episodes;
- conservatively falls back to `known_mask.all()` when no explicit episode flag is supplied;
- does not treat partially unknown episodes as negative;
- aligns episode losses with the planned 2-of-3 persistence trigger using a differentiable k-of-n score;
- uses active weight-mass normalization for weighted masked BCE;
- validates tensor shapes and episode flag cardinality.

This fixes the prior violation of the Teacher-v2 rule that unknown or unreplayed windows must never be converted to negatives.

### 4. Dataset and control scaffolding

`tools/multisuite_detector/c2g_dataset_scaffold.py` now adds:

- fold label-coverage summaries preserving unknown counts;
- hard viability gates for train/validation/test positive, negative, and episode support;
- deterministic `wrong-language-cross-task` donors that remain inside the same split but must come from a different task;
- identity consistency checks for donor generation.

This prevents training/calibration on folds with no known positives or negatives and makes the language shortcut control meaningful even when ordinary within-split shuffling would preserve the same task instruction.

### 5. Static tests

`tests/test_c2g_static.py` now covers:

- clean-success unknown semantics;
- Teacher primary field/role disagreement accounting;
- split viability failure;
- cross-task wrong-language donors;
- active weight-mass BCE normalization;
- partial-unknown episodes excluded from the negative-episode loss;
- fully-known negative episodes retaining the persistence-aligned penalty.

## Scientific interpretation after these fixes

These changes do not create a trained C2g detector. They only make the static design internally consistent.

The project state remains:

```text
TRACK_A_SMOKE5_STATIC_READINESS       = PASS_PENDING_SERVER_TEST
TRACK_A_FULL_MATRIX_STATIC_READINESS = PASS_PENDING_SERVER_TEST
TEACHER_V1_AUDIT_CODE                = PASS_STATIC
TEACHER_V1_ARTIFACT_REGENERATION     = NOT_STARTED
TEACHER_V1_FOR_TRAINING              = HOLD
COUNTERFACTUAL_TEACHER_V2            = SPEC_ONLY
C2G_DATASET_MATERIALIZATION          = NOT_STARTED
C2G_TRAINING                         = NOT_STARTED
C2G_ONLINE_VALIDATION                = NOT_STARTED
GPU_EPISODES_LAUNCHED                = 0
```

## Codex execution prompt

Work from branch `assistant/c2g-static-fixes-20260710`. Audit before modifying code.

### First response required

Return:

```text
CURRENT_HEAD
DIFF_FROM_3D06FA55
STATIC_CODE_FINDINGS
TEST_PLAN
SERVER_ARTIFACTS_REQUIRED
GPU_JOBS_PLANNED = 0
GO/HOLD
```

### Stage A — validate the assistant fixes

Do not launch GPU work.

1. Run CPU tests:

```bash
python -m unittest tests.test_c2g_static tests.test_c2f_track_a_static
```

2. Run `py_compile` on:

```text
src/gripper_attack/c2g_causal_vulnerability_detector.py
tools/multisuite_detector/c2g_dataset_scaffold.py
tools/multisuite_detector/audit_c2f_teacher_v1_labels.py
scripts/stageb/audit_c2f_track_a_run.py
scripts/stageb/run_c2f_canary_worker.py
```

3. Run Bash syntax checks on:

```text
scripts/stageb/run_c2f_track_a_smoke5.sh
scripts/stageb/run_c2f_table1_candidate_gpu17.sh
```

4. Specifically verify:

- a partial-known negative episode contributes zero negative-episode penalty;
- a fully-known negative episode contributes a finite persistence-aligned penalty;
- weighted BCE is normalized by active weight mass;
- split viability rejects a fold with zero known positives or zero known negatives;
- `wrong-language-cross-task` never selects a donor from the same task and never crosses a split;
- the full matrix launcher passes expected full commit, parent, and condition in every skip/retry/success path and final audit.

### Stage B — regenerate server-mounted Teacher-v1 audit artifacts

This stage is CPU-only but requires the server evidence roots. Do not load OpenVLA or LIBERO.

Rerun the Teacher-v1 audit with the same frozen inputs and Object override used by the existing report. Write to a new temporary output directory first. Verify:

- 2,000 episodes;
- 393,513 step rows unless the frozen source manifest proves a justified difference;
- zero read errors;
- clean-success unknown count is reported separately;
- field/role primary disagreement count is reported;
- prior Spatial/Object/L10/Goal pathology conclusions are unchanged or any difference is explained.

Then update:

```text
reports/c2f_teacher_v1_audit/teacher_v1_audit_report.json
reports/c2f_teacher_v1_audit/teacher_v1_by_suite_task.csv
reports/c2f_teacher_v1_audit/teacher_v1_reason_codes.csv
reports/C2F_TEACHER_V1_LABEL_AUDIT.md
```

Record commands, source roots, source hashes, output hashes, and exact commit.

### Stage C — continue only server-independent Teacher-v2 preparation

After Stage A and B pass, implement only pure/static components:

1. A structured target-resolution module that accepts parsed task/BDDL metadata and returns:

```text
resolved_target_objects
resolved_receptacles
ordered_subgoals
resolution_source
resolution_confidence
reason_code
```

2. Pure canonicalization helpers for MuJoCo geom/body names and contact-pair identity. They may use synthetic fixtures only; do not start LIBERO.

3. A frozen Teacher-v2 reason-code enum/schema and row validator.

4. A counterfactual replay manifest schema covering:

```text
snapshot hash
restore parity fields
candidate reason
matched-action short horizon
closed-loop continuation flag
T10 delivery semantics
known/unknown mask
threshold configuration
code/model/simulator provenance
```

5. Unit tests for ambiguous target resolution, multi-object tasks, mesh/link canonicalization, missing target metadata, and unknown masking.

Do not implement large replay collection or train C2g in this stage.

### Required return

```text
BRANCH
BASE_SHA
HEAD_SHA
COMMITS
CPU_TESTS
PY_COMPILE
BASH_SYNTAX
FULL_MATRIX_BINDING = PASS/HOLD
TEACHER_AUDIT_REGEN = PASS/HOLD
CLEAN_SUCCESS_TRI_STATE = PASS/HOLD
PRIMARY_FIELD_ROLE_CONSISTENCY = PASS/HOLD
UNKNOWN_SAFE_EPISODE_LOSS = PASS/HOLD
PERSISTENCE_ALIGNED_LOSS = PASS/HOLD
SPLIT_VIABILITY_GATE = PASS/HOLD
WRONG_LANGUAGE_CROSS_TASK = PASS/HOLD
TEACHER_V2_STATIC_SCHEMA = PASS/HOLD/NOT_STARTED
FILES_AND_HASHES
GPU_EPISODES_LAUNCHED = 0
NEXT_GPU_EXPERIMENTS = NONE_PENDING_REVIEW
```

## Hard boundaries for Codex

- Do not run the 5-episode smoke until a new review explicitly authorizes it.
- Do not run the 144-episode matrix.
- Do not run counterfactual replay.
- Do not materialize embeddings or train C2g.
- Do not change D7 Table1.
- Do not treat Teacher-v1 as the final vulnerability target.
- Do not treat missing labels or missing clean outcomes as negatives/failures.
