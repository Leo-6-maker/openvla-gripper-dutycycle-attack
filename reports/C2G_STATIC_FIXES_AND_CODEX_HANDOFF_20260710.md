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

# Why C2f did not close scientifically

The previous C2f detector was not simply "too small" or "undertrained". Its main failure modes were upstream of model capacity.

## 1. The target was a heuristic phase proxy, not causal vulnerability

Teacher-v1 labeled `stable_carry + target match` as attackable. That proxy can correlate with gripper-open vulnerability, but it does not answer the deployed question:

```text
Would forcing the gripper open for T10 at this exact step cause contact loss,
object drop, progress regression, or a clean-success-to-failure flip?
```

A model can achieve high window-level Teacher-v1 recall while still triggering at phases that are broad, benign, recoverable, or irrelevant to final task success.

## 2. Teacher-v1 has suite- and task-specific pathologies

The full audit already showed:

- Spatial labels are extremely dense and strongly associated with the absolute-EEF-z fallback, explaining episode-wide over-emission.
- Object has sparse positives from both phase scarcity and failed object/target grounding.
- several LIBERO-10 tasks have no primary labels because the stable-carry ontology does not cover their manipulation pattern;
- Goal contains tasks with stable-carry rows but no target-matched primary rows;
- release-safe supervision is almost absent and is based on a gripper transition rather than target-relative placement.

No threshold choice can repair labels that are structurally missing, overbroad, or semantically wrong.

## 3. Raw context enabled task-prior shortcuts

The legacy 108D context exposed suite identity, task index, and hashed task identity. Because positive-label density differs sharply by suite/task, the easiest solution was often to learn:

```text
this task usually emits / this task usually never emits
```

rather than infer the current physical vulnerability state. Random episode splits within observed tasks do not reveal this shortcut.

## 4. Global pooled visual features are weak for target grounding

Mean-pooled SigLIP features encode scene semantics well, but discard much of the spatial information needed to answer:

- which object is actually in the gripper;
- whether it is the language-specified target;
- where the target receptacle is;
- whether opening now would drop or correctly release the object.

This is particularly limiting in multi-object LIBERO-10 and Object tasks.

## 5. Window metrics did not match first-trigger deployment

C2f was optimized and thresholded per window, while deployment used the first emitted window to launch an irreversible T10 intervention. Important deployment errors therefore were not directly optimized:

- one early false trigger anywhere in a long episode;
- no emit in an attackable episode;
- timing error relative to the first causal interval;
- emit during a valid release-safe interval;
- persistent versus isolated score spikes.

## 6. The four-head conjunctive gate accumulated calibration error

The old decision required four independently trained proxy heads to satisfy a conjunction. A suite bias or calibration error in any head could dominate the final decision. The suppress/release head was especially weak because Teacher-v1 barely supplied valid release-safe examples.

## 7. The previous online control did not isolate detector timing value

`TRUE_CMDOPEN_T10_C2F` used a large, deterministic force-open payload, while `RAND_ACTION_NOISE_T10_C2F` used small action-space noise. Their payloads and postprocessed gripper effects were not matched. A TRUE/RAND gap therefore showed payload harm, but did not prove that the detector chose a better time than a random-time force-open control.

# Detailed detector upgrade roadmap

The upgrade is organized so that each stage answers one scientific question and has a hard GO/HOLD gate. Codex must not skip stages or train C2g before causal labels and split viability exist.

## Phase 0 — Runtime and provenance closure

Goal: prove that the frozen Track A plumbing produces complete, attributable artifacts before using it for any detector comparison.

Required:

- server CPU validation of the assistant static fixes;
- regenerated Teacher-v1 audit artifacts with corrected semantics;
- exact commit/protocol/parent/condition binding in smoke and full-matrix launchers;
- joint metadata and non-empty step-record completion;
- runtime-invalid attempt archiving and retry;
- Goal model manifest verification;
- action-level executed-command evidence.

Gate:

```text
TRACK_A_SMOKE5_STATIC_READINESS       = PASS
TRACK_A_FULL_MATRIX_STATIC_READINESS = PASS
TEACHER_V1_ARTIFACT_REGENERATION     = PASS
REMOTE_DIFF_REVIEW                   = PASS
```

This phase does not prove detector quality.

## Phase 1 — Teacher-v2 grounding foundation

Goal: replace nearest-body/language-substring heuristics with explicit privileged grounding, while keeping all privileged information out of student inputs.

Implement pure/static components first:

1. Structured target resolver:
   - parse task/BDDL object identities;
   - resolve target objects, receptacles, sites, and ordered subgoals;
   - use language matching only as a recorded fallback;
   - return confidence and reason codes.
2. Contact identity helpers:
   - canonicalize MuJoCo geom/body/link/mesh names;
   - map finger-object contact pairs to canonical task objects;
   - handle composite objects and ambiguous multi-contact cases.
3. Teacher-v2 row schema:
   - `teacher_confidence`;
   - `teacher_reason_code`;
   - contacted object and grounding source;
   - resolved target and receptacle;
   - object-relative lift evidence;
   - target-relative release evidence;
   - explicit unknown/abstain status.
4. Candidate strata:
   - close onset;
   - stable grasp;
   - persistent object contact;
   - relative object movement;
   - stable carry;
   - pre-release;
   - a frozen random noncandidate sample for recall auditing.

Gate:

```text
STRUCTURED_TARGET_RESOLUTION = PASS_STATIC
CONTACT_CANONICALIZATION     = PASS_STATIC
TEACHER_V2_SCHEMA            = PASS_STATIC
CANDIDATE_STRATA_SPEC        = PASS
```

## Phase 2 — Deterministic counterfactual replay smoke

Goal: verify that labels can be generated from matched causal comparisons rather than heuristic phase membership.

For each candidate step, restore the same simulator snapshot and compare:

### Tier A: matched-action short horizon

- clean recorded arm actions and clean gripper command;
- identical arm actions with force-open gripper for exactly T10;
- horizon approximately 10–30 steps;
- labels: contact loss, object drop, short-term progress regression.

### Tier B: closed-loop continuation

- clean policy continuation from the restored snapshot;
- force-open T10 followed by policy continuation and recovery;
- labels: long-term progress regression and clean-success-to-failure flip.

Snapshot provenance must include qpos, qvel, actuator state, mocap/userdata, simulation time, task/wrapper/controller state, termination counters, relevant RNG state, and a restoration hash/parity test.

Unknown rules:

- restore mismatch, action misalignment, unresolved target, ambiguous effect, or missing evidence must set `label_known_mask=0`;
- unknown rows must never be converted to negatives.

Initial smoke size should be small and stratified, not a full replay job.

Gate:

```text
RESTORE_PARITY                    = PASS
MATCHED_ACTION_ALIGNMENT          = PASS
EXACT_T10_DELIVERY                = PASS
UNKNOWN_MASKING                   = PASS
SHORT_HORIZON_EFFECT_MEASUREMENT  = PASS
CLOSED_LOOP_CONTINUATION_SMOKE    = PASS/HOLD_WITH_REASON
```

## Phase 3 — Causal dataset materialization

Goal: build a provenance-heavy C2g dataset whose primary label is command-open vulnerability.

Required labels:

```text
y_cmdopen_vulnerable
y_contact_loss
y_object_drop
y_progress_regression
y_success_flip
y_release_safe
y_contact_stable
y_grounding_confident
label_known_mask
teacher_confidence
teacher_reason_code
```

Primary dataset mode:

```text
NO_CONTEXT = temporal 25D + visual + language
```

Diagnostics:

```text
TEMPORAL_ONLY
TEMPORAL_PLUS_GLOBAL_VISUAL
TEMPORAL_PLUS_PATCH_VISUAL
NO_LANGUAGE
SHUFFLED_LANGUAGE
WRONG_LANGUAGE_CROSS_TASK
SUITE_ONLY
FULL_CONTEXT_LEGACY
PERMUTED_TASK_CONTEXT
```

Split modes:

- within-task episode split as an in-distribution reference;
- leave-one-task-out as the primary generalization test;
- leave-one-suite-out as a harder diagnostic.

Every fold must pass viability checks for known positives, known negatives, unknown rows, episode count, and task coverage. Split manifests and hashes must be frozen before training.

Sampling/weighting:

- equal total mass per task;
- equal total mass per episode within task;
- unknown masks separate from weights;
- cap or subsample highly redundant windows;
- report effective suite/task/label mass.

Gate:

```text
DATASET_PROVENANCE            = PASS
NO_EPISODE_LEAKAGE            = PASS
FOLD_VIABILITY                = PASS
KNOWN_UNKNOWN_ACCOUNTING      = PASS
TASK_EPISODE_BALANCE          = PASS
DATASET_MATERIALIZATION       = PASS
```

## Phase 4 — C2g model ladder

Goal: determine which added components provide real held-out-task value. Do not jump directly to the most complex model.

### Model A — C2g-Temporal

- causal 25D encoder only;
- establishes how much vulnerability can be inferred from command and physical state.

### Model B — C2g-Global

- causal 25D encoder;
- current/global pooled SigLIP feature;
- no language initially;
- measures visual value without task semantics.

### Model C — C2g-Global-Lang

- Model B plus language-conditioned FiLM/gating;
- no raw task index/hash;
- must beat wrong-language and language-dropout controls.

### Model D — C2g-PatchAttn primary candidate

- causal temporal encoder;
- compact projected SigLIP patch/spatial tokens;
- language-query cross-attention or top-k target-conditioned pooling;
- direct vulnerability head;
- auxiliary release-safe, contact-stable, grounding-confidence heads.

### Optional temporal refinement

Compare a single GRU with a dual-stream encoder:

- policy/action stream: action xyz, gripper command, streaks;
- physical-state stream: qpos/opening, EEF pose/velocity, deltas;
- fuse only after each stream is encoded.

The direct primary output is:

```text
p(command-open causes harm | clean history, current vision, language)
```

The deployment decision is based primarily on this calibrated probability, with release-safe and low-grounding-confidence used as vetoes.

Gate:

```text
TEMPORAL_BASELINE_TRAINED        = PASS
GLOBAL_VISUAL_INCREMENT          = MEASURED
LANGUAGE_INCREMENT               = MEASURED_WITH_CONTROLS
PATCH_ATTN_INCREMENT             = MEASURED
HELD_OUT_TASK_GENERALIZATION     = PASS/HOLD
```

## Phase 5 — Loss and trigger alignment

Primary losses:

- weighted masked BCE or focal loss for causal vulnerability;
- auxiliary contact/release/grounding losses;
- optional confidence weighting only after calibration analysis.

Episode losses:

- early emit before first known causal interval;
- miss of every known causal interval;
- any persistent emit in a fully-known negative episode;
- emit during release-safe intervals;
- optional temporal smoothness/hysteresis regularization.

The training-time persistence surrogate must match the online trigger, initially 2-of-3. A single isolated spike must not satisfy the episode miss objective.

Calibration:

- fit thresholds and temperature/isotonic calibration on validation tasks only;
- no per-task thresholds for the primary claim;
- suite-specific thresholds may appear only as secondary diagnostics.

Primary deployment metrics:

```text
episode any-emit false positive
attackable-episode no-emit
first-emit precision
first-emit timing error
early-trigger rate
release-safe emit fraction
2-of-3 persistent-trigger coverage
Brier score / ECE
suite macro and task macro
leave-one-task-out performance
```

Gate:

```text
WINDOW_METRICS                = REPORTED
EPISODE_METRICS               = REPORTED
CALIBRATION_ON_HELD_OUT_TASKS = PASS
PERSISTENCE_ALIGNMENT         = PASS
```

## Phase 6 — Online validation with matched controls

Primary conditions:

```text
CLEAN
TRUE_CMDOPEN_T10_C2G
CTRL_RANDOM_TIME_CMDOPEN_T10
ORACLE_CMDOPEN_T10
```

Optional placebo:

```text
RAND_ACTION_NOISE_T10_C2F
```

The critical comparison is TRUE versus random-time command-open with identical payload and T10 horizon. ORACLE provides an upper bound on timing quality.

Required:

- identical parent/init state;
- deterministic random-time seed;
- identical force-open payload;
- pre-trigger trace parity;
- action-level executed evidence;
- exact T10 delivery accounting;
- separate original and replication cohorts;
- exact paired contingency and McNemar/binomial tests.

Suggested sequence:

1. five-episode Track A plumbing smoke;
2. small Teacher-v2 grounding/replay smoke;
3. offline C2g pilot;
4. four-condition Object pilot on four parents;
5. review;
6. only then a preregistered Object replication;
7. Goal handled separately after clean-model smoke;
8. Spatial remains offline-audit-first until over-emission is resolved.

# Confidence assessment

These are engineering/scientific confidence estimates, not guaranteed performance claims.

## High confidence: 90% — diagnosis of the main C2f failure modes

Confidence is high that the dominant problems are label semantics, target grounding, context shortcuts, and window/deployment mismatch rather than insufficient model capacity. This is supported by the full Teacher-v1 audit and the suite-specific online pattern: Spatial over-emits under dense labels, Object under-emits under sparse/failed grounding, and several tasks have structurally zero positives.

## High confidence: 85% — Teacher-v2 will produce more scientifically valid targets

Matched counterfactual replay directly measures the deployed intervention and should be substantially better aligned than stable-carry heuristics. The remaining uncertainty is implementation complexity: deterministic restoration, target-progress metrics, and closed-loop divergence must be validated carefully.

## Moderately high confidence: 75% — no-context, task-held-out evaluation will reduce shortcut inflation

Removing raw task identity and requiring leave-one-task-out evaluation should expose genuine state inference. It may reduce headline in-distribution metrics, but that decrease would be scientifically healthy. The main risk is that language itself still carries task identity, hence the required wrong-language controls.

## Moderate confidence: 65% — C2g will improve online timing over random-time matched payload

The plan is designed to optimize the exact online question, so improvement is plausible. Confidence is not higher because some tasks may be vulnerable across broad intervals, OpenVLA may recover after opening, and causal labels may be sparse.

## Moderate confidence: 55% — patch-attention visual grounding will add material value

Patch/spatial tokens are better suited than global pooling for target-object grounding, but the camera view, resolution, and object scale may still limit performance. Temporal physical features may remain the dominant signal.

## Low-to-moderate confidence: 40% — one universal detector/gate will work equally well across all four suites

The task families differ substantially in manipulation type, release semantics, and vulnerability duration. A single primary model remains the goal, but a negative result here would not invalidate C2g; it may justify a shared backbone with explicit suite-agnostic state heads or carefully labeled secondary calibration diagnostics.

## What would falsify the upgrade hypothesis

The upgrade hypothesis should be rejected or revised if:

- counterfactual replay shows stable-carry timing has little relation to actual force-open harm;
- no-context models collapse while full-context models alone perform well on held-out tasks;
- wrong-language inputs do not reduce language-conditioned models, indicating task-prior reliance;
- ORACLE timing is not more harmful than random-time force-open, meaning detector timing has little exploitable value;
- the trained C2g fails to approach ORACLE first-trigger coverage despite adequate known labels;
- observed online differences disappear under matched-payload controls and replication.

# Codex execution prompt

Work from branch `assistant/c2g-static-fixes-20260710`. Audit before modifying code.

### First response required

Return:

```text
CURRENT_HEAD
DIFF_FROM_3D06FA55
STATIC_CODE_FINDINGS
TEST_PLAN
SERVER_ARTIFACTS_REQUIRED
DETECTOR_UPGRADE_PLAN_REVIEW
CONFIDENCE_RISK_REVIEW
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

6. Extend the architecture specification, without training, to define:

```text
C2g-Temporal
C2g-Global
C2g-Global-Lang
C2g-PatchAttn
optional dual-stream temporal encoder
```

For each model define exact inputs, parameter budget target, outputs, loss terms, required ablations, and the gate needed to advance it.

7. Extend the dataset specification, without materialization, to define:

- candidate strata and random noncandidate audit sampling;
- Tier-A and Tier-B causal labels;
- fold viability thresholds;
- wrong-language and language-dropout controls;
- per-task/episode effective-weight reporting;
- split manifest and hash schema.

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
DETECTOR_MODEL_LADDER_SPEC = PASS/HOLD/NOT_STARTED
CAUSAL_DATASET_SPEC = PASS/HOLD/NOT_STARTED
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
- Do not claim C2g performance from architecture or static tests.
- Do not collapse matched-action and closed-loop counterfactual effects into one label without preserving both components.
