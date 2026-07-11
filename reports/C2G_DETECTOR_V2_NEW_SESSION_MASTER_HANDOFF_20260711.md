# C2g Detector-v2 New-Session Master Handoff

Date: 2026-07-11

Repository: `Leo-6-maker/openvla-gripper-dutycycle-attack`

This document is the authoritative continuity handoff for a new ChatGPT/Codex-style session. It is deliberately self-contained, but the new session **must not trust it as a substitute for reading the repository**. The first task is an independent, read-only audit of the current repository, active branches, Draft PRs, core code, tests, reports, and server evidence before any additional implementation or execution.

---

## 0. Mandatory behavior for the new session

Before changing code, launching a server phase, or making a scientific claim, the new session must:

1. Query the actual remote heads of PRs #58, #59, and #60.
2. Fetch the latest head of `assistant/c2g-r5-provenance-materialization-20260711`.
3. Read this handoff and all documents listed in Section 5.
4. Inspect the repository tree and read the core modules listed in Section 6.
5. Inspect the full diff of PR #60 relative to its base.
6. Inspect the current CI status of the actual PR #60 head.
7. Verify that PRs #58, #59, and #60 remain Draft and unmerged.
8. Reconstruct the scientific contract from code and tests rather than relying only on prose.
9. Produce a read-only audit report before proposing or making any change.
10. Stop at the first unresolved provenance, semantic, leakage, resource, or compatibility gate.

The new session must separate three categories in every report:

- **repository/static evidence**;
- **live server/runtime evidence**;
- **scientific effectiveness evidence**.

Passing static tests is not live LIBERO/OpenVLA validation. Passing one runtime smoke is not evidence that Detector-v2 is effective.

---

## 1. Repository and PR topology

### 1.1 PR #58 — repository-side Detector-v2 pipeline

- PR: `#58`
- Branch: `assistant/c2g-clean-window-v2-20260710`
- Known head: `093efd9140d827a0df268c4106b35eafb32227d8`
- Base: `assistant/c2g-p0-static-patch-20260710`
- State: Draft, open, unmerged

PR #58 introduced the original repository-side end-to-end C2g pipeline:

```text
clean observation + clean causal history
  -> clean OpenVLA decode and policy intent
  -> clean gripper-critical detector + clean susceptibility gate
  -> fixed B-frame gripper-targeted VIS-PGD
  -> adversarial OpenVLA re-decode
  -> matched-load audit and paired analysis
```

It is a broad implementation PR and must not be merged merely because later server validation passed individual gates.

### 1.2 PR #59 — server remediation and active event tracking

- PR: `#59`
- Branch: `codex/c2g-strict-server-smoke-20260710`
- Known head: `ec35cabdef6c11ffcd8ee45b690334b20f8cbe32`
- Base: `assistant/c2g-clean-window-v2-20260710`
- State: Draft, open, unmerged

PR #59 contains server compatibility fixes, Goal-model provenance handling, BDDL/MuJoCo semantic fixes, active goal-event tracking, multi-target support, articulated-joint binding, and the one-shot `y_attack_start_b` correction.

### 1.3 PR #60 — provenance-bound R5 materialization gate

- PR: `#60`
- Branch: `assistant/c2g-r5-provenance-materialization-20260711`
- Base: `codex/c2g-strict-server-smoke-20260710`
- Known code head immediately before this handoff document was added:
  `2526871d72f428d5983f7f9529d15fd06acf690f`
- State: Draft, open, unmerged

The handoff-file commit itself advances the branch beyond the above known code head. The new session must query the actual remote head and bind all new artifacts to `git rev-parse HEAD`.

PR #60 adds a narrow repository-owned R5 gate:

- formal R4 dual-head provenance binder/verifier;
- exact HOLD-to-PASS one-shot drift validation;
- clean-collection mutation rejection;
- provenance-bound R5 multisuite materialization wrapper;
- CPU/read-only `preview` phase;
- focused pure-CPU tests;
- DeepSeek server-validation instructions.

No PR is authorized for merge at the time of this handoff.

---

## 2. Scientific objective and non-negotiable contract

### 2.1 Primary research objective

Detector-v2 is a **clean-only online detector of gripper-critical windows**. It operates before any attack and identifies clean trajectory windows in which a fixed-budget, gripper-targeted visual PGD attack is likely to disrupt gripper-dependent manipulation.

The actual attack remains:

```text
VIS-PGD on the visual input
-> target OpenVLA gripper OPEN token/logit behavior
-> adversarial OpenVLA re-decode
-> execute adversarially decoded action
```

The primary method does **not** directly overwrite the gripper actuator command.

### 2.2 Clean-only boundary

The following may be used by the offline Teacher:

- clean full trajectory;
- clean privileged simulator state;
- structured BDDL goal information;
- clean MuJoCo contacts;
- clean object/target positions and fixture joints;
- clean release/support evidence.

The following may be used by the online Student:

- clean RGB;
- causal 25D proprio/action history;
- clean 9D OpenVLA gripper policy-intent history;
- task language;
- clean visual/language embeddings;
- clean policy logits derived before the attack.

The following are forbidden from Teacher training labels, Student inputs, split selection, threshold calibration, susceptibility calibration, feature selection, and model selection:

- attacked images;
- attacked actions;
- attack outcomes;
- post-intervention state;
- counterfactual outcome labels;
- manual attacked-failure labels;
- future clean steps as Student input;
- task index/hash or suite identity as model features.

Counterfactual replay is optional post-hoc/oracle analysis only.

### 2.3 Teacher-v2 critical-window rule

Conceptually:

```text
y_gripper_critical_window =
    target_relevant
    AND gripper_dependency
    AND clean_close_intent
    AND lift_transport_or_constraint
    AND NOT release_safe
```

Unknown evidence remains unknown/null. It must never be silently converted to negative.

### 2.4 Physical criticality and susceptibility

The design decomposes attack opportunity into:

```text
Attack opportunity
= clean physical criticality
  x clean OpenVLA gripper flip susceptibility
```

The policy-intent stream includes OPEN/CLOSE probability mass, margin, entropy, top-1 semantics, and token ranks derived from the clean OpenVLA forward pass.

### 2.5 Fixed burst and persistence

Primary runtime behavior is frozen as:

- burst length: `B = 10` frames;
- persistence: contiguous 2-of-3 trigger rule;
- one-shot scheduling per episode;
- once triggered, run exactly B attack frames;
- no attack-result-dependent early stop or extension.

### 2.6 Multiple windows in LIBERO-10

Multiple gripper-critical intervals in a multi-stage episode are scientifically valid and are preserved.

The current label semantics are:

```text
y_gripper_critical_window:
    positive on every valid critical interval

y_burst_feasible:
    positive at every local position where a full B-frame burst can fit

y_attack_start_b:
    exactly one episode-global positive at the earliest burst-feasible interval
```

Therefore a LIBERO-10 episode may have two or more real critical intervals, but the primary matched-load protocol attacks at most once per episode. The second and later intervals remain training positives for criticality and burst feasibility; they are not deleted or relabeled as negative.

The online Student does not see Teacher privileged fields. It triggers at the first window that satisfies predicted criticality, release/grounding vetoes, susceptibility, and 2-of-3 persistence. It may miss the Teacher's first window and still trigger on a later real critical interval.

### 2.7 Matched-load primary experiment

Frozen core conditions:

```text
CLEAN
DET_GRIPPER_VIS_PGD
DET_RANDOM_VIS_ATTACK
RANDTIME_GRIPPER_VIS_PGD
RANDTIME_RANDOM_VIS_ATTACK
```

Primary compute-matched control:

```text
SHUFFLED_GRIPPER_GRADIENT
```

Matched conditions must freeze parent/init state, attacked-frame count, burst length, epsilon, step size, PGD iterations, preprocessing, projection/cast, temporal initialization, forward/backward/decode counts, and paired seeds.

### 2.8 Frozen evidence boundary

```text
D7_TABLE1 = STILL_FROZEN
```

No Detector-v2 work in PRs #58–#60 is authorized to rewrite the previously frozen D7/Table-1 evidence or claim Detector-v2 effectiveness.

---

## 3. Chronological implementation and validation history

### 3.1 Initial static pipeline

PR #58 implemented the broad repository-side pipeline, including:

- clean Teacher-v2 schema and label builder;
- Detector-v2 model and losses;
- clean policy-signal extraction;
- materializers and dataset audit;
- training and susceptibility calibration;
- clean-timing runner;
- VIS-PGD online integration;
- matched-load job builder, runner, audit, and analysis.

At that point, live server compatibility and real Teacher coverage were not established.

### 3.2 First server validation and S1 remediation

Codex server validation stopped correctly on:

- stale Goal shard manifest hashes;
- unsupported compact BDDL operator `turnon`;
- unresolved Panda `finger_joint1/2[_tip]` aliases.

PR #59 added:

- `turnon -> turn_on`, `turnoff -> turn_off` aliases;
- deterministic numbered-finger jaw identities;
- full Goal safetensors/index/header/hash audit;
- explicit load-only Goal manifest-v2 finalization;
- current-byte re-verification in later launchers.

### 3.3 First live A3 Teacher failure

The first tiny four-suite clean cohort produced 682/682 unknown rows. LIBERO-10 contained many target contacts but no target-bound progress evidence.

Root causes included:

- incomplete parsing of `:fixtures` and `:regions`;
- missing region-site to owner mapping;
- episode-level target compression;
- a single `primary_target` variable;
- no per-step active-target selection;
- no per-target progress baselines;
- articulated joint averaging instead of region-specific selection.

### 3.4 Event-aware scientific remediation

PR #59 then added:

- official objects/fixtures/regions parsing;
- fully qualified region sites;
- ordered goal-event bindings:
  `(operator, target, destination, interaction_site)`;
- per-step active-target selection from clean finger-target contacts;
- independent per-target lift/distance/joint baselines;
- region-derived articulated-joint selection;
- release-safe evidence bound to the active event;
- unknown-safe ambiguity handling;
- hard Student-feature denylist for Teacher-only event fields;
- mandatory goal-event scientific audit before materialization.

### 3.5 Live event-aware R3/R4 result before one-shot correction

A later tiny live collection succeeded technically and semantically:

- Object rows: 166
- Spatial rows: 72
- Goal rows: 248
- LIBERO-10 rows: 300
- total rows: 786
- known Teacher rows: 689
- unknown rows: 97
- known negatives: 474
- critical positives: 215
- triggerable/burst-feasible rows: 159
- contacted-target unresolved rows: 0
- active-progress unresolved rows: 0
- unknown-to-negative conversions: 0

The only remaining R4 violation was a LIBERO-10 episode with two disjoint critical intervals and two `y_attack_start_b` rows.

### 3.6 One-shot start correction

Commit chain on PR #59:

```text
34d428af263c52a2ce45df73496d82214315b34e
  make attack-start supervision episode-global one-shot

ec35cabdef6c11ffcd8ee45b690334b20f8cbe32
  add disjoint multi-target one-shot regression test
```

The correction preserves all critical and burst-feasible intervals but assigns `y_attack_start_b=true` only to the earliest episode-global feasible interval.

### 3.7 DeepSeek read-only live R4 re-audit

DeepSeek independently re-audited the frozen R3 collection without recollection, model loading, LIBERO execution, training, or attack.

Accepted result:

```text
CANONICAL_STATUS = PASS_C2G_CLEAN_WINDOW_V2_DRY_AUDIT
GOAL_EVENT_STATUS = PASS_C2G_GOAL_EVENT_TRACKING_AUDIT

ACTIVE_TARGET_KNOWN_ROWS = 689
CONTACTED_TARGET_UNRESOLVED_ROWS = 0
ACTIVE_PROGRESS_UNRESOLVED_ROWS = 0
KNOWN_TEACHER_ROWS = 689
CRITICAL_POSITIVE_ROWS = 215
TRIGGERABLE_WINDOWS = 159
UNKNOWN_ROWS = 97
KNOWN_NEGATIVE_ROWS = 474
UNKNOWN_CONVERTED_TO_NEGATIVE = 0

LIBERO_10_CRITICAL_INTERVAL_COUNT = 2
LIBERO_10_BURST_FEASIBLE_INTERVAL_COUNT = 2
LIBERO_10_BURST_FEASIBLE_ROWS = 94
LIBERO_10_ATTACK_START_ROWS = 1
MULTIPLE_ATTACK_START_VIOLATIONS = 0

EXPECTED_ONLY_CHANGE_CONFIRMED = true
UNEXPECTED_LABEL_DRIFT = false
SOURCE_COLLECTION_UNCHANGED = true
```

This establishes the live scientific correctness of event tracking and the one-shot label correction on the tiny cohort. It does not establish Detector-v2 effectiveness.

### 3.8 DeepSeek test-only compatibility commit

DeepSeek used an optional branch and commit:

```text
cf7a6a4849c518046e0bd8815324bca39007b329
```

It changed tests only:

- guarded a test-side torch import;
- changed one test import to `src.gripper_attack`.

This commit is **not** part of the production R5 branch and should not be merged automatically. Skipping torch-dependent tests is not proof of detector compatibility. Server reports must list passed, failed, and skipped tests separately.

### 3.9 PR #60 R5 code gate

PR #60 formalizes the accepted R4 evidence and blocks unsafe materialization through:

```text
tools/multisuite_detector/bind_c2g_r4_dual_head_provenance.py

tools/multisuite_detector/materialize_c2g_multisuite_dataset_bound.py

scripts/stageb/run_c2g_r5_bound_materialization.sh

tests/test_c2g_r4_dual_head_provenance.py

tests/test_c2g_r5_bound_materialization.py

reports/C2G_R5_DEEPSEEK_VALIDATION_HANDOFF_20260711.md
```

PR #60's existing GitHub workflows passed on the known pre-handoff code head. The new session must re-check CI on the actual current head after this documentation commit.

---

## 4. Exact accepted server evidence and hashes

### 4.1 Frozen event-aware R3 collection

Collection root:

```text
/mnt/sdc/dty_user/openvla_attack_evidence/c2g/
  c2g_event_v2_f5b2b2d1_20260711/clean_collection
```

Collection head:

```text
f5b2b2d14cdcd3359f8e3a1afa39a976df98ccc0
```

Frozen hashes:

```text
collection report =
c8a361bc0785669bad7263deaf024103efd3e7ae62d2178247c6ab9e3b5c2843

collection input manifest =
cf23edcbdb6506927dd2ec772d2bcb07c146f0851e9d55a61916c017e685849e

model binding report =
c159456b297720796c7fd0fa04c0ba3f6e77870f84947962f9efc4da07b80cf3

source file manifest before/after =
3f375669763087f74dd2c2d9966662b16e8777720b196add07e32ae83a7e55d9
```

The source file set consists of four `episode_metadata.json` files and four `step_records.jsonl` files.

### 4.2 Previous R4 HOLD artifacts

```text
old canonical HOLD report =
a4a5d133f000543375cd64633233033d50a3e1fd701370abe09fced397ec910b

old goal-event HOLD report =
7bad3273e153c43b5f100093441a6e27fd6f23f29463766c30cdb3efc605a62b

old HOLD binding =
14d79b89f4e06ee2509c971a9f7975e78ce9ffbfe1d4a5221daef021be76d0a1
```

These files must remain preserved and unmodified.

### 4.3 New live R4 PASS artifacts from DeepSeek

```text
canonical PASS report =
22e8572777203103c6ac78ca3e374b85110362bd47ebd068188664c332255a33

goal-event PASS report =
8fdeceea3bcc88d786d0fad1946c8ab8248f56866a223cf41538703ae85a1ffe

external DeepSeek dual-head binding =
d8c62c098a38d6f2108395a90b7987e0d7253f643ffd3da7b6e33b4ca067cd12

label builder =
5354e0fddc7e12c5c6f39d300ee366efdf98ad114e508efbc43a572443107555
```

The new session must locate the full absolute paths rather than infer them from hashes or abbreviated report strings.

### 4.4 Live resource ledger

Reported campaign totals before R5 preview:

```text
historical invalid clean episodes = 4
new event-aware clean episodes launched = 8
cumulative clean training episodes launched = 12
clean evaluation episodes launched = 0
attacked episodes launched = 0
training epochs = 0
datasets materialized = 0
counterfactual replays = 0
```

Reported cumulative OpenVLA model loads before the DeepSeek R4 read-only pass were 13, including prior attempts and one Goal load-only finalizer. DeepSeek's R4 re-audit added zero model loads.

### 4.5 Disk-space discrepancy requiring audit

Earlier server reports showed approximately 25.9 GiB free. The DeepSeek R4 report later showed approximately 396 GiB free.

Do not assume space increased. Before any materialization, report for both the collection filesystem and intended output filesystem:

```bash
df -B1T <collection path> <output parent>
df -i <collection path> <output parent>
```

Record device, filesystem type, total bytes, available bytes, and inodes. A likely explanation is that different paths or mount points were measured.

---

## 5. Required reading order for the new session

The new session must read these documents in order:

1. `reports/C2G_DETECTOR_V2_NEW_SESSION_MASTER_HANDOFF_20260711.md`
2. `reports/C2G_R5_DEEPSEEK_VALIDATION_HANDOFF_20260711.md`
3. `reports/C2G_A3_EVENT_TRACKING_REMEDIATION_AND_LONG_RANGE_PLAN_20260711.md`
4. `reports/C2G_DETECTOR_V2_LONG_RANGE_CODEX_PLAN_20260711.md`
5. `reports/C2G_SERVER_RESUME_FIX_SUMMARY_20260711.md`
6. `reports/C2G_S1_REMEDIATION_AND_RESUME_20260710.md`
7. `reports/C2G_CLEAN_WINDOW_STRICT_CODEX_EXECUTION_HANDOFF_20260710.md`
8. `reports/C2G_CLEAN_WINDOW_CANONICAL_PIPELINE_20260710.md`, if present on the inherited branch
9. PR #58 body and diff
10. PR #59 body, comments, and diff
11. PR #60 body, comments, and diff

The new session must note conflicts or stale statements across documents and resolve them using current code plus live evidence.

---

## 6. Required repository audit map

The new session must inspect at least the following areas before continuing.

### 6.1 Core clean-only schema and model

```text
src/gripper_attack/c2g_clean_window_schema.py
src/gripper_attack/c2g_clean_policy_signals.py
src/gripper_attack/c2g_gripper_critical_window_detector.py
src/gripper_attack/c2g_matched_load_manifest.py
```

Confirm:

- allowed/forbidden Student features;
- clean policy feature ordering;
- model heads;
- causal behavior;
- loss masking;
- one-shot scheduler;
- fixed burst behavior;
- matched-load fields.

### 6.2 Teacher/event semantics

```text
src/gripper_attack/c2g_bddl_metadata.py
src/gripper_attack/c2g_clean_event_tracking.py
src/gripper_attack/c2g_clean_mechanism.py
src/gripper_attack/c2g_semantic_aliases.py
src/gripper_attack/c2g_teacher_v2_contact_identity.py
src/gripper_attack/c2g_teacher_v2_target_resolution.py
tools/multisuite_detector/c2g_clean_window_label_builder.py
```

Confirm:

- BDDL objects/fixtures/regions parsing;
- region-site owner binding;
- ordered goal-event representation;
- per-step active-target selection;
- bilateral target contact semantics;
- target-specific progress;
- articulated-joint selection;
- release-safe veto;
- unknown-safe behavior;
- all intervals retained;
- one episode-global start only.

### 6.3 Collection and provenance

```text
scripts/stageb/collect_c2g_clean_window_rollouts_event_v2.py
scripts/stageb/collect_c2g_clean_window_rollouts_strict.py
scripts/stageb/collect_c2g_clean_window_rollouts_release.py
scripts/stageb/bind_c2g_collection_model_provenance.py
scripts/stageb/verify_c2g_collection_model_provenance.py
scripts/stageb/build_c2g_suite_model_map.py
scripts/stageb/build_c2g_suite_model_map_strict.py
scripts/stageb/verify_c2g_suite_model_map_strict.py
scripts/stageb/finalize_c2g_goal_model_manifest_v2.py
```

Confirm:

- suite-isolated subprocess behavior;
- exact model-byte binding;
- Goal manifest-v2 contract;
- clean collection immutability;
- no attack/outcome access.

### 6.4 Audits and R4/R5 binding

```text
tools/multisuite_detector/audit_c2g_clean_window_v2.py
tools/multisuite_detector/audit_c2g_goal_event_tracking.py
tools/multisuite_detector/bind_c2g_r4_dual_head_provenance.py
```

Confirm:

- canonical and goal-event PASS fields;
- read errors and violations are fail-closed;
- exact expected one-shot drift only;
- source collection before/after hashes;
- collection head and audit head are both explicit;
- old HOLD artifacts are hash-bound;
- attack/outcome and resource counters are zero.

### 6.5 Materialization and trainability

```text
tools/multisuite_detector/materialize_c2g_clean_window_dataset.py
tools/multisuite_detector/materialize_c2g_multisuite_dataset.py
tools/multisuite_detector/materialize_c2g_multisuite_dataset_bound.py
tools/multisuite_detector/validate_c2g_clean_window_dataset.py
```

Confirm:

- only clean artifacts are read;
- suite-specific OpenVLA/SigLIP embeddings are extracted with the correct model;
- no task/suite identity enters feature tensors;
- source artifacts and code are hash-bound;
- output must be external and empty;
- R4 binding is reverified before and after materialization;
- dataset split and trainability gates are fail-closed.

### 6.6 Training and calibration

```text
tools/multisuite_detector/train_c2g_clean_window_detector.py
tools/multisuite_detector/calibrate_c2g_clean_susceptibility.py
tools/multisuite_detector/run_c2g_clean_window_folds.py
```

Confirm:

- training never uses attack outcomes;
- checkpoint binds dataset/config/code hashes;
- thresholds are selected from clean validation data;
- susceptibility uses clean policy signals;
- one-epoch smoke can reload the checkpoint and produce finite outputs.

Training has not started and is not currently authorized.

### 6.7 Online runtime and attack

Inspect the current C2g online runner, clean-timing runner, existing OpenVLA evaluation runner, and:

```text
src/gripper_attack/attack_adapter.py
scripts/stageb/run_c2g_clean_timing_jobs_strict.py
scripts/stageb/build_c2g_matched_load_jobs_release.py
scripts/stageb/run_c2g_matched_load_jobs_map_release.py
scripts/stageb/audit_c2g_matched_load_run_release.py
scripts/stageb/analyze_c2g_matched_load_results.py
```

Confirm:

- clean decode occurs before detector evaluation;
- OPEN/CLOSE semantics are derived from executable model decoding;
- visual PGD is the primary payload;
- adversarial re-decode occurs before environment execution;
- fixed B-frame attack is immutable;
- frozen CLEAN is never rerun or overwritten;
- closed-world audit covers all expected jobs.

### 6.8 Canonical launchers

```text
scripts/stageb/run_c2g_clean_window_pipeline.sh
scripts/stageb/run_c2g_clean_window_pipeline_strict.sh
scripts/stageb/run_c2g_r5_bound_materialization.sh
```

For the immediate next stage, use only:

```text
scripts/stageb/run_c2g_r5_bound_materialization.sh
```

The currently authorized phase is `preview`, not `run`.

### 6.9 Tests

Read all `tests/test_c2g*.py`. At minimum inspect:

```text
tests/test_c2g_event_teacher_integration.py
tests/test_c2g_goal_event_tracking.py
tests/test_c2g_clean_window_v2.py
tests/test_c2g_r4_dual_head_provenance.py
tests/test_c2g_r5_bound_materialization.py
```

Report passed, failed, and skipped tests separately. Missing torch that skips detector tests is a HOLD.

---

## 7. Current stage status

```text
R0 repository/static                    = PASS historically; rerun on actual head
R1 live assets/model bytes              = PASS
R2 four-suite manifests                 = PASS
R3 event-aware clean collection         = PASS
R4 canonical Teacher audit              = PASS live
R4 goal-event audit                     = PASS live
R4 external dual-head binding           = PASS live
R4 repository-owned binder              = PASS static/CI
R5 provenance-bound materializer        = PASS static/CI
R5 server preview                       = NOT RUN
R5 actual materialization               = NOT AUTHORIZED / NOT RUN
Dataset trainability audit              = NOT RUN
Detector training                       = NOT STARTED
Training epochs                         = 0
Susceptibility calibration              = NOT RUN
Detector-only clean timing              = NOT RUN
Matched-load online smoke               = NOT RUN
Attacked episodes                       = 0
Scientific effectiveness                = NOT ESTABLISHED
D7_TABLE1                               = STILL FROZEN
```

---

## 8. Immediate authorized next work

The immediate objective is **not** training. It is to validate PR #60 on the server through a read-only R5 preview.

### 8.1 N0 — independent repository audit

The new session must first produce a report containing:

```text
ACTUAL_PR58_HEAD
ACTUAL_PR59_HEAD
ACTUAL_PR60_HEAD
PR58_STATE
PR59_STATE
PR60_STATE
PR60_BASE_ANCESTOR_CHECK
PR60_CHANGED_FILES
PR60_DIFF_REVIEW
CURRENT_CI_STATUS
SCIENTIFIC_CONTRACT_RECONSTRUCTED
REPO_AUDIT_FINDINGS
GO_HOLD_TO_BEGIN_SERVER_PREVIEW
```

No code change before this report unless a read-only inspection is impossible due to a trivial connector/path problem.

### 8.2 N1 — server static gate

Checkout the actual PR #60 head and run:

```bash
git fetch origin --prune
git checkout assistant/c2g-r5-provenance-materialization-20260711
git reset --hard origin/assistant/c2g-r5-provenance-materialization-20260711

export EXECUTED_HEAD="$(git rev-parse HEAD)"
export PYTHONPATH="$(git rev-parse --show-toplevel)/src:$(git rev-parse --show-toplevel)${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""

git status --short
git diff --check

python -m unittest -v \
  tests.test_c2g_r4_dual_head_provenance \
  tests.test_c2g_r5_bound_materialization \
  tests.test_c2g_event_teacher_integration \
  tests.test_c2g_goal_event_tracking

python -m unittest discover -s tests -p 'test_c2g*.py' -v

python -m py_compile \
  tools/multisuite_detector/bind_c2g_r4_dual_head_provenance.py \
  tools/multisuite_detector/materialize_c2g_multisuite_dataset_bound.py

bash -n scripts/stageb/run_c2g_r5_bound_materialization.sh
```

Requirements:

- zero failed tests;
- zero inappropriate skipped detector tests;
- clean worktree;
- no GPU/model load/rollout/materialization.

### 8.3 N2 — rebuild repository-owned R4 binding

Use the frozen R3 collection and the existing live PASS reports plus preserved HOLD reports.

Run the repository binder in `build` and `verify` modes. Bind:

```text
collection_head = f5b2b2d14cdcd3359f8e3a1afa39a976df98ccc0
audit_head = actual PR #60 git rev-parse HEAD
```

Required status:

```text
PASS_C2G_R4_DUAL_HEAD_PROVENANCE_BINDING
```

The binding must be written outside the repository and outside the frozen collection.

### 8.4 N3 — R5 preview only

Set the exact verified paths and run:

```bash
bash scripts/stageb/run_c2g_r5_bound_materialization.sh preview
```

Required status:

```text
PASS_C2G_R5_BOUND_MATERIALIZATION_DRY_RUN
```

Preview may hash model files. It must not:

- load model tensors;
- create LIBERO environments;
- launch clean rollouts;
- write an NPZ dataset;
- train;
- calibrate;
- attack.

### 8.5 Current resource caps

For N0–N3:

```text
NEW_CLEAN_EPISODES = 0
OPENVLA_MODEL_LOADS = 0
LIBERO_ENVIRONMENTS = 0
DATASETS_MATERIALIZED = 0
TRAINING_EPOCHS = 0
CLEAN_EVAL_EPISODES = 0
ATTACKED_EPISODES = 0
GPU_JOBS = 0
COUNTERFACTUAL_REPLAYS = 0
D7_MODIFICATIONS = 0
```

---

## 9. Work that remains unauthorized

Until the R5 preview report is reviewed and an explicit new authorization is given, do not run:

- R5 materialization `run`;
- any OpenVLA/SigLIP embedding extraction;
- new clean collection;
- dataset trainability audit on a newly generated dataset;
- detector training;
- threshold or susceptibility calibration;
- clean timing;
- parent binding;
- matched-load job execution;
- VIS-PGD;
- attacked rollouts;
- full folds;
- full CLEAN2000;
- full online matrix;
- PR merge;
- D7 changes.

---

## 10. Long-range plan after R5 preview

The following plan is informative only. Each expensive transition requires a fresh review and authorization.

### 10.1 R5 actual bounded materialization

Potential next authorization:

- reuse the four frozen event-aware episodes;
- materialize at most one episode per suite initially;
- use suite-specific OpenVLA/SigLIP models in suite-isolated subprocesses;
- generate a provenance-bound combined NPZ;
- write outputs externally;
- preserve at least 15 GiB free-space headroom;
- run no LIBERO rollout and no attack.

### 10.2 Dataset trainability audit

Require:

- episode-level split leakage = 0;
- nonempty train/val/test;
- known positives and known negatives in required splits;
- 2-of-3 triggerable support;
- finite arrays;
- target/mask cardinality closure;
- no task/suite identity in model tensors;
- clean-only provenance.

The four-episode tiny dataset may fail trainability due to insufficient support. That is not a reason to weaken the gate.

### 10.3 Adaptive clean support

Campaign historical cap was 40 clean training episodes.

Already reported:

```text
historical invalid = 4
new event-aware launched = 8
cumulative launched = 12
```

Before collecting more, independently verify the ledger and remaining authorized count. Earlier planning allowed up to 36 new event-aware episodes after the four historical invalid episodes, but 8 event-aware episodes have already been launched. Do not infer the remaining budget without reconstructing the exact campaign ledger.

If expansion is later authorized, add clean data in bounded stages and rerun all collection binding, Teacher audits, materialization, and trainability gates after every expansion.

### 10.4 One-epoch training smoke

Only after a PASS trainability report:

- exactly 1 epoch;
- strict checkpoint reload;
- one validation batch;
- finite losses/logits/probabilities;
- clean-only threshold calibration;
- clean-only susceptibility calibration;
- checkpoint/dataset/config/code hash closure.

This would establish trainability and runtime compatibility, not effectiveness.

### 10.5 Detector-only clean timing

At most four clean evaluation parents in the bounded smoke:

- zero attacks;
- freeze clean trajectories/init states/checkpoint/config;
- retain no-emit and burst-infeasible parents;
- require at least one emitted and burst-feasible parent before online smoke;
- never tune thresholds using attack outcomes.

### 10.6 One-parent matched-load runtime smoke

Only after separate authorization:

- reuse frozen CLEAN parent;
- do not rerun CLEAN;
- execute four attacked conditions;
- exactly B contiguous attack frames;
- verify Linf budget;
- verify forward/backward/decode counts;
- verify objective/seed pairings;
- verify pre-trigger parity;
- run closed-world audit immediately.

An `N=1` smoke proves runtime closure only.

### 10.7 Scientific detector evaluation

Requires later authorization and larger data:

- Teacher coverage tables by suite/task/mechanism;
- model ladder and ablations;
- within-task reference;
- leave-one-task-out primary evaluation;
- leave-one-suite-out diagnostic;
- at least three seeds;
- matched-load online pilot;
- confirmatory 2x2 timing x objective matrix.

---

## 11. Known risks and audit questions

The new session must explicitly assess these risks.

### 11.1 Test environment correctness

A test suite with torch-dependent classes skipped is not a full detector test. Verify the actual server Python environment and report skips separately.

### 11.2 Import-path consistency

The repository uses `src` packaging with pytest paths. Direct scripts bootstrap repository paths in several places. Check that tests and server launchers use consistent import conventions without hiding production import failures.

### 11.3 R4 audit provenance

The clean collection was produced at one head and audited at a later head. The formal dual-head binder is mandatory before materialization.

### 11.4 Model-byte provenance

All four suite models and the Goal manifest must be rehashed at preview time. A PASS JSON without current-byte verification is insufficient.

### 11.5 Tiny-cohort representativeness

R4 PASS demonstrates semantic correctness on four tiny episodes, not broad suite coverage. Materialization/training may still fail due to support or split imbalance.

### 11.6 Multiple-window semantics

Do not regress to one start per interval, and do not delete later critical intervals. Keep:

```text
all critical intervals
all burst-feasible intervals
one earliest episode-global start
one runtime burst per episode
```

### 11.7 Split semantics

The current tiny pipeline uses within-task splitting for the immediate smoke. Formal science requires LOTO primary evaluation. Do not interpret within-task tiny results as generalization.

### 11.8 Disk/mount interpretation

Resolve the 25.9 GiB versus 396 GiB discrepancy before writing embeddings or datasets.

### 11.9 PR structure

PR #60 is layered on PR #59, which is layered on PR #58. Do not merge the top PR without understanding base dependencies and reviewing the combined diff.

---

## 12. Allowed code changes for the next session

Before R5 preview, only minimal code-owned fixes are acceptable:

- direct-script import/path compatibility;
- deterministic report formatting;
- R4/R5 hash/provenance verification;
- output-path safety;
- read-only preview behavior;
- focused tests for the above.

Any fix must:

1. be made on a new branch based on the actual PR #60 head;
2. have a narrow diff;
3. include a focused regression test;
4. rerun the entire `test_c2g*.py` suite;
5. rerun affected compile/Bash gates;
6. preserve Draft/unmerged PR status;
7. be reported with exact commit SHA and rationale.

The following require explicit scientific review and must not be changed silently:

- Teacher-v2 label semantics;
- active-target selection;
- progress/contact/release definitions;
- unknown handling;
- Detector inputs or architecture;
- loss;
- threshold objective;
- susceptibility objective;
- 2-of-3 persistence;
- burst length;
- one-shot behavior;
- attack objective;
- condition matrix;
- matched-load contract;
- interpretation of outcomes.

---

## 13. Hard stop conditions

Return HOLD immediately for:

- remote/local head mismatch;
- dirty worktree;
- failed or improperly skipped detector tests;
- source collection mutation;
- ambiguous frozen collection root;
- R4 PASS report/hash mismatch;
- old HOLD artifact mismatch;
- collection-head mismatch;
- audit-head mismatch;
- label-builder mismatch;
- current model-byte mismatch;
- Goal manifest mismatch;
- nonempty output directory;
- less than 15 GiB free on the actual output filesystem;
- attack/outcome leakage;
- unknown-to-negative conversion;
- any model load, rollout, dataset write, training, or attack during preview;
- unexpected label drift;
- second LIBERO-10 critical interval disappearing;
- request to weaken scientific gates.

---

## 14. Required first response from the new session

Before any work, the new session should respond with:

```text
REPOSITORY =
DEFAULT_BRANCH =
PR58_ACTUAL_HEAD =
PR59_ACTUAL_HEAD =
PR60_ACTUAL_HEAD =
PR58_STATE =
PR59_STATE =
PR60_STATE =
PR60_BASE =
PR60_BASE_ANCESTOR_CHECK =
PR60_CURRENT_CI =

DOCUMENTS_READ =
CORE_MODULES_READ =
TESTS_REVIEWED =
PR_DIFFS_REVIEWED =

SCIENTIFIC_OBJECTIVE_RESTATED =
CLEAN_ONLY_BOUNDARY_RESTATED =
MULTI_WINDOW_ONE_SHOT_RULE_RESTATED =
ATTACK_PAYLOAD_RESTATED =
MATCHED_LOAD_MATRIX_RESTATED =

LIVE_EVIDENCE_ACCEPTED =
LIVE_EVIDENCE_NOT_YET_AVAILABLE =
DETECTOR_TRAINING_STARTED = NO
DATASETS_MATERIALIZED = 0
ATTACKED_EPISODES = 0
D7_TABLE1 = STILL_FROZEN

P0_REPO_FINDINGS =
P1_REPO_FINDINGS =
GO_HOLD_TO_START_R5_PREVIEW =
```

The new session must not skip the repository audit and immediately run server commands.

---

## 15. Required server-preview final report

After N0–N3, return:

```text
REMOTE_HEAD =
EXECUTED_HEAD =
WORKTREE_CLEAN =

TESTS_PASSED =
TESTS_FAILED =
TESTS_SKIPPED =
PY_COMPILE =
BASH_SYNTAX =

COLLECTION_ROOT =
COLLECTION_HEAD =
COLLECTION_REPORT_SHA256 =
COLLECTION_INPUT_MANIFEST_SHA256 =
MODEL_BINDING_REPORT_SHA256 =
COLLECTION_UNCHANGED =

R4_BINDING_STATUS =
R4_BINDING_PATH =
R4_BINDING_SHA256 =
R4_BINDING_AUDIT_HEAD =

R5_PREVIEW_STATUS =
R5_PREVIEW_COMMAND =
MODEL_BYTES_VERIFIED =
OUTPUT_DIRECTORY_EMPTY =
COLLECTION_FILESYSTEM =
OUTPUT_FILESYSTEM =
FREE_BYTES_BEFORE =
FREE_BYTES_AFTER =
INODES_AVAILABLE =

NEW_CLEAN_EPISODES = 0
OPENVLA_MODEL_LOADS = 0
LIBERO_ENVIRONMENTS = 0
DATASETS_MATERIALIZED = 0
TRAINING_EPOCHS = 0
CLEAN_EVAL_EPISODES = 0
ATTACKED_EPISODES = 0
GPU_JOBS = 0

P0_FINDINGS =
P1_FINDINGS =
SCIENTIFIC_CONTRACT_CHANGES = NONE
D7_TABLE1 = STILL_FROZEN
GO_HOLD_NEXT_STAGE = GO_R5_MATERIALIZATION_RUN_REVIEW | HOLD_<REASON>
```

---

## 16. Copyable prompt for the new conversation

```text
You are taking over the C2g Detector-v2 work in:
Leo-6-maker/openvla-gripper-dutycycle-attack

Do not continue from memory or from the chat summary alone.
Use GitHub to independently inspect the entire repository state first.

Mandatory first steps:
1. Fetch PRs #58, #59, and #60, including current heads, bases, states, diffs, comments, and CI.
2. Read:
   reports/C2G_DETECTOR_V2_NEW_SESSION_MASTER_HANDOFF_20260711.md
   reports/C2G_R5_DEEPSEEK_VALIDATION_HANDOFF_20260711.md
   reports/C2G_A3_EVENT_TRACKING_REMEDIATION_AND_LONG_RANGE_PLAN_20260711.md
   reports/C2G_DETECTOR_V2_LONG_RANGE_CODEX_PLAN_20260711.md
3. Inspect the core Teacher, event-tracking, Detector, materialization, training,
   online VIS-PGD, matched-load, provenance, launcher, and test files listed in the
   master handoff.
4. Reconstruct and report the clean-only scientific contract and the multi-window
   one-shot logic from code/tests.
5. Do not change code or run the server until you return the required first audit
   report in Section 14 of the master handoff.

Accepted live progress:
- R0–R3 passed on the bounded live server path.
- Event-aware Teacher-v2 live R4 canonical audit passed.
- Goal-event live R4 audit passed.
- 689 known rows, 215 critical positives, 159 triggerable rows, 97 unknown rows,
  474 known negatives, zero unknown-to-negative conversion.
- LIBERO-10 retains two critical intervals and two burst-feasible intervals, but
  only the earliest episode-global y_attack_start_b remains positive.
- Frozen R3 collection was unchanged by the read-only re-audit.

Not completed:
- R5 server preview on PR #60.
- Actual dataset materialization.
- Dataset trainability audit.
- Detector training.
- Calibration.
- Clean timing.
- Online VIS-PGD smoke.
- Scientific effectiveness evaluation.

Current code branch to audit:
assistant/c2g-r5-provenance-materialization-20260711
Draft PR #60.
Always use the actual remote head, not a copied SHA.

Immediate authorized target after repository audit:
- rebuild the repository-owned R4 dual-head provenance binding;
- run only:
  bash scripts/stageb/run_c2g_r5_bound_materialization.sh preview
- no model loading, rollout, dataset write, training, or attack.

Roles:
- ChatGPT/new session owns repository review, code design, and debugging.
- DeepSeek owns bounded server execution and evidence collection.
- DeepSeek must not redesign scientific semantics.

Maintain:
SCIENTIFIC_CONTRACT_CHANGES = NONE
D7_TABLE1 = STILL_FROZEN
```

---

## 17. Final authoritative status at handoff

```text
REPOSITORY_SIDE_DETECTOR_V2_PIPELINE = IMPLEMENTED
LIVE_LIBERO_OPENVLA_COLLECTION = PASS_BOUNDED
LIVE_EVENT_AWARE_TEACHER = PASS_BOUNDED
LIVE_MULTI_TARGET_TRACKING = PASS_BOUNDED
LIVE_ONE_SHOT_LABEL_CORRECTION = PASS_BOUNDED
R4_CANONICAL = PASS
R4_GOAL_EVENT = PASS
R4_EXTERNAL_DUAL_HEAD_BINDING = PASS
R4_REPOSITORY_BINDER = PASS_STATIC
R5_BOUND_MATERIALIZER = PASS_STATIC
R5_SERVER_PREVIEW = NOT_RUN
DATASET_MATERIALIZATION = NOT_RUN
DETECTOR_TRAINING = NOT_STARTED
TRAINING_EPOCHS = 0
CLEAN_TIMING = NOT_RUN
ONLINE_VIS_PGD = NOT_RUN
SCIENTIFIC_EFFECTIVENESS = NOT_ESTABLISHED
PR58 = DRAFT_UNMERGED
PR59 = DRAFT_UNMERGED
PR60 = DRAFT_UNMERGED
D7_TABLE1 = STILL_FROZEN
NEXT_GATE = INDEPENDENT_REPO_AUDIT_THEN_R5_PREVIEW
```
