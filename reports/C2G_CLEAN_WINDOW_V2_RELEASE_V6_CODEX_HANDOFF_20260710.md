# C2g Clean-Window Detector v2 — Release-v6 Codex Handoff

Date: 2026-07-10

## Repository

```text
repository = Leo-6-maker/openvla-gripper-dutycycle-attack
branch = assistant/c2g-clean-window-v2-20260710
pull_request = #58 (Draft)
canonical_entry_point = scripts/stageb/run_c2g_clean_window_release_v6.sh
```

Always record and bind the exact current `git rev-parse HEAD`. Do not infer the
current head from this document.

## Scientific objective

The detector uses only clean rollout information to select a gripper-critical
window for a fixed visual PGD burst:

```text
clean RGB + clean causal proprio/action history + clean OpenVLA policy intent
  -> target-relevant gripper-critical detector
  -> clean-only policy susceptibility gate
  -> fixed B-frame gripper-targeted VIS-PGD
  -> adversarial OpenVLA re-decode
  -> executed action
```

Detector labels, threshold selection, susceptibility calibration, split selection,
and model selection must not use attacked outcomes, post-intervention state,
counterfactual replay outcomes, or manual attacked-failure labels.

## Canonical phases

```bash
bash scripts/stageb/run_c2g_clean_window_release_v6.sh <phase>
```

```text
models
manifests
collect
audit
materialize
dataset_audit
train
calibrate
folds
clean_timing
bind_parents
build_jobs
run_jobs
audit_jobs
analyze
all
```

Do not run `all` during initial validation.

## Hard boundaries for first Codex pass

```text
GPU_EPISODES_AUTHORIZED = 0
LIBERO_ROLLOUTS_AUTHORIZED = 0
OPENVLA_INFERENCE_AUTHORIZED = 0
FULL_DATASET_MATERIALIZATION_AUTHORIZED = 0
DETECTOR_TRAINING_AUTHORIZED = 0
ONLINE_ATTACK_MATRIX_AUTHORIZED = 0
D7_TABLE1 = STILL_FROZEN
```

The first pass is CPU/static/read-only only.

## V0 — Repository and CI audit

Record:

```bash
git rev-parse HEAD
git rev-parse HEAD^
git merge-base --is-ancestor e3bec3b82ac104c633d8dacc5fd27f9cf30a7e85 HEAD
git status --short
git diff --check
```

Run all GitHub-equivalent C2g tests, including:

```text
test_c2g_clean_window_v2
test_c2g_clean_window_server_audit
test_c2g_clean_window_e2e
test_c2g_clean_window_pipeline_tools
test_c2g_clean_susceptibility
test_c2g_dataset_trainability
test_c2g_matched_load_strict
test_c2g_result_analysis
test_c2g_clean_collector_strict
test_c2g_release_manifest_builder
test_c2g_release_job_builder
test_c2g_release_v2
test_c2g_model_provenance
```

Run Bash syntax checks on:

```text
run_c2g_clean_window_end_to_end.sh
run_c2g_clean_window_full.sh
run_c2g_clean_window_production.sh
run_c2g_clean_window_release_v2.sh
run_c2g_clean_window_release_v3.sh
run_c2g_clean_window_release_v4.sh
run_c2g_clean_window_release_v5.sh
run_c2g_clean_window_release_v6.sh
```

Only release-v6 is canonical; earlier wrappers remain compatibility/history layers.

## V1 — Read-only model and asset audit

Identify the audited Goal model manifest and export:

```bash
export WORK_ROOT=/ABSOLUTE/EXTERNAL/c2g_release_v6_validation
export GOAL_MODEL_MANIFEST=/ABSOLUTE/PATH/goal_model_manifest.json
```

Run only:

```bash
bash scripts/stageb/run_c2g_clean_window_release_v6.sh models
```

Required:

```text
PASS_C2G_STRICT_SUITE_MODEL_MAP
all four suite model paths exist
all referenced model weight shards exist
full weight-shard SHA256 recorded
Goal model manifest status PASS and path matched
OpenVLA models loaded = 0
GPU jobs launched = 0
```

Then run the existing static BDDL/MuJoCo inventory against official mounted roots.
Require zero parser errors, supported-or-explicitly-held operators, and resolved
left/right gripper identities.

## V2 — Manifest preregistration only

Use tiny counts first:

```bash
export TRAIN_STATES_PER_TASK=1
export EVAL_STATES_PER_TASK=1
export MAX_TASKS_PER_SUITE=1
bash scripts/stageb/run_c2g_clean_window_release_v6.sh manifests
```

Audit:

```text
four suites represented
exact five-part parent keys
train/eval state identities disjoint
no attacked outcome use
no rollout launched
manifest hashes frozen
```

## V3 — One explicitly authorized clean episode per suite

This stage requires separate authorization because it loads OpenVLA and creates
LIBERO clean rollouts, though it executes no attack.

Before authorization, inspect:

```text
collect_c2g_clean_window_rollouts_release.py
collect_c2g_clean_window_rollouts_strict.py
collect_c2g_clean_window_rollouts.py
```

Required runtime properties:

```text
strict suite model verification occurs before loading
collector internal suite paths equal frozen map
Goal manifest is valid
canonical 25D feature names/order
no visual attacker constructed
no attacked observation/action/outcome field
clean policy logits are obtained before any detector/attack path
```

After authorization:

```bash
export MAX_TRAIN_EPISODES=1
bash scripts/stageb/run_c2g_clean_window_release_v6.sh collect
bash scripts/stageb/run_c2g_clean_window_release_v6.sh audit
```

Do not continue if any suite has zero known rows solely because the collector failed
to materialize available clean privileged evidence.

## V4 — Tiny four-suite materialization

Requires separate authorization to load the four suite-specific OpenVLA encoders.
Run:

```bash
bash scripts/stageb/run_c2g_clean_window_release_v6.sh materialize
bash scripts/stageb/run_c2g_clean_window_release_v6.sh dataset_audit
```

Require:

```text
suite-specific model bytes reverified before materialization
no episode split leakage
known positive and negative support in train/val/test
triggerable 2-of-3 positive support
feature dimensions identical across merged suites
student feature payload contains no task/suite shortcut or privileged Teacher field
```

Tiny manifests may legitimately HOLD for insufficient split support. That is not a
reason to weaken the gate; increase only the clean sample count after review.

## V5 — One-epoch training smoke

Requires separate authorization. Use one epoch and a small batch first. Verify:

```text
checkpoint schema = c2g.clean_window_checkpoint.2026-07-10.v1
checkpoint reload strict=true
checkpoint dataset SHA matches
validation-only detector thresholds exported
no attacked outcome input or metric used for selection
```

Then run clean susceptibility calibration:

```bash
bash scripts/stageb/run_c2g_clean_window_release_v6.sh calibrate
```

Require:

```text
schema = c2g.clean_susceptibility_calibration.2026-07-10.v1
uses_attack_outcomes = false
checkpoint and training-report hashes updated
runtime prefers checkpoint_clean_validation
```

## V6 — Detector-only online timing smoke

Requires separate authorization for one clean parent per suite. No attack is
delivered.

```bash
bash scripts/stageb/run_c2g_clean_window_release_v6.sh clean_timing
bash scripts/stageb/run_c2g_clean_window_release_v6.sh bind_parents
bash scripts/stageb/run_c2g_clean_window_release_v6.sh build_jobs
```

Audit:

```text
CLEAN attack_delivery_count = 0
CLEAN parent never rewritten
no-emit parents remain in denominator ledger
burst-infeasible late starts remain in denominator ledger
clean parent SHA and official init-state SHA frozen
CLEAN objective_seed equals preregistered eval_seed
attack objective seeds paired across DET/RANDTIME
random timing differs from detector timing
```

## V7 — One-parent five-condition command dry run

Run the matched launcher with its dry-run option directly. Verify command closure,
exact model path, Goal manifest argument, condition names, attack load, seeds, and
output paths. Do not launch the GPU smoke until this report is reviewed.

## V8 — Explicitly authorized one-parent online GPU smoke

Primary control currently implemented:

```text
SHUFFLED_GRIPPER_GRADIENT
```

Do not select a different control family unless its runtime implementation and
compute audit are separately completed.

After execution, immediately run:

```bash
bash scripts/stageb/run_c2g_clean_window_release_v6.sh audit_jobs
bash scripts/stageb/run_c2g_clean_window_release_v6.sh analyze
```

Require the runtime audit to PASS before interpreting any task outcome.

## Required Codex return format

```text
BRANCH
BASE_SHA
HEAD_SHA
WORKTREE_CLEAN
REMOTE_HEAD_MATCH

CPU_TESTS
PY_COMPILE
BASH_SYNTAX
GITHUB_CI

STRICT_SUITE_MODEL_MAP
FULL_WEIGHT_SHARD_BINDING
GOAL_MODEL_MANIFEST_BINDING
LIVE_BDDL_OPERATOR_CENSUS
LIVE_MUJOCO_CONTACT_ALIAS_CENSUS

RELEASE_PARENT_MANIFESTS
TRAIN_EVAL_PARENT_OVERLAP
CANONICAL_25D_ORDER
CLEAN_ONLY_FIELD_AUDIT
TINY_CLEAN_COLLECTION
TINY_TEACHER_DRY_AUDIT
TINY_DATASET_MATERIALIZATION
DATASET_TRAINABILITY
ONE_EPOCH_CHECKPOINT_RELOAD
CLEAN_SUSCEPTIBILITY_CALIBRATION
DETECTOR_ONLY_TIMING
CLEAN_PARENT_BINDING
MATCHED_JOB_BUILD
MATCHED_COMMAND_DRY_RUN
ONLINE_GPU_SMOKE
RUNTIME_AUDIT
RESULT_ANALYSIS

GPU_EPISODES_LAUNCHED
LIBERO_ROLLOUTS_LAUNCHED
OPENVLA_INFERENCE_RUNS
DETECTORS_TRAINED
DATASETS_MATERIALIZED

FILES_AND_SHA256
P0_FINDINGS
P1_FINDINGS
GO_HOLD_NEXT_STAGE
```

Do not report a later stage as PASS when it was not run. Stop at the first failed
fail-closed gate and provide the exact file, parent, step, field, expected value, and
actual value.
