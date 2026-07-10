# C2g Clean-Window Detector v2 — End-to-End Train and Run Runbook

Date: 2026-07-10

## Scope

This runbook covers the complete executable path:

```text
preregistered clean parent manifest
  -> clean privileged rollout collection
  -> clean Teacher-v2 audit and labels
  -> frozen clean dataset materialization
  -> detector training / validation threshold calibration
  -> optional LOTO and LOSO folds
  -> detector-only CLEAN timing pass
  -> clean parent and initial-state hash binding
  -> five-condition matched-load job construction
  -> Detector-v2 + VIS-PGD online execution
  -> closed-world runtime audit
```

The primary detector remains clean-only. No attacked outcome, post-intervention state,
manual failure label, or counterfactual replay result enters its labels, inputs,
threshold calibration, split construction, or model selection.

## Repository identity

```text
repository = Leo-6-maker/openvla-gripper-dutycycle-attack
branch = assistant/c2g-clean-window-v2-20260710
base = e3bec3b82ac104c633d8dacc5fd27f9cf30a7e85
```

Always bind every server command to the current exact `git rev-parse HEAD` and require
a clean worktree.

## Primary entry point

```bash
bash scripts/stageb/run_c2g_clean_window_end_to_end.sh <phase>
```

Available phases:

```text
collect
 audit
 materialize
 train
 folds
 clean_timing
 bind_parents
 build_jobs
 run_jobs
 audit_jobs
 all
```

`all` is executable but must not be used until each preceding gate is independently
reviewed. The safer operating mode is one phase at a time.

## Required manifests

### Training clean-rollout manifest

`TRAIN_EPISODE_MANIFEST` must contain one object per intended clean rollout:

```json
{
  "parent_key": "libero_object/task_0/state_0/train_000",
  "suite": "libero_object",
  "task_index": 0,
  "state_id": 0
}
```

The collector resolves language and BDDL from the official LIBERO task object and
loads the official init state identified by `state_id`.

### Evaluation parent manifest

`EVAL_PARENT_MANIFEST` must be preregistered before attacked execution:

```json
{
  "parent_key": "libero_object/task_0/state_0/eval_000",
  "suite": "libero_object",
  "task_index": 0,
  "state_id": 0,
  "eval_seed": 42000,
  "max_steps": 300
}
```

The detector timing pass, random-time sampling, initial-state hashes, and all five
conditions are derived from this frozen parent set.

## Minimal environment

```bash
export TRAIN_EPISODE_MANIFEST=/absolute/read_only/train_episodes.jsonl
export EVAL_PARENT_MANIFEST=/absolute/read_only/eval_parents.jsonl
export WORK_ROOT=/absolute/external/c2g_clean_window_v2_$(date +%Y%m%d_%H%M%S)
export DEVICE=cuda:0
export EMBEDDING_BACKEND=openvla_siglip
export OPENVLA_MODEL_PATH=/absolute/model/path
export MODEL_PATH_TEMPLATE='/absolute/models/{suite}'
export WINDOW=16
export BURST_LENGTH=10
export EPOCHS=40
export BATCH_SIZE=128
```

`WORK_ROOT` must be outside the repository.

## Stage E0 — Repository and live asset validation

Before collection:

```bash
git status --short
git diff --check
python -m unittest -v \
  tests.test_c2g_clean_window_v2 \
  tests.test_c2g_clean_window_server_audit \
  tests.test_c2g_clean_window_e2e \
  tests.test_c2g_clean_window_pipeline_tools
```

Run the existing read-only static asset census on official BDDL and MuJoCo XML roots.
No collection should start with unsupported goal operators or unresolved left/right
finger aliases unless those tasks are explicitly placed into an abstain stratum.

## Stage E1 — Clean privileged rollout collection

```bash
bash scripts/stageb/run_c2g_clean_window_end_to_end.sh collect
```

Implementation:

```text
scripts/stageb/collect_c2g_clean_window_rollouts.py
```

Per step it records:

```text
student-visible clean fields:
  RGB path
  25D causal proprio/action vector
  task language
  clean OpenVLA gripper policy-intent 9D

teacher-only clean privileged fields:
  MuJoCo contact geom pairs
  target and destination pose evidence
  object-relative lift
  target-distance progress
  fixture/joint motion
  near-target / support / release-safe evidence
```

It never launches an attack and never reads an attacked result.

Required collection gate:

```text
status = PASS_CLEAN_COLLECTION
attacks_launched = 0
attack_outcomes_read = false
all selected episodes have nonempty metadata and step records
all RGB paths resolve
all 25D and policy-intent features are finite
```

## Stage E2 — Tiny clean Teacher-v2 audit

```bash
bash scripts/stageb/run_c2g_clean_window_end_to_end.sh audit
```

The audit must report:

```text
zero attacked/outcome field use
zero unknown-to-negative conversion
zero absolute-EEF-z-only critical positives
zero release-safe critical positives
target/distractor identity consistency
one-or-zero fixed-B start per episode
explicit unknown/abstain for unsupported mechanisms
```

## Stage E3 — Dataset materialization

```bash
bash scripts/stageb/run_c2g_clean_window_end_to_end.sh materialize
```

Implementation:

```text
tools/multisuite_detector/materialize_c2g_clean_window_dataset.py
```

Output includes:

```text
X_proprio [N,T,25]
X_policy [N,T,9]
X_visual [N,Dv]
X_language [N,Dl]
y_* and m_* [N,T] for every detector head
sample_weight [N,T]
episode_fully_known_negative [N]
suite / task_index / episode_key / step / split
input artifact manifest and dataset SHA256
```

`task_index` and `suite` are audit metadata, not model inputs.

Required materialization gate:

```text
status = PASS_MATERIALIZED
n_episode_errors = 0
nonempty train / val / test splits
known positive and known negative support
nonzero triggerable positive intervals
all arrays finite and cardinality-aligned
input manifest and dataset SHA256 frozen
```

## Stage E4 — Training and clean validation calibration

```bash
bash scripts/stageb/run_c2g_clean_window_end_to_end.sh train
```

Implementation:

```text
tools/multisuite_detector/train_c2g_clean_window_detector.py
```

The trainer performs:

```text
masked clean multi-head training
unknown-safe episode losses
2-of-3 persistence-aligned penalties
validation-only threshold sweep
episode any-trigger FP constraint
release-safe trigger constraint
checkpoint/config/dataset SHA binding
```

Checkpoint:

```text
c2g.clean_window_checkpoint.2026-07-10.v1
```

Required training gate:

```text
checkpoint loads with strict state_dict matching
validation thresholds are selected without attacked outcomes
validation has positive and fully-known-negative episodes
no NaN/Inf loss or probability
checkpoint and report hashes frozen
```

## Stage E5 — LOTO / LOSO folds

Primary task-generalization evaluation:

```bash
bash scripts/stageb/run_c2g_clean_window_end_to_end.sh folds
```

Implementation:

```text
tools/multisuite_detector/run_c2g_clean_window_folds.py
```

Each fold changes only the episode split array. Features and labels remain identical
to the frozen base dataset. The held-out task or suite never enters train/validation.

## Stage E6 — Detector-only CLEAN timing pass

```bash
bash scripts/stageb/run_c2g_clean_window_end_to_end.sh clean_timing
```

The online worker executes a normal clean OpenVLA policy. The detector and scheduler
run, but no attack is delivered. `trigger_started` is extracted into a frozen timing
manifest.

Required gate:

```text
all preregistered evaluation parents runtime-valid
no attacked frame delivered
exactly one detector start per included parent
start is fixed-B burst-feasible
clean timing manifest SHA256 frozen
```

Parents with no detector trigger must be reported explicitly. They may be retained
as detector no-emit outcomes, but they cannot be silently removed after attacked
results are observed.

## Stage E7 — Parent and initial-state binding

```bash
bash scripts/stageb/run_c2g_clean_window_end_to_end.sh bind_parents
```

Implementation:

```text
scripts/stageb/prepare_c2g_eval_parents.py
```

It binds each parent to:

```text
CLEAN metadata + step-record combined SHA256
official LIBERO init-state dtype/shape/content SHA256
suite/task/state identity
evaluation seed
clean detector start
```

## Stage E8 — Five-condition matched-load manifest

```bash
bash scripts/stageb/run_c2g_clean_window_end_to_end.sh build_jobs
```

Frozen conditions:

```text
CLEAN
DET_GRIPPER_VIS_PGD
DET_RANDOM_VIS_ATTACK
RANDTIME_GRIPPER_VIS_PGD
RANDTIME_RANDOM_VIS_ATTACK
```

The primary random/non-gripper control is compute-matched. Default:

```text
SHUFFLED_GRIPPER_GRADIENT
```

Every parent has exact matching of:

```text
init state
clean parent
checkpoint/config
burst length
epsilon / step size / PGD iterations
projection and cast
preprocessing and input size
random-start and temporal-init policies
loss forward / backward / adversarial decode counts
```

## Stage E9 — Online Detector-v2 + VIS-PGD execution

```bash
bash scripts/stageb/run_c2g_clean_window_end_to_end.sh run_jobs
```

Implementation:

```text
scripts/stageb/run_c2g_clean_window_vis_pgd.py
scripts/stageb/run_c2g_matched_load_jobs.py
```

Per-step order:

```text
clean RGB
  -> clean OpenVLA generation
  -> clean gripper score row
  -> 25D + clean policy-intent + visual/language detector
  -> detector or frozen random-time gate
  -> fixed B-frame TokenPrefixPGDAttacker
  -> adversarial OpenVLA re-decode
  -> execute adversarial action
```

The worker does not apply direct command-space force-open.

## Stage E10 — Closed-world online audit

```bash
bash scripts/stageb/run_c2g_clean_window_end_to_end.sh audit_jobs
```

Implementation:

```text
scripts/stageb/audit_c2g_matched_load_run.py
```

It verifies:

```text
exact expected job set; no unexpected jobs
runtime-valid metadata and nonempty parseable records
protocol and identity binding
exact contiguous B-frame delivery
DET pair uses identical detector timing
RANDTIME pair uses identical deterministic random timing
random time differs from detector time
identical compute counts and Linf budget
pre-trigger clean trajectory parity
per-parent success contingency retained
```

## Required comparison plan

Timing value:

```text
DET_GRIPPER_VIS_PGD
vs
RANDTIME_GRIPPER_VIS_PGD
```

Objective specificity:

```text
DET_GRIPPER_VIS_PGD
vs
DET_RANDOM_VIS_ATTACK
```

Factorial interaction:

```text
Detector timing × gripper-targeted objective
```

No detector architecture, threshold, split, or label may be changed based on attacked
outcomes from this matrix.

## Current authorization boundary

The repository now contains the executable collection, materialization, training,
fold evaluation, runtime integration, job construction, launcher, and auditor.
This code state does not prove live server compatibility or scientific performance.

Before full execution, Codex/server must independently validate:

```text
official BDDL parsing and target resolution
real MuJoCo finger/contact naming
clean collector field coverage on all four suites
OpenVLA global embedding parity between materializer and runtime
checkpoint strict loading
one small CPU/single-parent dry command per entry point
one explicitly authorized GPU smoke before any matrix
```

Frozen boundaries remain:

```text
D7_TABLE1 = STILL_FROZEN
TEACHER_V1_FOR_TRAINING = HOLD
COUNTERFACTUAL_REPLAY = OPTIONAL_POSTHOC_ONLY
ATTACKED_OUTCOME_DETECTOR_TRAINING = FORBIDDEN
```
