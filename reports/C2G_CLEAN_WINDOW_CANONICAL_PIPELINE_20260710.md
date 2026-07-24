# C2g Clean-Window Detector v2 — Canonical End-to-End Pipeline

Date: 2026-07-10

## Canonical entry point

```bash
bash scripts/stageb/run_c2g_clean_window_pipeline.sh <phase>
```

This script directly wires the strict components. Historical `release_v2` through
`release_v8` wrappers remain repository history and compatibility layers; they are
not the preferred server entry point.

## Scientific contract

```text
clean RGB + clean causal proprio/action history + clean OpenVLA policy intent
  -> clean Teacher-v2 gripper-critical labels
  -> clean-only Detector-v2 training and validation calibration
  -> clean detector timing
  -> fixed B-frame gripper-targeted visual PGD
  -> adversarial OpenVLA re-decode
  -> matched-load execution, audit, and paired analysis
```

Attacked outcomes, post-intervention state, counterfactual replay outcomes, and
manual attacked-failure labels are forbidden from detector labels, features,
threshold selection, susceptibility calibration, split selection, and model
selection.

## Direct phases

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

`all` is executable for reproducibility but is not authorization for expensive work.
Server validation must proceed one reviewed phase at a time.

## Required environment

```bash
export WORK_ROOT=/absolute/external/c2g_run_root
export GOAL_MODEL_MANIFEST=/absolute/audited/goal_model_manifest.json
export DEVICE=cuda:0
```

Optional controls:

```text
WINDOW=16
BURST_LENGTH=10
EPOCHS=40
BATCH_SIZE=128
HIDDEN=128
TRAIN_STATES_PER_TASK=40
EVAL_STATES_PER_TASK=10
MAX_TASKS_PER_SUITE=0
MAX_STEPS=300
PARENT_SELECTION_SEED=42
MASTER_ATTACK_SEED=42
MAX_TRAIN_EPISODES=0
MAX_EVAL_JOBS=0
SUSCEPTIBILITY_POSITIVE_RETENTION=0.80
```

## Phase gates

### models

- resolves the exact four-suite OpenVLA model map;
- requires the audited Goal model-integrity manifest;
- hashes every referenced model weight shard and selected processor/tokenizer file;
- immediately recomputes the full manifest.

### manifests

- selects official LIBERO init states deterministically;
- emits disjoint training and evaluation cohorts;
- uses runtime-compatible five-part parent keys;
- launches no rollout.

### collect

- verifies full policy bytes before collection;
- uses the strict canonical 25D ordering;
- runs only clean OpenVLA actions;
- records clean privileged Teacher-v2 evidence;
- binds every episode metadata file to the full suite model manifest after collection.

### audit

- runs a small four-suite Teacher-v2 dry audit;
- rejects attacked fields, unknown-to-negative conversion, absolute-z-only positives,
  release-safe positives, and missing eligible evidence.

### materialize / dataset_audit

- uses exact suite-specific OpenVLA visual/language encoders;
- writes a hash-bound NPZ dataset;
- rejects episode leakage and insufficient positive, negative, or triggerable support.

### train / calibrate / folds

- trains only from clean labels and deployment-visible inputs;
- selects detector thresholds on validation data;
- calibrates clean policy susceptibility without attacked outcomes;
- exports a strict runtime checkpoint;
- runs LOTO diagnostic folds.

### clean_timing / bind_parents

- executes detector-only CLEAN parents with zero attacked frames;
- binds the clean parent artifact and official init-state hashes;
- preserves no-emit and late/burst-infeasible parents in the denominator.

### build_jobs / run_jobs

- constructs the five-condition matrix;
- pairs stochastic seeds across detector/random timing for each objective;
- uses `SHUFFLED_GRIPPER_GRADIENT` as the implemented compute-matched primary control;
- verifies frozen CLEAN parents but never rewrites them;
- runs only the four attack rows after CLEAN timing is frozen.

### audit_jobs / analyze

- permits only excluded-ledger-bound CLEAN artifacts outside the attacked matrix;
- verifies exact delivery, load, compute counts, Linf budget, objective/seed/checkpoint
  binding, and pre-trigger parity;
- performs paired exact McNemar/binomial analysis only after runtime audit PASS;
- reports per-suite effects and detector coverage denominators.

## Codex validation sequence

```text
C0  branch/head/worktree and all CPU workflows
C1  models
C2  manifests with one train and one eval state per task
C3  one explicitly authorized clean episode per suite
C4  audit
C5  tiny materialization and dataset_audit
C6  one-epoch training and strict checkpoint reload
C7  clean susceptibility calibration
C8  one detector-only clean timing parent per suite
C9  bind_parents and build_jobs
C10 matched launcher command dry run
C11 explicitly authorized one-parent five-condition GPU smoke
C12 audit_jobs before analyze
```

At the first HOLD, stop and report the exact file, parent, step, field, expected
value, and actual value. Do not weaken a gate to make a smoke run pass.

## Current claim boundary

```text
REPOSITORY_SIDE_E2E_IMPLEMENTATION = COMPLETE
CANONICAL_DIRECT_ORCHESTRATOR = IMPLEMENTED
STATIC_AND_SYNTHETIC_VALIDATION = REQUIRED_GREEN
LIVE_LIBERO_OPENVLA_COMPATIBILITY = NOT_YET_VERIFIED
FULL_CLEAN_COLLECTION = NOT_RUN
DETECTOR_TRAINING = NOT_RUN
ONLINE_MATCHED_MATRIX = NOT_RUN
SCIENTIFIC_EFFECTIVENESS = NOT_ESTABLISHED
D7_TABLE1 = STILL_FROZEN
```
