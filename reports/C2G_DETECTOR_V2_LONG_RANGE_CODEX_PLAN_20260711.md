# C2g Detector-v2 — Long-Range Codex Execution Plan

Date: 2026-07-11

## 0. Mission and current branch

The operational target is to run the clean-only Detector-v2 end to end:

```text
clean LIBERO/OpenVLA rollout
  -> clean privileged Teacher-v2 labels
  -> clean-only multisuite dataset
  -> detector training and validation-only calibration
  -> detector-only CLEAN timing
  -> frozen matched-load VIS-PGD jobs
  -> one-parent runtime smoke
  -> closed-world audit
  -> later, separately approved multisuite replication
```

Use the exact current remote head of:

```text
codex/c2g-strict-server-smoke-20260710
```

Source PR #58 and remediation PR #59 remain Draft and unmerged. Bind every artifact to
`git rev-parse HEAD`; never hard-code a stale head from this document.

## 1. Scientific contract

The detector is clean-only. Its primary positive target is:

```text
target relevant
AND gripper dependent
AND clean close intent
AND lift / transport / constrained manipulation
AND NOT release safe
```

Allowed student inputs:

```text
clean RGB
clean 25D proprio/action history
clean OpenVLA gripper policy-intent history
current task language
```

Forbidden for labels, features, calibration, model selection, or timing selection:

```text
attacked observations/actions
post-intervention state
attack outcome
counterfactual outcome
manual attacked-failure labels
future clean student inputs
task-index/hash or suite one-hot model features
normalized episode step as an identity shortcut
```

The detector selects only the attack start. The payload remains the existing visual
`TokenPrefixPGDAttacker`; a fixed B-frame burst is immutable after trigger.

Frozen primary matrix:

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

Do not replace the visual attack with command-space force-open and do not substitute an
unmatched uniform-noise/action-noise control.

## 2. Accepted engineering repairs already on the server branch

The branch includes the following repairs without scientific-contract changes:

```text
module-safe release job-builder invocation
suite provenance recovery from the frozen parent-key namespace
turnon/turnoff -> turn_on/turn_off explicit BDDL syntax aliases
Panda finger_joint1/2[_tip] deterministic two-jaw identities
Goal v2 static safetensors/index/header/hash audit
Goal load-only v2 manifest finalizer with explicit rebase token
byte-verifying legacy/v2 Goal manifest validator
Goal v2 validation in clean timing and matched-load map launchers
suite-isolated clean collection subprocesses to avoid retaining four 7B models
combined closed clean-collection manifest reconstruction
mapping-key-only outcome leakage scan to avoid documentation-value false positives
```

The collector isolation repair is operationally important: one suite model is loaded per
subprocess and released when that process exits. Do not collapse collection back into a
single four-model Python process.

## 3. Resource and storage policy

The first server report showed approximately 42 GiB free on `/mnt/sdc`.

Rules:

1. Do not duplicate the four policy model directories.
2. Search for frozen Goal bytes read-only; do not copy candidates until their complete
   directory passes the strict audit.
3. Put all generated artifacts under one external `WORK_ROOT`, never inside the repo.
4. Record `df -B1`, inode availability, and GPU memory before every live phase.
5. Stop before a live phase when projected free space after completion is below 15 GiB.
6. Preserve reports, manifests, checkpoints, and SHA files. Raw RGB may be pruned only
   after a materialized dataset and its immutable input manifest both pass and only with
   explicit user approval.
7. Never delete historical D7 evidence or another experiment root to make space.

## 4. Branch and patch discipline

Start from a fresh checkout of the current remote server branch. Minimal server fixes are
allowed only for:

```text
path discovery
Python/import compatibility
LIBERO/OpenVLA/MuJoCo API compatibility
dtype/serialization/checkpoint compatibility
provenance checks
runtime logging required by an existing audit
resume/output-path correctness
```

After any code patch:

1. commit the smallest coherent change;
2. rerun S0 completely;
3. rerun the affected live stage from its beginning;
4. report the commit SHA and exact evidence.

Return `HOLD_SCIENTIFIC_CONTRACT` instead of changing Teacher semantics, feature sets,
splits, losses, calibration objectives, 2-of-3 persistence, B, attack objective, matrix,
load-matching rules, or outcome interpretation.

## 5. Track A — bounded end-to-end smoke

Track A is the immediate objective. It retains the existing caps:

```text
new clean training episodes <= 40
clean evaluation parents <= 4
training epochs = 1
matched online parents <= 1
attacked runs for selected parent = 4
full replication matrix = 0
counterfactual replays = 0
D7 modifications = 0
```

### A0 — repository identity and static validation

```bash
git fetch origin --prune
git checkout codex/c2g-strict-server-smoke-20260710
git reset --hard origin/codex/c2g-strict-server-smoke-20260710
export C2G_HEAD="$(git rev-parse HEAD)"
git status --short
git diff --check

python -m py_compile \
  scripts/stageb/collect_c2g_clean_window_rollouts_release.py \
  scripts/stageb/run_c2g_clean_timing_jobs_strict.py \
  scripts/stageb/run_c2g_matched_load_jobs_map_release.py \
  scripts/stageb/build_c2g_suite_model_map.py \
  scripts/stageb/finalize_c2g_goal_model_manifest_v2.py \
  tools/multisuite_detector/audit_c2g_goal_model_integrity_v2.py \
  tools/multisuite_detector/audit_c2g_static_assets_strict.py

python -m unittest discover -s tests -p 'test_c2g*.py' -v
bash -n scripts/stageb/run_c2g_clean_window_pipeline.sh
bash -n scripts/stageb/run_c2g_clean_window_pipeline_strict.sh
```

Gate:

```text
all C2g tests PASS
worktree clean
no generated repository output
remote/local head match
```

### A1 — live S1 asset and Goal provenance closure

Use the same 40 BDDL and 107 XML roots from the first S1 run:

```bash
python tools/multisuite_detector/audit_c2g_static_assets_strict.py \
  --bddl-root "$OFFICIAL_BDDL_ROOT" \
  --xml-root "$OFFICIAL_XML_ROOT" \
  --output-json "$WORK_ROOT/s1/static_asset_inventory_strict.json"
```

Required:

```text
40 BDDL files
107 XML files
0 parse errors
unsupported operators = []
unresolved finger candidates = []
left/right jaw aliases nonempty
OpenVLA loads = 0
LIBERO rollouts = 0
```

Resolve Goal provenance in this order.

#### A1a — frozen-byte search

Search targeted model/cache roots for a complete directory matching all old shard hashes.
Do not run an unrestricted full-filesystem hash sweep. Prefer name/size filtering before
SHA256. A candidate is accepted only if all four shards, index, config, tokenizer, and
processor files form one complete directory and pass the current strict model audit.

#### A1b — explicit C2g-only current-byte rebase

Use only when no complete frozen copy exists:

```bash
python tools/multisuite_detector/audit_c2g_goal_model_integrity_v2.py \
  --model-path /mnt/sdc/dty_user/openvla_attack/models/libero-goal \
  --previous-manifest artifacts/goal_model_manifest.json \
  --output-report "$WORK_ROOT/s1/goal_model_static_integrity_v2.json"

python scripts/stageb/finalize_c2g_goal_model_manifest_v2.py \
  --static-report "$WORK_ROOT/s1/goal_model_static_integrity_v2.json" \
  --model-path /mnt/sdc/dty_user/openvla_attack/models/libero-goal \
  --output-manifest "$WORK_ROOT/s1/goal_model_manifest_v2.json" \
  --device cuda:0 \
  --rebase-approval C2G_GOAL_MODEL_REBASE_20260710

export GOAL_MODEL_MANIFEST="$WORK_ROOT/s1/goal_model_manifest_v2.json"
```

The one load-only validation is allowed; it must create zero LIBERO environments,
rollouts, or attacks. The mismatch ledger against the old manifest must remain present.

Then:

```bash
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh models
```

Gate:

```text
PASS_C2G_STRICT_SUITE_MODEL_MAP
PASS_C2G_STRICT_SUITE_MODEL_VERIFICATION
all model shards hash-bound
Goal current-byte ledger verified
```

### A2 — tiny preregistration

Create a new empty root for each incompatible manifest configuration.

```bash
export DEVICE=cuda:0
export WINDOW=16
export BURST_LENGTH=10
export EPOCHS=1
export BATCH_SIZE=32
export HIDDEN=128
export MAX_STEPS=300
export PARENT_SELECTION_SEED=42
export MASTER_ATTACK_SEED=42
export SUSCEPTIBILITY_POSITIVE_RETENTION=0.80
export TRAIN_STATES_PER_TASK=1
export EVAL_STATES_PER_TASK=1
export MAX_TASKS_PER_SUITE=1
export MAX_TRAIN_EPISODES=0
export MAX_EVAL_JOBS=0

bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh manifests
```

Gate:

```text
all four suites represented
exact five-part parent keys
train/eval state overlap = 0
one clean train and one clean eval parent per selected suite task
```

### A3 — four-suite clean compatibility collection

```bash
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh collect
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh audit
```

The release collector must report:

```text
execution_mode = SUITE_ISOLATED_SUBPROCESSES
one subprocess per represented suite
four runtime-valid clean parents
zero attacked frames
zero attack outcomes
combined artifact manifest covers every episode metadata and step file
```

Inspect one complete episode per suite. Verify RGB, canonical 25D ordering, 9D policy
features, target/contact/progress/release evidence, suite model hash, and Goal manifest
binding.

Additional mandatory diagnostic before scaling:

```text
MULTI_TARGET_TASK_COUNT
MULTI_TARGET_KNOWN_POSITIVE_ROWS
MULTI_TARGET_TRIGGERABLE_WINDOWS
ARTICULATED_TASK_COUNT
ARTICULATED_KNOWN_POSITIVE_ROWS
```

Current collector logic is strongest for one active target. If real multi-target tasks
produce contact-positive but systematically progress-negative rows, stop with
`HOLD_MULTI_TARGET_EVENT_TRACKING` rather than treating those rows as evidence that no
critical window exists. Implement per-step active-target/subgoal tracking as a separate
reviewed engineering patch before large-scale collection.

### A4 — adaptive clean support and materialization

First attempt the tiny cohort:

```bash
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh materialize
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh dataset_audit
```

Never weaken the audit. If the only failure is insufficient clean support, expand using a
new empty `WORK_ROOT`:

```text
Expansion A: 3 states/task, 2 tasks/suite, <=24 episodes
Expansion B: 5 states/task, 2 tasks/suite, <=40 episodes
```

Stop after 40 episodes with `HOLD_INSUFFICIENT_CLEAN_SUPPORT` if required train/val/test
positive, negative, and 2-of-3-triggerable support still does not exist.

Gate:

```text
PASS_C2G_DATASET_TRAINABILITY
four suites represented
no split leakage
unknown remains masked
student arrays contain no privileged or identity-shortcut fields
finite arrays and compatible dimensions
known positive and negative support in required splits
2-of-3 triggerable positive windows exist
```

### A5 — one-epoch detector and clean susceptibility

```bash
export EPOCHS=1
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh train
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh calibrate
```

Independently reload the checkpoint with `strict=True` and run one validation batch.

Gate:

```text
training epochs = 1
checkpoint/dataset/config hashes agree
finite logits, losses, probabilities, and thresholds
threshold selection uses clean validation only
susceptibility uses clean validation only
uses_attack_outcomes = false
```

### A6 — detector-only clean timing

```bash
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh clean_timing
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh bind_parents
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh build_jobs
```

The clean-timing launcher now accepts and byte-verifies Goal v2 manifests. Required:

```text
<=4 clean evaluation parents
0 attacked frames
clean parent and init-state hashes frozen
checkpoint/config hashes frozen
explicit no-emit retained
late burst-infeasible emit retained in denominator ledger
at least one detector-emitted burst-feasible parent
```

Do not lower thresholds using attacked outcomes when no parent emits.

### A7 — one-parent command dry run

Select one eligible parent, prefer LIBERO-Object, and create a five-row closed-world smoke
manifest. Copy the frozen CLEAN bytes without rerunning CLEAN.

Run the model-map launcher with `--dry-run`. It must byte-verify the Goal manifest before
launch planning and plan exactly four attacked workers.

Gate:

```text
five conditions exactly
CLEAN not launched or rewritten
four attacked commands only
same B/epsilon/step-size/K/preprocess/projection/cast/temporal-init
paired objective seeds
control = SHUFFLED_GRIPPER_GRADIENT
```

### A8 — one-parent online VIS-PGD smoke

Only after A0-A7 PASS, run the same command without `--dry-run`, then immediately run:

```bash
python scripts/stageb/audit_c2g_matched_load_run_release.py ...
python scripts/stageb/analyze_c2g_matched_load_results.py ...
```

Gate:

```text
frozen CLEAN unchanged
four attacked workers runtime-valid
exactly B contiguous attacked frames each
processor-space Linf within budget
forward/backward/decode counts match manifest
objective families and seeds match
paired timing starts match
random start differs from detector start
pre-trigger clean parity PASS
closed-world runtime audit PASS
```

N=1 is a runtime smoke only; make no effectiveness claim.

## 6. Track B — detector-quality closure after Track A PASS

Track B requires a new explicit approval because it exceeds the bounded smoke.

### B1 — clean Teacher coverage audit

Produce per-suite, per-task, and per-mechanism tables:

```text
rows/episodes
target resolved/unknown
grasp/contact known
progress known
release known
critical positives
known negatives
2-of-3 triggerable windows
window durations
reason codes
abstain rates
```

Hard requirements before production training:

1. no task is silently all-negative due to unresolved target/contact/progress;
2. multi-target and articulated tasks are either correctly event-tracked or explicitly
   preregistered as abstain/unsupported;
3. release-safe rows never become critical positives;
4. absolute EEF-z alone never creates positives;
5. suite/task density imbalance is reported and controlled by sampling/weights, not task
   identity features.

### B2 — reuse and collection policy

Prefer reusing existing clean rollouts only when they contain the exact raw fields,
OpenVLA checkpoint binding, preprocessing, and task-state provenance required by Teacher
v2. Re-label clean data; never reuse Teacher-v1 primary labels as Detector-v2 truth.

Collect new clean episodes only for missing task/mechanism support. Freeze train/eval
parents before collection. Do not use attacked outcomes to decide which clean episodes are
kept.

### B3 — model ladder

Train and compare:

```text
Temporal/proprio only
Clean-logit susceptibility only
Temporal + susceptibility
+ global SigLIP
+ patch SigLIP + language
no-language
wrong-language
legacy task-context diagnostic only
no release veto
no persistence
active-window head vs start-window head
```

Primary model remains no-context. Legacy task/suite identity may appear only as a shortcut
diagnostic, never as the main result.

### B4 — generalization and seeds

After one-epoch smoke, run separately approved training with:

```text
within-task reference
leave-one-task-out primary
leave-one-suite-out diagnostic
>=3 training seeds for the selected architecture
```

Freeze checkpoints, dataset hashes, thresholds, calibration, and feature schemas before
online evaluation.

### B5 — online pilot and confirmatory matrix

Pilot first with a small preregistered parent set. Scale only after runtime audits PASS.
The confirmatory design must retain the 2x2 timing/objective factorization:

```text
timing effect: DET_GRIPPER vs RANDTIME_GRIPPER
objective effect: DET_GRIPPER vs DET_RANDOM
interaction: detector timing x gripper targeting
```

Match attacked frame count and compute load. Report task SR, paired success flips,
gripper-token/logit changes, executed open duty cycle, qpos/width response, arm drift,
and effect per attacked frame/backward pass.

## 7. Stop conditions

Stop immediately and return HOLD for:

```text
dirty/divergent worktree
Goal file/hash/load mismatch
unsupported BDDL operator without explicit abstain handling
unresolved bilateral jaw identity
attack/outcome field entering clean labels or calibration
canonical 25D mismatch
unknown converted to negative
collection provenance mutation
multi-target systematic false-negative labeling
split leakage
trainability failure after approved cap
non-finite training/calibration values
checkpoint/config/dataset hash mismatch
CLEAN attacked frame count > 0
frozen CLEAN overwritten
no burst-feasible clean parent
matched-load/seed/budget mismatch
pre-trigger parity failure
closed-world runtime audit HOLD
free-space safety threshold violation
```

## 8. Required Codex status cadence

Return a structured status at every gate, not only at the end:

```text
REMOTE_HEAD
EXECUTED_HEAD
NEW_FIX_COMMITS
WORKTREE_CLEAN
STAGE
EXACT_COMMANDS
EXIT_CODES
REPORT_PATHS
REPORT_SHA256
PASS_INVARIANTS
HOLD_INVARIANTS
CLEAN_TRAIN_EPISODES_LAUNCHED
CLEAN_EVAL_EPISODES_LAUNCHED
ATTACKED_EPISODES_LAUNCHED
OPENVLA_MODEL_LOADS
TRAINING_EPOCHS
DATASETS_MATERIALIZED
FREE_BYTES_BEFORE_AFTER
P0_FINDINGS
P1_FINDINGS
SCIENTIFIC_CONTRACT_CHANGES = NONE / HOLD
D7_TABLE1 = STILL_FROZEN
GO_HOLD_NEXT_STAGE
```

Do not call skipped stages PASS. Do not describe static tests as live validation. Do not
interpret online outcomes before the closed-world audit passes.
