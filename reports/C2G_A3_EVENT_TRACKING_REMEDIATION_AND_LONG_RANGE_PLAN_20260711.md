# C2g Detector-v2 — A3 Event-Tracking Remediation and Long-Range Codex Plan

Date: 2026-07-11

## 0. Decision on the returned A3 HOLD

The server stop at `HOLD_MULTI_TARGET_EVENT_TRACKING` was correct and must remain in the
record. A3 established runtime compatibility, but it did **not** establish a usable
Teacher-v2 label path:

```text
runtime-valid clean episodes = 4
known Teacher rows = 0
unknown Teacher rows = 682
multi-target contact-positive rows = 150
multi-target progress-positive rows = 0
```

The root cause was broader than one missing `primary_target` assignment:

1. official LIBERO `:fixtures` and `:regions` were not represented in the clean BDDL
   metadata;
2. movable destination objects such as baskets and plates were incorrectly treated as a
   disjoint receptacle class;
3. official goals refer to qualified sites such as `basket_1_contain_region` and
   `wooden_cabinet_1_middle_region`, but no site-owner relation was available;
4. exact BDDL goals were weakened into an episode-level target summary before collection;
5. contextual language mentions in Spatial were treated as fatal conflicts despite an exact
   structured goal;
6. collection used a single episode-level `primary_target`, so multi-target progress could
   not be bound to the currently manipulated object;
7. articulated progress could average multiple fixture joints instead of selecting the joint
   named by the goal region.

The previous A3 artifacts remain immutable evidence of the HOLD. They must not be edited,
relabelled as PASS, or used for Detector-v2 training.

## 1. Remediation now present on the server branch

Use the latest remote head of:

```text
codex/c2g-strict-server-smoke-20260710
```

Never hard-code a SHA from this document. Bind every new artifact to the actual
`git rev-parse HEAD` after checkout.

The remediation adds:

```text
official :objects / :fixtures / :regions parsing
qualified region site -> owner mapping
exact ordered goal predicates
per-predicate (operator, target, destination, interaction-site) bindings
movable destination objects preserved as object identities
language conflict downgraded to a diagnostic when exact BDDL is available
per-step active target selection from current clean finger-target contacts
independent per-target lift/distance/joint baselines
region-derived joint selector for articulated tasks
null/unknown evidence for unresolved active events
privileged active-event fields blocked from student inputs
closed event-tracking audit required before materialization
```

The strict collector now routes through:

```text
scripts/stageb/collect_c2g_clean_window_rollouts_event_v2.py
```

The strict `audit` phase now runs both:

```text
audit_c2g_clean_window_v2.py
audit_c2g_goal_event_tracking.py
```

The strict `materialize` phase refuses to run unless the second report is exactly:

```text
PASS_C2G_GOAL_EVENT_TRACKING_AUDIT
```

## 2. Scientific contract remains frozen

The patch changes clean privileged evidence extraction, not the detector claim.

Primary clean Teacher target:

```text
target relevant
AND gripper dependent
AND clean close intent
AND clean manipulation progress
AND NOT release safe
```

Allowed student inputs:

```text
clean RGB
clean causal 25D proprio/action history
clean OpenVLA gripper policy-intent history
task language
```

Teacher-only, forbidden student inputs include:

```text
active_target_entity
active_subgoal_index
active_operator
active_destination_entity
active_interaction_site
goal_event_bindings
contact identities
object/target positions
per-target progress
fixture joint motion
release-safe evidence
```

Attacked observations, attacked actions, post-intervention state, attack outcomes,
counterfactual outcomes, and manual attacked-failure labels remain forbidden for Teacher
labels, student features, threshold calibration, architecture selection, and trigger timing.

The detector selects only the attack start. The payload remains fixed-length visual
`TokenPrefixPGDAttacker`; the primary compute-matched control remains
`SHUFFLED_GRIPPER_GRADIENT`.

## 3. Resource ledger and authorization boundary

Counts already consumed by the returned run:

```text
historical invalid-for-training clean train episodes = 4
clean eval episodes = 0
attacked episodes = 0
training epochs = 0
```

Continue to enforce the original cumulative Track-A cap:

```text
all clean train episodes launched in this bounded campaign <= 40
therefore new event-aware clean train episodes <= 36
clean evaluation parents <= 4
training epochs = 1
matched online parents <= 1
attacked runs for selected parent = 4
full replication matrix = 0
counterfactual replays = 0
D7 modifications = 0
```

All new outputs must use a fresh external root. The old b624 A3 root is read-only evidence.
Do not delete D7 or other experiment roots. Record free bytes and inodes before and after
every live phase; stop before a phase whose projected completion would leave less than
15 GiB free.

## 4. Resume plan: R0 through R10

### R0 — fresh checkout and complete static gate

```bash
git fetch origin --prune
git checkout codex/c2g-strict-server-smoke-20260710
git reset --hard origin/codex/c2g-strict-server-smoke-20260710
export C2G_HEAD="$(git rev-parse HEAD)"
git status --short
git diff --check

python -m py_compile \
  src/gripper_attack/c2g_bddl_metadata.py \
  src/gripper_attack/c2g_teacher_v2_target_resolution.py \
  src/gripper_attack/c2g_clean_event_tracking.py \
  src/gripper_attack/c2g_clean_mechanism.py \
  src/gripper_attack/c2g_clean_window_schema.py \
  scripts/stageb/collect_c2g_clean_window_rollouts_event_v2.py \
  scripts/stageb/collect_c2g_clean_window_rollouts_strict.py \
  tools/multisuite_detector/audit_c2g_goal_event_tracking.py

python -m unittest discover -s tests -p 'test_c2g*.py' -v
bash -n scripts/stageb/run_c2g_clean_window_pipeline.sh
bash -n scripts/stageb/run_c2g_clean_window_pipeline_strict.sh
```

Gate:

```text
all C2g tests PASS
worktree clean
local head = remote head
no repository-generated artifacts
```

A server-only compatibility patch must be minimal, committed, pushed, and followed by a
complete R0 rerun.

### R1 — revalidate live assets and model bytes

Rerun the same 40-BDDL / 107-XML strict inventory because the target-role parser changed.
The prior Goal v2 manifest may be reused only when its path, ledger, file sizes, and hashes
still verify against current bytes. Do not perform a second Goal load-only finalization when
the existing audited v2 manifest remains byte-valid.

Required:

```text
40 BDDL files
107 XML files
0 parse errors
unsupported operators = []
unresolved finger candidates = []
PASS_C2G_STRICT_SUITE_MODEL_MAP
PASS_C2G_STRICT_SUITE_MODEL_VERIFICATION
OpenVLA rollout count = 0 in the asset audit
```

### R2 — create a fresh event-aware Track-A root

Use a new empty root, for example:

```bash
export WORK_ROOT=/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_event_v2_${C2G_HEAD:0:8}_20260711
export GOAL_MODEL_MANIFEST=/ABSOLUTE/AUDITED/goal_model_manifest_v2.json
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

bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh models
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh manifests
```

Required:

```text
four suites represented
4 clean train parents + 4 clean eval parents
train/eval state identity overlap = 0
all model and manifest artifacts bind C2G_HEAD
```

### R3 — recollect the four tiny clean parents

```bash
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh collect
```

The collector must report:

```text
execution_mode = SUITE_ISOLATED_SUBPROCESSES
event_tracking_schema = c2g.clean_goal_event_tracking.2026-07-11.v1
4 runtime-valid clean episodes
0 attacked frames
attack_outcomes_read = false
one model process at a time
combined collection manifest closed over every metadata/step artifact
```

Expected structured role audit for the selected task-0 cohort:

```text
LIBERO-Object   mechanism = pick_place_transfer; goal_event_count = 1
LIBERO-Spatial  mechanism = pick_place_transfer; goal_event_count = 1
LIBERO-Goal     mechanism = articulated_object; goal_event_count = 1
LIBERO-10       mechanism = multi_object_transfer; goal_event_count = 2
```

These are role-resolution expectations, not effectiveness claims.

### R4 — run both clean scientific audits

```bash
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh audit
```

Required reports:

```text
clean_window_v2_audit_report.json
c2g_goal_event_tracking_audit.json
```

Hard gate:

```text
PASS_C2G_CLEAN_WINDOW_V2_DRY_AUDIT
PASS_C2G_GOAL_EVENT_TRACKING_AUDIT
eligible episodes with zero known Teacher rows = 0
contacted goal target unresolved rows = 0
active target contact with progress unresolved rows = 0
unknown converted to negative = 0
multiple attack-start rows per episode = 0
absolute-EEF-z-only critical positives = 0
release-safe critical positives = 0
```

For the multi-target episode, report separately:

```text
contact-positive rows
active-target-known rows
bilateral rows
progress-known rows
known Teacher rows
critical-positive rows
burst-feasible rows
```

A clean policy may fail to produce a critical positive in one episode. That alone is not an
implementation failure. The required distinction is:

```text
NO_PHYSICAL_PROGRESS_BY_CLEAN_POLICY
versus
TRACKER_OR_PROGRESS_EVIDENCE_UNRESOLVED
```

The second case is a HOLD. Do not convert it to a known negative.

For the articulated episode, require the region-derived joint selector to resolve and require
non-null joint-progress evidence. Zero positive rows are allowed only when the selected joint
truly did not move beyond the frozen threshold.

### R5 — adaptive clean support without exceeding the cumulative cap

Attempt materialization and trainability on the R3 cohort:

```bash
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh materialize
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh dataset_audit
```

Do not weaken the audit. When the only HOLD is insufficient positive/negative/split support,
add clean parents deterministically to the same event-aware collection through incremental,
non-overlapping manifests:

```text
E0: 4 event-aware episodes
E1: add at most 20, event-aware total <= 24
E2: add at most 12, event-aware total <= 36
campaign cumulative including the old 4 invalid episodes <= 40
```

Each incremental manifest must have unique parent keys, preserve the frozen train/eval
separation, and be bound to the same current model bytes and code head. After each increment,
rerun collection binding, canonical audit, event audit, materialization, and trainability from
the appropriate phase.

Stop at 36 event-aware episodes with:

```text
HOLD_INSUFFICIENT_CLEAN_SUPPORT
```

when train/val/test known positives, known negatives, or contiguous 2-of-3 triggerable support
remain insufficient. Do not relax split or label rules.

Trainability gate:

```text
PASS_C2G_DATASET_TRAINABILITY
four suites represented
no episode/split leakage
unknown remains masked
finite compatible arrays
no privileged/identity-shortcut student fields
known positive and negative support in required splits
contiguous 2-of-3 triggerable positives exist
```

### R6 — one-epoch checkpoint and clean-only calibration

```bash
export EPOCHS=1
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh train
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh calibrate
```

Independently reload the checkpoint with `strict=True` and run one validation batch.

Required:

```text
training epochs = 1
finite losses/logits/probabilities/thresholds
checkpoint, dataset, config, and code hashes agree
threshold selection uses clean validation only
susceptibility calibration uses clean validation only
uses_attack_outcomes = false
```

### R7 — detector-only clean timing on at most four eval parents

```bash
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh clean_timing
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh bind_parents
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh build_jobs
```

Required:

```text
clean eval parents <= 4
attack delivery count = 0
clean parent/init-state/checkpoint/config hashes frozen
explicit no-emit parents retained
burst-infeasible late emits retained in denominator ledger
at least one detector-emitted, burst-feasible parent
```

Do not use attacked outcomes to lower thresholds or choose a checkpoint.

### R8 — one-parent five-condition command dry run

Choose one emitted, burst-feasible parent, preferring LIBERO-Object for the first runtime
smoke. Copy the frozen CLEAN bytes; never rerun or overwrite CLEAN. Produce exactly:

```text
CLEAN
DET_GRIPPER_VIS_PGD
DET_RANDOM_VIS_ATTACK
RANDTIME_GRIPPER_VIS_PGD
RANDTIME_RANDOM_VIS_ATTACK
```

Dry-run requirements:

```text
CLEAN worker count = 0
attacked worker count = 4
same B/epsilon/step-size/PGD-K/preprocessing/projection/cast/temporal-init
paired objective seeds
control objective = SHUFFLED_GRIPPER_GRADIENT
```

### R9 — one-parent online visual-attack smoke

Only after R0-R8 PASS, execute the four attacked workers and immediately run the strict
closed-world audit.

Required:

```text
frozen CLEAN bytes unchanged
4 runtime-valid attacked rows
exactly B contiguous attacked frames per row
processor-space Linf within budget
forward/backward/adversarial-decode counts match
objective family and seed pairing match
DET timing pair matches
RANDTIME timing pair matches
random timing differs from detector timing
pre-trigger clean parity PASS
closed-world runtime audit PASS
```

`N=1` is only an execution smoke. Do not make an effectiveness claim.

### R10 — separately approved detector-quality program

R10 is not authorized by this bounded resume. After R9 PASS, return a proposal for:

```text
per-suite/task/mechanism Teacher coverage
multi-target and articulated event coverage
model ladder and no-context primary model
within-task reference
leave-one-task-out primary
leave-one-suite-out diagnostic
>=3 seeds for selected architecture
small preregistered online pilot
confirmatory 2x2 timing x objective matrix
```

The primary comparisons remain:

```text
timing effect:    DET_GRIPPER vs RANDTIME_GRIPPER
objective effect: DET_GRIPPER vs DET_RANDOM
interaction:      detector timing x gripper targeting
```

## 5. Stop conditions

Return HOLD immediately for:

```text
dirty or divergent worktree
Goal/model byte mismatch
unsupported operator or unresolved jaw alias
eligible task without exact goal-event bindings
contacted goal target unresolved
active target contact without explicit progress semantics
multi-target rows forced to false instead of null
articulated region unable to bind a unique joint
attack/outcome leakage into clean labels/features/calibration/model selection
canonical 25D mismatch
unknown converted to negative
collection or model provenance mutation
split leakage
trainability failure after the cumulative cap
non-finite training/calibration
checkpoint/config/dataset hash mismatch
CLEAN attacked frame count > 0
frozen CLEAN overwritten
no burst-feasible emitted eval parent
matched-load/seed/budget mismatch
pre-trigger parity failure
closed-world runtime audit HOLD
projected free space below 15 GiB
```

## 6. Required Codex status cadence

Return at every gate:

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
HISTORICAL_INVALID_CLEAN_EPISODES = 4
NEW_EVENT_AWARE_CLEAN_EPISODES_LAUNCHED
CUMULATIVE_CLEAN_TRAIN_EPISODES_LAUNCHED
CLEAN_EVAL_EPISODES_LAUNCHED
ATTACKED_EPISODES_LAUNCHED
OPENVLA_MODEL_LOADS
TRAINING_EPOCHS
DATASETS_MATERIALIZED
FREE_BYTES_BEFORE_AFTER
MECHANISM_COUNTS
GOAL_EVENT_BINDING_COUNTS
ACTIVE_TARGET_KNOWN_ROWS
CONTACTED_TARGET_UNRESOLVED_ROWS
ACTIVE_PROGRESS_UNRESOLVED_ROWS
KNOWN_TEACHER_ROWS
CRITICAL_POSITIVE_ROWS
TRIGGERABLE_WINDOWS
P0_FINDINGS
P1_FINDINGS
SCIENTIFIC_CONTRACT_CHANGES = NONE / HOLD
D7_TABLE1 = STILL_FROZEN
GO_HOLD_NEXT_STAGE
```

Do not call skipped phases PASS. Do not describe GitHub CPU tests as live LIBERO/OpenVLA
validation. Do not interpret online outcomes before the runtime audit passes.
