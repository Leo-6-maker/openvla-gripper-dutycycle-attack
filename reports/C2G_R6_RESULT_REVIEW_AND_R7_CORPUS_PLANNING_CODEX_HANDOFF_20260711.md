# C2g Detector-v2 R6 Result Review and R7 Corpus-Planning Codex Handoff

Date: 2026-07-11

Repository: `Leo-6-maker/openvla-gripper-dutycycle-attack`

Accepted R6 head:

```text
10c0c40fc241d277d637a6441c64eeae9dac6148
```

R7 code branch:

```text
assistant/c2g-r7-corpus-planning-20260711
```

## 1. Accepted R6 result

The reported R6A-R6C execution is accepted:

```text
REMOTE_HEAD = EXECUTED_HEAD = 10c0c40fc241d277d637a6441c64eeae9dac6148
WORKTREE_CLEAN = YES
TESTS = 208 passed / 0 failed / 0 skipped
R6_PREVIEW = PASS_C2G_R6_DATASET_AUDIT_PREVIEW
R6_AUDIT = PASS_C2G_R6_BOUND_DATASET_AUDIT
INTEGRITY = PASS
ENGINEERING_SMOKE = PASS
SCIENTIFIC_TRAINABILITY = HOLD
TRAINING_AUTHORIZATION = HOLD_INSUFFICIENT_SCIENTIFIC_SUPPORT
```

R6 report:

```text
/mnt/sdc/dty_user/openvla_attack_evidence/c2g/
c2g_r6_dataset_audit_10c0c40_20260711/
c2g_r6_bound_dataset_audit.json
```

R6 report SHA256:

```text
c7d9fc92eafff80845bdbb3c8cbb941917693eccd550a613dc08c76db791c1b7
```

The immutable R5 19-file tree hash remained:

```text
d910da3f90c9adba3a893de346efb2740e8a60d911954e2cb1f4251c2888cbab
```

R6 established the exact current split:

```text
train = LIBERO-10 + Goal
val   = Spatial
test  = Object
```

The 726 overlapping windows reconstruct to only four independent clean episodes
and four suite-namespaced tasks. Each suite contributes one episode, one task,
and one split. The Goal episode contains zero positive and zero triggerable
critical support, with 248 known-negative unique steps.

Therefore the following remain not established:

```text
within-task generalization
leave-one-task-out generalization
leave-one-suite-out generalization
threshold calibration
independent clean test evidence
```

No model, environment, rollout, training, calibration, attack, deletion, merge,
or D7 modification occurred.

## 2. R6 scientific decision

The formal interpretation is:

```text
R5/R6 materialization integrity = PASS
R5/R6 engineering readability = PASS
R5/R6 scientific training data = HOLD
Detector-v2 training = NOT AUTHORIZED
```

Window count must never be substituted for episode support. The next action is
not a one-epoch run and not new collection. The next action is to freeze a
scientific parent registry and perform a read-only eligibility census of all
currently available artifact-rich clean episodes.

## 3. R7 corpus protocol

R7 freezes one deterministic registry over the official LIBERO init-state
universe. The default per-task allocation is:

```text
30 DETECTOR_TRAIN
 5 DETECTOR_VAL
 5 DETECTOR_TEST_WITHIN_TASK
10 ATTACK_EVAL_PREREGISTERED
--------------------------------
50 official states per task
```

Across four suites and ten tasks per suite, the expected registry is:

```text
DETECTOR_TRAIN                 1,200 episodes
DETECTOR_VAL                     200 episodes
DETECTOR_TEST_WITHIN_TASK        200 episodes
ATTACK_EVAL_PREREGISTERED        400 episodes
TOTAL                          2,000 episodes
```

These are planned parents, not collected data. R7 reports planned and actually
available counts separately.

The 10 attack-evaluation states per task are permanently excluded from:

```text
model fitting
checkpoint selection
threshold calibration
clean detector test
architecture tuning
```

They remain preregistered for later downstream timing/attack evaluation only.

### Primary pooled within-task protocol

```text
train = the 30 DETECTOR_TRAIN states from every task
val   = the 5 DETECTOR_VAL states from every task
test  = the 5 DETECTOR_TEST_WITHIN_TASK states from every task
```

Every suite and every task is therefore represented in all three splits by
construction, without episode or state overlap.

### LOTO protocol

For held-out `(suite, task)`:

```text
train = DETECTOR_TRAIN from all non-held tasks
val   = DETECTOR_VAL from all non-held tasks
test  = all 40 detector-development states from the held task
attack-eval states remain excluded
```

### LOSO protocol

For held-out suite:

```text
train = DETECTOR_TRAIN from the three non-held suites
val   = DETECTOR_VAL from the three non-held suites
test  = all 400 detector-development states from the held suite
attack-eval states remain excluded
```

The same frozen registry is used for all protocols. No fold may redraw states.

## 4. R7 code contract

Planner:

```text
tools/multisuite_detector/plan_c2g_scientific_corpus.py
```

It:

- reads official LIBERO benchmark task/state metadata only;
- creates no environment;
- loads no OpenVLA model;
- deterministically allocates exact per-task cohorts;
- emits a 2,000-parent registry when official 50-state coverage is present;
- emits four cohort manifests;
- emits 40 LOTO fold definitions and four LOSO fold definitions;
- rejects state/cohort overlap and duplicate parent keys;
- records training authorization as HOLD.

Read-only source inventory auditor:

```text
tools/multisuite_detector/audit_c2g_clean_source_inventory.py
```

It:

- binds the exact R7 plan report and registry bytes;
- scans existing clean `episode_metadata.json` + `step_records.jsonl` pairs;
- checks CLEAN/runtime-valid identity;
- verifies finite 25D and 9D causal features;
- verifies task language and all RGB paths;
- rejects attacked/post-intervention/outcome fields;
- rebuilds Teacher-v2 labels from clean privileged fields;
- measures unique-step positive, negative, unknown, and 2-of-3 support;
- detects duplicate suite/task/state identities;
- separates registered reusable, registered ineligible, and unregistered assets;
- emits exact missing-parent deficits by cohort/suite/task;
- never authorizes training.

Launcher:

```text
scripts/stageb/run_c2g_r7_corpus_planning.sh
```

Modes:

```text
preview-plan
plan
preview-audit
audit
```

Regression tests:

```text
tests/test_c2g_r7_scientific_corpus.py
```

## 5. Codex authorization

Codex is authorized only for:

```text
R7A current-head static validation
R7B exact corpus-plan preview
R7C one official-metadata corpus-plan materialization
R7D exact source-audit preview
R7E one read-only source-inventory audit of the frozen four-episode collection
STOP
```

Codex is not authorized to:

- patch repository or server code;
- lower or change the 30/5/5/10 allocation;
- change selection seed 42;
- create a LIBERO environment;
- load any OpenVLA model;
- launch any clean rollout;
- collect new data;
- materialize SigLIP/language embeddings;
- train or calibrate Detector-v2;
- run clean timing or VIS-PGD;
- merge any PR;
- modify D7;
- delete or clean storage.

If a compatibility issue appears, stop and report it exactly. Do not patch on
the server.

## 6. R7A current-head static gate

```bash
git fetch origin --prune
git checkout assistant/c2g-r7-corpus-planning-20260711
git reset --hard origin/assistant/c2g-r7-corpus-planning-20260711

export REMOTE_HEAD="$(git rev-parse origin/assistant/c2g-r7-corpus-planning-20260711)"
export EXECUTED_HEAD="$(git rev-parse HEAD)"
export PYTHONPATH="$(git rev-parse --show-toplevel)/src:$(git rev-parse --show-toplevel)${PYTHONPATH:+:$PYTHONPATH}"

test "$REMOTE_HEAD" = "$EXECUTED_HEAD"
test -z "$(git status --short)"
git diff --check

python -m unittest discover -s tests -p 'test_c2g*.py' -v

python -m py_compile \
  tools/multisuite_detector/plan_c2g_scientific_corpus.py \
  tools/multisuite_detector/audit_c2g_clean_source_inventory.py \
  tests/test_c2g_r7_scientific_corpus.py

bash -n scripts/stageb/run_c2g_r7_corpus_planning.sh
```

Required:

```text
REMOTE_HEAD = EXECUTED_HEAD
failed = 0
skipped = 0
py_compile = PASS
bash syntax = PASS
worktree clean
```

## 7. R7B plan preview

```bash
export R6_AUDIT_REPORT=/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r6_dataset_audit_10c0c40_20260711/c2g_r6_bound_dataset_audit.json
export EXPECTED_R6_AUDIT_SHA256=c7d9fc92eafff80845bdbb3c8cbb941917693eccd550a613dc08c76db791c1b7
export EXPECTED_R6_HEAD=10c0c40fc241d277d637a6441c64eeae9dac6148

export AUDIT_HEAD="$EXECUTED_HEAD"
export R7_OUTPUT_ROOT=/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r7_corpus_plan_${EXECUTED_HEAD:0:8}_20260711

export SELECTION_SEED=42
export MAX_STEPS=300
export TRAIN_STATES_PER_TASK=30
export VAL_STATES_PER_TASK=5
export TEST_STATES_PER_TASK=5
export ATTACK_EVAL_STATES_PER_TASK=10

test ! -e "$R7_OUTPUT_ROOT"

bash scripts/stageb/run_c2g_r7_corpus_planning.sh preview-plan \
  | tee /tmp/c2g_r7_corpus_plan_preview.json
```

Preview requirements:

```text
status = PASS_C2G_R7_CORPUS_PLAN_PREVIEW
command uses --from-libero
command binds current R7 head
command freezes seed 42 and 30/5/5/10
R7 output root remains absent
model loads = 0
environments = 0
rollouts = 0
```

## 8. R7C official-metadata corpus plan

The planner may import LIBERO benchmark metadata and read task init-state array
lengths. It may not create an environment.

```bash
bash scripts/stageb/run_c2g_r7_corpus_planning.sh plan \
  | tee /tmp/c2g_r7_corpus_plan_stdout.json

export R7_PLAN_REPORT="$R7_OUTPUT_ROOT/c2g_scientific_corpus_plan_report.json"
export R7_REGISTRY="$R7_OUTPUT_ROOT/c2g_parent_registry.jsonl"

test -f "$R7_PLAN_REPORT"
test -f "$R7_REGISTRY"
sha256sum "$R7_PLAN_REPORT" "$R7_REGISTRY"

export EXPECTED_R7_PLAN_REPORT_SHA256="$(sha256sum "$R7_PLAN_REPORT" | awk '{print $1}')"
```

Required review fields:

```text
status
expected_git_commit
inventory_source
selection_seed
cohort_counts_per_task
summary.episode_count
summary.task_count
summary.suite_count
summary.cohort_counts
summary.split_counts
summary.per_suite
loto_fold_count
loso_fold_count
training_authorization
boundaries
```

Expected official shape, but do not force it:

```text
suite_count = 4
task_count = 40
episode_count = 2000
train = 1200
val = 200
test = 200
attack_eval = 400
LOTO folds = 40
LOSO folds = 4
```

Any task with fewer than 50 available init states is a hard HOLD. Do not reduce
counts to make the plan pass.

## 9. R7D source-audit preview

Frozen source collection to inspect:

```text
/mnt/sdc/dty_user/openvla_attack_evidence/c2g/
c2g_event_v2_f5b2b2d1_20260711/clean_collection
```

Configure:

```bash
export R7_SOURCE_ROOTS=/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_event_v2_f5b2b2d1_20260711/clean_collection
export R7_SOURCE_AUDIT_ROOT=/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r7_source_inventory_${EXECUTED_HEAD:0:8}_20260711
export R7_SOURCE_AUDIT_REPORT="$R7_SOURCE_AUDIT_ROOT/c2g_r7_clean_source_inventory_audit.json"
export R7_REUSABLE_MANIFEST="$R7_SOURCE_AUDIT_ROOT/c2g_r7_reusable_clean_parents.jsonl"
export HASH_RGB=1

test ! -e "$R7_SOURCE_AUDIT_REPORT"
test ! -e "$R7_REUSABLE_MANIFEST"

bash scripts/stageb/run_c2g_r7_corpus_planning.sh preview-audit \
  | tee /tmp/c2g_r7_source_audit_preview.json
```

Preview requirements:

```text
status = PASS_C2G_R7_SOURCE_AUDIT_PREVIEW
plan report SHA is explicitly bound
registry path is explicitly bound
source root is the frozen clean collection
outputs are outside source and repository trees
source tree remains unchanged
```

## 10. R7E one read-only source inventory audit

```bash
before_tree="$(find "$R7_SOURCE_ROOTS" -type f -printf '%P %s\n' | sort | sha256sum | awk '{print $1}')"

bash scripts/stageb/run_c2g_r7_corpus_planning.sh audit \
  | tee /tmp/c2g_r7_source_audit_stdout.json

after_tree="$(find "$R7_SOURCE_ROOTS" -type f -printf '%P %s\n' | sort | sha256sum | awk '{print $1}')"

test "$before_tree" = "$after_tree"
test -f "$R7_SOURCE_AUDIT_REPORT"
test -f "$R7_REUSABLE_MANIFEST"
sha256sum "$R7_SOURCE_AUDIT_REPORT" "$R7_REUSABLE_MANIFEST"
```

Print the required fields:

```bash
python - "$R7_SOURCE_AUDIT_REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in (
    "status",
    "audit_head",
    "source_episode_candidate_count",
    "registered_reusable_episode_count",
    "registered_ineligible_episode_count",
    "unregistered_episode_count",
    "duplicate_identity_count",
    "detector_source_corpus_status",
    "attack_eval_source_corpus_status",
    "detector_required_episode_count",
    "detector_available_episode_count",
    "detector_missing_episode_count",
    "attack_eval_required_episode_count",
    "attack_eval_available_episode_count",
    "attack_eval_missing_episode_count",
    "within_task_generalization_ready",
    "loto_generalization_ready",
    "loso_generalization_ready",
    "training_authorization",
    "next_stage",
    "boundaries",
):
    print(f"{key}={report.get(key)!r}")
print("per_cohort=")
for key, value in report["per_cohort"].items():
    print(key, value)
print("per_suite=")
for key, value in report["per_suite"].items():
    print(key, value)
print("per_split=")
for key, value in report["per_split"].items():
    print(key, value)
print("episodes=")
for value in report["episode_audits"]:
    print(value)
PY
```

## 11. Expected interpretation

The exact result must come from R7. The likely current source result is:

```text
corpus plan = PASS
planned parent count = 2000
existing source candidates = 4
registered reusable = between 0 and 4, determined by exact identities/schema
source corpus completeness = HOLD
within-task readiness = false
LOTO readiness = false
LOSO readiness = false
training authorization = HOLD
```

Do not force the four episodes to be reusable. Any duplicate identity, forbidden
field, missing RGB, malformed 25D/9D vector, or Teacher-v2 failure must remain
ineligible and be reported.

## 12. Required final result format

```text
R7_HEAD
REMOTE_HEAD
EXECUTED_HEAD
WORKTREE_CLEAN

TESTS_PASSED
TESTS_FAILED
TESTS_SKIPPED
PY_COMPILE
BASH_SYNTAX

R7_PLAN_PREVIEW_STATUS
R7_PLAN_STATUS
R7_PLAN_REPORT
R7_PLAN_REPORT_SHA256
R7_REGISTRY
R7_REGISTRY_SHA256

PLANNED_EPISODES
PLANNED_TASKS
PLANNED_SUITES
PLANNED_COHORT_COUNTS
PLANNED_SPLIT_COUNTS
LOTO_FOLD_COUNT
LOSO_FOLD_COUNT

R7_SOURCE_PREVIEW_STATUS
R7_SOURCE_AUDIT_STATUS
R7_SOURCE_AUDIT_REPORT
R7_SOURCE_AUDIT_REPORT_SHA256
R7_REUSABLE_MANIFEST
R7_REUSABLE_MANIFEST_SHA256

SOURCE_EPISODE_CANDIDATES
REGISTERED_REUSABLE_EPISODES
REGISTERED_INELIGIBLE_EPISODES
UNREGISTERED_EPISODES
DUPLICATE_IDENTITIES
DETECTOR_REQUIRED
DETECTOR_AVAILABLE
DETECTOR_MISSING
ATTACK_EVAL_REQUIRED
ATTACK_EVAL_AVAILABLE
ATTACK_EVAL_MISSING
PER_COHORT_SUPPORT
PER_SUITE_SUPPORT
PER_SPLIT_SUPPORT
EPISODE_AUDITS

WITHIN_TASK_READY
LOTO_READY
LOSO_READY
TRAINING_AUTHORIZATION
SOURCE_TREE_UNCHANGED

MODEL_LOADS
LIBERO_ENVIRONMENTS
CLEAN_ROLLOUTS
ATTACKED_ROLLOUTS
TRAINING_EPOCHS
CALIBRATION_RUNS
STORAGE_DELETIONS
D7_TABLE1

P0_FINDINGS
P1_FINDINGS
FINAL_DECISION
```

Required final decision:

```text
GO_R7_RESULT_REVIEW
```

or:

```text
HOLD_<EXACT_PROVENANCE_OR_COMPATIBILITY_REASON>
```

After posting the complete result to the R7 Draft PR, stop. No collection or
training stage is automatically authorized.
