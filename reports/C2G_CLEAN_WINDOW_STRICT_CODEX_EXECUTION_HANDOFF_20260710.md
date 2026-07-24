# C2g Clean-Window Detector v2 — Strict Codex Execution Handoff

Date: 2026-07-10

## 0. Repository identity

```text
repository = Leo-6-maker/openvla-gripper-dutycycle-attack
branch = assistant/c2g-clean-window-v2-20260710
pull_request = #58 (Draft, unmerged)
base_branch = assistant/c2g-p0-static-patch-20260710
base_sha = e3bec3b82ac104c633d8dacc5fd27f9cf30a7e85
pre_handoff_remote_head = eeed04fc8439d41521166ae6fbb85083fe0d90c5
canonical_entry_point = scripts/stageb/run_c2g_clean_window_pipeline_strict.sh
canonical_direct_orchestrator = scripts/stageb/run_c2g_clean_window_pipeline.sh
```

The branch may move when this handoff itself is committed. Codex must fetch the remote
branch and bind every run to the exact current output of `git rev-parse HEAD`. Do not
hard-code `pre_handoff_remote_head` as the execution head.

## 1. Mission

Take over the server-side validation and bounded execution of the complete clean-only
C2g Detector-v2 pipeline:

```text
strict four-suite model provenance
  -> deterministic disjoint clean train/eval parent manifests
  -> clean privileged LIBERO/OpenVLA collection
  -> clean Teacher-v2 audit
  -> four-suite OpenVLA/SigLIP dataset materialization
  -> fail-closed dataset trainability audit
  -> one-epoch detector training smoke
  -> clean-only susceptibility calibration
  -> detector-only CLEAN timing
  -> frozen parent/init/checkpoint binding
  -> matched-load job construction
  -> one-parent five-condition command dry run
  -> one-parent bounded online VIS-PGD smoke
  -> closed-world runtime audit
  -> paired result analysis
```

This handoff authorizes a **bounded end-to-end server smoke**, not the full production
experiment.

## 2. Scientific contract — do not change

Primary detector target:

```text
target relevant
AND gripper dependent
AND clean close intent
AND lift / transport / constrained manipulation
AND NOT release safe
```

Primary student inputs are causal clean-only inputs:

```text
clean RGB
clean 25D proprio/action history
clean OpenVLA gripper policy-intent history
current task language
```

Forbidden for labels, student inputs, split selection, threshold selection,
susceptibility calibration, and model selection:

```text
attacked observations or actions
post-intervention state
attack outcome
manual attacked-failure labels
counterfactual replay outcome
future clean student inputs
task-index/hash or suite one-hot shortcut features
normalized episode step as task identity proxy
```

The detector selects only the start time. Once triggered, the attack is an immutable
fixed `B`-frame visual burst. The existing `TokenPrefixPGDAttacker` remains the
payload. Do not replace the primary attack with command-space force-open.

Frozen primary online matrix:

```text
CLEAN
DET_GRIPPER_VIS_PGD
DET_RANDOM_VIS_ATTACK
RANDTIME_GRIPPER_VIS_PGD
RANDTIME_RANDOM_VIS_ATTACK
```

The only currently authorized primary compute-matched control is:

```text
SHUFFLED_GRIPPER_GRADIENT
```

Do not substitute uniform noise or an unmatched action-noise control.

## 3. Hard boundaries

Authorized by this handoff:

```text
CPU/static/read-only validation
strict model and asset hashing
up to 40 new CLEAN training episodes total
up to 4 detector-only CLEAN evaluation parents
one-epoch detector training smoke
one clean susceptibility calibration
one selected parent for the matched online smoke
four attacked runs for that selected parent; the existing frozen CLEAN parent is reused
```

Not authorized:

```text
full CLEAN2000 collection
more than 40 new CLEAN training episodes
more than 1 training epoch
full LOTO/LOSO training matrix
more than 4 detector-only CLEAN evaluation parents
more than one matched online parent
full online replication matrix
counterfactual replay as a primary label source
publication claim updates
merging PR #58
modifying main
modifying or replacing frozen D7 evidence
```

Always preserve:

```text
D7_TABLE1 = STILL_FROZEN
TEACHER_V1_FOR_PRIMARY_TRAINING = HOLD
COUNTERFACTUAL_REPLAY = OPTIONAL_POSTHOC_ONLY
PR58 = DRAFT_AND_UNMERGED
```

Stop immediately at the first fail-closed gate that cannot be fixed without changing
the scientific contract.

## 4. Allowed code fixes during server validation

Codex may create a new branch from the fetched PR head and commit minimal fixes for:

```text
server path discovery
Python/import compatibility
LIBERO/OpenVLA/MuJoCo API compatibility
serialization and dtype compatibility
checkpoint loading compatibility
strict provenance checks
missing runtime logging required by an existing audit contract
launcher/resume/output-path correctness
```

Codex must not silently change:

```text
Teacher-v2 label semantics
student feature set
split rules
dataset trainability gates
loss definition
threshold objective
susceptibility objective
2-of-3 persistence
burst length
attack objective
condition matrix
matched-load requirements
outcome interpretation
```

If one of those scientific elements appears wrong, return `HOLD_SCIENTIFIC_CONTRACT`
with evidence rather than patching it during the run.

## 5. Mandatory operating rules

1. Use a fresh clean checkout.
2. Work from a dedicated server branch only when fixes are needed:

```text
codex/c2g-strict-server-smoke-20260710
```

3. Keep every generated artifact outside the repository.
4. Never use historical `release_v2` through `release_v8` wrappers as the primary
   entry point.
5. Use only:

```bash
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh <phase>
```

6. Never run `all` during this smoke.
7. Record every exact command, exit code, report path, and SHA256.
8. Do not call a stage PASS merely because the process exited zero; inspect its report
   status and invariants.
9. Never convert unknown labels into negatives.
10. Never rerun or overwrite a frozen CLEAN online parent during the matched smoke.

## 6. Stage S0 — checkout, identity, and repository validation

Run:

```bash
git fetch origin --prune
git checkout -B codex/c2g-strict-server-smoke-20260710 \
  origin/assistant/c2g-clean-window-v2-20260710

export C2G_HEAD="$(git rev-parse HEAD)"
git rev-parse HEAD
git rev-parse HEAD^
git merge-base --is-ancestor e3bec3b82ac104c633d8dacc5fd27f9cf30a7e85 HEAD
git status --short
git diff --check
```

Record:

```text
remote branch head
local head
base ancestor result
worktree cleanliness
Python version
PyTorch version
CUDA version
GPU inventory
CPU/RAM/free-space inventory
LIBERO/OpenVLA environment path
```

Run repository-side validation:

```bash
python -m py_compile \
  scripts/stageb/run_c2g_clean_window_pipeline.sh \
  scripts/stageb/collect_c2g_clean_window_rollouts_release.py \
  scripts/stageb/run_c2g_clean_timing_jobs_strict.py \
  scripts/stageb/build_c2g_matched_load_jobs_release.py \
  scripts/stageb/run_c2g_matched_load_jobs_map_release.py \
  scripts/stageb/audit_c2g_matched_load_run_release.py \
  scripts/stageb/analyze_c2g_matched_load_results.py \
  tools/multisuite_detector/materialize_c2g_multisuite_dataset.py \
  tools/multisuite_detector/validate_c2g_clean_window_dataset.py \
  tools/multisuite_detector/train_c2g_clean_window_detector.py \
  tools/multisuite_detector/calibrate_c2g_clean_susceptibility.py

bash -n scripts/stageb/run_c2g_clean_window_pipeline.sh
bash -n scripts/stageb/run_c2g_clean_window_pipeline_strict.sh

python -m unittest discover -s tests -p 'test_c2g*.py' -v
```

S0 gate:

```text
all selected Python files compile
both canonical shell scripts pass bash -n
all discovered C2g tests pass
worktree remains clean
no protected/generated repository output is created
```

Any static failure must be fixed and the whole S0 gate rerun before continuing.

## 7. Stage S1 — read-only live asset discovery

Before loading OpenVLA or launching LIBERO:

1. Locate the four exact policy model directories.
2. Locate the audited Goal model manifest.
3. Locate official LIBERO BDDL roots.
4. Locate MuJoCo XML/model assets.
5. Run the repository's existing static BDDL/MuJoCo inventory.
6. Freeze a SHA256 inventory in a new external directory.

Required live-asset findings:

```text
all four suite model directories exist
all referenced weight shards exist
Goal model manifest exists and reports PASS
Goal manifest path/model path agree
all required BDDL files parse or are explicitly held
left/right gripper contact aliases resolve
no server artifact is written into the repository
OpenVLA models loaded during read-only inventory = 0
LIBERO rollouts launched during read-only inventory = 0
```

Unsupported task operators or unresolved contact identity are a HOLD for affected
tasks. Do not guess using language-only matching.

## 8. Common bounded environment

After S0 and S1 pass, define:

```bash
export WORK_ROOT=/ABSOLUTE/EXTERNAL/c2g_strict_smoke_${C2G_HEAD:0:12}
export GOAL_MODEL_MANIFEST=/ABSOLUTE/AUDITED/goal_model_manifest.json
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
```

`WORK_ROOT` must be outside the repository and initially empty. Record its filesystem,
free space, owner, and permissions.

## 9. Stage S2 — strict model map and tiny preregistered manifests

Start with one task and one state per suite:

```bash
export TRAIN_STATES_PER_TASK=1
export EVAL_STATES_PER_TASK=1
export MAX_TASKS_PER_SUITE=1
export MAX_TRAIN_EPISODES=0
export MAX_EVAL_JOBS=0

bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh models
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh manifests
```

Inspect and hash:

```text
$WORK_ROOT/config/c2g_suite_model_map.json
$WORK_ROOT/config/c2g_suite_model_map_report.json
$WORK_ROOT/config/c2g_suite_model_verification_report.json
$WORK_ROOT/manifests/c2g_train_clean_parents.jsonl
$WORK_ROOT/manifests/c2g_eval_preregistered_parents.jsonl
```

S2 gate:

```text
all four suites represented
all model weight shards hash-bound
Goal model integrity bound
exact five-part parent keys
train/eval state identity overlap = 0
one train and one eval parent per selected task
no model inference or rollout during manifest generation
```

## 10. Stage S3 — four-suite CLEAN compatibility collection

The initial collection must contain exactly one clean training parent per suite. It is
allowed to load OpenVLA and launch clean LIBERO rollouts, but no attack object may be
constructed or executed.

Run:

```bash
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh collect
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh audit
```

For each suite inspect at least one full episode:

```text
metadata exists and is valid JSON
step_records.jsonl exists and is nonempty
RGB paths resolve
canonical features_25d length/order is exact
clean_policy_intent_9d is finite
suite model path and full model digest match the frozen model map
Goal episode binds the Goal manifest
uses_attack_outcome = false
attacks_launched = 0
no attacked observation/action/result field exists
privileged target/contact/progress/release fields are either explicit or unknown
```

S3 gate:

```text
4 selected clean parents attempted
4 runtime-valid clean episodes completed
0 attacked frames
0 attack outcomes read
0 model-provenance mismatches
Teacher dry audit contains no unknown-to-negative conversion
release-safe is never critical-positive
absolute EEF-z alone never creates a positive
```

A live API incompatibility may be minimally patched under Section 4. Rerun S0 and S3
after every code patch.

## 11. Stage S4 — real tiny dataset and trainability gate

Run the strict materializer only after the immutable collection/model binding report
passes:

```bash
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh materialize
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh dataset_audit
```

The one-parent-per-suite compatibility cohort may legitimately HOLD because it is too
small to provide train/val/test positive and negative support. Never weaken the audit.

If and only if the HOLD reason is insufficient clean support, expand the clean training
cohort in this order:

```text
Expansion A:
  TRAIN_STATES_PER_TASK=3
  MAX_TASKS_PER_SUITE=2
  maximum new clean episodes = 24

Expansion B, only if A remains insufficient:
  TRAIN_STATES_PER_TASK=5
  MAX_TASKS_PER_SUITE=2
  maximum new clean episodes = 40
```

For an expansion, use a new empty `WORK_ROOT` suffix or rebuild manifests and collection
without mixing incompatible frozen manifests. Never exceed 40 clean training episodes.

S4 gate:

```text
collection provenance reverified immediately before materialization
episode metadata closure unchanged
model shard bytes unchanged
four suites represented
feature dimensions compatible across suites
no episode appears in multiple splits
known positives and known negatives exist in required splits
2-of-3 triggerable positive support exists
unknown remains masked
student payload contains no privileged or identity-shortcut field
status = PASS_C2G_DATASET_TRAINABILITY
```

If the 40-episode cap still cannot satisfy the gate, return
`HOLD_INSUFFICIENT_CLEAN_SUPPORT` and stop.

## 12. Stage S5 — one-epoch checkpoint and clean susceptibility calibration

Run only after S4 PASS:

```bash
export EPOCHS=1
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh train
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh calibrate
```

Then independently reload the exported checkpoint with `strict=True` and run at least
one validation batch through the runtime model.

S5 gate:

```text
checkpoint schema accepted
checkpoint reload strict=true
checkpoint dataset SHA matches the materialized dataset
training report and checkpoint hashes agree
model selection uses clean validation only
detector thresholds are validation-only
susceptibility schema accepted
uses_attack_outcomes = false
runtime source = checkpoint_clean_validation or equivalent frozen clean source
no NaN/Inf in logits, probabilities, losses, or thresholds
training epochs executed = 1
```

Do not increase epochs in this handoff.

## 13. Stage S6 — detector-only CLEAN timing and frozen job build

Use the preregistered tiny evaluation manifest: one parent per suite, four total.

Run:

```bash
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh clean_timing
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh bind_parents
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh build_jobs
```

S6 gate:

```text
4 preregistered evaluation parents attempted
all CLEAN runs deliver 0 attacked frames
CLEAN outputs are runtime-valid
clean parent metadata/steps hashes frozen
official init-state dtype/shape/content hashes frozen
checkpoint/config hashes frozen
explicit detector no-emit retained in timing manifest
late burst-infeasible emit retained in denominator ledger
random-time start differs from detector start
two objective conditions at each timing share paired objective seeds
CLEAN objective seed equals preregistered eval_seed
```

If no parent is detector-emitted and burst-feasible, return
`HOLD_NO_ATTACKABLE_PARENT_IN_TINY_EVAL` and stop. Do not lower thresholds using attacked
outcomes.

## 14. Stage S7 — one-parent closed-world command dry run

Select exactly one detector-emitted, burst-feasible parent from the frozen job manifest.
Prefer `libero_object`; otherwise use the first eligible parent in deterministic
suite/task/state order.

Create outside the repository:

```text
$WORK_ROOT/smoke_one_parent/jobs.jsonl
$WORK_ROOT/smoke_one_parent/jobs.excluded.jsonl
$WORK_ROOT/smoke_one_parent/online/
```

Rules:

1. `jobs.jsonl` contains exactly the five frozen rows for the selected parent.
2. Validate it with `validate_core_2x2_manifest(..., strict_objective_seed_pairing=True)`.
3. `jobs.excluded.jsonl` is empty for this one-parent closed world.
4. Copy the selected frozen `CLEAN` directory byte-for-byte into the new smoke online
   root. Recompute hashes and confirm equality. Do not rerun CLEAN.
5. Execute the model-map launcher with `--dry-run` against the five-row manifest.
6. Verify it will validate/reuse CLEAN and launch only the four attacked conditions.

Direct dry-run command:

```bash
python scripts/stageb/run_c2g_matched_load_jobs_map_release.py \
  --jobs "$WORK_ROOT/smoke_one_parent/jobs.jsonl" \
  --suite-model-map "$WORK_ROOT/config/c2g_suite_model_map.json" \
  --goal-model-manifest "$GOAL_MODEL_MANIFEST" \
  --output-root "$WORK_ROOT/smoke_one_parent/online" \
  --checkpoint "$WORK_ROOT/training/c2g_clean_window_detector.pt" \
  --expected-git-commit "$C2G_HEAD" \
  --device "$DEVICE" \
  --resume \
  --dry-run
```

S7 gate:

```text
exactly five conditions in the smoke manifest
frozen CLEAN bytes unchanged
no CLEAN worker launch
exactly four attacked commands planned
correct suite model selected
Goal manifest passed only when required
same B/epsilon/step-size/K/projection/cast/preprocess/temporal-init across attacks
paired objective seeds correct
control objective = SHUFFLED_GRIPPER_GRADIENT
```

## 15. Stage S8 — bounded one-parent online VIS-PGD smoke

S8 is authorized only when S0–S7 all PASS. Execute the same command without
`--dry-run`:

```bash
python scripts/stageb/run_c2g_matched_load_jobs_map_release.py \
  --jobs "$WORK_ROOT/smoke_one_parent/jobs.jsonl" \
  --suite-model-map "$WORK_ROOT/config/c2g_suite_model_map.json" \
  --goal-model-manifest "$GOAL_MODEL_MANIFEST" \
  --output-root "$WORK_ROOT/smoke_one_parent/online" \
  --checkpoint "$WORK_ROOT/training/c2g_clean_window_detector.pt" \
  --expected-git-commit "$C2G_HEAD" \
  --device "$DEVICE" \
  --resume
```

Immediately audit and analyze:

```bash
python scripts/stageb/audit_c2g_matched_load_run_release.py \
  --jobs "$WORK_ROOT/smoke_one_parent/jobs.jsonl" \
  --output-root "$WORK_ROOT/smoke_one_parent/online" \
  --excluded-ledger "$WORK_ROOT/smoke_one_parent/jobs.excluded.jsonl" \
  --report "$WORK_ROOT/smoke_one_parent/runtime_audit.json"

python scripts/stageb/analyze_c2g_matched_load_results.py \
  --audit-report "$WORK_ROOT/smoke_one_parent/runtime_audit.json" \
  --output "$WORK_ROOT/smoke_one_parent/result_analysis.json"
```

If the analyzer requires a job-build denominator report, create a smoke-local report
from the already frozen parent/job metadata; do not infer or edit outcomes.

S8 runtime gate:

```text
CLEAN is reused, not overwritten
exactly four attacked workers launched
all four attacked jobs runtime-valid
exactly B contiguous attacked frames per attacked condition
reported processor-space Linf <= epsilon + tolerance
reported forward/backward/decode counts match the manifest
objective family and objective seed match the manifest
detector timing pair starts identically
random-time pair starts identically
random-time start differs from detector start
pre-trigger clean trajectory parity passes
closed-world audit status = PASS
```

A one-parent result is a runtime smoke only. Do not make an effectiveness claim or
update the paper from `n=1`.

## 16. Stop conditions

Stop and return HOLD immediately for any of the following:

```text
dirty or divergent worktree
unresolved current remote head
model shard or Goal manifest mismatch
unsupported BDDL operator without explicit abstain route
unresolved bilateral gripper contact identity
attacked/outcome field entering clean labels or calibration
canonical 25D mismatch
unknown converted to negative
collection provenance mutation
split leakage
trainability gate failure after the 40-episode cap
non-finite training/calibration output
checkpoint/config/dataset hash mismatch
CLEAN attack delivery count > 0
CLEAN parent overwritten
no eligible tiny-eval parent
manifest condition/load/seed mismatch
budget violation
pre-trigger parity failure
runtime audit HOLD
```

For every HOLD provide:

```text
exact stage
exact command
exit code
file/parent/condition/step
expected value
actual value
smallest safe proposed fix
whether the fix is engineering-only or changes the scientific contract
```

## 17. Required first Codex response

Before running anything, Codex must reply with:

```text
CURRENT_REMOTE_HEAD
LOCAL_CHECKOUT_HEAD
BASE_ANCESTOR_CHECK
WORKTREE_STATUS
CANONICAL_ENTRY_POINT
SERVER_ENVIRONMENT_DISCOVERY_PLAN
GOAL_MODEL_MANIFEST_CANDIDATE
LIVE_ASSET_ROOT_CANDIDATES
STATIC_TEST_COMMANDS
BOUNDED_RESOURCE_CAPS
STAGE_PLAN_S0_TO_S8
GPU_JOBS_PLANNED
CLEAN_ROLLOUTS_PLANNED
ATTACKED_RUNS_PLANNED
FULL_MATRIX_PLANNED = 0
D7_MODIFICATIONS_PLANNED = 0
GO/HOLD_TO_START_S0
```

## 18. Required final Codex report

Return one structured report containing:

```text
BRANCH
BASE_SHA
REMOTE_HEAD_AT_START
EXECUTED_HEAD
NEW_FIX_COMMITS
WORKTREE_CLEAN
REMOTE_HEAD_MATCH

S0_REPOSITORY_STATIC
S1_LIVE_ASSET_INVENTORY
S2_MODEL_MAP_AND_MANIFESTS
S3_FOUR_SUITE_CLEAN_COMPATIBILITY
S4_REAL_DATASET_AND_TRAINABILITY
S5_ONE_EPOCH_CHECKPOINT
S5_CLEAN_SUSCEPTIBILITY
S6_DETECTOR_ONLY_TIMING
S6_PARENT_AND_JOB_BINDING
S7_ONE_PARENT_COMMAND_DRY_RUN
S8_ONE_PARENT_ONLINE_SMOKE
S8_RUNTIME_AUDIT
S8_RESULT_ANALYSIS

CLEAN_TRAIN_EPISODES_LAUNCHED
CLEAN_EVAL_EPISODES_LAUNCHED
ATTACKED_EPISODES_LAUNCHED
OPENVLA_MODEL_LOADS
DETECTORS_TRAINED
TRAINING_EPOCHS
DATASETS_MATERIALIZED
COUNTERFACTUAL_REPLAYS

MODEL_MAP_SHA256
GOAL_MODEL_MANIFEST_SHA256
TRAIN_MANIFEST_SHA256
EVAL_MANIFEST_SHA256
COLLECTION_BINDING_SHA256
DATASET_SHA256
DATASET_AUDIT_SHA256
CHECKPOINT_SHA256
TRAIN_REPORT_SHA256
SUSCEPTIBILITY_REPORT_SHA256
TIMING_MANIFEST_SHA256
BOUND_PARENT_SHA256
JOB_MANIFEST_SHA256
EXCLUDED_LEDGER_SHA256
RUNTIME_AUDIT_SHA256
RESULT_ANALYSIS_SHA256

P0_FINDINGS
P1_FINDINGS
ENGINEERING_FIXES
SCIENTIFIC_CONTRACT_CHANGES = NONE / HOLD
D7_TABLE1 = STILL_FROZEN
GO_HOLD_NEXT_STAGE
```

Do not mark a stage PASS if it was skipped. Do not report static/synthetic validation as
live LIBERO/OpenVLA validation. Do not interpret an online outcome unless the closed-
world runtime audit passes first.
