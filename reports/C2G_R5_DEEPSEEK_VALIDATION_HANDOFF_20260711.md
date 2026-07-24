# C2g Detector-v2 R5 DeepSeek Validation Handoff

Date: 2026-07-11

## 1. Role split

- Repository code, scientific contract, and debugging are owned by the assistant branch/PR.
- DeepSeek is the independent server executor and evidence collector.
- DeepSeek must not redesign Teacher-v2, Detector-v2, the loss, the one-shot scheduler, or the attack protocol.

## 2. Accepted R4 evidence

The read-only R4 re-audit is accepted:

- canonical audit: PASS;
- goal-event audit: PASS;
- known rows: 689;
- critical positives: 215;
- triggerable rows: 159;
- unknown rows: 97;
- known negatives: 474;
- unknown-to-negative conversions: 0;
- LIBERO-10 retains two critical intervals and two burst-feasible intervals;
- episode-global attack-start rows changed only from 2 to 1;
- the eight frozen collection files were byte-identical before/after.

The DeepSeek commit `cf7a6a4...` changed tests only. It is not part of the production R5 branch. In particular, skipping torch-dependent test classes is not equivalent to proving torch compatibility. Server validation must report skipped tests and use the correct Python environment for detector tests.

## 3. Code under validation

Repository:
`Leo-6-maker/openvla-gripper-dutycycle-attack`

Branch:
`assistant/c2g-r5-provenance-materialization-20260711`

Draft PR:
`#60`

Always fetch and bind the actual remote head. Do not rely on a head copied from this document.

Added code:

- `tools/multisuite_detector/bind_c2g_r4_dual_head_provenance.py`
- `tools/multisuite_detector/materialize_c2g_multisuite_dataset_bound.py`
- `scripts/stageb/run_c2g_r5_bound_materialization.sh`
- focused R4/R5 CPU tests.

## 4. Current authorization

Authorized now:

- checkout and static tests;
- read-only verification of the existing R3 collection;
- rebuild of the R4 dual-head binding with the repository tool;
- R5 `preview` only;
- model-byte hashing/verification without model loading.

Not authorized now:

- actual dataset materialization;
- OpenVLA model loading;
- LIBERO environment creation;
- new clean rollout;
- detector training;
- calibration;
- clean timing;
- VIS-PGD;
- attacked rollout;
- PR merge;
- D7 modification.

Resource counters for this validation must remain zero:

```text
NEW_CLEAN_EPISODES = 0
OPENVLA_MODEL_LOADS = 0
LIBERO_ENVIRONMENTS = 0
DATASETS_MATERIALIZED = 0
TRAINING_EPOCHS = 0
CLEAN_EVAL_EPISODES = 0
ATTACKED_EPISODES = 0
GPU_JOBS = 0
```

## 5. Static gate

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

Report passed, failed, and skipped test counts separately. A torch-dependent test skipped because the wrong Python environment was used is a HOLD, not a PASS.

## 6. Rebuild the repository-owned R4 binding

Use the frozen collection:

`/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_event_v2_f5b2b2d1_20260711/clean_collection`

Required frozen hashes:

```text
collection report = c8a361bc0785669bad7263deaf024103efd3e7ae62d2178247c6ab9e3b5c2843
collection input manifest = cf23edcbdb6506927dd2ec772d2bcb07c146f0851e9d55a61916c017e685849e
model binding report = c159456b297720796c7fd0fa04c0ba3f6e77870f84947962f9efc4da07b80cf3
```

Use the full paths of the new PASS reports and the three preserved old HOLD artifacts. Write the new binding outside the repository and outside the frozen collection.

```bash
python tools/multisuite_detector/bind_c2g_r4_dual_head_provenance.py build \
  --collection-root "$COLLECTION_ROOT" \
  --collection-report "$COLLECTION_ROOT/c2g_clean_collection_report.json" \
  --collection-binding-report "$COLLECTION_BINDING_REPORT" \
  --canonical-audit "$NEW_CANONICAL_PASS_REPORT" \
  --goal-event-audit "$NEW_GOAL_EVENT_PASS_REPORT" \
  --label-builder tools/multisuite_detector/c2g_clean_window_label_builder.py \
  --collection-head f5b2b2d14cdcd3359f8e3a1afa39a976df98ccc0 \
  --audit-head "$EXECUTED_HEAD" \
  --previous-canonical-hold "$OLD_CANONICAL_HOLD_REPORT" \
  --previous-goal-event-hold "$OLD_GOAL_EVENT_HOLD_REPORT" \
  --previous-hold-binding "$OLD_HOLD_BINDING" \
  --output-report "$R4_PROVENANCE_BINDING"

python tools/multisuite_detector/bind_c2g_r4_dual_head_provenance.py verify \
  --binding "$R4_PROVENANCE_BINDING" \
  --collection-root "$COLLECTION_ROOT" \
  --expected-audit-head "$EXECUTED_HEAD"
```

Required status:

`PASS_C2G_R4_DUAL_HEAD_PROVENANCE_BINDING`

## 7. R5 preview

Choose a new, external, nonexistent output directory. Do not use the previous audit directory as the materialization output.

```bash
export WORK_ROOT=/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_event_v2_f5b2b2d1_20260711
export COLLECTION_ROOT="$WORK_ROOT/clean_collection"
export R4_PROVENANCE_BINDING=/ABSOLUTE/PASS/c2g_r4_dual_head_provenance_binding.json
export GOAL_MODEL_MANIFEST=/ABSOLUTE/AUDITED/goal_manifest_v2.json
export R5_OUTPUT_ROOT=/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r5_preview_${EXECUTED_HEAD:0:8}_20260711/dataset
export DEVICE=cuda:0
export MAX_EPISODES_PER_SUITE=1

bash scripts/stageb/run_c2g_r5_bound_materialization.sh preview
```

The preview may hash model files. It must not load model tensors or write a dataset.

Required preview status:

`PASS_C2G_R5_BOUND_MATERIALIZATION_DRY_RUN`

Required invariants:

- current four-suite model verification is PASS;
- R4 binding is PASS and binds the actual audit head;
- collection bytes remain unchanged;
- output directory remains absent or empty;
- command contains the verified collection, suite map, R4 binding, and actual audit head;
- free bytes exceed 15 GiB;
- all runtime/resource counters remain zero.

## 8. Stop conditions

Return HOLD immediately for:

- source collection mutation;
- R4 binding hash/status mismatch;
- current model-byte mismatch;
- audit-head mismatch;
- nonempty output directory;
- less than 15 GiB free space;
- skipped detector tests caused by missing torch;
- any attempted model load, rollout, materialization, training, or attack;
- any request to weaken scientific gates.

## 9. Required response

```text
REMOTE_HEAD =
EXECUTED_HEAD =
WORKTREE_CLEAN =
TESTS_PASSED =
TESTS_FAILED =
TESTS_SKIPPED =
R4_BINDING_STATUS =
R4_BINDING_PATH =
R4_BINDING_SHA256 =
R5_PREVIEW_STATUS =
R5_PREVIEW_COMMAND =
COLLECTION_UNCHANGED =
MODEL_BYTES_VERIFIED =
OUTPUT_DIRECTORY_EMPTY =
FREE_BYTES_BEFORE =
FREE_BYTES_AFTER =
NEW_CLEAN_EPISODES = 0
OPENVLA_MODEL_LOADS = 0
LIBERO_ENVIRONMENTS = 0
DATASETS_MATERIALIZED = 0
TRAINING_EPOCHS = 0
ATTACKED_EPISODES = 0
GPU_JOBS = 0
P0_FINDINGS =
P1_FINDINGS =
SCIENTIFIC_CONTRACT_CHANGES = NONE
D7_TABLE1 = STILL_FROZEN
GO_HOLD_NEXT_STAGE = GO_R5_MATERIALIZATION_RUN_REVIEW | HOLD_<REASON>
```

Do not run materialization until a later explicit authorization is issued after review of this preview.
