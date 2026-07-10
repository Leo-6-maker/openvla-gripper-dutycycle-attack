#!/usr/bin/env bash
set -euo pipefail

# End-to-end C2g clean-window pipeline orchestrator.
#
# Required environment:
#   TRAIN_EPISODE_MANIFEST   clean training rollout manifest
#   EVAL_PARENT_MANIFEST     preregistered online evaluation parents
#   WORK_ROOT                external output root (must be outside repository)
#
# Four-suite OpenVLA materialization:
#   SUITE_MODEL_MAP=/absolute/suite_model_map.json
# The JSON object must map all four LIBERO suite names to their exact policy model
# directories. OPENVLA_MODEL_PATH is retained only for a single-model diagnostic.

PHASE="${1:-}"
if [[ -z "$PHASE" ]]; then
  echo "usage: $0 <collect|audit|materialize|train|folds|clean_timing|bind_parents|build_jobs|run_jobs|audit_jobs|all>" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
HEAD_SHA="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing C2g pipeline from dirty worktree" >&2
  git status --short >&2
  exit 2
fi

: "${WORK_ROOT:?set WORK_ROOT to an external output directory}"
WORK_ROOT="$(python -c 'from pathlib import Path; import os; print(Path(os.environ["WORK_ROOT"]).resolve())')"
REPO_RESOLVED="$(python -c 'from pathlib import Path; print(Path(".").resolve())')"
case "$WORK_ROOT/" in
  "$REPO_RESOLVED/"*) echo "WORK_ROOT must be outside the repository" >&2; exit 2 ;;
esac
mkdir -p "$WORK_ROOT"

DEVICE="${DEVICE:-cuda:0}"
WINDOW="${WINDOW:-16}"
BURST_LENGTH="${BURST_LENGTH:-10}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-128}"
HIDDEN="${HIDDEN:-128}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-openvla_siglip}"
OPENVLA_MODEL_PATH="${OPENVLA_MODEL_PATH:-}"
SUITE_MODEL_MAP="${SUITE_MODEL_MAP:-}"
MODEL_PATH_TEMPLATE="${MODEL_PATH_TEMPLATE:-}"
MAX_TRAIN_EPISODES="${MAX_TRAIN_EPISODES:-0}"
MAX_EVAL_JOBS="${MAX_EVAL_JOBS:-0}"
CONTROL_OBJECTIVE="${CONTROL_OBJECTIVE:-SHUFFLED_GRIPPER_GRADIENT}"

COLLECTION_ROOT="$WORK_ROOT/clean_collection"
DRY_AUDIT_ROOT="$WORK_ROOT/clean_dry_audit"
DATASET_ROOT="$WORK_ROOT/dataset"
TRAIN_ROOT="$WORK_ROOT/training"
FOLD_ROOT="$WORK_ROOT/folds"
ONLINE_ROOT="$WORK_ROOT/online"
TIMING_MANIFEST="$WORK_ROOT/detector_timing.jsonl"
BOUND_PARENTS="$WORK_ROOT/eval_parents_bound.jsonl"
JOB_MANIFEST="$WORK_ROOT/c2g_matched_load_jobs.jsonl"
RUN_AUDIT="$WORK_ROOT/c2g_matched_load_run_audit.json"
DATASET_PATH="$DATASET_ROOT/c2g_clean_window_w$(printf '%02d' "$WINDOW")_${EMBEDDING_BACKEND}_within_task.npz"
CHECKPOINT_PATH="$TRAIN_ROOT/c2g_clean_window_detector.pt"
TRAIN_CONFIG_PATH="$TRAIN_ROOT/c2g_clean_window_training_report.json"

require_file() {
  [[ -f "$1" ]] || { echo "required file missing: $1" >&2; exit 2; }
}

phase_collect() {
  : "${TRAIN_EPISODE_MANIFEST:?set TRAIN_EPISODE_MANIFEST}"
  require_file "$TRAIN_EPISODE_MANIFEST"
  python scripts/stageb/collect_c2g_clean_window_rollouts.py \
    --manifest "$TRAIN_EPISODE_MANIFEST" \
    --output-root "$COLLECTION_ROOT" \
    --expected-git-commit "$HEAD_SHA" \
    --device "$DEVICE" \
    --max-episodes "$MAX_TRAIN_EPISODES"
}

phase_audit() {
  python tools/multisuite_detector/audit_c2g_clean_window_v2.py \
    --input-root "$COLLECTION_ROOT" \
    --output-dir "$DRY_AUDIT_ROOT" \
    --repo-root "$REPO_ROOT" \
    --episodes-per-suite 2 \
    --burst-length "$BURST_LENGTH" \
    --strict-four-suites
}

phase_materialize() {
  if [[ -n "$SUITE_MODEL_MAP" ]]; then
    require_file "$SUITE_MODEL_MAP"
    python tools/multisuite_detector/materialize_c2g_multisuite_dataset.py \
      --input-root "$COLLECTION_ROOT" \
      --output-dir "$DATASET_ROOT" \
      --suite-model-map "$SUITE_MODEL_MAP" \
      --window "$WINDOW" \
      --burst-length "$BURST_LENGTH" \
      --backend "$EMBEDDING_BACKEND" \
      --device "$DEVICE" \
      --max-episodes-per-suite "$MAX_TRAIN_EPISODES" \
      --split-mode within_task \
      --git-commit "$HEAD_SHA"
    return
  fi
  local extra=()
  if [[ -n "$OPENVLA_MODEL_PATH" ]]; then
    extra+=(--openvla-model-path "$OPENVLA_MODEL_PATH")
  fi
  python tools/multisuite_detector/materialize_c2g_clean_window_dataset.py \
    --input-root "$COLLECTION_ROOT" \
    --output-dir "$DATASET_ROOT" \
    --window "$WINDOW" \
    --burst-length "$BURST_LENGTH" \
    --backend "$EMBEDDING_BACKEND" \
    --device "$DEVICE" \
    --max-episodes "$MAX_TRAIN_EPISODES" \
    --split-mode within_task \
    --git-commit "$HEAD_SHA" \
    --require-zero-errors \
    "${extra[@]}"
}

phase_train() {
  require_file "$DATASET_PATH"
  python tools/multisuite_detector/train_c2g_clean_window_detector.py \
    --dataset "$DATASET_PATH" \
    --output-dir "$TRAIN_ROOT" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --hidden "$HIDDEN" \
    --git-commit "$HEAD_SHA"
}

phase_folds() {
  require_file "$DATASET_PATH"
  python tools/multisuite_detector/run_c2g_clean_window_folds.py \
    --dataset "$DATASET_PATH" \
    --output-root "$FOLD_ROOT" \
    --mode loto \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --hidden "$HIDDEN" \
    --git-commit "$HEAD_SHA"
}

phase_clean_timing() {
  : "${EVAL_PARENT_MANIFEST:?set EVAL_PARENT_MANIFEST}"
  require_file "$EVAL_PARENT_MANIFEST"
  require_file "$CHECKPOINT_PATH"
  local extra=()
  if [[ -n "$MODEL_PATH_TEMPLATE" ]]; then
    extra+=(--model-path "$MODEL_PATH_TEMPLATE")
  fi
  python scripts/stageb/run_c2g_clean_timing_jobs.py \
    --parents "$EVAL_PARENT_MANIFEST" \
    --checkpoint "$CHECKPOINT_PATH" \
    --output-root "$ONLINE_ROOT" \
    --expected-git-commit "$HEAD_SHA" \
    --device "$DEVICE" \
    --burst-length "$BURST_LENGTH" \
    --max-jobs "$MAX_EVAL_JOBS" \
    "${extra[@]}"
  python scripts/stageb/extract_c2g_detector_timing.py \
    --clean-output-root "$ONLINE_ROOT" \
    --output "$TIMING_MANIFEST" \
    --require-trigger
}

phase_bind_parents() {
  : "${EVAL_PARENT_MANIFEST:?set EVAL_PARENT_MANIFEST}"
  require_file "$EVAL_PARENT_MANIFEST"
  python scripts/stageb/prepare_c2g_eval_parents.py \
    --parents "$EVAL_PARENT_MANIFEST" \
    --clean-output-root "$ONLINE_ROOT" \
    --output "$BOUND_PARENTS" \
    --expected-git-commit "$HEAD_SHA"
}

phase_build_jobs() {
  require_file "$BOUND_PARENTS"
  require_file "$TIMING_MANIFEST"
  require_file "$CHECKPOINT_PATH"
  require_file "$TRAIN_CONFIG_PATH"
  python scripts/stageb/build_c2g_matched_load_jobs.py \
    --parents "$BOUND_PARENTS" \
    --detector-timing "$TIMING_MANIFEST" \
    --checkpoint "$CHECKPOINT_PATH" \
    --detector-config "$TRAIN_CONFIG_PATH" \
    --output "$JOB_MANIFEST" \
    --burst-length "$BURST_LENGTH" \
    --control-objective "$CONTROL_OBJECTIVE"
}

phase_run_jobs() {
  require_file "$JOB_MANIFEST"
  local extra=()
  if [[ -n "$MODEL_PATH_TEMPLATE" ]]; then
    extra+=(--model-path "$MODEL_PATH_TEMPLATE")
  fi
  python scripts/stageb/run_c2g_matched_load_jobs.py \
    --jobs "$JOB_MANIFEST" \
    --output-root "$ONLINE_ROOT" \
    --checkpoint "$CHECKPOINT_PATH" \
    --expected-git-commit "$HEAD_SHA" \
    --device "$DEVICE" \
    --max-jobs "$MAX_EVAL_JOBS" \
    --resume \
    "${extra[@]}"
}

phase_audit_jobs() {
  require_file "$JOB_MANIFEST"
  python scripts/stageb/audit_c2g_matched_load_run.py \
    --jobs "$JOB_MANIFEST" \
    --output-root "$ONLINE_ROOT" \
    --report "$RUN_AUDIT"
}

case "$PHASE" in
  collect) phase_collect ;;
  audit) phase_audit ;;
  materialize) phase_materialize ;;
  train) phase_train ;;
  folds) phase_folds ;;
  clean_timing) phase_clean_timing ;;
  bind_parents) phase_bind_parents ;;
  build_jobs) phase_build_jobs ;;
  run_jobs) phase_run_jobs ;;
  audit_jobs) phase_audit_jobs ;;
  all)
    phase_collect
    phase_audit
    phase_materialize
    phase_train
    phase_folds
    phase_clean_timing
    phase_bind_parents
    phase_build_jobs
    phase_run_jobs
    phase_audit_jobs
    ;;
  *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac
