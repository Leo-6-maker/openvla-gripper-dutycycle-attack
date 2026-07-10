#!/usr/bin/env bash
set -euo pipefail

# Canonical C2g clean-window Detector-v2 pipeline.
#
# This script directly wires every strict release component without delegating
# through the historical release-v2...v8 compatibility wrappers. It is the only
# entry point that new server validation and production runs should use.
#
# Required:
#   WORK_ROOT=/absolute/external/output/root
#   GOAL_MODEL_MANIFEST=/absolute/audited/goal_model_manifest.json
#
# Expensive phases remain authorization-gated operationally. `all` is executable
# for reproducibility but must not be launched without explicit approval.

PHASE="${1:-}"
if [[ -z "$PHASE" ]]; then
  echo "usage: $0 <models|manifests|collect|audit|materialize|dataset_audit|train|calibrate|folds|clean_timing|bind_parents|build_jobs|run_jobs|audit_jobs|analyze|all>" >&2
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
: "${GOAL_MODEL_MANIFEST:?set GOAL_MODEL_MANIFEST to the audited Goal model manifest}"
WORK_ROOT="$(python -c 'from pathlib import Path; import os; print(Path(os.environ["WORK_ROOT"]).resolve())')"
GOAL_MODEL_MANIFEST="$(python -c 'from pathlib import Path; import os; print(Path(os.environ["GOAL_MODEL_MANIFEST"]).resolve())')"
REPO_RESOLVED="$(python -c 'from pathlib import Path; print(Path(".").resolve())')"
case "$WORK_ROOT/" in
  "$REPO_RESOLVED/"*) echo "WORK_ROOT must be outside the repository" >&2; exit 2 ;;
esac
[[ -f "$GOAL_MODEL_MANIFEST" ]] || { echo "Goal model manifest missing: $GOAL_MODEL_MANIFEST" >&2; exit 2; }
mkdir -p "$WORK_ROOT"

DEVICE="${DEVICE:-cuda:0}"
WINDOW="${WINDOW:-16}"
BURST_LENGTH="${BURST_LENGTH:-10}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-128}"
HIDDEN="${HIDDEN:-128}"
MAX_TRAIN_EPISODES="${MAX_TRAIN_EPISODES:-0}"
MAX_EVAL_JOBS="${MAX_EVAL_JOBS:-0}"
SUSCEPTIBILITY_POSITIVE_RETENTION="${SUSCEPTIBILITY_POSITIVE_RETENTION:-0.80}"
TRAIN_STATES_PER_TASK="${TRAIN_STATES_PER_TASK:-40}"
EVAL_STATES_PER_TASK="${EVAL_STATES_PER_TASK:-10}"
MAX_TASKS_PER_SUITE="${MAX_TASKS_PER_SUITE:-0}"
MAX_STEPS="${MAX_STEPS:-300}"
PARENT_SELECTION_SEED="${PARENT_SELECTION_SEED:-42}"
MASTER_ATTACK_SEED="${MASTER_ATTACK_SEED:-42}"

CONFIG_ROOT="$WORK_ROOT/config"
MANIFEST_ROOT="$WORK_ROOT/manifests"
COLLECTION_ROOT="$WORK_ROOT/clean_collection"
DRY_AUDIT_ROOT="$WORK_ROOT/clean_dry_audit"
DATASET_ROOT="$WORK_ROOT/dataset"
TRAIN_ROOT="$WORK_ROOT/training"
FOLD_ROOT="$WORK_ROOT/folds"
ONLINE_ROOT="$WORK_ROOT/online"

SUITE_MODEL_MAP="$CONFIG_ROOT/c2g_suite_model_map.json"
SUITE_MODEL_REPORT="$CONFIG_ROOT/c2g_suite_model_map_report.json"
MODEL_VERIFICATION_REPORT="$CONFIG_ROOT/c2g_suite_model_verification_report.json"
COLLECTION_BINDING_REPORT="$CONFIG_ROOT/c2g_clean_collection_model_binding_report.json"
TRAIN_MANIFEST="${TRAIN_EPISODE_MANIFEST:-$MANIFEST_ROOT/c2g_train_clean_parents.jsonl}"
EVAL_MANIFEST="${EVAL_PARENT_MANIFEST:-$MANIFEST_ROOT/c2g_eval_preregistered_parents.jsonl}"
DATASET_PATH="$DATASET_ROOT/c2g_clean_window_w$(printf '%02d' "$WINDOW")_openvla_siglip_within_task.npz"
DATASET_AUDIT="$DATASET_ROOT/c2g_clean_window_dataset_trainability.json"
CHECKPOINT_PATH="$TRAIN_ROOT/c2g_clean_window_detector.pt"
TRAIN_REPORT="$TRAIN_ROOT/c2g_clean_window_training_report.json"
SUSCEPTIBILITY_REPORT="$TRAIN_ROOT/c2g_clean_susceptibility_report.json"
TIMING_MANIFEST="$WORK_ROOT/detector_timing.jsonl"
BOUND_PARENTS="$WORK_ROOT/eval_parents_bound.jsonl"
JOB_MANIFEST="$WORK_ROOT/c2g_matched_load_jobs.jsonl"
EXCLUDED_LEDGER="$JOB_MANIFEST.excluded.jsonl"
JOB_BUILD_REPORT="$JOB_MANIFEST.report.json"
RUN_AUDIT="$WORK_ROOT/c2g_matched_load_run_audit.json"
RESULT_ANALYSIS="$WORK_ROOT/c2g_matched_load_result_analysis.json"

require_file() {
  [[ -f "$1" ]] || { echo "required file missing: $1" >&2; exit 2; }
}

verify_models() {
  require_file "$SUITE_MODEL_MAP"
  require_file "$SUITE_MODEL_REPORT"
  python scripts/stageb/verify_c2g_suite_model_map_strict.py \
    --model-map "$SUITE_MODEL_MAP" \
    --model-report "$SUITE_MODEL_REPORT" \
    --goal-model-manifest "$GOAL_MODEL_MANIFEST" \
    --output-report "$MODEL_VERIFICATION_REPORT"
}

phase_models() {
  mkdir -p "$CONFIG_ROOT"
  python scripts/stageb/build_c2g_suite_model_map_strict.py \
    --output-map "$SUITE_MODEL_MAP" \
    --output-report "$SUITE_MODEL_REPORT" \
    --goal-model-manifest "$GOAL_MODEL_MANIFEST"
  verify_models
}

phase_manifests() {
  mkdir -p "$MANIFEST_ROOT"
  python scripts/stageb/build_c2g_clean_manifests_release.py \
    --output-dir "$MANIFEST_ROOT" \
    --train-states-per-task "$TRAIN_STATES_PER_TASK" \
    --eval-states-per-task "$EVAL_STATES_PER_TASK" \
    --max-tasks-per-suite "$MAX_TASKS_PER_SUITE" \
    --max-steps "$MAX_STEPS" \
    --seed "$PARENT_SELECTION_SEED"
}

phase_collect() {
  verify_models
  require_file "$TRAIN_MANIFEST"
  python scripts/stageb/collect_c2g_clean_window_rollouts_release.py \
    --suite-model-map "$SUITE_MODEL_MAP" \
    --suite-model-report "$SUITE_MODEL_REPORT" \
    --goal-model-manifest "$GOAL_MODEL_MANIFEST" \
    --model-verification-report "$MODEL_VERIFICATION_REPORT" \
    --manifest "$TRAIN_MANIFEST" \
    --output-root "$COLLECTION_ROOT" \
    --expected-git-commit "$HEAD_SHA" \
    --device "$DEVICE" \
    --max-episodes "$MAX_TRAIN_EPISODES"
  python scripts/stageb/bind_c2g_collection_model_provenance.py \
    --collection-root "$COLLECTION_ROOT" \
    --model-map "$SUITE_MODEL_MAP" \
    --model-report "$SUITE_MODEL_REPORT" \
    --goal-model-manifest "$GOAL_MODEL_MANIFEST" \
    --model-verification-report "$MODEL_VERIFICATION_REPORT" \
    --output-report "$COLLECTION_BINDING_REPORT"
}

phase_audit() {
  require_file "$COLLECTION_BINDING_REPORT"
  python tools/multisuite_detector/audit_c2g_clean_window_v2.py \
    --input-root "$COLLECTION_ROOT" \
    --output-dir "$DRY_AUDIT_ROOT" \
    --repo-root "$REPO_ROOT" \
    --episodes-per-suite 2 \
    --burst-length "$BURST_LENGTH" \
    --strict-four-suites
}

phase_materialize() {
  verify_models
  require_file "$COLLECTION_BINDING_REPORT"
  python tools/multisuite_detector/materialize_c2g_multisuite_dataset.py \
    --input-root "$COLLECTION_ROOT" \
    --output-dir "$DATASET_ROOT" \
    --suite-model-map "$SUITE_MODEL_MAP" \
    --window "$WINDOW" \
    --burst-length "$BURST_LENGTH" \
    --backend openvla_siglip \
    --device "$DEVICE" \
    --max-episodes-per-suite "$MAX_TRAIN_EPISODES" \
    --split-mode within_task \
    --git-commit "$HEAD_SHA"
}

phase_dataset_audit() {
  require_file "$DATASET_PATH"
  python tools/multisuite_detector/validate_c2g_clean_window_dataset.py \
    --dataset "$DATASET_PATH" \
    --report "$DATASET_AUDIT" \
    --persistence-window 3 \
    --persistence-required 2 \
    --require-test-support
}

phase_train() {
  require_file "$DATASET_PATH"
  require_file "$DATASET_AUDIT"
  python - "$DATASET_AUDIT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], "r", encoding="utf-8"))
if report.get("status") != "PASS_C2G_DATASET_TRAINABILITY":
    raise SystemExit("dataset trainability report is not PASS")
PY
  python tools/multisuite_detector/train_c2g_clean_window_detector.py \
    --dataset "$DATASET_PATH" \
    --output-dir "$TRAIN_ROOT" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --hidden "$HIDDEN" \
    --git-commit "$HEAD_SHA"
}

phase_calibrate() {
  require_file "$DATASET_PATH"
  require_file "$CHECKPOINT_PATH"
  require_file "$TRAIN_REPORT"
  python tools/multisuite_detector/calibrate_c2g_clean_susceptibility.py \
    --dataset "$DATASET_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --training-report "$TRAIN_REPORT" \
    --output-report "$SUSCEPTIBILITY_REPORT" \
    --split val \
    --positive-retention "$SUSCEPTIBILITY_POSITIVE_RETENTION" \
    --require-clean-close
}

phase_folds() {
  require_file "$DATASET_PATH"
  require_file "$DATASET_AUDIT"
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
  verify_models
  require_file "$EVAL_MANIFEST"
  require_file "$CHECKPOINT_PATH"
  require_file "$SUSCEPTIBILITY_REPORT"
  python scripts/stageb/run_c2g_clean_timing_jobs_strict.py \
    --parents "$EVAL_MANIFEST" \
    --suite-model-map "$SUITE_MODEL_MAP" \
    --goal-model-manifest "$GOAL_MODEL_MANIFEST" \
    --checkpoint "$CHECKPOINT_PATH" \
    --output-root "$ONLINE_ROOT" \
    --expected-git-commit "$HEAD_SHA" \
    --device "$DEVICE" \
    --burst-length "$BURST_LENGTH" \
    --max-jobs "$MAX_EVAL_JOBS" \
    --resume
  python scripts/stageb/extract_c2g_detector_timing.py \
    --clean-output-root "$ONLINE_ROOT" \
    --output "$TIMING_MANIFEST" \
    --no-require-trigger
}

phase_bind_parents() {
  require_file "$EVAL_MANIFEST"
  require_file "$TIMING_MANIFEST"
  python scripts/stageb/prepare_c2g_eval_parents.py \
    --parents "$EVAL_MANIFEST" \
    --clean-output-root "$ONLINE_ROOT" \
    --output "$BOUND_PARENTS" \
    --expected-git-commit "$HEAD_SHA"
}

phase_build_jobs() {
  require_file "$BOUND_PARENTS"
  require_file "$TIMING_MANIFEST"
  require_file "$CHECKPOINT_PATH"
  require_file "$TRAIN_REPORT"
  python scripts/stageb/build_c2g_matched_load_jobs_release.py \
    --parents "$BOUND_PARENTS" \
    --detector-timing "$TIMING_MANIFEST" \
    --checkpoint "$CHECKPOINT_PATH" \
    --detector-config "$TRAIN_REPORT" \
    --output "$JOB_MANIFEST" \
    --master-seed "$MASTER_ATTACK_SEED" \
    --burst-length "$BURST_LENGTH" \
    --control-objective SHUFFLED_GRIPPER_GRADIENT
}

phase_run_jobs() {
  verify_models
  require_file "$JOB_MANIFEST"
  require_file "$CHECKPOINT_PATH"
  python scripts/stageb/run_c2g_matched_load_jobs_map_release.py \
    --jobs "$JOB_MANIFEST" \
    --suite-model-map "$SUITE_MODEL_MAP" \
    --goal-model-manifest "$GOAL_MODEL_MANIFEST" \
    --output-root "$ONLINE_ROOT" \
    --checkpoint "$CHECKPOINT_PATH" \
    --expected-git-commit "$HEAD_SHA" \
    --device "$DEVICE" \
    --max-jobs "$MAX_EVAL_JOBS" \
    --resume
}

phase_audit_jobs() {
  require_file "$JOB_MANIFEST"
  require_file "$EXCLUDED_LEDGER"
  python scripts/stageb/audit_c2g_matched_load_run_release.py \
    --jobs "$JOB_MANIFEST" \
    --output-root "$ONLINE_ROOT" \
    --excluded-ledger "$EXCLUDED_LEDGER" \
    --report "$RUN_AUDIT"
}

phase_analyze() {
  require_file "$RUN_AUDIT"
  require_file "$JOB_BUILD_REPORT"
  python scripts/stageb/analyze_c2g_matched_load_results.py \
    --audit-report "$RUN_AUDIT" \
    --job-build-report "$JOB_BUILD_REPORT" \
    --output "$RESULT_ANALYSIS"
}

run_phase() {
  case "$1" in
    models) phase_models ;;
    manifests) phase_manifests ;;
    collect) phase_collect ;;
    audit) phase_audit ;;
    materialize) phase_materialize ;;
    dataset_audit) phase_dataset_audit ;;
    train) phase_train ;;
    calibrate) phase_calibrate ;;
    folds) phase_folds ;;
    clean_timing) phase_clean_timing ;;
    bind_parents) phase_bind_parents ;;
    build_jobs) phase_build_jobs ;;
    run_jobs) phase_run_jobs ;;
    audit_jobs) phase_audit_jobs ;;
    analyze) phase_analyze ;;
    *) echo "unknown phase: $1" >&2; exit 2 ;;
  esac
}

if [[ "$PHASE" == "all" ]]; then
  for stage in models manifests collect audit materialize dataset_audit train calibrate folds clean_timing bind_parents build_jobs run_jobs audit_jobs analyze
  do
    echo "=== C2g canonical phase: $stage ==="
    run_phase "$stage"
  done
else
  run_phase "$PHASE"
fi
