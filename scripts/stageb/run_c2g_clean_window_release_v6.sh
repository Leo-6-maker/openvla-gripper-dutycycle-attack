#!/usr/bin/env bash
set -euo pipefail

# Canonical C2g release-v6 entry point.
#
# Adds full model-byte verification before clean collection, visual embedding
# materialization, detector-only timing, and online matched VIS-PGD execution.

PHASE="${1:-}"
if [[ -z "$PHASE" ]]; then
  echo "usage: $0 <models|manifests|collect|audit|materialize|dataset_audit|train|calibrate|folds|clean_timing|bind_parents|build_jobs|run_jobs|audit_jobs|analyze|all>" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
HEAD_SHA="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing C2g release-v6 pipeline from dirty worktree" >&2
  git status --short >&2
  exit 2
fi

: "${WORK_ROOT:?set WORK_ROOT to an external output directory}"
: "${GOAL_MODEL_MANIFEST:?set GOAL_MODEL_MANIFEST to the audited Goal model manifest}"
WORK_ROOT="$(python -c 'from pathlib import Path; import os; print(Path(os.environ["WORK_ROOT"]).resolve())')"
GOAL_MODEL_MANIFEST="$(python -c 'from pathlib import Path; import os; print(Path(os.environ["GOAL_MODEL_MANIFEST"]).resolve())')"
export WORK_ROOT GOAL_MODEL_MANIFEST

CONFIG_ROOT="$WORK_ROOT/config"
SUITE_MODEL_MAP="$CONFIG_ROOT/c2g_suite_model_map.json"
SUITE_MODEL_REPORT="$CONFIG_ROOT/c2g_suite_model_map_report.json"
MODEL_VERIFICATION_REPORT="$CONFIG_ROOT/c2g_suite_model_verification_report.json"
TRAIN_MANIFEST="${TRAIN_EPISODE_MANIFEST:-$WORK_ROOT/manifests/c2g_train_clean_parents.jsonl}"
COLLECTION_ROOT="$WORK_ROOT/clean_collection"

require_file() {
  [[ -f "$1" ]] || { echo "required file missing: $1" >&2; exit 2; }
}

verify_models() {
  require_file "$SUITE_MODEL_MAP"
  require_file "$SUITE_MODEL_REPORT"
  require_file "$GOAL_MODEL_MANIFEST"
  python scripts/stageb/verify_c2g_suite_model_map_strict.py \
    --model-map "$SUITE_MODEL_MAP" \
    --model-report "$SUITE_MODEL_REPORT" \
    --goal-model-manifest "$GOAL_MODEL_MANIFEST" \
    --output-report "$MODEL_VERIFICATION_REPORT"
}

phase_collect_release() {
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
    --device "${DEVICE:-cuda:0}" \
    --max-episodes "${MAX_TRAIN_EPISODES:-0}"
}

delegate_v5() {
  bash scripts/stageb/run_c2g_clean_window_release_v5.sh "$1"
}

case "$PHASE" in
  models|manifests|audit|dataset_audit|train|calibrate|folds|bind_parents|build_jobs|audit_jobs|analyze)
    delegate_v5 "$PHASE"
    ;;
  collect)
    phase_collect_release
    ;;
  materialize|clean_timing|run_jobs)
    verify_models
    delegate_v5 "$PHASE"
    ;;
  all)
    delegate_v5 models
    delegate_v5 manifests
    phase_collect_release
    for stage in audit materialize dataset_audit train calibrate folds clean_timing bind_parents build_jobs run_jobs audit_jobs analyze
    do
      echo "=== C2g release-v6 phase: $stage ==="
      if [[ "$stage" == "materialize" || "$stage" == "clean_timing" || "$stage" == "run_jobs" ]]; then
        verify_models
      fi
      delegate_v5 "$stage"
    done
    ;;
  *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac
