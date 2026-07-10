#!/usr/bin/env bash
set -euo pipefail

# Final C2g release entry point.
# - full weight-shard suite model binding;
# - runtime-compatible five-part clean parent keys;
# - strict collection, training, clean timing, matched VIS-PGD, audit, analysis.

PHASE="${1:-}"
if [[ -z "$PHASE" ]]; then
  echo "usage: $0 <models|manifests|collect|audit|materialize|dataset_audit|train|calibrate|folds|clean_timing|bind_parents|build_jobs|run_jobs|audit_jobs|analyze|all>" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing C2g release-v4 pipeline from dirty worktree" >&2
  git status --short >&2
  exit 2
fi

: "${WORK_ROOT:?set WORK_ROOT to an external output directory}"
WORK_ROOT="$(python -c 'from pathlib import Path; import os; print(Path(os.environ["WORK_ROOT"]).resolve())')"
export WORK_ROOT

phase_models() {
  bash scripts/stageb/run_c2g_clean_window_release_v3.sh models
}

phase_manifests() {
  local manifest_root="$WORK_ROOT/manifests"
  mkdir -p "$manifest_root"
  python scripts/stageb/build_c2g_clean_manifests_release.py \
    --output-dir "$manifest_root" \
    --train-states-per-task "${TRAIN_STATES_PER_TASK:-40}" \
    --eval-states-per-task "${EVAL_STATES_PER_TASK:-10}" \
    --max-tasks-per-suite "${MAX_TASKS_PER_SUITE:-0}" \
    --max-steps "${MAX_STEPS:-300}" \
    --seed "${PARENT_SELECTION_SEED:-42}"
}

case "$PHASE" in
  models) phase_models ;;
  manifests) phase_manifests ;;
  all)
    phase_models
    phase_manifests
    for stage in collect audit materialize dataset_audit train calibrate folds clean_timing bind_parents build_jobs run_jobs audit_jobs analyze
    do
      echo "=== C2g release-v4 phase: $stage ==="
      bash scripts/stageb/run_c2g_clean_window_release_v2.sh "$stage"
    done
    ;;
  collect|audit|materialize|dataset_audit|train|calibrate|folds|clean_timing|bind_parents|build_jobs|run_jobs|audit_jobs|analyze)
    bash scripts/stageb/run_c2g_clean_window_release_v2.sh "$PHASE"
    ;;
  *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac
