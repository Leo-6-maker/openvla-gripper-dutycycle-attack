#!/usr/bin/env bash
set -euo pipefail

# Full C2g clean-window pipeline entry point.
#
# This wrapper closes the only input gap left by the staged orchestrator: it can
# deterministically preregister disjoint training/evaluation parent manifests and
# then delegates each reviewed phase to run_c2g_clean_window_end_to_end.sh.
# Expensive execution remains phase-gated; the `all` mode must not be launched
# without explicit server authorization.

PHASE="${1:-}"
if [[ -z "$PHASE" ]]; then
  echo "usage: $0 <manifests|collect|audit|materialize|dataset_audit|train|calibrate|folds|clean_timing|bind_parents|build_jobs|run_jobs|audit_jobs|analyze|all>" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing C2g full pipeline from dirty worktree" >&2
  git status --short >&2
  exit 2
fi

: "${WORK_ROOT:?set WORK_ROOT to an external output directory}"
WORK_ROOT="$(python -c 'from pathlib import Path; import os; print(Path(os.environ["WORK_ROOT"]).resolve())')"
REPO_RESOLVED="$(python -c 'from pathlib import Path; print(Path(".").resolve())')"
case "$WORK_ROOT/" in
  "$REPO_RESOLVED/"*) echo "WORK_ROOT must be outside the repository" >&2; exit 2 ;;
esac

MANIFEST_ROOT="${MANIFEST_ROOT:-$WORK_ROOT/manifests}"
GENERATED_TRAIN_MANIFEST="$MANIFEST_ROOT/c2g_train_clean_parents.jsonl"
GENERATED_EVAL_MANIFEST="$MANIFEST_ROOT/c2g_eval_preregistered_parents.jsonl"
TRAIN_STATES_PER_TASK="${TRAIN_STATES_PER_TASK:-40}"
EVAL_STATES_PER_TASK="${EVAL_STATES_PER_TASK:-10}"
MAX_TASKS_PER_SUITE="${MAX_TASKS_PER_SUITE:-0}"
MAX_STEPS="${MAX_STEPS:-300}"
PARENT_SELECTION_SEED="${PARENT_SELECTION_SEED:-42}"

phase_manifests() {
  mkdir -p "$MANIFEST_ROOT"
  python scripts/stageb/build_c2g_clean_manifests.py \
    --output-dir "$MANIFEST_ROOT" \
    --train-states-per-task "$TRAIN_STATES_PER_TASK" \
    --eval-states-per-task "$EVAL_STATES_PER_TASK" \
    --max-tasks-per-suite "$MAX_TASKS_PER_SUITE" \
    --max-steps "$MAX_STEPS" \
    --seed "$PARENT_SELECTION_SEED"
}

ensure_manifests() {
  if [[ -z "${TRAIN_EPISODE_MANIFEST:-}" ]]; then
    export TRAIN_EPISODE_MANIFEST="$GENERATED_TRAIN_MANIFEST"
  fi
  if [[ -z "${EVAL_PARENT_MANIFEST:-}" ]]; then
    export EVAL_PARENT_MANIFEST="$GENERATED_EVAL_MANIFEST"
  fi
  [[ -f "$TRAIN_EPISODE_MANIFEST" ]] || {
    echo "training manifest missing: $TRAIN_EPISODE_MANIFEST; run manifests phase first" >&2
    exit 2
  }
  [[ -f "$EVAL_PARENT_MANIFEST" ]] || {
    echo "evaluation manifest missing: $EVAL_PARENT_MANIFEST; run manifests phase first" >&2
    exit 2
  }
}

delegate() {
  ensure_manifests
  bash scripts/stageb/run_c2g_clean_window_end_to_end.sh "$1"
}

case "$PHASE" in
  manifests)
    phase_manifests
    ;;
  collect|audit|materialize|dataset_audit|train|calibrate|folds|clean_timing|bind_parents|build_jobs|run_jobs|audit_jobs|analyze)
    delegate "$PHASE"
    ;;
  all)
    phase_manifests
    export TRAIN_EPISODE_MANIFEST="$GENERATED_TRAIN_MANIFEST"
    export EVAL_PARENT_MANIFEST="$GENERATED_EVAL_MANIFEST"
    for stage in \
      collect audit materialize dataset_audit train calibrate folds \
      clean_timing bind_parents build_jobs run_jobs audit_jobs analyze
    do
      echo "=== C2g phase: $stage ==="
      bash scripts/stageb/run_c2g_clean_window_end_to_end.sh "$stage"
    done
    ;;
  *)
    echo "unknown phase: $PHASE" >&2
    exit 2
    ;;
esac
