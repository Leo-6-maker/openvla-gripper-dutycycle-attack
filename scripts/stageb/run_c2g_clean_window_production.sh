#!/usr/bin/env bash
set -euo pipefail

# Strict production entry point for the C2g clean-window pipeline.
#
# It differs from the generic full wrapper only for collection: the strict
# collector enforces the frozen canonical 25D feature order before any dataset is
# written. All later phases delegate to the reviewed end-to-end orchestrator.

PHASE="${1:-}"
if [[ -z "$PHASE" ]]; then
  echo "usage: $0 <manifests|collect|audit|materialize|dataset_audit|train|calibrate|folds|clean_timing|bind_parents|build_jobs|run_jobs|audit_jobs|analyze|all>" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing C2g production pipeline from dirty worktree" >&2
  git status --short >&2
  exit 2
fi

: "${WORK_ROOT:?set WORK_ROOT to an external output directory}"
WORK_ROOT="$(python -c 'from pathlib import Path; import os; print(Path(os.environ["WORK_ROOT"]).resolve())')"
MANIFEST_ROOT="${MANIFEST_ROOT:-$WORK_ROOT/manifests}"
TRAIN_EPISODE_MANIFEST="${TRAIN_EPISODE_MANIFEST:-$MANIFEST_ROOT/c2g_train_clean_parents.jsonl}"
EVAL_PARENT_MANIFEST="${EVAL_PARENT_MANIFEST:-$MANIFEST_ROOT/c2g_eval_preregistered_parents.jsonl}"
export WORK_ROOT TRAIN_EPISODE_MANIFEST EVAL_PARENT_MANIFEST

phase_manifests() {
  bash scripts/stageb/run_c2g_clean_window_full.sh manifests
}

phase_collect_strict() {
  [[ -f "$TRAIN_EPISODE_MANIFEST" ]] || {
    echo "training manifest missing: $TRAIN_EPISODE_MANIFEST" >&2
    exit 2
  }
  HEAD_SHA="$(git rev-parse HEAD)"
  DEVICE="${DEVICE:-cuda:0}"
  MAX_TRAIN_EPISODES="${MAX_TRAIN_EPISODES:-0}"
  python scripts/stageb/collect_c2g_clean_window_rollouts_strict.py \
    --manifest "$TRAIN_EPISODE_MANIFEST" \
    --output-root "$WORK_ROOT/clean_collection" \
    --expected-git-commit "$HEAD_SHA" \
    --device "$DEVICE" \
    --max-episodes "$MAX_TRAIN_EPISODES"
}

delegate() {
  bash scripts/stageb/run_c2g_clean_window_end_to_end.sh "$1"
}

case "$PHASE" in
  manifests) phase_manifests ;;
  collect) phase_collect_strict ;;
  audit|materialize|dataset_audit|train|calibrate|folds|clean_timing|bind_parents|build_jobs|run_jobs|audit_jobs|analyze)
    delegate "$PHASE"
    ;;
  all)
    phase_manifests
    phase_collect_strict
    for stage in \
      audit materialize dataset_audit train calibrate folds \
      clean_timing bind_parents build_jobs run_jobs audit_jobs analyze
    do
      echo "=== C2g strict phase: $stage ==="
      delegate "$stage"
    done
    ;;
  *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac
