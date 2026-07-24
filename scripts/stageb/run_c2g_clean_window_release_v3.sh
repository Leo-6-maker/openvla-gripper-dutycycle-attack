#!/usr/bin/env bash
set -euo pipefail

# Final release wrapper: identical to release-v2 except that model provenance is
# bound to every referenced weight shard rather than only lightweight metadata.

PHASE="${1:-}"
if [[ -z "$PHASE" ]]; then
  echo "usage: $0 <models|manifests|collect|audit|materialize|dataset_audit|train|calibrate|folds|clean_timing|bind_parents|build_jobs|run_jobs|audit_jobs|analyze|all>" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing C2g release-v3 pipeline from dirty worktree" >&2
  git status --short >&2
  exit 2
fi

: "${WORK_ROOT:?set WORK_ROOT to an external output directory}"
: "${GOAL_MODEL_MANIFEST:?set GOAL_MODEL_MANIFEST to the audited Goal model manifest}"
WORK_ROOT="$(python -c 'from pathlib import Path; import os; print(Path(os.environ["WORK_ROOT"]).resolve())')"
GOAL_MODEL_MANIFEST="$(python -c 'from pathlib import Path; import os; print(Path(os.environ["GOAL_MODEL_MANIFEST"]).resolve())')"
export WORK_ROOT GOAL_MODEL_MANIFEST

phase_models_strict() {
  local config_root="$WORK_ROOT/config"
  mkdir -p "$config_root"
  python scripts/stageb/build_c2g_suite_model_map_strict.py \
    --output-map "$config_root/c2g_suite_model_map.json" \
    --output-report "$config_root/c2g_suite_model_map_report.json" \
    --goal-model-manifest "$GOAL_MODEL_MANIFEST"
}

case "$PHASE" in
  models)
    phase_models_strict
    ;;
  all)
    phase_models_strict
    for stage in manifests collect audit materialize dataset_audit train calibrate folds clean_timing bind_parents build_jobs run_jobs audit_jobs analyze
    do
      echo "=== C2g release-v3 phase: $stage ==="
      bash scripts/stageb/run_c2g_clean_window_release_v2.sh "$stage"
    done
    ;;
  manifests|collect|audit|materialize|dataset_audit|train|calibrate|folds|clean_timing|bind_parents|build_jobs|run_jobs|audit_jobs|analyze)
    bash scripts/stageb/run_c2g_clean_window_release_v2.sh "$PHASE"
    ;;
  *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac
