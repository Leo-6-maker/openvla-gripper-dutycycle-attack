#!/usr/bin/env bash
set -euo pipefail

# Canonical C2g release-v8 entry point.
# After strict clean collection, every episode metadata file is atomically bound to
# the frozen full suite model manifest before Teacher audit or materialization.

PHASE="${1:-}"
if [[ -z "$PHASE" ]]; then
  echo "usage: $0 <models|manifests|collect|audit|materialize|dataset_audit|train|calibrate|folds|clean_timing|bind_parents|build_jobs|run_jobs|audit_jobs|analyze|all>" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing C2g release-v8 pipeline from dirty worktree" >&2
  git status --short >&2
  exit 2
fi

: "${WORK_ROOT:?set WORK_ROOT to an external output directory}"
: "${GOAL_MODEL_MANIFEST:?set GOAL_MODEL_MANIFEST to the audited Goal model manifest}"
WORK_ROOT="$(python -c 'from pathlib import Path; import os; print(Path(os.environ["WORK_ROOT"]).resolve())')"
GOAL_MODEL_MANIFEST="$(python -c 'from pathlib import Path; import os; print(Path(os.environ["GOAL_MODEL_MANIFEST"]).resolve())')"
export WORK_ROOT GOAL_MODEL_MANIFEST

CONFIG_ROOT="$WORK_ROOT/config"
COLLECTION_ROOT="$WORK_ROOT/clean_collection"
SUITE_MODEL_MAP="$CONFIG_ROOT/c2g_suite_model_map.json"
SUITE_MODEL_REPORT="$CONFIG_ROOT/c2g_suite_model_map_report.json"
MODEL_VERIFICATION_REPORT="$CONFIG_ROOT/c2g_suite_model_verification_report.json"
COLLECTION_BINDING_REPORT="$CONFIG_ROOT/c2g_clean_collection_model_binding_report.json"

phase_bind_collection() {
  python scripts/stageb/bind_c2g_collection_model_provenance.py \
    --collection-root "$COLLECTION_ROOT" \
    --model-map "$SUITE_MODEL_MAP" \
    --model-report "$SUITE_MODEL_REPORT" \
    --goal-model-manifest "$GOAL_MODEL_MANIFEST" \
    --model-verification-report "$MODEL_VERIFICATION_REPORT" \
    --output-report "$COLLECTION_BINDING_REPORT"
}

delegate_v7() {
  bash scripts/stageb/run_c2g_clean_window_release_v7.sh "$1"
}

case "$PHASE" in
  collect)
    delegate_v7 collect
    phase_bind_collection
    ;;
  materialize)
    [[ -f "$COLLECTION_BINDING_REPORT" ]] || {
      echo "clean collection model binding report missing: $COLLECTION_BINDING_REPORT" >&2
      exit 2
    }
    delegate_v7 materialize
    ;;
  all)
    for stage in models manifests; do delegate_v7 "$stage"; done
    delegate_v7 collect
    phase_bind_collection
    for stage in audit materialize dataset_audit train calibrate folds clean_timing bind_parents build_jobs run_jobs audit_jobs analyze
    do
      echo "=== C2g release-v8 phase: $stage ==="
      if [[ "$stage" == "materialize" && ! -f "$COLLECTION_BINDING_REPORT" ]]; then
        echo "clean collection model binding report missing" >&2
        exit 2
      fi
      delegate_v7 "$stage"
    done
    ;;
  models|manifests|audit|dataset_audit|train|calibrate|folds|clean_timing|bind_parents|build_jobs|run_jobs|audit_jobs|analyze)
    delegate_v7 "$PHASE"
    ;;
  *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac
