#!/usr/bin/env bash
set -euo pipefail

# Final release-v5 wrapper. It delegates release-v4 except for matched job
# construction, where each CLEAN row's seed is bound to the actual detector-only
# clean execution seed before the runtime audit is allowed.

PHASE="${1:-}"
if [[ -z "$PHASE" ]]; then
  echo "usage: $0 <models|manifests|collect|audit|materialize|dataset_audit|train|calibrate|folds|clean_timing|bind_parents|build_jobs|run_jobs|audit_jobs|analyze|all>" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing C2g release-v5 pipeline from dirty worktree" >&2
  git status --short >&2
  exit 2
fi

: "${WORK_ROOT:?set WORK_ROOT to an external output directory}"
WORK_ROOT="$(python -c 'from pathlib import Path; import os; print(Path(os.environ["WORK_ROOT"]).resolve())')"
export WORK_ROOT

phase_build_jobs_release() {
  local bound_parents="$WORK_ROOT/eval_parents_bound.jsonl"
  local timing_manifest="$WORK_ROOT/detector_timing.jsonl"
  local checkpoint="$WORK_ROOT/training/c2g_clean_window_detector.pt"
  local training_report="$WORK_ROOT/training/c2g_clean_window_training_report.json"
  local output="$WORK_ROOT/c2g_matched_load_jobs.jsonl"
  for path in "$bound_parents" "$timing_manifest" "$checkpoint" "$training_report"; do
    [[ -f "$path" ]] || { echo "required file missing: $path" >&2; exit 2; }
  done
  python scripts/stageb/build_c2g_matched_load_jobs_release.py \
    --parents "$bound_parents" \
    --detector-timing "$timing_manifest" \
    --checkpoint "$checkpoint" \
    --detector-config "$training_report" \
    --output "$output" \
    --burst-length "${BURST_LENGTH:-10}" \
    --control-objective SHUFFLED_GRIPPER_GRADIENT
}

case "$PHASE" in
  build_jobs)
    phase_build_jobs_release
    ;;
  all)
    for stage in models manifests collect audit materialize dataset_audit train calibrate folds clean_timing bind_parents
    do
      echo "=== C2g release-v5 phase: $stage ==="
      bash scripts/stageb/run_c2g_clean_window_release_v4.sh "$stage"
    done
    phase_build_jobs_release
    for stage in run_jobs audit_jobs analyze
    do
      echo "=== C2g release-v5 phase: $stage ==="
      bash scripts/stageb/run_c2g_clean_window_release_v4.sh "$stage"
    done
    ;;
  models|manifests|collect|audit|materialize|dataset_audit|train|calibrate|folds|clean_timing|bind_parents|run_jobs|audit_jobs|analyze)
    bash scripts/stageb/run_c2g_clean_window_release_v4.sh "$PHASE"
    ;;
  *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac
