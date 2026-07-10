#!/usr/bin/env bash
set -euo pipefail

# Canonical C2g release-v7 entry point.
# Adds an audit that allows only ledger-bound CLEAN artifacts from detector no-emit
# or burst-infeasible parents while preserving strict closure for attacked jobs.

PHASE="${1:-}"
if [[ -z "$PHASE" ]]; then
  echo "usage: $0 <models|manifests|collect|audit|materialize|dataset_audit|train|calibrate|folds|clean_timing|bind_parents|build_jobs|run_jobs|audit_jobs|analyze|all>" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing C2g release-v7 pipeline from dirty worktree" >&2
  git status --short >&2
  exit 2
fi

: "${WORK_ROOT:?set WORK_ROOT to an external output directory}"
WORK_ROOT="$(python -c 'from pathlib import Path; import os; print(Path(os.environ["WORK_ROOT"]).resolve())')"
export WORK_ROOT

phase_audit_release() {
  local jobs="$WORK_ROOT/c2g_matched_load_jobs.jsonl"
  local excluded="$jobs.excluded.jsonl"
  local online="$WORK_ROOT/online"
  local report="$WORK_ROOT/c2g_matched_load_run_audit.json"
  for path in "$jobs" "$excluded"; do
    [[ -f "$path" ]] || { echo "required file missing: $path" >&2; exit 2; }
  done
  python scripts/stageb/audit_c2g_matched_load_run_release.py \
    --jobs "$jobs" \
    --output-root "$online" \
    --excluded-ledger "$excluded" \
    --report "$report"
}

delegate_v6() {
  bash scripts/stageb/run_c2g_clean_window_release_v6.sh "$1"
}

case "$PHASE" in
  audit_jobs)
    phase_audit_release
    ;;
  all)
    for stage in models manifests collect audit materialize dataset_audit train calibrate folds clean_timing bind_parents build_jobs run_jobs
    do
      echo "=== C2g release-v7 phase: $stage ==="
      delegate_v6 "$stage"
    done
    phase_audit_release
    delegate_v6 analyze
    ;;
  models|manifests|collect|audit|materialize|dataset_audit|train|calibrate|folds|clean_timing|bind_parents|build_jobs|run_jobs|analyze)
    delegate_v6 "$PHASE"
    ;;
  *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac
