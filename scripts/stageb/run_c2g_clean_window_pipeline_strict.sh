#!/usr/bin/env bash
set -euo pipefail

# Strict canonical C2g pipeline wrapper.
#
# Delegates direct phases to run_c2g_clean_window_pipeline.sh, but inserts immutable
# collection/model verification before materialization and an event-tracking scientific
# audit after the canonical clean-label audit.

PHASE="${1:-}"
if [[ -z "$PHASE" ]]; then
  echo "usage: $0 <models|manifests|collect|audit|materialize|dataset_audit|train|calibrate|folds|clean_timing|bind_parents|build_jobs|run_jobs|audit_jobs|analyze|all>" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing strict C2g pipeline from dirty worktree" >&2
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
DRY_AUDIT_ROOT="$WORK_ROOT/clean_dry_audit"
SUITE_MODEL_MAP="$CONFIG_ROOT/c2g_suite_model_map.json"
SUITE_MODEL_REPORT="$CONFIG_ROOT/c2g_suite_model_map_report.json"
MODEL_VERIFICATION_REPORT="$CONFIG_ROOT/c2g_suite_model_verification_report.json"
COLLECTION_BINDING_REPORT="$CONFIG_ROOT/c2g_clean_collection_model_binding_report.json"
COLLECTION_BINDING_VERIFICATION_REPORT="$CONFIG_ROOT/c2g_clean_collection_model_binding_verification.json"
EVENT_TRACKING_AUDIT="$DRY_AUDIT_ROOT/c2g_goal_event_tracking_audit.json"
BASE="$REPO_ROOT/scripts/stageb/run_c2g_clean_window_pipeline.sh"
BURST_LENGTH="${BURST_LENGTH:-10}"

verify_collection_binding() {
  for path in \
    "$SUITE_MODEL_MAP" \
    "$SUITE_MODEL_REPORT" \
    "$MODEL_VERIFICATION_REPORT" \
    "$COLLECTION_BINDING_REPORT" \
    "$GOAL_MODEL_MANIFEST"
  do
    [[ -f "$path" ]] || { echo "required file missing: $path" >&2; exit 2; }
  done
  python scripts/stageb/verify_c2g_collection_model_provenance.py \
    --collection-root "$COLLECTION_ROOT" \
    --binding-report "$COLLECTION_BINDING_REPORT" \
    --model-map "$SUITE_MODEL_MAP" \
    --model-report "$SUITE_MODEL_REPORT" \
    --goal-model-manifest "$GOAL_MODEL_MANIFEST" \
    --model-verification-report "$MODEL_VERIFICATION_REPORT" \
    --output-report "$COLLECTION_BINDING_VERIFICATION_REPORT"
}

run_phase() {
  case "$1" in
    audit)
      bash "$BASE" audit
      python tools/multisuite_detector/audit_c2g_goal_event_tracking.py \
        --input-root "$COLLECTION_ROOT" \
        --output-report "$EVENT_TRACKING_AUDIT" \
        --burst-length "$BURST_LENGTH"
      ;;
    materialize)
      verify_collection_binding
      [[ -f "$EVENT_TRACKING_AUDIT" ]] || {
        echo "event-tracking audit missing: $EVENT_TRACKING_AUDIT" >&2
        exit 2
      }
      python - "$EVENT_TRACKING_AUDIT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], "r", encoding="utf-8"))
if report.get("status") != "PASS_C2G_GOAL_EVENT_TRACKING_AUDIT":
    raise SystemExit("event-tracking scientific audit is not PASS")
PY
      bash "$BASE" materialize
      ;;
    models|manifests|collect|dataset_audit|train|calibrate|folds|clean_timing|bind_parents|build_jobs|run_jobs|audit_jobs|analyze)
      bash "$BASE" "$1"
      ;;
    *) echo "unknown phase: $1" >&2; exit 2 ;;
  esac
}

if [[ "$PHASE" == "all" ]]; then
  for stage in models manifests collect audit materialize dataset_audit train calibrate folds clean_timing bind_parents build_jobs run_jobs audit_jobs analyze
  do
    echo "=== C2g strict canonical phase: $stage ==="
    run_phase "$stage"
  done
else
  run_phase "$PHASE"
fi
