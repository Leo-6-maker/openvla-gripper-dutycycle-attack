#!/usr/bin/env bash
set -euo pipefail

# Provenance-bound R5 materialization entry point.
#
# This script intentionally stops after dataset materialization. It never trains a
# detector, creates a LIBERO environment, launches a clean rollout, or runs an
# attack. Use `preview` before the separately authorized `run` phase.

PHASE="${1:-}"
if [[ "$PHASE" != "preview" && "$PHASE" != "run" ]]; then
  echo "usage: $0 <preview|run>" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
AUDIT_HEAD="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing R5 materialization from dirty worktree" >&2
  git status --short >&2
  exit 2
fi

: "${WORK_ROOT:?set WORK_ROOT to the external R3 configuration root}"
: "${R4_PROVENANCE_BINDING:?set R4_PROVENANCE_BINDING to the PASS dual-head report}"
: "${GOAL_MODEL_MANIFEST:?set GOAL_MODEL_MANIFEST to the audited Goal manifest}"
: "${R5_OUTPUT_ROOT:?set R5_OUTPUT_ROOT to a new external empty output directory}"

resolve_path() {
  python - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
}

WORK_ROOT="$(resolve_path "$WORK_ROOT")"
R4_PROVENANCE_BINDING="$(resolve_path "$R4_PROVENANCE_BINDING")"
GOAL_MODEL_MANIFEST="$(resolve_path "$GOAL_MODEL_MANIFEST")"
R5_OUTPUT_ROOT="$(resolve_path "$R5_OUTPUT_ROOT")"
COLLECTION_ROOT="$(resolve_path "${COLLECTION_ROOT:-$WORK_ROOT/clean_collection}")"

CONFIG_ROOT="$WORK_ROOT/config"
SUITE_MODEL_MAP="${SUITE_MODEL_MAP:-$CONFIG_ROOT/c2g_suite_model_map.json}"
SUITE_MODEL_REPORT="${SUITE_MODEL_REPORT:-$CONFIG_ROOT/c2g_suite_model_map_report.json}"
MODEL_VERIFICATION_REPORT="${MODEL_VERIFICATION_REPORT:-$CONFIG_ROOT/c2g_suite_model_verification_report.json}"

for path in \
  "$COLLECTION_ROOT" \
  "$R4_PROVENANCE_BINDING" \
  "$GOAL_MODEL_MANIFEST" \
  "$SUITE_MODEL_MAP" \
  "$SUITE_MODEL_REPORT" \
  "$MODEL_VERIFICATION_REPORT"
do
  [[ -e "$path" ]] || { echo "required R5 input missing: $path" >&2; exit 2; }
done

DEVICE="${DEVICE:-cuda:0}"
WINDOW="${WINDOW:-16}"
BURST_LENGTH="${BURST_LENGTH:-10}"
MAX_EPISODES_PER_SUITE="${MAX_EPISODES_PER_SUITE:-0}"
MIN_FREE_BYTES="${MIN_FREE_BYTES:-16106127360}"

COMMAND=(
  python tools/multisuite_detector/materialize_c2g_multisuite_dataset_bound.py
  --input-root "$COLLECTION_ROOT"
  --output-dir "$R5_OUTPUT_ROOT"
  --r4-provenance-binding "$R4_PROVENANCE_BINDING"
  --audit-head "$AUDIT_HEAD"
  --suite-model-map "$SUITE_MODEL_MAP"
  --suite-model-report "$SUITE_MODEL_REPORT"
  --goal-model-manifest "$GOAL_MODEL_MANIFEST"
  --model-verification-report "$MODEL_VERIFICATION_REPORT"
  --backend openvla_siglip
  --device "$DEVICE"
  --window "$WINDOW"
  --burst-length "$BURST_LENGTH"
  --split-mode within_task
  --max-episodes-per-suite "$MAX_EPISODES_PER_SUITE"
  --min-free-bytes "$MIN_FREE_BYTES"
)

if [[ "$PHASE" == "preview" ]]; then
  "${COMMAND[@]}" --dry-run
else
  "${COMMAND[@]}"
fi
