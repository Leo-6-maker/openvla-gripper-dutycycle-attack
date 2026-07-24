#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preview-plan}"
case "$MODE" in
  preview-plan|plan|preview-audit|audit) ;;
  *)
    echo "usage: $0 [preview-plan|plan|preview-audit|audit]" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

required_base=(
  R6_AUDIT_REPORT
  EXPECTED_R6_AUDIT_SHA256
  EXPECTED_R6_HEAD
  R7_OUTPUT_ROOT
)
for name in "${required_base[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 2
  fi
done

AUDIT_HEAD="${AUDIT_HEAD:-$(git rev-parse HEAD)}"
SELECTION_SEED="${SELECTION_SEED:-42}"
MAX_STEPS="${MAX_STEPS:-300}"
TRAIN_STATES_PER_TASK="${TRAIN_STATES_PER_TASK:-30}"
VAL_STATES_PER_TASK="${VAL_STATES_PER_TASK:-5}"
TEST_STATES_PER_TASK="${TEST_STATES_PER_TASK:-5}"
ATTACK_EVAL_STATES_PER_TASK="${ATTACK_EVAL_STATES_PER_TASK:-10}"

actual_r6_sha="$(sha256sum "$R6_AUDIT_REPORT" | awk '{print $1}')"
if [[ "$actual_r6_sha" != "$EXPECTED_R6_AUDIT_SHA256" ]]; then
  echo "R6 audit report SHA mismatch: $actual_r6_sha != $EXPECTED_R6_AUDIT_SHA256" >&2
  exit 2
fi

python - "$R6_AUDIT_REPORT" "$EXPECTED_R6_HEAD" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
expected_head = sys.argv[2]
report = json.loads(path.read_text(encoding="utf-8"))
required = {
    "status": "PASS_C2G_R6_BOUND_DATASET_AUDIT",
    "integrity_status": "PASS_C2G_R6_DATASET_INTEGRITY",
    "scientific_trainability_status": "HOLD_C2G_R6_SCIENTIFIC_TRAINABILITY",
    "training_authorization": "HOLD_INSUFFICIENT_SCIENTIFIC_SUPPORT",
}
for key, expected in required.items():
    if report.get(key) != expected:
        raise SystemExit(f"R6 boundary mismatch: {key}={report.get(key)!r}, expected {expected!r}")
if report.get("audit_head") != expected_head:
    raise SystemExit(
        f"R6 audit head mismatch: {report.get('audit_head')!r} != {expected_head!r}"
    )
if int(report.get("episode_count", -1)) != 4:
    raise SystemExit("R6 report no longer binds the accepted four-episode smoke")
PY

plan_cmd=(
  python
  tools/multisuite_detector/plan_c2g_scientific_corpus.py
  --from-libero
  --output-dir "$R7_OUTPUT_ROOT"
  --train-states-per-task "$TRAIN_STATES_PER_TASK"
  --val-states-per-task "$VAL_STATES_PER_TASK"
  --test-states-per-task "$TEST_STATES_PER_TASK"
  --attack-eval-states-per-task "$ATTACK_EVAL_STATES_PER_TASK"
  --max-steps "$MAX_STEPS"
  --seed "$SELECTION_SEED"
  --expected-git-commit "$AUDIT_HEAD"
)

json_command() {
  local status="$1"
  shift
  printf '{"status":"%s","audit_head":"%s","command":[' "$status" "$AUDIT_HEAD"
  local separator=""
  local argument
  for argument in "$@"; do
    printf '%s"%s"' "$separator" "$(python -c 'import json,sys; print(json.dumps(sys.argv[1])[1:-1])' "$argument")"
    separator=","
  done
  printf ']}\n'
}

if [[ "$MODE" == "preview-plan" ]]; then
  test ! -e "$R7_OUTPUT_ROOT"
  json_command "PASS_C2G_R7_CORPUS_PLAN_PREVIEW" "${plan_cmd[@]}"
  exit 0
fi

if [[ "$MODE" == "plan" ]]; then
  test ! -e "$R7_OUTPUT_ROOT"
  "${plan_cmd[@]}"
  exit 0
fi

required_audit=(
  EXPECTED_R7_PLAN_REPORT_SHA256
  R7_SOURCE_ROOTS
  R7_SOURCE_AUDIT_REPORT
  R7_REUSABLE_MANIFEST
)
for name in "${required_audit[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 2
  fi
done

R7_PLAN_REPORT="${R7_PLAN_REPORT:-$R7_OUTPUT_ROOT/c2g_scientific_corpus_plan_report.json}"
R7_REGISTRY="${R7_REGISTRY:-$R7_OUTPUT_ROOT/c2g_parent_registry.jsonl}"
PERSISTENCE_WINDOW="${PERSISTENCE_WINDOW:-3}"
PERSISTENCE_REQUIRED="${PERSISTENCE_REQUIRED:-2}"
BURST_LENGTH="${BURST_LENGTH:-10}"
HASH_RGB="${HASH_RGB:-1}"

IFS=':' read -r -a source_roots <<< "$R7_SOURCE_ROOTS"
if [[ "${#source_roots[@]}" -eq 0 ]]; then
  echo "R7_SOURCE_ROOTS produced no roots" >&2
  exit 2
fi

audit_cmd=(
  python
  tools/multisuite_detector/audit_c2g_clean_source_inventory.py
  --registry "$R7_REGISTRY"
  --plan-report "$R7_PLAN_REPORT"
  --expected-plan-report-sha256 "$EXPECTED_R7_PLAN_REPORT_SHA256"
  --output-report "$R7_SOURCE_AUDIT_REPORT"
  --reusable-manifest "$R7_REUSABLE_MANIFEST"
  --persistence-window "$PERSISTENCE_WINDOW"
  --persistence-required "$PERSISTENCE_REQUIRED"
  --burst-length "$BURST_LENGTH"
  --audit-head "$AUDIT_HEAD"
)
for root in "${source_roots[@]}"; do
  audit_cmd+=(--source-root "$root")
done
if [[ "$HASH_RGB" == "0" ]]; then
  audit_cmd+=(--no-hash-rgb)
else
  audit_cmd+=(--hash-rgb)
fi

if [[ "$MODE" == "preview-audit" ]]; then
  test -f "$R7_PLAN_REPORT"
  test -f "$R7_REGISTRY"
  test ! -e "$R7_SOURCE_AUDIT_REPORT"
  test ! -e "$R7_REUSABLE_MANIFEST"
  json_command "PASS_C2G_R7_SOURCE_AUDIT_PREVIEW" "${audit_cmd[@]}"
  exit 0
fi

test -f "$R7_PLAN_REPORT"
test -f "$R7_REGISTRY"
test ! -e "$R7_SOURCE_AUDIT_REPORT"
test ! -e "$R7_REUSABLE_MANIFEST"
"${audit_cmd[@]}"
