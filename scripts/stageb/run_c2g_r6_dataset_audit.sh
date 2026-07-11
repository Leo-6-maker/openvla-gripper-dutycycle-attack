#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
case "$MODE" in
  preview|run) ;;
  *)
    echo "usage: $0 [preview|run]" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

required=(
  R5_DATASET
  R5_BOUND_REPORT
  R5_BASE_REPORT
  EXPECTED_DATASET_SHA256
  EXPECTED_BOUND_REPORT_SHA256
  EXPECTED_BASE_REPORT_SHA256
  EXPECTED_MATERIALIZATION_HEAD
  R6_AUDIT_REPORT
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 2
  fi
done

AUDIT_HEAD="${AUDIT_HEAD:-$(git rev-parse HEAD)}"
PERSISTENCE_WINDOW="${PERSISTENCE_WINDOW:-3}"
PERSISTENCE_REQUIRED="${PERSISTENCE_REQUIRED:-2}"
MIN_TOTAL_EPISODES="${MIN_TOTAL_EPISODES:-12}"
MIN_TOTAL_TASKS="${MIN_TOTAL_TASKS:-8}"
MIN_EPISODES_PER_SUITE="${MIN_EPISODES_PER_SUITE:-3}"
MIN_TASKS_PER_SUITE="${MIN_TASKS_PER_SUITE:-2}"
MIN_SPLITS_PER_SUITE="${MIN_SPLITS_PER_SUITE:-3}"
MIN_TRAIN_EPISODES="${MIN_TRAIN_EPISODES:-4}"
MIN_VAL_EPISODES="${MIN_VAL_EPISODES:-2}"
MIN_TEST_EPISODES="${MIN_TEST_EPISODES:-2}"
MIN_TRAIN_SUITES="${MIN_TRAIN_SUITES:-4}"
MIN_VAL_SUITES="${MIN_VAL_SUITES:-2}"
MIN_TEST_SUITES="${MIN_TEST_SUITES:-2}"

cmd=(
  python
  tools/multisuite_detector/audit_c2g_r5_bound_dataset.py
  --dataset "$R5_DATASET"
  --bound-materialization-report "$R5_BOUND_REPORT"
  --base-materialization-report "$R5_BASE_REPORT"
  --expected-dataset-sha256 "$EXPECTED_DATASET_SHA256"
  --expected-bound-report-sha256 "$EXPECTED_BOUND_REPORT_SHA256"
  --expected-base-report-sha256 "$EXPECTED_BASE_REPORT_SHA256"
  --expected-materialization-head "$EXPECTED_MATERIALIZATION_HEAD"
  --audit-head "$AUDIT_HEAD"
  --output-report "$R6_AUDIT_REPORT"
  --persistence-window "$PERSISTENCE_WINDOW"
  --persistence-required "$PERSISTENCE_REQUIRED"
  --min-total-episodes "$MIN_TOTAL_EPISODES"
  --min-total-tasks "$MIN_TOTAL_TASKS"
  --min-episodes-per-suite "$MIN_EPISODES_PER_SUITE"
  --min-tasks-per-suite "$MIN_TASKS_PER_SUITE"
  --min-splits-per-suite "$MIN_SPLITS_PER_SUITE"
  --min-train-episodes "$MIN_TRAIN_EPISODES"
  --min-val-episodes "$MIN_VAL_EPISODES"
  --min-test-episodes "$MIN_TEST_EPISODES"
  --min-train-suites "$MIN_TRAIN_SUITES"
  --min-val-suites "$MIN_VAL_SUITES"
  --min-test-suites "$MIN_TEST_SUITES"
)

if [[ "$MODE" == "preview" ]]; then
  printf '{"status":"PASS_C2G_R6_DATASET_AUDIT_PREVIEW","audit_head":"%s","command":[' "$AUDIT_HEAD"
  separator=""
  for argument in "${cmd[@]}"; do
    printf '%s"%s"' "$separator" "$(python -c 'import json,sys; print(json.dumps(sys.argv[1])[1:-1])' "$argument")"
    separator=","
  done
  printf ']}\n'
  exit 0
fi

test ! -e "$R6_AUDIT_REPORT"
"${cmd[@]}"
