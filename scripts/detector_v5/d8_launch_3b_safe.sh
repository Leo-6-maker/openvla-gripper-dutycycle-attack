#!/usr/bin/env bash
# D8-3B launcher: explicit environment, separate pre-existing log root, unique run root.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CACHE_ROOT="${D8_CACHE_ROOT:?set D8_CACHE_ROOT to the sealed Cache A root}"
CACHE_SEAL="${D8_CACHE_SEAL:?set D8_CACHE_SEAL to Cache A SHA256SUMS.sha256}"
CACHE_A_SEAL="${D8_CACHE_A_SEAL:?set D8_CACHE_A_SEAL to the expected Cache A seal}"
CACHE_B_SEAL="${D8_CACHE_B_SEAL:?set D8_CACHE_B_SEAL to the expected Cache B seal}"
EXPECTED_SOURCE_COMMIT="${D8_EXPECTED_SOURCE_COMMIT:?set D8_EXPECTED_SOURCE_COMMIT to the approved 40-character commit SHA}"
EXPECTED_SOURCE_TREE="${D8_EXPECTED_SOURCE_TREE:?set D8_EXPECTED_SOURCE_TREE to the approved 40-character tree SHA}"
H1_SOURCE_COMMIT="${D8_H1_SOURCE_COMMIT:?set D8_H1_SOURCE_COMMIT to the H1 source commit SHA}"
H1_SOURCE_TREE="${D8_H1_SOURCE_TREE:?set D8_H1_SOURCE_TREE to the H1 source tree SHA}"
SOURCE_SNAPSHOT_SHA256="${D8_SOURCE_SNAPSHOT_SHA256:?set D8_SOURCE_SNAPSHOT_SHA256 to the H1 source snapshot SHA256}"
COMPARATOR_SEAL="${D8_COMPARATOR_SEAL:?set D8_COMPARATOR_SEAL to the comparator seal}"
P5_ARTIFACT_SEAL="${D8_P5_ARTIFACT_SEAL:?set D8_P5_ARTIFACT_SEAL to the P5 artifact seal}"
H1_REVIEW_SEAL="${D8_H1_REVIEW_SEAL:?set D8_H1_REVIEW_SEAL to the H1 review seal}"
LOG_ROOT="${D8_LOG_ROOT:?set D8_LOG_ROOT to an existing dispatcher-log directory}"
PYTHON_BIN="${D8_PYTHON_BIN:?set D8_PYTHON_BIN to an absolute validated Python executable}"
GPUS="${D8_GPUS:?set D8_GPUS to the explicitly approved non-negative GPU list}"
SHELL_SCRIPT_SHA256="$(sha256sum "${REPO_ROOT}/scripts/detector_v5/d8_launch_3b_safe.sh" | cut -d ' ' -f1)"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "D8_PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -d "${LOG_ROOT}" ]]; then
    echo "D8_LOG_ROOT must already exist: ${LOG_ROOT}" >&2
    exit 2
fi

SOURCE_PREFIX="$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD)"
UTC="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${D8_RUN_ROOT:-${LOG_ROOT}/d8_3b_${SOURCE_PREFIX}_${UTC}}"
if [[ -e "${RUN_ROOT}" ]]; then
    echo "run root already exists; refusing to clobber: ${RUN_ROOT}" >&2
    exit 2
fi

DISPATCH_LOG="${LOG_ROOT}/d8_3b_${SOURCE_PREFIX}_${UTC}.dispatch.log"
PID_FILE="${LOG_ROOT}/d8_3b_${SOURCE_PREFIX}_${UTC}.dispatcher.pid"

echo "=== D8-3B fail-closed dispatch ==="
echo "Cache:   ${CACHE_ROOT}"
echo "Run:     ${RUN_ROOT}"
echo "Log:     ${DISPATCH_LOG}"
echo "Python:  ${PYTHON_BIN}"
echo "GPUs:    ${GPUS}"
echo "================================="

cd "${REPO_ROOT}"
nohup "${PYTHON_BIN}" -u "${REPO_ROOT}/scripts/detector_v5/run_d8_2_cv_parallel.py" \
    --cache-root "${CACHE_ROOT}" \
    --cache-seal "${CACHE_SEAL}" \
    --cache-a-seal "${CACHE_A_SEAL}" \
    --cache-b-seal "${CACHE_B_SEAL}" \
    --comparator-seal "${COMPARATOR_SEAL}" \
    --p5-artifact-seal "${P5_ARTIFACT_SEAL}" \
    --h1-review-seal "${H1_REVIEW_SEAL}" \
    --h1-source-commit "${H1_SOURCE_COMMIT}" \
    --h1-source-tree "${H1_SOURCE_TREE}" \
    --source-snapshot-sha256 "${SOURCE_SNAPSHOT_SHA256}" \
    --expected-source-commit "${EXPECTED_SOURCE_COMMIT}" \
    --expected-source-tree "${EXPECTED_SOURCE_TREE}" \
    --shell-script-sha256 "${SHELL_SCRIPT_SHA256}" \
    --output-root "${RUN_ROOT}" \
    --log-root "${LOG_ROOT}" \
    --python-bin "${PYTHON_BIN}" \
    --gpus "${GPUS}" \
    --seeds "20260720,20260721,20260722,20260723,20260724,20260725,20260726,20260727,20260728,20260729" \
    --epochs 100 \
    --configs "B3" \
    > "${DISPATCH_LOG}" 2>&1 &
DISPATCHER_PID="$!"
printf '%s\n' "${DISPATCHER_PID}" > "${PID_FILE}"

sleep 1
if ! kill -0 "${DISPATCHER_PID}" 2>/dev/null; then
    echo "dispatcher exited during launcher startup: ${DISPATCHER_PID}" >&2
    tail -n 80 "${DISPATCH_LOG}" >&2 || true
    exit 1
fi
if [[ ! -f "${RUN_ROOT}/EXECUTION_RECEIPT.json" ]]; then
    echo "dispatcher did not create the execution receipt during launcher startup: ${RUN_ROOT}" >&2
    tail -n 80 "${DISPATCH_LOG}" >&2 || true
    exit 1
fi

echo "Launcher PID: ${DISPATCHER_PID}"
echo "Kill switch:  touch ${RUN_ROOT}/STOP_D8_3B"
echo "Manifest:     ${RUN_ROOT}/JOB_MANIFEST.json"
