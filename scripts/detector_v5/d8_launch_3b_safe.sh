#!/usr/bin/env bash
# D8-3B launcher: explicit environment, separate pre-existing log root, unique run root.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CACHE_ROOT="${D8_CACHE_ROOT:?set D8_CACHE_ROOT to the sealed Cache A root}"
CACHE_SEAL="${D8_CACHE_SEAL:?set D8_CACHE_SEAL to Cache A SHA256SUMS.sha256}"
LOG_ROOT="${D8_LOG_ROOT:?set D8_LOG_ROOT to an existing dispatcher-log directory}"
PYTHON_BIN="${D8_PYTHON_BIN:?set D8_PYTHON_BIN to an absolute validated Python executable}"
GPUS="${D8_GPUS:?set D8_GPUS to the explicitly approved non-negative GPU list}"

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

echo "Launcher PID: ${DISPATCHER_PID}"
echo "Kill switch:  touch ${RUN_ROOT}/STOP_D8_3B"
echo "Manifest:     ${RUN_ROOT}/JOB_MANIFEST.json"
