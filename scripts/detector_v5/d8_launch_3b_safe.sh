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
DISPATCHER_SCRIPT="${REPO_ROOT}/scripts/detector_v5/run_d8_2_cv_parallel.py"
STARTUP_TIMEOUT_SECONDS="${D8_STARTUP_TIMEOUT_SECONDS:-900}"
STARTUP_POLL_SECONDS="${D8_STARTUP_POLL_SECONDS:-1}"
STARTUP_GRACE_SECONDS="${D8_STARTUP_GRACE_SECONDS:-30}"

case "${STARTUP_TIMEOUT_SECONDS}" in
    ''|*[!0-9]*) echo "D8_STARTUP_TIMEOUT_SECONDS must be a non-negative integer" >&2; exit 2 ;;
esac
case "${STARTUP_GRACE_SECONDS}" in
    ''|*[!0-9]*) echo "D8_STARTUP_GRACE_SECONDS must be a non-negative integer" >&2; exit 2 ;;
esac
if [[ -z "${STARTUP_POLL_SECONDS}" ]]; then
    echo "D8_STARTUP_POLL_SECONDS must be non-empty" >&2
    exit 2
fi

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
PREFLIGHT_LOG="${LOG_ROOT}/d8_3b_${SOURCE_PREFIX}_${UTC}.preflight.json"

COMMON_ARGS=(
    --cache-root "${CACHE_ROOT}"
    --cache-seal "${CACHE_SEAL}"
    --cache-a-seal "${CACHE_A_SEAL}"
    --cache-b-seal "${CACHE_B_SEAL}"
    --comparator-seal "${COMPARATOR_SEAL}"
    --p5-artifact-seal "${P5_ARTIFACT_SEAL}"
    --h1-review-seal "${H1_REVIEW_SEAL}"
    --h1-source-commit "${H1_SOURCE_COMMIT}"
    --h1-source-tree "${H1_SOURCE_TREE}"
    --source-snapshot-sha256 "${SOURCE_SNAPSHOT_SHA256}"
    --expected-source-commit "${EXPECTED_SOURCE_COMMIT}"
    --expected-source-tree "${EXPECTED_SOURCE_TREE}"
    --shell-script-sha256 "${SHELL_SCRIPT_SHA256}"
    --output-root "${RUN_ROOT}"
    --log-root "${LOG_ROOT}"
    --python-bin "${PYTHON_BIN}"
    --gpus "${GPUS}"
    --seeds "20260720,20260721,20260722,20260723,20260724,20260725,20260726,20260727,20260728,20260729"
    --epochs 100
    --configs B3
)

startup_manifest_ready() {
    local manifest="$1"
    local job_count status_count allowed_count
    [[ -f "${manifest}" ]] || return 1
    job_count="$(grep -Ec '"job_id":' "${manifest}" || true)"
    status_count="$(grep -Ec '"status": "' "${manifest}" || true)"
    allowed_count="$(grep -Ec '"status": "(PENDING|RUNNING)"' "${manifest}" || true)"
    (( job_count == 50 && status_count == 50 && allowed_count == 50 )) \
        && grep -Eq '"planned_jobs": 50' "${manifest}"
}

reap_dispatcher() {
    local pid="$1"
    local grace_seconds="$2"
    local deadline
    if kill -0 "${pid}" 2>/dev/null; then
        kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
        deadline=$((SECONDS + grace_seconds))
        while kill -0 "${pid}" 2>/dev/null && (( SECONDS < deadline )); do
            sleep 0.2
        done
        if kill -0 "${pid}" 2>/dev/null; then
            kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
        fi
    fi
    wait "${pid}" 2>/dev/null || true
    ! kill -0 "${pid}" 2>/dev/null
}

echo "=== D8-3B fail-closed dispatch ==="
echo "Cache:   ${CACHE_ROOT}"
echo "Run:     ${RUN_ROOT}"
echo "Log:     ${DISPATCH_LOG}"
echo "Python:  ${PYTHON_BIN}"
echo "GPUs:    ${GPUS}"
echo "Preflight: ${PREFLIGHT_LOG}"
echo "================================="

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -u "${DISPATCHER_SCRIPT}" "${COMMON_ARGS[@]}" --preflight-only \
    > "${PREFLIGHT_LOG}" 2>&1 || {
    echo "D8-3B preflight failed" >&2
    tail -n 120 "${PREFLIGHT_LOG}" >&2 || true
    exit 1
}

if ! command -v setsid >/dev/null 2>&1; then
    echo "setsid is required to isolate the dispatcher process group" >&2
    exit 2
fi

nohup setsid "${PYTHON_BIN}" -u "${DISPATCHER_SCRIPT}" "${COMMON_ARGS[@]}" \
    > "${DISPATCH_LOG}" 2>&1 &
DISPATCHER_PID="$!"
printf '%s\n' "${DISPATCHER_PID}" > "${PID_FILE}"

STARTUP_DEADLINE=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
STARTUP_READY=0
while (( SECONDS < STARTUP_DEADLINE )); do
    if ! kill -0 "${DISPATCHER_PID}" 2>/dev/null; then
        rc=1
        if wait "${DISPATCHER_PID}"; then
            rc=1
        else
            rc=$?
        fi
        echo "dispatcher exited during preflight: rc=${rc}" >&2
        tail -n 120 "${DISPATCH_LOG}" >&2 || true
        exit "${rc}"
    fi
    if [[ -f "${RUN_ROOT}/EXECUTION_RECEIPT.json" ]] \
        && startup_manifest_ready "${RUN_ROOT}/JOB_MANIFEST.json"; then
        STARTUP_READY=1
        break
    fi
    sleep "${STARTUP_POLL_SECONDS}"
done

if (( STARTUP_READY == 0 )); then
    echo "dispatcher startup timeout after ${STARTUP_TIMEOUT_SECONDS}s" >&2
    if ! reap_dispatcher "${DISPATCHER_PID}" "${STARTUP_GRACE_SECONDS}"; then
        echo "dispatcher PID remains alive after TERM/KILL cleanup" >&2
        exit 1
    fi
    if kill -0 "${DISPATCHER_PID}" 2>/dev/null; then
        echo "dispatcher PID still exists after reap" >&2
        exit 1
    fi
    tail -n 120 "${DISPATCH_LOG}" >&2 || true
    exit 1
fi

echo "Launcher PID: ${DISPATCHER_PID}"
echo "Kill switch:  touch ${RUN_ROOT}/STOP_D8_3B"
echo "Manifest:     ${RUN_ROOT}/JOB_MANIFEST.json"
