#!/bin/bash
# D8-3B Safe Launch: single dispatcher, B3-only, 6-GPU queue, kill switch.
# Usage: bash d8_launch_3b_safe.sh
set -euo pipefail

CACHE_ROOT="/mnt/sdc/dty_user/d8_h1_9dd3_cache_A"
OUTPUT_ROOT="/mnt/sdc/dty_user/d8_3b_cv_safe"
SEEDS="20260720,20260721,20260722,20260723,20260724,20260725,20260726,20260727,20260728,20260729"
EPOCHS=100
GPUS="0,1,2,3,6,7"
PYTHON="$(which python3 || which python)"

echo "=== D8-3B Safe Dispatch ==="
echo "Seeds:   ${SEEDS}"
echo "Epochs:  ${EPOCHS}"
echo "GPUs:    ${GPUS}"
echo "Output:  ${OUTPUT_ROOT}"
echo "==========================="

nohup "${PYTHON}" -u scripts/detector_v5/run_d8_2_cv_parallel.py \
    --cache-root "${CACHE_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --seeds "${SEEDS}" \
    --gpus "${GPUS}" \
    --epochs "${EPOCHS}" \
    --configs "B3" \
    > "${OUTPUT_ROOT}/dispatch.log" 2>&1 &

echo "Launcher PID: $!"
echo "Kill switch:  touch ${OUTPUT_ROOT}/STOP_D8_3B"
echo "Log:          ${OUTPUT_ROOT}/dispatch.log"
