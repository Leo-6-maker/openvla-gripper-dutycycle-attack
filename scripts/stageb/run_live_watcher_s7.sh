#!/bin/bash
# S7 live watcher — monitor only, no VIS/RAND launch
set -euo pipefail

OUT_DIR=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/action_hidden_full
LOG=$OUT_DIR/watcher_s7_live.log
SHARD10_LOG=$OUT_DIR/shard10.log
RETRY_DIR=$OUT_DIR/retry_gpu26_shard45
STAMP() { date '+%Y-%m-%d %H:%M:%S'; }

RETRY_RESTARTED=0
MAX_RETRY_RESTARTS=1

echo "[$(STAMP)] S7 watcher started" | tee -a $LOG

while true; do
    NOW=$(STAMP)
    ALERTS=""

    # Check shard10
    SHARD10_PID=$(pgrep -f 'extract_action_hidden_full.py.*start 0.*count 20' | head -1 || true)
    if [ -z "$SHARD10_PID" ]; then
        ALERTS="${ALERTS}[ALERT] shard10 process NOT FOUND\n"
    fi

    # Check retry
    RETRY_PID=$(pgrep -f 'extract_action_hidden_retry.py' | head -1 || true)
    if [ -z "$RETRY_PID" ]; then
        ALERTS="${ALERTS}[ALERT] retry_gpu26 process NOT FOUND\n"
    fi

    # GPU memory using awk (portable, no grep -P)
    GPU_MEM=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null)
    GPU0_MEM=$(echo "$GPU_MEM" | awk -F', ' '/^0,/ {print $2}' | awk '{print $1}')
    GPU2_MEM=$(echo "$GPU_MEM" | awk -F', ' '/^2,/ {print $2}' | awk '{print $1}')
    GPU4_MEM=$(echo "$GPU_MEM" | awk -F', ' '/^4,/ {print $2}' | awk '{print $1}')
    GPU6_MEM=$(echo "$GPU_MEM" | awk -F', ' '/^6,/ {print $2}' | awk '{print $1}')

    # Memory alerts
    if [ -n "$GPU0_MEM" ] && [ "$GPU0_MEM" -gt 10500 ] 2>/dev/null; then
        ALERTS="${ALERTS}[ALERT] GPU 0 memory high: ${GPU0_MEM} MiB\n"
    fi
    if [ -n "$GPU2_MEM" ] && [ "$GPU2_MEM" -gt 10500 ] 2>/dev/null; then
        ALERTS="${ALERTS}[ALERT] GPU 2 memory high: ${GPU2_MEM} MiB\n"
    fi
    if [ -n "$GPU4_MEM" ] && [ "$GPU4_MEM" -gt 2000 ] 2>/dev/null; then
        ALERTS="${ALERTS}[ALERT] GPU 4 memory not released: ${GPU4_MEM} MiB\n"
    fi

    # Scan logs for errors
    SHARD10_ERRS=$(grep -ciE 'Error|Traceback|CUDA error|CUBLAS|EGL|MuJoCo|Xid|illegal|CUBLAS_STATUS' "$SHARD10_LOG" 2>/dev/null || echo 0)
    RETRY_ERRS=0
    for f in "$RETRY_DIR"/*.log; do
        [ -f "$f" ] && RETRY_ERRS=$((RETRY_ERRS + $(grep -ciE 'Error|Traceback|CUDA error|CUBLAS|EGL|MuJoCo|Xid|illegal|CUBLAS_STATUS' "$f" 2>/dev/null || echo 0)))
    done

    if [ "$SHARD10_ERRS" -gt 0 ] 2>/dev/null; then
        ALERTS="${ALERTS}[ALERT] shard10 log has ${SHARD10_ERRS} error lines\n"
    fi
    if [ "$RETRY_ERRS" -gt 0 ] 2>/dev/null; then
        ALERTS="${ALERTS}[ALERT] retry_gpu26 log has ${RETRY_ERRS} error lines\n"
    fi

    # Progress
    SHARD10_PROG=$(grep -c 'dim=' "$SHARD10_LOG" 2>/dev/null || echo 0)
    RETRY_COMPLETED=$(wc -l < "$RETRY_DIR/completed_window_ids.txt" 2>/dev/null || echo 0)
    RETRY_FAILED=$(wc -l < "$RETRY_DIR/failed_window_ids.txt" 2>/dev/null || echo 0)
    CSV_ROWS=$(wc -l < "$RETRY_DIR/action_hidden_full_features_w20_retry_gpu26.csv" 2>/dev/null || echo 0)

    STATUS="[$NOW] shard10=${SHARD10_PROG}/20 retry_completed=${RETRY_COMPLETED} retry_failed=${RETRY_FAILED} csv_rows=${CSV_ROWS} GPU0=${GPU0_MEM}MiB GPU2=${GPU2_MEM}MiB GPU6=${GPU6_MEM}MiB GPU4=${GPU4_MEM}MiB"

    if [ -n "$ALERTS" ]; then
        echo -e "[$NOW] *** ALERTS ***\n$ALERTS$STATUS" | tee -a $LOG
    else
        echo "$STATUS" | tee -a $LOG
    fi

    sleep 300
done
