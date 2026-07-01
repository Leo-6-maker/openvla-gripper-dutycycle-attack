#!/bin/bash
# CHAIN Stage 3: TMA Full — Student (8 GPU) → Random-Time (8 GPU) sequential, hard gates
set -uo pipefail
EXEC=/mnt/sdc/dty_user/table1_sota_execution_v1; LOGDIR=$EXEC/logs; mkdir -p $LOGDIR
PYTHON=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
SOTA_WORKER=$EXEC/commands/run_sota_worker.py
SOTA_MF=$EXEC/manifests; NEXT_SCRIPT=$EXEC/commands/chain_stage4_uma_shuffled.sh

log() { echo "$(date -Iseconds) [STAGE3] $*" | tee -a $LOGDIR/chain.log; }

# ── TMA Student: 8 GPUs, 162 jobs ──
log "=== TMA STUDENT (8 GPUs, 162 jobs) ==="
for gpu in 0 1 2 3 4 5 6 7; do
    MF=$SOTA_MF/TMA/manifest_gpu${gpu}.jsonl
    [ -f "$MF" ] && nohup $PYTHON -u $SOTA_WORKER $gpu $MF > $LOGDIR/full_tma_student_gpu${gpu}.log 2>&1 &
    log "GPU $gpu TMA_Student PID=$!"
done

log "Waiting for TMA Student workers (8 PIDs)..."
wait
DONE=0; FAILED=0
for gpu in 0 1 2 3 4 5 6 7; do
    lf="$LOGDIR/full_tma_student_gpu${gpu}.log"
    [ -f "$lf" ] && DONE=$((DONE + $(grep -c 'COMPLETE' "$lf" 2>/dev/null || echo 0)))
    [ -f "$lf" ] && FAILED=$((FAILED + $(grep -c 'FAILED' "$lf" 2>/dev/null || echo 0)))
done
log "TMA Student: $DONE/162 COMPLETE, $FAILED FAILED"
if [ "$FAILED" -gt 0 ] || [ "$DONE" -ne 162 ]; then
    log "FATAL: TMA Student gate FAILED"
    exit 1
fi

# ── TMA Random-Time: 8 GPUs, 162 jobs ──
log "=== TMA RANDOM-TIME (8 GPUs, 162 jobs) ==="
for gpu in 0 1 2 3 4 5 6 7; do
    MF=$SOTA_MF/TMA_RANDOM_TIME/manifest_gpu${gpu}.jsonl
    [ -f "$MF" ] && nohup $PYTHON -u $SOTA_WORKER $gpu $MF > $LOGDIR/full_tma_random_gpu${gpu}.log 2>&1 &
    log "GPU $gpu TMA_RandomTime PID=$!"
done

log "Waiting for TMA Random-Time workers..."
wait
DONE=0; FAILED=0
for gpu in 0 1 2 3 4 5 6 7; do
    lf="$LOGDIR/full_tma_random_gpu${gpu}.log"
    [ -f "$lf" ] && DONE=$((DONE + $(grep -c 'COMPLETE' "$lf" 2>/dev/null || echo 0)))
    [ -f "$lf" ] && FAILED=$((FAILED + $(grep -c 'FAILED' "$lf" 2>/dev/null || echo 0)))
done
log "TMA Random-Time: $DONE/162 COMPLETE, $FAILED FAILED"
if [ "$FAILED" -gt 0 ] || [ "$DONE" -ne 162 ]; then
    log "FATAL: TMA Random-Time gate FAILED"
    exit 1
fi

log "TMA Full gate PASS. Advancing to Stage 4..."
if [ -x "$NEXT_SCRIPT" ]; then
    exec bash "$NEXT_SCRIPT"
else
    log "ERROR: $NEXT_SCRIPT not found"; exit 1
fi
