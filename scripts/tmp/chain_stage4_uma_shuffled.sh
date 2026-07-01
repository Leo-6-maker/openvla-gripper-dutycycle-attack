#!/bin/bash
# CHAIN Stage 4: UMA Full (8 GPU, 162) → SHUFFLED Full (8 GPU, 162) sequential, hard gates
set -uo pipefail
EXEC=/mnt/sdc/dty_user/table1_sota_execution_v1; LOGDIR=$EXEC/logs; mkdir -p $LOGDIR
PYTHON=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
SOTA_WORKER=$EXEC/commands/run_sota_worker.py
SOTA_MF=$EXEC/manifests; NEXT_SCRIPT=$EXEC/commands/chain_stage5_aggregate.sh

log() { echo "$(date -Iseconds) [STAGE4] $*" | tee -a $LOGDIR/chain.log; }

# ── UMA: 8 GPUs, 162 jobs ──
log "=== UMA (8 GPUs, 162 jobs) ==="
for gpu in 0 1 2 3 4 5 6 7; do
    MF=$SOTA_MF/UMA/manifest_gpu${gpu}.jsonl
    [ -f "$MF" ] && nohup $PYTHON -u $SOTA_WORKER $gpu $MF > $LOGDIR/full_uma_gpu${gpu}.log 2>&1 &
    log "GPU $gpu UMA PID=$!"
done

log "Waiting for UMA workers..."
wait
DONE=0; FAILED=0
for gpu in 0 1 2 3 4 5 6 7; do
    lf="$LOGDIR/full_uma_gpu${gpu}.log"
    [ -f "$lf" ] && DONE=$((DONE + $(grep -c 'COMPLETE' "$lf" 2>/dev/null || echo 0)))
    [ -f "$lf" ] && FAILED=$((FAILED + $(grep -c 'FAILED' "$lf" 2>/dev/null || echo 0)))
done
log "UMA: $DONE/162 COMPLETE, $FAILED FAILED"
if [ "$FAILED" -gt 0 ] || [ "$DONE" -ne 162 ]; then
    log "FATAL: UMA gate FAILED"; exit 1
fi

# ── SHUFFLED: 8 GPUs, 162 jobs ──
log "=== SHUFFLED (8 GPUs, 162 jobs) ==="
for gpu in 0 1 2 3 4 5 6 7; do
    MF=$SOTA_MF/SHUFFLED/manifest_gpu${gpu}.jsonl
    [ -f "$MF" ] && nohup $PYTHON -u $SOTA_WORKER $gpu $MF > $LOGDIR/full_shuffled_gpu${gpu}.log 2>&1 &
    log "GPU $gpu SHUFFLED PID=$!"
done

log "Waiting for SHUFFLED workers..."
wait
DONE=0; FAILED=0
for gpu in 0 1 2 3 4 5 6 7; do
    lf="$LOGDIR/full_shuffled_gpu${gpu}.log"
    [ -f "$lf" ] && DONE=$((DONE + $(grep -c 'COMPLETE' "$lf" 2>/dev/null || echo 0)))
    [ -f "$lf" ] && FAILED=$((FAILED + $(grep -c 'FAILED' "$lf" 2>/dev/null || echo 0)))
done
log "SHUFFLED: $DONE/162 COMPLETE, $FAILED FAILED"
if [ "$FAILED" -gt 0 ] || [ "$DONE" -ne 162 ]; then
    log "FATAL: SHUFFLED gate FAILED"; exit 1
fi

log "UMA+SHUFFLED gates PASS. Advancing to Stage 5..."
if [ -x "$NEXT_SCRIPT" ]; then
    exec bash "$NEXT_SCRIPT"
else
    log "ERROR: $NEXT_SCRIPT not found"; exit 1
fi
