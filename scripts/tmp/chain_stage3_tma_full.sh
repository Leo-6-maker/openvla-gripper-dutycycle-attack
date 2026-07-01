#!/bin/bash
# CHAIN Stage 3: Launch TMA Full (Student + Random-Time), sleep 3h, call stage4
set -uo pipefail
EXEC=/mnt/sdc/dty_user/table1_sota_execution_v1
LOGDIR=$EXEC/logs; mkdir -p $LOGDIR
PYTHON=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
SOTA_WORKER=$EXEC/commands/run_sota_worker.py
SOTA_MF=$EXEC/manifests
NEXT_SCRIPT=$EXEC/commands/chain_stage4_uma_shuffled.sh

log() { echo "$(date -Iseconds) [STAGE3] $*" | tee -a $LOGDIR/chain.log; }

log "=== LAUNCHING TMA FULL (Student GPU0-3 + RandomTime GPU4-7) ==="
# TMA Student: GPU 0-3 (4 GPUs × ~21 jobs = ~84 min), TMA Random-Time: GPU 4-7
for pair in "TMA:tma:0:4" "TMA_RANDOM_TIME:tma_random_time:4:8"; do
    IFS=':' read -r cond label g_start g_end <<< "$pair"
    for gpu in $(seq $g_start $((g_end - 1))); do
        MF=$SOTA_MF/$cond/manifest_gpu${gpu}.jsonl
        [ -f "$MF" ] && nohup $PYTHON -u $SOTA_WORKER $gpu $MF > $LOGDIR/full_${label}_gpu${gpu}.log 2>&1 &
        log "GPU $gpu ($cond) PID=$!"
    done
done

TMA_MIN=120
log "Sleeping ${TMA_MIN}min for TMA full (324 jobs × 4min/4GPU ≈ 84min + buffer)..."
sleep $((TMA_MIN * 60))

DONE=0; FAILED=0
for label in tma tma_random_time; do
    for gpu in 0 1 2 3 4 5 6 7; do
        lf="$LOGDIR/full_${label}_gpu${gpu}.log"
        [ -f "$lf" ] && DONE=$((DONE + $(grep -c 'COMPLETE' "$lf" 2>/dev/null || echo 0)))
        [ -f "$lf" ] && FAILED=$((FAILED + $(grep -c 'FAILED' "$lf" 2>/dev/null || echo 0)))
    done
done
log "TMA Full: $DONE/324 COMPLETE, $FAILED FAILED"

if [ -x "$NEXT_SCRIPT" ]; then
    log "Calling Stage 4..."
    exec bash "$NEXT_SCRIPT"
else
    log "ERROR: Next script not found: $NEXT_SCRIPT"
fi
