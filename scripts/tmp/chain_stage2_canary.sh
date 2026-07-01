#!/bin/bash
# CHAIN Stage 2: Launch canary wave, sleep 60min, then call stage3
set -uo pipefail
EXEC=/mnt/sdc/dty_user/table1_sota_execution_v1
LOGDIR=$EXEC/logs; mkdir -p $LOGDIR
PYTHON=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
SOTA_WORKER=$EXEC/commands/run_sota_worker.py
CANARY=$EXEC/canary
NEXT_SCRIPT=$EXEC/commands/chain_stage3_tma_full.sh

log() { echo "$(date -Iseconds) [STAGE2] $*" | tee -a $LOGDIR/chain.log; }

log "=== LAUNCHING CANARY WAVE ==="
for pair in "0:TMA:tma_student" "1:TMA_RANDOM_TIME:tma_random" "2:UMA:uma" "3:SHUFFLED:shuffled"; do
    IFS=':' read -r gpu cond label <<< "$pair"
    MF=$CANARY/$cond/manifest_canary.jsonl
    [ -f "$MF" ] && nohup $PYTHON -u $SOTA_WORKER $gpu $MF > $LOGDIR/canary_${label}_gpu${gpu}.log 2>&1 &
    log "GPU $gpu ($cond) PID=$!"
done

CANARY_MIN=60
log "Sleeping ${CANARY_MIN}min for canary (9 jobs × 5min + buffer)..."
sleep $((CANARY_MIN * 60))

DONE=0; FAILED=0
for label in tma_student tma_random uma shuffled; do
    for gpu in 0 1 2 3; do
        lf="$LOGDIR/canary_${label}_gpu${gpu}.log"
        [ -f "$lf" ] && DONE=$((DONE + $(grep -c 'COMPLETE' "$lf" 2>/dev/null || echo 0)))
        [ -f "$lf" ] && FAILED=$((FAILED + $(grep -c 'FAILED' "$lf" 2>/dev/null || echo 0)))
    done
done
log "Canary: $DONE/36 COMPLETE, $FAILED FAILED"

if [ -x "$NEXT_SCRIPT" ]; then
    log "Calling Stage 3..."
    exec bash "$NEXT_SCRIPT"
else
    log "ERROR: Next script not found: $NEXT_SCRIPT"
fi
