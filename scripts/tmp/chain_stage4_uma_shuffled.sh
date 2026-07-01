#!/bin/bash
# CHAIN Stage 4: Launch UMA + SHUFFLED Full, sleep 3h, call stage5
set -uo pipefail
EXEC=/mnt/sdc/dty_user/table1_sota_execution_v1
LOGDIR=$EXEC/logs; mkdir -p $LOGDIR
PYTHON=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
SOTA_WORKER=$EXEC/commands/run_sota_worker.py
SOTA_MF=$EXEC/manifests
NEXT_SCRIPT=$EXEC/commands/chain_stage5_aggregate.sh

log() { echo "$(date -Iseconds) [STAGE4] $*" | tee -a $LOGDIR/chain.log; }

log "=== LAUNCHING UMA (GPU0-3) + SHUFFLED (GPU4-7) ==="
for pair in "UMA:uma:0:4" "SHUFFLED:shuffled:4:8"; do
    IFS=':' read -r cond label g_start g_end <<< "$pair"
    for gpu in $(seq $g_start $((g_end - 1))); do
        MF=$SOTA_MF/$cond/manifest_gpu${gpu}.jsonl
        [ -f "$MF" ] && nohup $PYTHON -u $SOTA_WORKER $gpu $MF > $LOGDIR/full_${label}_gpu${gpu}.log 2>&1 &
        log "GPU $gpu ($cond) PID=$!"
    done
done

US_MIN=180
log "Sleeping ${US_MIN}min for UMA+SHUFFLED (324 jobs)..."
sleep $((US_MIN * 60))

DONE=0; FAILED=0
for label in uma shuffled; do
    for gpu in 0 1 2 3 4 5 6 7; do
        lf="$LOGDIR/full_${label}_gpu${gpu}.log"
        [ -f "$lf" ] && DONE=$((DONE + $(grep -c 'COMPLETE' "$lf" 2>/dev/null || echo 0)))
        [ -f "$lf" ] && FAILED=$((FAILED + $(grep -c 'FAILED' "$lf" 2>/dev/null || echo 0)))
    done
done
log "UMA+SHUFFLED: $DONE/324 COMPLETE, $FAILED FAILED"

if [ -x "$NEXT_SCRIPT" ]; then
    log "Calling Stage 5..."
    exec bash "$NEXT_SCRIPT"
else
    log "ERROR: Next script not found: $NEXT_SCRIPT"
fi
