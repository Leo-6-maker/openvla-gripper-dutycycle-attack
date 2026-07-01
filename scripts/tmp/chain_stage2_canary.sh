#!/bin/bash
# CHAIN Stage 2: Launch canary wave (4 conditions × 1 GPU each), wait for PID, validate, call stage3
set -uo pipefail
EXEC=/mnt/sdc/dty_user/table1_sota_execution_v1; LOGDIR=$EXEC/logs; mkdir -p $LOGDIR
PYTHON=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
SOTA_WORKER=$EXEC/commands/run_sota_worker.py
CANARY=$EXEC/canary; NEXT_SCRIPT=$EXEC/commands/chain_stage3_tma_full.sh

log() { echo "$(date -Iseconds) [STAGE2] $*" | tee -a $LOGDIR/chain.log; }

log "=== LAUNCHING CANARY WAVE (4 GPUs, 1 per condition) ==="
declare -A PIDS
for pair in "0:TMA:tma_student" "1:TMA_RANDOM_TIME:tma_random" "2:UMA:uma" "3:SHUFFLED:shuffled"; do
    IFS=':' read -r gpu cond label <<< "$pair"
    MF=$CANARY/$cond/manifest_canary.jsonl
    if [ -f "$MF" ]; then
        nohup $PYTHON -u $SOTA_WORKER $gpu $MF > $LOGDIR/canary_${label}_gpu${gpu}.log 2>&1 &
        PIDS[$cond]=$!
        log "GPU $gpu ($cond) PID=${PIDS[$cond]}"
    fi
done

# Wait for ALL canary workers to complete
log "Waiting for canary workers..."
FAILED_CONDS=""
for cond in TMA TMA_RANDOM_TIME UMA SHUFFLED; do
    pid="${PIDS[$cond]:-}"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        log "  Waiting for $cond (PID $pid)..."
        if ! wait "$pid"; then
            FAILED_CONDS="$FAILED_CONDS $cond"
            log "  $cond worker FAILED"
        else
            log "  $cond worker DONE"
        fi
    fi
done

# Count results
TOTAL_DONE=0; TOTAL_FAILED=0
for label in tma_student tma_random uma shuffled; do
    for gpu in 0 1 2 3; do
        lf="$LOGDIR/canary_${label}_gpu${gpu}.log"
        [ -f "$lf" ] && TOTAL_DONE=$((TOTAL_DONE + $(grep -c 'COMPLETE' "$lf" 2>/dev/null || echo 0)))
        [ -f "$lf" ] && TOTAL_FAILED=$((TOTAL_FAILED + $(grep -c 'FAILED' "$lf" 2>/dev/null || echo 0)))
    done
done
log "Canary results: $TOTAL_DONE/36 COMPLETE, $TOTAL_FAILED FAILED"

# Hard gate
if [ -n "$FAILED_CONDS" ] || [ "$TOTAL_FAILED" -gt 0 ] || [ "$TOTAL_DONE" -ne 36 ]; then
    log "FATAL: Canary gate FAILED — not advancing to Stage 3"
    log "  Failed conditions:$FAILED_CONDS"
    log "  Done=$TOTAL_DONE/36 Failed=$TOTAL_FAILED"
    exit 1
fi

log "Canary gate PASS. Advancing to Stage 3..."
if [ -x "$NEXT_SCRIPT" ]; then
    exec bash "$NEXT_SCRIPT"
else
    log "ERROR: $NEXT_SCRIPT not found"
    exit 1
fi
