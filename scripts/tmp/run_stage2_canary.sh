#!/bin/bash
# Stage 2: Canary wave launcher + fixed-duration wait
set -uo pipefail
EXEC=/mnt/sdc/dty_user/table1_sota_execution_v1
LOGDIR=$EXEC/logs
PYTHON=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
SOTA_WORKER=$EXEC/commands/run_sota_worker.py
CANARY=$EXEC/canary

log() { echo "$(date -Iseconds) $*" | tee -a $LOGDIR/stage2.log; }

log "=== STAGE 2: CANARY WAVE ==="

for pair in "0:TMA:tma_student" "1:TMA_RANDOM_TIME:tma_random" "2:UMA:uma" "3:SHUFFLED:shuffled"; do
    IFS=':' read -r gpu cond label <<< "$pair"
    MF=$CANARY/$cond/manifest_canary.jsonl
    if [ -f "$MF" ]; then
        nohup $PYTHON -u $SOTA_WORKER $gpu $MF > $LOGDIR/canary_${label}_gpu${gpu}.log 2>&1 &
        log "GPU $gpu ($cond) PID=$!"
    fi
done

# Wait fixed duration: 9 jobs × 5 min = 45 min, add 15 min buffer = 60 min
SLEEP_SEC=3600
log "Sleeping ${SLEEP_SEC}s for canary completion..."
sleep $SLEEP_SEC

# Check results
DONE=0; FAILED=0
for label in tma_student tma_random uma shuffled; do
    for gpu in 0 1 2 3; do
        lf="$LOGDIR/canary_${label}_gpu${gpu}.log"
        if [ -f "$lf" ]; then
            DONE=$((DONE + $(grep -c 'COMPLETE' "$lf" 2>/dev/null || echo 0)))
            FAILED=$((FAILED + $(grep -c 'FAILED' "$lf" 2>/dev/null || echo 0)))
        fi
    done
done
log "Canary results: $DONE/36 COMPLETE, $FAILED FAILED"
log "=== STAGE 2 DONE ==="
