#!/bin/bash
# CHAIN Stage 4: UMA Full (8 GPU, 162) → SHUFFLED Full (8 GPU, 162) sequential, hard gates
set -uo pipefail
EXEC=/mnt/sdc/dty_user/table1_sota_execution_v1; LOGDIR=$EXEC/logs; mkdir -p $LOGDIR
PYTHON=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
SOTA_WORKER=$EXEC/commands/run_sota_worker.py
SOTA_MF=$EXEC/manifests; NEXT_SCRIPT=$EXEC/commands/chain_stage5_aggregate.sh
VALIDATOR=$EXEC/commands/validate_sota_condition.py
FORMAL_PASS_DIR=$EXEC/formal_pass; CANARY_PASS_DIR=$EXEC/canary_pass
mkdir -p $FORMAL_PASS_DIR

log() { echo "$(date -Iseconds) [STAGE4] $*" | tee -a $LOGDIR/chain.log; }

# ── Authorization gate: check UMA + SHUFFLED CANARY_PASS ──
for pair in "UMA_STUDENT:UMA" "SHUFFLED_STUDENT:SHUFFLED"; do
    IFS=':' read -r cond_name cond_ns <<< "$pair"
    pass_file="$CANARY_PASS_DIR/${cond_name}_CANARY_PASS.json"
    if [ ! -f "$pass_file" ]; then
        log "FATAL: Missing canary authorization: $pass_file"; exit 1
    fi
    gate=$(python3 -c "import json; print(json.load(open('$pass_file')).get('gate_pass', False))" 2>/dev/null || echo "False")
    if [ "$gate" != "True" ]; then
        log "FATAL: Canary gate not passed: $pass_file"; exit 1
    fi
    log "Authorization: $(basename $pass_file) PASS"
done

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
    log "FATAL: UMA process gate FAILED"; exit 1
fi
# Formal validator
log "Running formal validator on UMA..."
python3 "$VALIDATOR" --condition UMA_STUDENT --manifest "$SOTA_MF/UMA/manifest_gpu0.jsonl" \
    --artifact_root /mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1 \
    --expected 162 --mode formal \
    --output "$FORMAL_PASS_DIR/UMA_STUDENT_FORMAL_PASS.json" || {
    log "FATAL: UMA formal validation FAILED"; exit 1
}
log "UMA formal validator PASS"

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
    log "FATAL: SHUFFLED process gate FAILED"; exit 1
fi
# Formal validator
log "Running formal validator on SHUFFLED..."
python3 "$VALIDATOR" --condition SHUFFLED_STUDENT --manifest "$SOTA_MF/SHUFFLED/manifest_gpu0.jsonl" \
    --artifact_root /mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1 \
    --expected 162 --mode formal \
    --output "$FORMAL_PASS_DIR/SHUFFLED_STUDENT_FORMAL_PASS.json" || {
    log "FATAL: SHUFFLED formal validation FAILED"; exit 1
}
log "SHUFFLED formal validator PASS"

log "UMA+SHUFFLED gates PASS. Advancing to Stage 5..."
if [ -x "$NEXT_SCRIPT" ]; then
    exec bash "$NEXT_SCRIPT"
else
    log "ERROR: $NEXT_SCRIPT not found"; exit 1
fi
