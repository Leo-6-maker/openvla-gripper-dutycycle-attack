#!/bin/bash
# CHAIN Stage 4: UMA Full (8 GPU, 162) → SHUFFLED Full (8 GPU, 162) sequential, hard gates
set -uo pipefail
EXEC=/mnt/sdc/dty_user/table1_sota_execution_v1; LOGDIR=$EXEC/logs; mkdir -p $LOGDIR
PYTHON=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
SOTA_WORKER=$EXEC/commands/run_sota_worker.py
SOTA_MF=$EXEC/manifests; NEXT_SCRIPT=$EXEC/commands/chain_stage5_aggregate.sh
VALIDATOR=$EXEC/commands/validate_sota_condition.py
MF_BUILDER=$EXEC/commands/build_formal_manifest.py
FORMAL_PASS_DIR=$EXEC/formal_pass; CANARY_PASS_DIR=$EXEC/canary_pass
FORMAL_MF_DIR=$EXEC/manifests/formal
mkdir -p $FORMAL_PASS_DIR $FORMAL_MF_DIR
CURRENT_COMMIT=$(cd /mnt/sdc/dty_user/openvla_attack && git rev-parse HEAD 2>/dev/null || echo "unknown")

log() { echo "$(date -Iseconds) [STAGE4] $*" | tee -a $LOGDIR/chain.log; }

# ── Authorization gate: SHA re-verification for UMA + SHUFFLED ──
for pair in "UMA_STUDENT:UMA" "SHUFFLED_STUDENT:SHUFFLED"; do
    IFS=':' read -r cond_name cond_ns <<< "$pair"
    pass_file="$CANARY_PASS_DIR/${cond_name}_CANARY_PASS.json"
    if [ ! -f "$pass_file" ]; then
        log "FATAL: Missing canary authorization: $pass_file"; exit 1
    fi
    pass_commit=$(python3 -c "import json; print(json.load(open('$pass_file')).get('commit_sha',''))" 2>/dev/null)
    pass_bridge=$(python3 -c "import json; print(json.load(open('$pass_file')).get('bridge_sha256',''))" 2>/dev/null)
    pass_worker=$(python3 -c "import json; print(json.load(open('$pass_file')).get('worker_sha256',''))" 2>/dev/null)
    current_bridge=$(python3 -c "import hashlib; print(hashlib.sha256(open('/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_v2_vis_sc5_mlp_bridge.py','rb').read()).hexdigest())")
    current_worker=$(python3 -c "import hashlib; print(hashlib.sha256(open('/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_sota_worker.py','rb').read()).hexdigest())")
    gate_pass=$(python3 -c "import json; print(json.load(open('$pass_file')).get('gate_pass', False))")

    sha_mismatch=0
    [ "$pass_commit" != "$CURRENT_COMMIT" ] && { log "  SHA mismatch: commit"; sha_mismatch=1; }
    [ "$pass_bridge" != "$current_bridge" ] && { log "  SHA mismatch: bridge"; sha_mismatch=1; }
    [ "$pass_worker" != "$current_worker" ] && { log "  SHA mismatch: worker"; sha_mismatch=1; }
    [ "$gate_pass" != "True" ] && { log "  Gate not passed"; sha_mismatch=1; }

    if [ "$sha_mismatch" -ne 0 ]; then
        log "FATAL: SHA verification failed for $cond_name"; exit 1
    fi
    log "Authorization: $(basename $pass_file) SHA-verified PASS"
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
python3 "$MF_BUILDER" "$SOTA_MF/UMA" "$FORMAL_MF_DIR/UMA_formal_162.jsonl" || { log "FATAL: manifest build failed"; exit 1; }
log "Running formal validator on UMA..."
python3 "$VALIDATOR" --condition UMA_STUDENT --manifest "$FORMAL_MF_DIR/UMA_formal_162.jsonl" \
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
python3 "$MF_BUILDER" "$SOTA_MF/SHUFFLED" "$FORMAL_MF_DIR/SHUFFLED_formal_162.jsonl" || { log "FATAL: manifest build failed"; exit 1; }
log "Running formal validator on SHUFFLED..."
python3 "$VALIDATOR" --condition SHUFFLED_STUDENT --manifest "$FORMAL_MF_DIR/SHUFFLED_formal_162.jsonl" \
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
