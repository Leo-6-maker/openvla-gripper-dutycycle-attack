#!/bin/bash
# CHAIN Stage 3: TMA Full — Student (8 GPU) → Random-Time (8 GPU) sequential, hard gates
set -uo pipefail
EXEC=/mnt/sdc/dty_user/table1_sota_execution_v1; LOGDIR=$EXEC/logs; mkdir -p $LOGDIR
PYTHON=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
SOTA_WORKER=$EXEC/commands/run_sota_worker.py
SOTA_MF=$EXEC/manifests; NEXT_SCRIPT=$EXEC/commands/chain_stage4_uma_shuffled.sh

log() { echo "$(date -Iseconds) [STAGE3] $*" | tee -a $LOGDIR/chain.log; }

# ── Authorization gate: check CANARY_PASS files, verify SHA bindings ──
CANARY_PASS_DIR=$EXEC/canary_pass
VALIDATOR=$EXEC/commands/validate_sota_condition.py
MF_BUILDER=$EXEC/commands/build_formal_manifest.py
FORMAL_PASS_DIR=$EXEC/formal_pass; FORMAL_MF_DIR=$EXEC/manifests/formal
mkdir -p $FORMAL_PASS_DIR $FORMAL_MF_DIR
CURRENT_COMMIT=$(cd /mnt/sdc/dty_user/openvla_attack && git rev-parse HEAD 2>/dev/null || echo "unknown")

for pair in "TMA_STUDENT:TMA" "TMA_RANDOM_TIME:TMA_RANDOM_TIME"; do
    IFS=':' read -r cond_name cond_ns <<< "$pair"
    pass_file="$CANARY_PASS_DIR/${cond_name}_CANARY_PASS.json"
    if [ ! -f "$pass_file" ]; then
        log "FATAL: Missing canary authorization: $pass_file"; exit 1
    fi
    # Full SHA re-verification
    pass_commit=$(python3 -c "import json; print(json.load(open('$pass_file')).get('commit_sha',''))" 2>/dev/null)
    pass_bridge=$(python3 -c "import json; print(json.load(open('$pass_file')).get('bridge_sha256',''))" 2>/dev/null)
    pass_worker=$(python3 -c "import json; print(json.load(open('$pass_file')).get('worker_sha256',''))" 2>/dev/null)
    pass_validator=$(python3 -c "import json; print(json.load(open('$pass_file')).get('validator_sha256',''))" 2>/dev/null)
    current_bridge=$(python3 -c "import hashlib; print(hashlib.sha256(open('/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_v2_vis_sc5_mlp_bridge.py','rb').read()).hexdigest())")
    current_worker=$(python3 -c "import hashlib; print(hashlib.sha256(open('/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_sota_worker.py','rb').read()).hexdigest())")
    current_validator=$(python3 -c "import hashlib; print(hashlib.sha256(open('$VALIDATOR','rb').read()).hexdigest())")
    gate_pass=$(python3 -c "import json; print(json.load(open('$pass_file')).get('gate_pass', False))")

    sha_mismatch=0
    [ "$pass_commit" != "$CURRENT_COMMIT" ] && { log "  SHA mismatch: commit ($pass_commit vs $CURRENT_COMMIT)"; sha_mismatch=1; }
    [ "$pass_bridge" != "$current_bridge" ] && { log "  SHA mismatch: bridge"; sha_mismatch=1; }
    [ "$pass_worker" != "$current_worker" ] && { log "  SHA mismatch: worker"; sha_mismatch=1; }
    [ "$pass_validator" != "$current_validator" ] && { log "  SHA mismatch: validator"; sha_mismatch=1; }
    [ "$gate_pass" != "True" ] && { log "  Gate not passed"; sha_mismatch=1; }

    if [ "$sha_mismatch" -ne 0 ]; then
        log "FATAL: SHA verification failed for $cond_name"; exit 1
    fi
    log "Authorization: $(basename $pass_file) SHA-verified PASS"
done

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
    log "FATAL: TMA Student process gate FAILED"; exit 1
fi

# Build formal manifest + formal validator for TMA Student
log "Building formal manifest for TMA Student..."
python3 "$MF_BUILDER" "$SOTA_MF/TMA" "$FORMAL_MF_DIR/TMA_formal_162.jsonl" || { log "FATAL: manifest build failed"; exit 1; }
log "Running formal validator on TMA Student..."
python3 "$VALIDATOR" --condition TMA_STUDENT --manifest "$FORMAL_MF_DIR/TMA_formal_162.jsonl" \
    --artifact_root /mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1 \
    --expected 162 --mode formal \
    --output "$FORMAL_PASS_DIR/TMA_STUDENT_FORMAL_PASS.json" || {
    log "FATAL: TMA Student formal validation FAILED"; exit 1
}
log "TMA Student formal validator PASS"

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
    log "FATAL: TMA Random-Time process gate FAILED"; exit 1
fi

# Build formal manifest + formal validator for TMA Random-Time
log "Building formal manifest for TMA Random-Time..."
python3 "$MF_BUILDER" "$SOTA_MF/TMA_RANDOM_TIME" "$FORMAL_MF_DIR/TMA_RANDOM_TIME_formal_162.jsonl" || { log "FATAL: manifest build failed"; exit 1; }
log "Running formal validator on TMA Random-Time..."
python3 "$VALIDATOR" --condition TMA_RANDOM_TIME --manifest "$FORMAL_MF_DIR/TMA_RANDOM_TIME_formal_162.jsonl" \
    --artifact_root /mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1 \
    --expected 162 --mode formal \
    --output "$FORMAL_PASS_DIR/TMA_RANDOM_TIME_FORMAL_PASS.json" || {
    log "FATAL: TMA Random-Time formal validation FAILED"; exit 1
}
log "TMA Random-Time formal validator PASS"

log "TMA Full gates PASS. Advancing to Stage 4..."
if [ -x "$NEXT_SCRIPT" ]; then
    exec bash "$NEXT_SCRIPT"
else
    log "ERROR: $NEXT_SCRIPT not found"; exit 1
fi
