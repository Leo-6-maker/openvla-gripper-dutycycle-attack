#!/bin/bash
# Supervisor: Oracle completion → audit → SOTA canary manifests → SOTA chain
set -uo pipefail

EXEC=/mnt/sdc/dty_user/table1_sota_execution_v1
LOGDIR=$EXEC/logs; mkdir -p $LOGDIR
PYTHON=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
ORACLE_DIR=/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/COMMAND_OPEN_ORACLE_T10/formal_v1

log() { echo "$(date -Iseconds) [SUPERVISOR] $*" | tee -a $LOGDIR/supervisor_oracle_sota.log; }

log "===== ORACLE→SOTA SUPERVISOR PID=$$ ====="

# ═══ Stage 0: Wait for Oracle completion ═══
log "Stage 0: Waiting for Oracle 141/141..."
while true; do
    ORACLE_EP=$(find $ORACLE_DIR -name 'episode_summary.json' 2>/dev/null | wc -l)
    ORACLE_WORKERS=$(ps aux | grep -c 'run_oracle_env_gripper_worker' 2>/dev/null || echo 0)
    ORACLE_FAILED=$(grep -r 'FAILED' $LOGDIR/oracle_full_gpu*.log 2>/dev/null | wc -l) || ORACLE_FAILED=0
    log "Oracle: $ORACLE_EP/141, workers=$ORACLE_WORKERS, failed=$ORACLE_FAILED"

    if [ "$ORACLE_WORKERS" -le 2 ] && [ "$ORACLE_EP" -ge 141 ]; then
        log "Oracle COMPLETE: $ORACLE_EP/141, failed=$ORACLE_FAILED"
        break
    fi
    sleep 120
done

# ═══ Stage 1: Oracle audit ═══
log "Stage 1: Running Oracle audit..."
if python3 $EXEC/commands/audit_oracle.py 2>&1 | tee -a $LOGDIR/supervisor_oracle_sota.log; then
    log "Oracle audit PASS — COMMAND_OPEN_ORACLE_V1 FROZEN"
else
    log "FATAL: Oracle audit FAILED — stopping"
    exit 1
fi

# Clean oracle workers
pkill -f 'run_oracle_env_gripper' 2>/dev/null || true
sleep 5

# ═══ Stage 2: Regenerate SOTA canary manifests ═══
log "Stage 2: Regenerating SOTA canary manifests with latest code..."
if python3 $EXEC/commands/prepare_sota_canaries.py 2>&1 | tee -a $LOGDIR/supervisor_oracle_sota.log; then
    log "SOTA manifests regenerated"
else
    log "FATAL: Manifest generation FAILED"
    exit 1
fi

# ═══ Stage 3: Clean old formal dirs + launch SOTA chain ═══
log "Stage 3: Cleaning old formal directories..."
for cond in TMA TMA_RANDOM_TIME UMA SHUFFLED; do
    rm -rf /mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/$cond/formal_v1/fold_* 2>/dev/null
    rm -rf /mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/$cond/canary_v1 2>/dev/null
done
rm -f $LOGDIR/canary_*.log $LOGDIR/full_*.log $LOGDIR/chain.log

log "Stage 3: Launching SOTA chain (canary → TMA full → UMA+SHUFFLED → aggregate)..."
CHAIN_START=$EXEC/commands/chain_stage2_canary.sh
if [ -x "$CHAIN_START" ]; then
    exec bash "$CHAIN_START"
else
    log "FATAL: $CHAIN_START not found"
    exit 1
fi
