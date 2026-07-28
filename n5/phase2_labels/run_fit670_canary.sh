#!/bin/bash
set -e
WORKTREE=/tmp/fresh670_v5_worktree
PYTHON=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
OUT=/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fresh670_v5_r2_canary

echo "=== FIT670 V5 8-GPU CANARY ==="
echo "Started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

launch_worker() {
    local GPU=$1
    local LOG="/tmp/canary_gpu_${GPU}.log"
    echo "Launching GPU $GPU..."
    $PYTHON -u $WORKTREE/n5/phase2_labels/run_fit670_atomic_worker.py \
        --shard-id $GPU --gpu $GPU \
        --model-path /mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10 \
        --official-worker /mnt/sdc/dty_user/openvla_attack_official_v3_20260716/scripts/official_clean_worker.py \
        --transition-receipt /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fit670_transition \
        --identity-allowlist /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fit670_allowlist/FIT670_IDENTITY_ALLOWLIST.json \
        --shard-plan /tmp/fit670_shard_plan.json \
        --registry-root /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_v2_r7/run_A/per_task \
        --alias-ledger /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_v2_r7/run_A/ALIAS_LEDGER.json \
        --upstream-root /mnt/sdc/dty_user/openvla_attack \
        --output-root $OUT --seed 20260717 \
        --max-identities 1 \
        > $LOG 2>&1 &
    echo "  GPU $GPU PID: $!"
}

for GPU in 0 1 2 3 4 5 6 7; do
    launch_worker $GPU
done

echo ""
echo "All 8 workers launched. Waiting for completion..."
wait

echo ""
echo "=== CANARY COMPLETE ==="
echo "Finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

FAILS=0
for GPU in 0 1 2 3 4 5 6 7; do
    LOG="/tmp/canary_gpu_${GPU}.log"
    echo "--- GPU $GPU ---"
    grep -E 'steps=|HOLD|ERROR|Traceback|SKIP|sealed|Worker shard' $LOG | tail -5
    if grep -q 'ERROR\|Traceback\|HOLD' $LOG 2>/dev/null; then
        FAILS=$((FAILS+1))
    fi
done
echo ""
echo "Failed workers: $FAILS / 8"
exit $FAILS
