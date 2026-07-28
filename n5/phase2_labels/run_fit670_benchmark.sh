#!/bin/bash
# FIT670 Concurrency Benchmark — 1/2/4/6/8 worker throughput test
# Non-consumable: each worker collects 1 libero_10 episode (~520 steps)
set -e

WORKTREE=/tmp/fresh670_v5_worktree
PYTHON=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
OUT=/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fresh670_v5
ALLOWLIST=/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fit670_allowlist/FIT670_IDENTITY_ALLOWLIST.json
TRANSITION=/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fit670_transition

echo "=== FIT670 CONCURRENCY BENCHMARK ==="
echo "Started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""

run_config() {
    local N=$1
    local GPUS=$2
    local LABEL="bench_${N}w"
    echo "====== $N workers (GPUs: $GPUS) ======"
    echo "Start: $(date -u '+%H:%M:%S')"

    rm -rf "$OUT" 2>/dev/null
    PIDS=()

    for GPU in $GPUS; do
        local LOG="/tmp/bench_${N}w_gpu_${GPU}.log"
        $PYTHON -u $WORKTREE/n5/phase2_labels/run_fit670_atomic_worker.py \
            --shard-id $GPU --gpu $GPU \
            --model-path /mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10 \
            --official-worker /mnt/sdc/dty_user/openvla_attack_official_v3_20260716/scripts/official_clean_worker.py \
            --transition-receipt $TRANSITION \
            --identity-allowlist $ALLOWLIST \
            --shard-plan /tmp/fit670_shard_plan.json \
            --registry-root /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_v2_r7/run_A/per_task \
            --alias-ledger /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_v2_r7/run_A/ALIAS_LEDGER.json \
            --upstream-root /mnt/sdc/dty_user/openvla_attack \
            --output-root $OUT --seed 20260717 \
            > $LOG 2>&1 &
        PIDS+=($!)
    done

    # Wait for all
    for PID in ${PIDS[@]}; do wait $PID; done

    echo "End: $(date -u '+%H:%M:%S')"

    # Report
    echo "--- Results ---"
    for GPU in $GPUS; do
        local LOG="/tmp/bench_${N}w_gpu_${GPU}.log"
        echo -n "GPU $GPU: "
        grep 'steps=' $LOG | tail -1
        grep 'cuda=' $LOG | tail -3
    done

    # Aggregate
    local TOTAL_STEPS=$(for GPU in $GPUS; do
        LOG="/tmp/bench_${N}w_gpu_${GPU}.log"
        grep 'steps=' $LOG | tail -1 | grep -oP 'steps=\K[0-9]+' || echo 0
    done | paste -sd+ | bc)
    echo "Total steps: $TOTAL_STEPS"
    echo ""
}

# Run configurations
# 1 worker: GPU 0 only
run_config 1 "0"

# 2 workers: GPUs 0,1
run_config 2 "0 1"

# 4 workers: GPUs 0,2,4,6 (spread across NUMA)
run_config 4 "0 2 4 6"

# 6 workers: GPUs 0,1,2,3,4,5
run_config 6 "0 1 2 3 4 5"

# 8 workers: all
run_config 8 "0 1 2 3 4 5 6 7"

echo "=== BENCHMARK COMPLETE ==="
echo "Finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
