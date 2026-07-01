#!/bin/bash
# TMA Canary Wave A — 4 conditions × 9-fold canary on 8 GPUs
#   GPU 0-1: TMA Student        (adapted TMA + student trigger)
#   GPU 2-3: TMA Random-Time     (adapted TMA + random-time)
#   GPU 4-5: UMA Student         (untargeted CE-PGD + student trigger)
#   GPU 6-7: SHUFFLED Student    (shuffled gradient + student trigger)
set -uo pipefail

WORKER=/mnt/sdc/dty_user/table1_sota_execution_v1/commands/run_sota_worker.py
CANARY=/mnt/sdc/dty_user/table1_sota_execution_v1/canary
LOGDIR=/mnt/sdc/dty_user/table1_sota_execution_v1/logs
PYTHON=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python

declare -A GPU_MAP
# TMA Student: GPU 0-1
GPU_MAP[0]="$CANARY/TMA/manifest_canary.jsonl"
GPU_MAP[1]="$CANARY/TMA/manifest_canary.jsonl"
# TMA Random-Time: GPU 2-3
GPU_MAP[2]="$CANARY/TMA_RANDOM_TIME/manifest_canary.jsonl"
GPU_MAP[3]="$CANARY/TMA_RANDOM_TIME/manifest_canary.jsonl"
# UMA Student: GPU 4-5
GPU_MAP[4]="$CANARY/UMA/manifest_canary.jsonl"
GPU_MAP[5]="$CANARY/UMA/manifest_canary.jsonl"
# SHUFFLED Student: GPU 6-7
GPU_MAP[6]="$CANARY/SHUFFLED/manifest_canary.jsonl"
GPU_MAP[7]="$CANARY/SHUFFLED/manifest_canary.jsonl"

echo "=== TMA CANARY WAVE A ==="
echo "Launch time: $(date -Iseconds)"

for gpu in 0 1 2 3 4 5 6 7; do
    MF="${GPU_MAP[$gpu]}"
    COND=$(basename $(dirname "$MF"))
    if [ -f "$MF" ]; then
        nohup $PYTHON -u $WORKER $gpu $MF > $LOGDIR/canary_${COND}_gpu${gpu}.log 2>&1 &
        echo "GPU $gpu ($COND): PID=$!"
    else
        echo "GPU $gpu ($COND): MANIFEST MISSING: $MF"
    fi
done

echo "=== ALL CANARIES LAUNCHED ==="
