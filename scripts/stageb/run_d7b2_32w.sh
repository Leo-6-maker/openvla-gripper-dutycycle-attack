#!/bin/bash
# D7B2: 32 workers (4/GPU) — staggered 4 waves
set -e

PYTHON=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3.10
PKG=/mnt/sdc/dty_user/openvla_attack_evidence/condition_matrix/d4c2e3_25d_baseline_package
MANIFEST=/mnt/sdc/dty_user/openvla_attack_evidence/condition_matrix/d7_table1_manifest/d7_table1_queue_manifest.csv
OUT=/mnt/sdc/dty_user/openvla_attack_evidence/condition_matrix/d7b2_table1_normalized_rollout
COMMIT=c94d809
SCRIPT=/mnt/sdc/dty_user/openvla_attack_codex_tools_pr50_f3e6b0/scripts/stageb/run_d7_table1_persistent_worker.py

rm -rf "$OUT"; mkdir -p "$OUT"
echo "=== D7B2: 32 workers (4/GPU) ==="
echo "commit=$COMMIT"

launch_one() {
    local gpu=$1 start=$2 end=$3
    local log="$OUT/worker_gpu${gpu}_r${start}.log"
    echo "[GPU $gpu] rows $start-$end"
    CUDA_VISIBLE_DEVICES=$gpu nohup $PYTHON -u "$SCRIPT" \
      --gpu $gpu --detector-package "$PKG" --manifest "$MANIFEST" \
      --output-root "$OUT" --source-commit "$COMMIT" \
      --start-row $start --end-row $end \
      > "$log" 2>&1 &
}

# 32 workers = 716/32 = 22 rows each (last gets remainder)
# Wave 1: GPU 0-7, first worker each
echo "=== Wave 1 ==="
launch_one 0 0 23;    launch_one 1 23 45;   launch_one 2 45 67
launch_one 3 67 90;   launch_one 4 90 112;  launch_one 5 112 135
launch_one 6 135 157; launch_one 7 157 180
sleep 120

# Wave 2: GPU 0-7, second worker each
echo "=== Wave 2 ==="
launch_one 0 180 202; launch_one 1 202 225; launch_one 2 225 247
launch_one 3 247 270; launch_one 4 270 292; launch_one 5 292 315
launch_one 6 315 337; launch_one 7 337 360
sleep 120

# Wave 3: GPU 0-7, third worker each
echo "=== Wave 3 ==="
launch_one 0 360 382; launch_one 1 382 405; launch_one 2 405 427
launch_one 3 427 450; launch_one 4 450 472; launch_one 5 472 495
launch_one 6 495 517; launch_one 7 517 540
sleep 120

# Wave 4: GPU 0-7, fourth worker each
echo "=== Wave 4 ==="
launch_one 0 540 562; launch_one 1 562 585; launch_one 2 585 607
launch_one 3 607 630; launch_one 4 630 652; launch_one 5 652 675
launch_one 6 675 697; launch_one 7 697 716

sleep 60
echo ""
echo "=== Launch complete: 32 workers ==="
for gpu in 0 1 2 3 4 5 6 7; do
    n=$(nvidia-smi --id=$gpu --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    mem=$(nvidia-smi --id=$gpu --query-gpu=memory.used --format=csv,noheader 2>/dev/null)
    echo "GPU $gpu: $n processes, $mem"
done
