#!/bin/bash
# set -e removed — one GPU OOM must not kill other shards
REPO=/mnt/sdc/dty_user/openvla_attack
PY=$REPO/envs/openvla-official-a800/bin/python3
BRIDGE=$REPO/scripts/stageb/run_v2_vis_sc5_mlp_bridge.py
CKPT=$REPO/artifacts/detector/sc5_mlp_s2.pt
OUTBASE=$REPO/evidence/m1c/sc5_v2_primary

export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HOME=/mnt/sdc/dty_user/openvla_attack/sandbox_home
export TMPDIR=/mnt/sdc/dty_user/openvla_attack/tmp
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager
export OPENVLA_MODEL_PATH=$REPO/models/openvla-7b-finetuned-libero-object

run_batch() {
    local gpu=$1 pool=$2 state_start=$3 state_end=$4
    local total=$(( (state_end - state_start + 1) * 10 ))
    local count=0
    for task in 0 1 2 3 4 5 6 7 8 9; do
        for state in $(seq $state_start $state_end); do
            local out=$OUTBASE/$pool/task${task}_state${state}
            if [ -f "$out/.done" ]; then
                count=$((count + 1))
                continue
            fi
            if [ -f "$out/RUNNING" ]; then
                continue
            fi
            count=$((count + 1))
            echo "=== [GPU$gpu] [$count/$total] $pool task=$task state=$state $(date) ==="
            rm -rf "$out"
            mkdir -p "$out"
            touch "$out/RUNNING"
            CUDA_VISIBLE_DEVICES=$gpu $PY $BRIDGE \
                --condition CLEAN --state_id $state --task_idx $task \
                --anchor 0 --seed_id 42 --output_dir "$out" \
                --render_gpu $gpu --mlp_path $CKPT \
                > "$out/stdout.log" 2>"$out/stderr.log"
            local rc; rc=$?
            if [ $rc -eq 0 ]; then
                local tsha=$(sha256sum "$out/step_telemetry.csv" 2>/dev/null | cut -d' ' -f1)
                echo "{\"exit_code\":0,\"telemetry_sha\":\"$tsha\",\"completed\":\"$(date -Iseconds)\"}" > "$out/.done"
            else
                echo "{\"exit_code\":$rc,\"error\":\"non-zero exit\"}" > "$out/.done"
            fi
            rm -f "$out/RUNNING"
        done
    done
    echo "=== GPU$gpu BATCH DONE $(date) ==="
}

echo "=== SC5-V2 PRIMARY COLLECTION $(date) ==="
echo "GPU0: train 3-5 (30)  GPU1: train 6-8 (30)  GPU2: train 9-11 (30)"
echo "GPU3: train 12-14 (30)  GPU4: train 15-17 (30)  GPU5: train 18-20 (30)"
echo "GPU7: train 21-22(20) + dev 23-27(50) = 70"
echo "Total: 7 GPUs, 250 cells"

run_batch 0 train 3 5 &
run_batch 1 train 6 8 &
run_batch 2 train 9 11 &
run_batch 3 train 12 14 &
run_batch 4 train 15 17 &
run_batch 5 train 18 20 &
run_batch 7 train 21 22 &
PID=$!
wait $PID
# Dev after train completes (reuse GPU0)
run_batch 0 dev 23 27
echo "=== DEV DONE $(date) ==="

wait
echo "=== ALL DONE $(date) ==="
