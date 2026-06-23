#!/bin/bash
set -e
REPO=/mnt/sdc/dty_user/openvla_attack
PY=$REPO/envs/openvla-official-a800/bin/python3
BRIDGE=$REPO/scripts/stageb/run_v2_vis_sc5_mlp_bridge.py
CKPT=$REPO/artifacts/detector/sc5_mlp_s2.pt
OUTBASE=$REPO/evidence/m1c/object_clean_corpus

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

echo "=== M1C COLLECTION V4 (non-overlapping, fixed rc) $(date) ==="
echo "GPU2: train 3-9 (70)  GPU5: train 10-15 (60)  GPU3: train 16-27 (120)  GPU4: val 28-37 (100)"
echo "Total: 4 GPUs, 350 cells"

run_batch 2 train 3 9 &
PID2=$!
run_batch 5 train 10 15 &
PID5=$!
run_batch 3 train 16 27 &
PID3=$!
run_batch 4 validation 28 37 &
PID4=$!

wait $PID2 $PID3 $PID4 $PID5
echo "=== ALL TRAIN+VAL DONE $(date) ==="
