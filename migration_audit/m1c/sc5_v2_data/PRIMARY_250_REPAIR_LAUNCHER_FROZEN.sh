#!/bin/bash
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

run_cell() {
    local gpu=$1 pool=$2 task=$3 state=$4
    local out=$OUTBASE/$pool/task${task}_state${state}
    echo "  [GPU$gpu] $pool task=$task state=$state $(date)"
    rm -rf "$out"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=$gpu $PY $BRIDGE \
        --condition CLEAN --state_id $state --task_idx $task \
        --anchor 0 --seed_id 42 --output_dir "$out" \
        --render_gpu $gpu --mlp_path $CKPT \
        > "$out/stdout.log" 2>"$out/stderr.log"
    local rc; rc=$?
    if [ $rc -eq 0 ]; then
        local tsha=$(sha256sum "$out/step_telemetry.csv" 2>/dev/null | cut -d' ' -f1)
        if [ -n "$tsha" ]; then
            echo "{\"exit_code\":0,\"telemetry_sha\":\"$tsha\",\"completed\":\"$(date -Iseconds)\"}" > "$out/.done"
            echo "    OK steps=$(wc -l < $out/step_telemetry.csv) sha=$tsha"
        else
            echo "{\"exit_code\":0,\"error\":\"no telemetry file\"}" > "$out/.done"
            echo "    FAIL: no telemetry produced"
        fi
    else
        echo "{\"exit_code\":$rc,\"error\":\"non-zero exit\"}" > "$out/.done"
        echo "    FAIL rc=$rc"
    fi
}

# Cell list
CELLS=(
"train 0 4" "train 0 5" "train 0 6" "train 0 8" "train 0 9"
"train 0 10" "train 0 12" "train 0 13" "train 0 14" "train 0 16"
"train 0 17" "train 0 18" "train 0 20" "train 0 21" "train 0 22"
"train 1 3" "train 1 4" "train 1 5" "train 1 8" "train 1 9"
"train 1 10" "train 1 11" "train 1 12" "train 1 13" "train 1 16"
"train 1 17" "train 1 18" "train 1 19" "train 1 20" "train 1 21"
"train 1 22"
"train 2 3" "train 2 4" "train 2 5" "train 2 6" "train 2 7"
"train 2 8" "train 2 9" "train 2 11" "train 2 12" "train 2 13"
"train 2 14" "train 2 15" "train 2 16" "train 2 17" "train 2 18"
"train 2 19" "train 2 20" "train 2 21" "train 2 22"
"train 3 3" "train 3 4" "train 3 5" "train 3 6" "train 3 7"
"train 3 8" "train 3 9" "train 3 10" "train 3 11" "train 3 12"
"train 3 13" "train 3 14" "train 3 15" "train 3 16" "train 3 21"
"train 3 22"
"train 4 3" "train 4 4" "train 4 9" "train 4 12" "train 4 21"
"train 4 22"
"train 5 21"
"train 6 21" "train 6 22"
"train 7 21" "train 7 22"
"train 8 21" "train 8 22"
"train 9 22"
"dev 0 24" "dev 0 25" "dev 0 26" "dev 0 27"
"dev 1 23" "dev 1 24" "dev 1 26" "dev 1 27"
"dev 2 23" "dev 2 24" "dev 2 25" "dev 2 26" "dev 2 27"
"dev 3 25" "dev 3 26" "dev 3 27"
"dev 4 23" "dev 4 24"
)

echo "=== SC5-V2 REPAIR $(date) ==="
echo "Total cells: ${#CELLS[@]} on GPUs 1-5"

count=0
batch=0
for cell in "${CELLS[@]}"; do
    read pool task state <<< "$cell"
    gpu=$(( (count % 5) + 1 ))
    run_cell $gpu $pool $task $state &
    count=$((count + 1))
    if [ $((count % 5)) -eq 0 ]; then
        batch=$((batch + 1))
        wait
        echo "=== BATCH $batch DONE ($count/${#CELLS[@]}) $(date) ==="
    fi
done
wait  # final batch

echo "=== REPAIR DONE $(date) ==="
echo "Total done: $(find $OUTBASE -name .done | wc -l)"
echo "Total valid: $(find $OUTBASE -name step_telemetry.csv -size +100c | wc -l)"
