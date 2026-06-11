#!/bin/bash
# S20I clean expansion: 6 new tasks, run sequentially when GPU45 is free
set +e
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENVLA_ATTN_IMPLEMENTATION=eager
export CUDA_VISIBLE_DEVICES=4,5
unset DISPLAY

OUT=/data/liuyu/outputs/stageb_s20i_clean_expansion_20260612
mkdir -p $OUT

PY=/data/aviary/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py
MODEL=/data/aviary/models/openvla/openvla-7b-finetuned-libero-object
MAX=280
SM=check_success
JOB=230000

TASKS="milk cream_cheese salad_dressing bbq_sauce orange_juice alphabet_soup"

echo "[$(date +%H:%M:%S)] S20I Clean Expansion: waiting for GPU45 worker to exit..."
while ps aux | grep -q "[s]20h_multiseed_worker.*gpu45"; do
    sleep 10
done
echo "[$(date +%H:%M:%S)] GPU45 free. Starting 6 clean episodes."

for TASK in $TASKS; do
    echo "[$(date +%H:%M:%S)] Clean: $TASK s1"
    $PY -u $S --task $TASK --state_ids 1 --condition clean \
        --max_steps_override $MAX --success_metric $SM \
        --num_steps_wait 10 --model_path $MODEL \
        --render_gpu_device_id 4 --model_gpu_device_id -1 \
        --output_dir $OUT --job_id $((JOB++)) --seed 0
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "FAIL $TASK (rc=$RC)"
    fi
done

echo "[$(date +%H:%M:%S)] Clean episodes done. Generating candidate universe..."
$PY -u scripts/stageb/run_s20i_build_expansion_universe.py

echo "[$(date +%H:%M:%S)] Clean expansion complete."
