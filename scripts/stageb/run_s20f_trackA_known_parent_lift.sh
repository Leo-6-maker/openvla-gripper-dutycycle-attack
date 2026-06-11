#!/bin/bash
# S20f Track A: known vulnerable parent lift to official/V4 Layer3
# GPU pair: 1,0 (dedicated, sequential, no parallel overlap)
# Parents: ketchup_s0_w150-160, tomato_sauce_s0_w70-80
set +e
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENVLA_ATTN_IMPLEMENTATION=eager
export CUDA_VISIBLE_DEVICES=1,0
unset DISPLAY

OUT=/data/liuyu/outputs/stageb_s20f_trackA_known_parent_lift_20260611
VID=$OUT/videos
mkdir -p $OUT $VID

PY=/data/aviary/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py
MODEL=/data/aviary/models/openvla/openvla-7b-finetuned-libero-object
MAX=280; SM=done
JOB=970000

run_ep() {
    local task=$1 sid=$2 cond=$3 ws=$4 we=$5 seed=$6 jid=$7 vid_dir=$8
    echo "[$(date +%H:%M:%S)] $task s$sid $cond w$ws-$we seed=$seed job=$jid"
    local extra=""
    if [ "$cond" = "random_linf" ]; then
        extra="--eps_raw_pixels 6 --attack_seed $seed --random_control_seed $seed"
    elif [ "$cond" = "vis_pgd" ]; then
        extra="--eps_raw_pixels 6 --attack_seed $seed --pgd_steps 20"
    fi
    $PY -u $S --task $task --state_ids $sid --condition $cond \
        --window_start $ws --window_end $we \
        --max_steps_override $MAX --success_metric $SM \
        --num_steps_wait 10 --model_path $MODEL \
        --render_gpu_device_id 0 --model_gpu_device_id -1 \
        --output_dir $OUT --save_video_dir "$vid_dir" \
        --job_id $jid --seed 0 $extra
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "FAIL $task s$sid $cond w$ws-$we seed=$seed (rc=$rc)"
        return 1
    fi
    return 0
}

check_rand_pass() {
    local task=$1 sid=$2 ws=$3 we=$4 seed=$5 jid=$6
    local f=$(ls $OUT/summary_${task}_s${sid}_w${ws}_${we}_s20d_random_linf_seed${seed}_job${jid}.json 2>/dev/null)
    if [ -z "$f" ]; then
        echo "  RAND summary not found, skipping VIS"
        return 1
    fi
    local open=$(/data/aviary/envs/openvla_official_libero_20260525/bin/python3 -c "import json; print(json.load(open('$f'))['decoded_open_count'])")
    local streak=$(/data/aviary/envs/openvla_official_libero_20260525/bin/python3 -c "import json; print(json.load(open('$f'))['max_open_streak'])")
    local succ=$(/data/aviary/envs/openvla_official_libero_20260525/bin/python3 -c "import json; print(json.load(open('$f'))['success_primary'])")
    echo "  RAND open=$open streak=$streak success=$succ"
    if [ "$succ" = "True" ] && [ "$open" -le 2 ] && [ "$streak" -le 1 ]; then
        return 0  # PASS
    fi
    return 1  # VETO or BORDERLINE
}

# ── Parent 1: ketchup s0 w150-160 ──
echo ""
echo "============================================================"
echo " PARENT 1: ketchup s0"
echo "============================================================"

# Clean
run_ep ketchup 0 clean 150 160 0 $((JOB++)) "$VID/ketchup_s0_clean" || echo "CLEAN_FAIL"
echo ""

# RAND seed80
run_ep ketchup 0 random_linf 150 160 80 $((JOB++)) "$VID/ketchup_s0_w150_160_rand80" || echo "RAND_FAIL"
echo ""

# VIS seed80 (only if RAND passes)
JID_VIS=$((JOB++))
if check_rand_pass ketchup 0 150 160 80 $((JID_VIS-1)); then
    run_ep ketchup 0 vis_pgd 150 160 80 $JID_VIS "$VID/ketchup_s0_w150_160_vis80" || echo "VIS_FAIL"
else
    echo "  RAND-veto FAIL or BORDERLINE — skipping VIS for ketchup s0"
fi

# ── Parent 2: tomato_sauce s0 w70-80 ──
echo ""
echo "============================================================"
echo " PARENT 2: tomato_sauce s0"
echo "============================================================"

# Clean
run_ep tomato_sauce 0 clean 70 80 0 $((JOB++)) "$VID/tomato_s0_clean" || echo "CLEAN_FAIL"
echo ""

# RAND seed80
run_ep tomato_sauce 0 random_linf 70 80 80 $((JOB++)) "$VID/tomato_s0_w70_80_rand80" || echo "RAND_FAIL"
echo ""

# VIS seed80 (only if RAND passes)
JID_VIS=$((JOB++))
if check_rand_pass tomato_sauce 0 70 80 80 $((JID_VIS-1)); then
    run_ep tomato_sauce 0 vis_pgd 70 80 80 $JID_VIS "$VID/tomato_s0_w70_80_vis80" || echo "VIS_FAIL"
else
    echo "  RAND-veto FAIL or BORDERLINE — skipping VIS for tomato_sauce s0"
fi

echo ""
echo "[$(date +%H:%M:%S)] Track A DONE"
