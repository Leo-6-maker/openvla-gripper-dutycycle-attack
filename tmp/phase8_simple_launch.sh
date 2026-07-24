#!/bin/bash
# Phase 8 Simple Launch — reuse Object bridge pattern
# Usage: bash phase8_simple_launch.sh <GPU> <SUITE> <JOB_INDEX>
# SUITE = SPATIAL | GOAL | LIBERO_10
set -e
GPU=$1; SUITE=$2; IDX=${3:-0}
BASE=/mnt/sdc/dty_user/openvla_attack

export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager TF_FORCE_GPU_ALLOW_GROWTH=true
export HOME=$BASE/sandbox_home TMPDIR=$BASE/tmp
PY=$BASE/envs/openvla-official-a800/bin/python3
BRIDGE=$BASE/scripts/stageb/run_v2_vis_sc5_mlp_bridge_telemetry_v2.py
DETECTOR=$BASE/outputs/sc5_v2_seed42/sc5_mlp_v2.pt
COMMIT=$(git -C $BASE rev-parse HEAD)

# Suite config
case $SUITE in
    SPATIAL) MODEL=$BASE/models/libero-spatial/spatial_c8f03f4_20260620; UK=libero_spatial; SN=libero_spatial ;;
    GOAL)    MODEL=$BASE/models/libero-goal; UK=libero_goal; SN=libero_goal ;;
    LIBERO_10) MODEL=$BASE/models/libero-10; UK=libero_10; SN=libero_10 ;;
    *) echo "Unknown suite: $SUITE"; exit 1 ;;
esac

# 7 conditions × 3 seeds × 10 tasks = 210 runs per suite
# JOB_INDEX maps to: condition=(IDX%7), task_seed=((IDX/7)%30), seed=task_seed%3, task=task_seed/3

CI=$((IDX % 7))
TS=$(( (IDX / 7) % 30 ))
SEED_IDX=$((TS % 3))
TASK=$((TS / 3))

CONDS=("CLEAN" "RANDOM" "UNTARGETED_CE" "TMA_NOLOCK" "TMA_ARMLOCK" "PREFIX_NOLOCK" "PREFIX_ARMLOCK")
COND_NAME=${CONDS[$CI]}
SEEDS=(42 123 456)
SEED_VAL=${SEEDS[$SEED_IDX]}
STATE_ID=$SEED_IDX

# Build args (same pattern as Object launcher)
OUT=$BASE/evidence/phase8_cross_suite_v1/runs/p8_${SUITE}_t${TASK}_s${STATE_ID}_c${COND_NAME}

if [ -f "$OUT/.done" ]; then echo "SKIP: $OUT"; exit 0; fi
rm -rf "$OUT" 2>/dev/null; mkdir -p "$OUT"

ARGS="--task_idx $TASK --state_id $STATE_ID --eval_seed 0 --seed_id $SEED_VAL"
ARGS="$ARGS --output_dir $OUT --render_gpu $GPU --mlp_path $DETECTOR"
ARGS="$ARGS --libero_preprocess_backend upstream_tf_jpeg --anchor 0"
ARGS="$ARGS --source_commit $COMMIT --save_video --video_fps 10 --frame_stride 2"
ARGS="$ARGS --unnorm_key $UK --suite_name $SN"

case $COND_NAME in
    CLEAN)         ARGS="$ARGS --condition CLEAN --attack_objective ''" ;;
    RANDOM)        ARGS="$ARGS --condition RAND_T10 --attack_objective ''" ;;
    UNTARGETED_CE) ARGS="$ARGS --condition TRUE_T10 --attack_objective untargeted_clean_token_ce" ;;
    TMA_NOLOCK)    ARGS="$ARGS --condition TRUE_T10 --attack_objective vanilla_tma_gripper_open_ce" ;;
    TMA_ARMLOCK)   ARGS="$ARGS --condition TRUE_T10 --attack_objective vanilla_tma_gripper_open_ce --arm_lock" ;;
    PREFIX_NOLOCK) ARGS="$ARGS --condition TRUE_T10 --attack_objective autoregressive_prefix_gripper_target_token_logratio_arm_v3" ;;
    PREFIX_ARMLOCK) ARGS="$ARGS --condition TRUE_T10 --attack_objective autoregressive_prefix_gripper_target_token_logratio_arm_v3 --arm_lock" ;;
esac

echo "$(date) GPU$GPU: $SUITE t$TASK s$STATE_ID seed$SEED_VAL $COND_NAME"
env CUDA_VISIBLE_DEVICES=$GPU OPENVLA_MODEL_PATH=$MODEL $PY -u $BRIDGE $ARGS > "$OUT/stdout.log" 2> "$OUT/stderr.log"
EC=$?
echo "$(date) GPU$GPU: $SUITE t$TASK s$STATE_ID $COND_NAME exit=$EC"
[ $EC -eq 0 ] && touch "$OUT/.done"
exit $EC
