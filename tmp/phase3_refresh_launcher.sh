#!/bin/bash
# Phase 3 (P20): Core metric telemetry refresh
# 4 conditions x 9 cells x 3 seeds = 108 runs
# Each GPU runs 2 workers in parallel (round-robin scheduling)
set -e
cd /mnt/sdc/dty_user/openvla_attack
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager
export OPENVLA_MODEL_PATH=/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object
export HOME=/mnt/sdc/dty_user/openvla_attack/sandbox_home TMPDIR=/mnt/sdc/dty_user/openvla_attack/tmp
export TF_FORCE_GPU_ALLOW_GROWTH=true
PY=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3
B=/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_v2_vis_sc5_mlp_bridge_telemetry_v2.py
C=/mnt/sdc/dty_user/openvla_attack/outputs/sc5_v2_seed42/sc5_mlp_v2.pt
COMMIT=$(git rev-parse HEAD)
BASE=/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/metric_refresh_v2

TMA="vanilla_tma_gripper_open_ce"
PREFIX="autoregressive_prefix_gripper_target_token_logratio_arm_v3"

# Task/anchor mapping: CELL TASK STATE ANCHOR
declare -A ANCHORS
ANCHORS["salad_dressing_s0"]="2 0 84"
ANCHORS["bbq_sauce_s0"]="3 0 128"
ANCHORS["ketchup_s0"]="4 0 95"
ANCHORS["milk_s4"]="7 4 92"
ANCHORS["butter_s2"]="6 2 100"
ANCHORS["alphabet_soup_s0"]="0 0 86"
ANCHORS["orange_juice_s0"]="9 0 167"
ANCHORS["butter_s0"]="6 0 85"
ANCHORS["tomato_sauce_s0"]="5 0 176"

CELLS=("salad_dressing_s0" "bbq_sauce_s0" "ketchup_s0" "milk_s4" "butter_s2" "alphabet_soup_s0" "orange_juice_s0" "butter_s0" "tomato_sauce_s0")
SEEDS=(42 123 456)

run_one() {
  local GPU=$1 CELL=$2 TASK=$3 STATE=$4 ANCHOR=$5 SEED=$6 OBJ=$7 LOCK=$8 TAG=$9
  export CUDA_VISIBLE_DEVICES=$GPU
  local OUT=${BASE}/${TAG}/${CELL}_s${SEED}
  if [ -f "$OUT/COMPLETE.json" ]; then echo "SKIP $TAG/$CELL s$SEED"; return 0; fi
  echo "=== GPU$GPU: $TAG $CELL s$SEED $(date) ==="
  rm -rf "$OUT"; mkdir -p "$OUT"
  local LOCK_FLAG=""; [ "$LOCK" = "1" ] && LOCK_FLAG="--arm_lock"
  $PY -u $B --condition TRUE_T10 --state_id $STATE --anchor $ANCHOR --seed_id $SEED --task_idx $TASK \
    --attack_objective "$OBJ" $LOCK_FLAG \
    --output_dir "$OUT" --render_gpu $GPU --mlp_path $C \
    --libero_preprocess_backend upstream_tf_jpeg \
    --save_video --source_commit $COMMIT --video_fps 10 --frame_stride 2 \
    > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  echo "=== DONE GPU$GPU: $TAG $CELL s$SEED $(date) ==="
}

# Usage: bash phase3_refresh_launcher.sh <GPU> <RUN_INDEX>
# 108 runs: 4 conditions x 9 cells x 3 seeds
# Index mapping:
#   0-26:   TMA no-lock (tag=tma_nolock)
#   27-53:  TMA ArmLock (tag=tma_armlock)
#   54-80:  Prefix no-lock (tag=prefix_nolock)
#   81-107: Prefix ArmLock (tag=prefix_armlock)

IDX=${2:-0}
GPU=$1

if [ $IDX -lt 27 ]; then
    TAG="tma_nolock"; OBJ="$TMA"; LOCK=0; BASE_IDX=$IDX
elif [ $IDX -lt 54 ]; then
    TAG="tma_armlock"; OBJ="$TMA"; LOCK=1; BASE_IDX=$((IDX - 27))
elif [ $IDX -lt 81 ]; then
    TAG="prefix_nolock"; OBJ="$PREFIX"; LOCK=0; BASE_IDX=$((IDX - 54))
else
    TAG="prefix_armlock"; OBJ="$PREFIX"; LOCK=1; BASE_IDX=$((IDX - 81))
fi

CELL_IDX=$((BASE_IDX / 3))
SEED_IDX=$((BASE_IDX % 3))
CELL=${CELLS[$CELL_IDX]}
SEED=${SEEDS[$SEED_IDX]}
read TASK STATE ANCHOR <<< "${ANCHORS[$CELL]}"

run_one $GPU "$CELL" $TASK $STATE $ANCHOR $SEED "$OBJ" $LOCK "$TAG"
