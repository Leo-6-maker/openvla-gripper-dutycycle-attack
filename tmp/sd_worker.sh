#!/bin/bash
# Stage D worker: 46 nolock jobs using same pattern as Stage C launcher
set -e
cd /mnt/sdc/dty_user/openvla_attack
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager TF_FORCE_GPU_ALLOW_GROWTH=true
export HOME=/mnt/sdc/dty_user/openvla_attack/sandbox_home TMPDIR=/mnt/sdc/dty_user/openvla_attack/tmp
PY=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3
B=/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_v2_vis_sc5_mlp_bridge_telemetry_v2.py
C=/mnt/sdc/dty_user/openvla_attack/outputs/sc5_v2_seed42/sc5_mlp_v2.pt
COMMIT=$(git rev-parse HEAD)
BASE=/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/breadth_120
TMA="vanilla_tma_gripper_open_ce"
PREFIX="autoregressive_prefix_gripper_target_token_logratio_arm_v3"

STATES=(
  "salad_dressing 2 1"
  "bbq_sauce 3 4"
  "ketchup 4 1"
  "milk 7 5"
  "butter 6 5"
  "orange_juice 9 2"
  "tomato_sauce 5 1"
  "butter 6 6"
)
SEEDS=(42 123 456)

run_one() {
  local GPU=$1 CELL=$2 TASK=$3 STATE=$4 SEED=$5 OBJ=$6 TAG=$7
  local OUT=${BASE}/${TAG}/${CELL}_s${STATE}_s${SEED}
  if [ -f "$OUT/COMPLETE.json" ]; then return 0; fi
  mkdir -p "$OUT"
  echo "GPU$GPU: $TAG $CELL s$STATE s$SEED $(date)"
  env CUDA_VISIBLE_DEVICES=$GPU $PY -u $B --condition TRUE_T10 --state_id $STATE \
    --anchor 0 --seed_id $SEED --task_idx $TASK --attack_objective "$OBJ" \
    --eval_seed 0 --output_dir "$OUT" --render_gpu $GPU --mlp_path $C \
    --libero_preprocess_backend upstream_tf_jpeg --save_video --source_commit $COMMIT \
    --video_fps 10 --frame_stride 2 > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  echo "GPU$GPU: $TAG $CELL s$STATE s$SEED DONE $(date)"
}

# Usage: bash sd_worker.sh <GPU> <JOB_INDEX>
# 0-22: TMA nolock (skip salad s1 s42 which was preflight)
# 23-45: Prefix nolock (skip salad s1 s42 which was preflight)
GPU=$1; IDX=$2

if [ $IDX -lt 23 ]; then
  TAG="tma_nolock"; OBJ="$TMA"
  ACTUAL=$((IDX + 1))
else
  TAG="prefix_nolock"; OBJ="$PREFIX"
  ACTUAL=$((IDX - 23 + 1))
fi

STATE_IDX=$((ACTUAL / 3))
SEED_IDX=$((ACTUAL % 3))
read CELL TASK STATE <<< "${STATES[$STATE_IDX]}"
SEED=${SEEDS[$SEED_IDX]}

run_one $GPU "$CELL" $TASK $STATE $SEED "$OBJ" "$TAG"
