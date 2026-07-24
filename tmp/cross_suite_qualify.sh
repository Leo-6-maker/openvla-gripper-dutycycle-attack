#!/bin/bash
# Cross-suite clean qualification: find 2 clean-success states for 5 gripper tasks
# Uses battle-tested run_sc5_cross_suite_clean.py
set -e
cd /mnt/sdc/dty_user/openvla_attack
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager TF_FORCE_GPU_ALLOW_GROWTH=true
export HOME=/mnt/sdc/dty_user/openvla_attack/sandbox_home TMPDIR=/mnt/sdc/dty_user/openvla_attack/tmp
PY=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3
B=/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_sc5_cross_suite_clean.py
C=/mnt/sdc/dty_user/openvla_attack/outputs/sc5_v2_seed42/sc5_mlp_v2.pt
COMMIT=$(git rev-parse HEAD)
BASE=/mnt/sdc/dty_user/openvla_attack/evidence/cross_suite/clean_qualify
mkdir -p $BASE

# 5 gripper-critical tasks across suites (pick from LIBERO benchmark)
# Task selection: grasp + lift + transport required
TASKS=(
  "libero_spatial pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
  "libero_spatial pick_up_the_butter_and_place_it_in_the_tray"
  "libero_goal put_the_black_bowl_on_the_plate"
  "libero_goal put_the_butter_on_the_tray"
  "libero_10 pick_up_the_butter_and_place_it_in_the_basket"
)
# Short names for output dirs
NAMES=(
  "spatial_black_bowl"
  "spatial_butter_tray"
  "goal_black_bowl"
  "goal_butter_tray"
  "libero10_butter"
)

run_clean() {
  local GPU=$1 NAME=$2 SUITE=$3 TASK_STR=$4 STATE=$5
  local OUT=${BASE}/${NAME}_s${STATE}
  if [ -f "$OUT/COMPLETE.json" ]; then
    local S=$(grep -oP '"task_success": \K[a-z]+' $OUT/episode_summary.json 2>/dev/null || echo '?')
    echo "EXISTING $NAME s$STATE success=$S"
    return 0
  fi
  echo "=== GPU$GPU: $NAME s$STATE ($SUITE) $(date) ==="
  rm -rf "$OUT"; mkdir -p "$OUT"
  env CUDA_VISIBLE_DEVICES=$GPU $PY -u $B --suite $SUITE --task "$TASK_STR" \
    --state_id $STATE --seed 42 --output_dir "$OUT" --render_gpu $GPU \
    --mlp_path $C --source_commit $COMMIT --save_video --video_fps 10 --frame_stride 2 \
    > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  local S=$(grep -oP '"task_success": \K[a-z]+' $OUT/episode_summary.json 2>/dev/null || echo '?')
  echo "=== DONE $NAME s$STATE success=$S $(date) ==="
  [ "$S" = "true" ]
}

# Usage: bash cross_suite_qualify.sh <GPU> <TASK_INDEX>
GPU=$1; TIDX=${2:-0}
NAME=${NAMES[$TIDX]}
SUITE=$(echo ${TASKS[$TIDX]} | cut -d' ' -f1)
TASK_STR=$(echo ${TASKS[$TIDX]} | cut -d' ' -f2-)

FOUND=0; STATE=0
while [ $FOUND -lt 2 ] && [ $STATE -lt 20 ]; do
  if run_clean $GPU "$NAME" $SUITE "$TASK_STR" $STATE; then
    echo "FROZEN: $NAME s$STATE"
    FOUND=$((FOUND + 1))
  else
    echo "INELIGIBLE: $NAME s$STATE"
  fi
  STATE=$((STATE + 1))
done
echo "$NAME: $FOUND/2 states frozen"
