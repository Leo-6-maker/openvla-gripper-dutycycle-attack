#!/bin/bash
# Cross-suite clean qualification v2: FIXED — no rm -rf, COMPLETE.json tracking
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

TASKS=(
  "libero_spatial pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
  "libero_spatial pick_up_the_butter_and_place_it_in_the_tray"
  "libero_goal put_the_black_bowl_on_the_plate"
  "libero_goal put_the_butter_on_the_tray"
  "libero_10 pick_up_the_butter_and_place_it_in_the_basket"
)
NAMES=(spatial_black_bowl spatial_butter_tray goal_black_bowl goal_butter_tray libero10_butter)

run_clean() {
  local G=$1 N=$2 SU=$3 TS=$4 ST=$5
  local OUT=${BASE}/${N}_s${ST}
  if [ -f "$OUT/COMPLETE.json" ]; then
    local S=$(grep -oP '"task_success": \K[a-z]+' $OUT/episode_summary.json 2>/dev/null || echo '?')
    if [ "$S" = "true" ]; then return 0; else return 1; fi
  fi
  if [ -d "$OUT" ] && [ -n "$(ls -A $OUT 2>/dev/null)" ]; then
    echo "WARNING: stale dir $OUT — cleaning"
    rm -rf "$OUT"
  fi
  mkdir -p "$OUT"
  echo "GPU$G: $N s$ST $(date)"
  env CUDA_VISIBLE_DEVICES=$G $PY -u $B --suite $SU --task "$TS" \
    --state_id $ST --seed 42 --output_dir "$OUT" --render_gpu $G \
    --mlp_path $C --source_commit $COMMIT --save_video --video_fps 10 --frame_stride 2 \
    > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  local S=$(grep -oP '"task_success": \K[a-z]+' $OUT/episode_summary.json 2>/dev/null || echo '?')
  echo "DONE $N s$ST success=$S"
  [ "$S" = "true" ]
}

G=$1; TIDX=${2:-0}
N=${NAMES[$TIDX]}
SU=$(echo ${TASKS[$TIDX]} | cut -d' ' -f1)
TS=$(echo ${TASKS[$TIDX]} | cut -d' ' -f2-)

FOUND=0; ST=0
while [ $FOUND -lt 2 ] && [ $ST -lt 20 ]; do
  echo "Testing $N s$ST (found=$FOUND/2)..."
  if run_clean $G "$N" $SU "$TS" $ST; then
    echo "FROZEN: $N s$ST"
    FOUND=$((FOUND + 1))
  else
    echo "INELIGIBLE: $N s$ST"
  fi
  ST=$((ST + 1))
done
echo "$N: $FOUND/2 states frozen"
