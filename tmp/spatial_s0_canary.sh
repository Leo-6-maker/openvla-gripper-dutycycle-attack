#!/bin/bash
# Spatial S0: Model+bridge canary — 3 CLEAN runs on 3 different Spatial tasks
# Usage: bash spatial_s0_canary.sh <GPU_ID>
set -e
cd /mnt/sdc/dty_user/openvla_attack
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager TF_FORCE_GPU_ALLOW_GROWTH=true
export HOME=/mnt/sdc/dty_user/openvla_attack/sandbox_home TMPDIR=/mnt/sdc/dty_user/openvla_attack/tmp
PY=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3
B=/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_sc5_cross_suite_clean.py
M=/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620
C=/mnt/sdc/dty_user/openvla_attack/outputs/sc5_v2_seed42/sc5_mlp_v2.pt
COMMIT=$(git rev-parse HEAD)
BASE=/mnt/sdc/dty_user/openvla_attack/evidence/phase9_spatial_transfer/canary_s0
mkdir -p $BASE

# 3 Spatial tasks × state0 × seed42 CLEAN
JOBS=(
  "libero_spatial 0 0 pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
  "libero_spatial 1 0 pick_up_the_butter_and_place_it_in_the_tray"
  "libero_spatial 2 0 pick_up_the_cheese_and_place_it_on_the_plate"
)

run_one() {
  local G=$1 SU=$2 TI=$3 ST=$4 TN=$5
  local OUT=${BASE}/${SU}_task${TI}_s${ST}
  if [ -f "$OUT/COMPLETE.json" ]; then
    echo "SKIP: $OUT already complete"
    return 0
  fi
  mkdir -p "$OUT"
  echo "GPU$G: $SU task$TI s$ST $(date)"
  env CUDA_VISIBLE_DEVICES=$G $PY -u $B \
    --suite $SU --task_idx $TI --state_id $ST --seed 42 \
    --model_path $M --unnorm_key libero_spatial \
    --output_dir "$OUT" --render_gpu $G \
    --mlp_path $C --source_commit $COMMIT \
    --save_video --video_fps 10 --frame_stride 2 \
    > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  local EC=$?
  echo "GPU$G: $SU task$TI s$ST exit=$EC $(date)"
  return $EC
}

GPU=$1
for job in "${JOBS[@]}"; do
  read SU TI ST TN <<< "$job"
  run_one $GPU $SU $TI $ST "$TN"
done
echo "S0 canary GPU$GPU done"
