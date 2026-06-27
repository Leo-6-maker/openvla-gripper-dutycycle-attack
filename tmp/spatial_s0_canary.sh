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
D=/mnt/sdc/dty_user/openvla_attack/outputs/sc5_v2_seed42/sc5_mlp_v2.pt
COMMIT=$(git rev-parse HEAD)
BASE=/mnt/sdc/dty_user/openvla_attack/evidence/phase9_spatial_transfer/canary_s0
mkdir -p $BASE

# 3 Spatial tasks (canonical index 0,1,2) × state0 × eval_seed0 CLEAN
run_one() {
  local G=$1 SU=$2 TI=$3 ST=$4
  local OUT=${BASE}/${SU}_task${TI}_s${ST}
  if [ -f "$OUT/COMPLETE.json" ]; then
    echo "SKIP: $OUT already complete"
    return 0
  fi
  rm -rf "$OUT" 2>/dev/null  # clean any stale partial dir
  echo "GPU$G: $SU task$TI s$ST $(date)"
  env CUDA_VISIBLE_DEVICES=$G $PY -u $B \
    --suite $SU --task_idx $TI --state_id $ST --eval_seed 0 \
    --model_path $M --unnorm_key libero_spatial \
    --output_dir "$OUT" --render_gpu $G \
    --detector_path $D --source_commit $COMMIT \
    --save_video \
    > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  local EC=$?
  echo "GPU$G: $SU task$TI s$ST exit=$EC $(date)"
  return $EC
}

GPU=$1
for TI in 0 1 2; do
  run_one $GPU "libero_spatial" $TI 0
done
echo "S0 canary GPU$GPU done"
