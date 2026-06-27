#!/bin/bash
# Tomato detector coverage: CLEAN sweep state0-9, 2 repeats each = 20 runs
# Usage: bash tomato_coverage.sh <GPU_ID>
set -e
cd /mnt/sdc/dty_user/openvla_attack
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager TF_FORCE_GPU_ALLOW_GROWTH=true
export HOME=/mnt/sdc/dty_user/openvla_attack/sandbox_home TMPDIR=/mnt/sdc/dty_user/openvla_attack/tmp
PY=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3
B=/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_v2_vis_sc5_mlp_bridge_telemetry_v2.py
C=/mnt/sdc/dty_user/openvla_attack/outputs/sc5_v2_seed42/sc5_mlp_v2.pt
COMMIT=$(git rev-parse HEAD)
BASE=/mnt/sdc/dty_user/openvla_attack/evidence/phase11_detector_coverage/tomato_sweep
mkdir -p $BASE

run_one() {
  local G=$1 ST=$2 RPT=$3
  local OUT=${BASE}/tomato_sauce_s${ST}_r${RPT}
  if [ -f "$OUT/COMPLETE.json" ]; then
    echo "SKIP: $OUT already complete"
    return 0
  fi
  mkdir -p "$OUT"
  echo "GPU$G: tomato s$ST r$RPT $(date)"
  env CUDA_VISIBLE_DEVICES=$G $PY -u $B \
    --condition CLEAN --state_id $ST --anchor 0 --seed_id 42 \
    --task_idx 5 --attack_objective "" \
    --eval_seed 0 --output_dir "$OUT" --render_gpu $G \
    --mlp_path $C --libero_preprocess_backend upstream_tf_jpeg \
    --save_video --source_commit $COMMIT \
    --video_fps 10 --frame_stride 2 \
    > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  local EC=$?
  echo "GPU$G: tomato s$ST r$RPT exit=$EC $(date)"
  return $EC
}

GPU=$1
for ST in 0 1 2 3 4 5 6 7 8 9; do
  for RPT in 1 2; do
    run_one $GPU $ST $RPT
  done
done
echo "Tomato coverage GPU$GPU done"
