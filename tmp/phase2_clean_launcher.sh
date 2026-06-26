#!/bin/bash
# Phase 2: CLEAN canonical + determinism audit
# Reuses battle-tested run_one() pattern — calls telemetry v2 bridge
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
BASE=/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/clean_determinism

run_one() {
  local GPU=$1 CELL=$2 TASK=$3 STATE=$4 SEED=$5 TAG=$6
  export CUDA_VISIBLE_DEVICES=$GPU
  local OUT=${BASE}/${TAG}/${CELL}_s${SEED}_r$(printf '%02d' ${7:-0})
  if [ -f "$OUT/COMPLETE.json" ]; then echo "SKIP $TAG/$CELL s$SEED"; return 0; fi
  echo "=== GPU$GPU: $TAG $CELL s$SEED $(date) ==="
  rm -rf "$OUT"; mkdir -p "$OUT"
  $PY -u $B --condition CLEAN --state_id $STATE --anchor 0 --seed_id $SEED --task_idx $TASK \
    --attack_objective vanilla_tma_gripper_open_ce \
    --output_dir "$OUT" --render_gpu $GPU --mlp_path $C \
    --libero_preprocess_backend upstream_tf_jpeg \
    --save_video --source_commit $COMMIT --video_fps 10 --frame_stride 2 \
    > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  echo "=== DONE GPU$GPU: $TAG $CELL s$SEED $(date) ==="
}

# Usage: bash phase2_clean_launcher.sh <GPU> <RUN_INDEX>
# RUN_INDEX maps to specific cell+tag; 2 workers per GPU means 2 run indices per GPU

case "${2:-0}" in
# --- Worker 0: Canonical CLEAN (9 cells, 1 run each) ---
0)  run_one $1 salad_dressing_s0 2 0 42 canonical ;;
1)  run_one $1 bbq_sauce_s0 3 0 42 canonical ;;
2)  run_one $1 ketchup_s0 4 0 42 canonical ;;
3)  run_one $1 milk_s4 7 4 42 canonical ;;
4)  run_one $1 butter_s2 6 2 42 canonical ;;
5)  run_one $1 alphabet_soup_s0 0 0 42 canonical ;;
6)  run_one $1 orange_juice_s0 9 0 42 canonical ;;
7)  run_one $1 butter_s0 6 0 42 canonical ;;
8)  run_one $1 tomato_sauce_s0 5 0 42 canonical ;;

# --- Worker 1: Determinism repeats (3 cells × 3 repeats) ---
9)  run_one $1 salad_dressing_s0 2 0 42 determinism 1 ;;
10) run_one $1 salad_dressing_s0 2 0 42 determinism 2 ;;
11) run_one $1 salad_dressing_s0 2 0 42 determinism 3 ;;
12) run_one $1 butter_s0 6 0 42 determinism 1 ;;
13) run_one $1 butter_s0 6 0 42 determinism 2 ;;
14) run_one $1 butter_s0 6 0 42 determinism 3 ;;
15) run_one $1 tomato_sauce_s0 5 0 42 determinism 1 ;;
16) run_one $1 tomato_sauce_s0 5 0 42 determinism 2 ;;
17) run_one $1 tomato_sauce_s0 5 0 42 determinism 3 ;;
esac
