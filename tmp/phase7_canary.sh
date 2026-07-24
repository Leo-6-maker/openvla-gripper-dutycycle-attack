#!/bin/bash
# Phase 7D: 3-run canary on GPU 5
set -e
GPU=5
cd /mnt/sdc/dty_user/openvla_attack
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager
export OPENVLA_MODEL_PATH=/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object
export HOME=/mnt/sdc/dty_user/openvla_attack/sandbox_home TMPDIR=/mnt/sdc/dty_user/openvla_attack/tmp
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=$GPU
PY=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3
B=/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_v2_vis_sc5_mlp_bridge.py
CKPT=/mnt/sdc/dty_user/openvla_attack/outputs/sc5_v2_seed42/sc5_mlp_v2.pt
BACKEND=upstream_tf_jpeg
COMMIT=$(git rev-parse HEAD)
OUT=/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/canary
mkdir -p $OUT

echo "=== CANARY GPU=$GPU commit=$COMMIT $(date) ==="

# Canary A: TV TRUE_T10 (butter_s0, known TV from official smoke)
echo "=== CANARY_A TV-VIS $(date) ==="
rm -rf $OUT/canary_a_tv_vis; mkdir -p $OUT/canary_a_tv_vis
$PY $B --condition TRUE_T10 --state_id 0 --anchor 85 --seed_id 42 --task_idx 6 \
  --output_dir $OUT/canary_a_tv_vis --render_gpu $GPU --mlp_path $CKPT \
  --libero_preprocess_backend $BACKEND \
  --save_video --source_commit $COMMIT --video_fps 10 --frame_stride 1 \
  > $OUT/canary_a_tv_vis/stdout.log 2> $OUT/canary_a_tv_vis/stderr.log
echo "A: $(tail -1 $OUT/canary_a_tv_vis/stdout.log)"

# Canary B: Same TV RAND_T10
echo "=== CANARY_B TV-RAND $(date) ==="
rm -rf $OUT/canary_b_tv_rand; mkdir -p $OUT/canary_b_tv_rand
$PY $B --condition RAND_T10 --state_id 0 --anchor 85 --seed_id 42 --task_idx 6 \
  --output_dir $OUT/canary_b_tv_rand --render_gpu $GPU --mlp_path $CKPT \
  --libero_preprocess_backend $BACKEND \
  --save_video --source_commit $COMMIT --video_fps 10 --frame_stride 1 \
  > $OUT/canary_b_tv_rand/stdout.log 2> $OUT/canary_b_tv_rand/stderr.log
echo "B: $(tail -1 $OUT/canary_b_tv_rand/stdout.log)"

# Canary C: First formal NC + TRUE_T10 (task/state TBD after census)
# This part is filled after formal NC manifest is frozen
echo "=== CANARY_C NC (TBD after census) $(date) ==="

echo "=== CANARY DONE $(date) ==="
