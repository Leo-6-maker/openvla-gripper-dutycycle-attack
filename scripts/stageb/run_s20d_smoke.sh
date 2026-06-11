#!/bin/bash
# S20d V4 clean clone smoke — verify clean baseline reproduction
# Required states: ketchup s0, s1, s3 + tomato_sauce s3, s5
set +e
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
export OPENVLA_ATTN_IMPLEMENTATION=eager

OUT=/data/liuyu/outputs/stageb_s20d_v4_official_l3_20260611/smoke
VID=/data/liuyu/outputs/stageb_s20d_v4_official_l3_20260611/smoke_videos
mkdir -p $OUT $VID

PY=/data/aviary/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py
MODEL=/data/aviary/models/openvla/openvla-7b-finetuned-libero-object
MAX=280
SM=done  # success_metric as prior baseline

# ── GPU 1,0: ketchup s0, s1, s3 ──
echo "[$(date +%H:%M:%S)] S20d SMOKE GPU(1,0) — ketchup s0"
export CUDA_VISIBLE_DEVICES=1,0
$PY -u $S \
  --task ketchup --state_ids 0 --condition clean \
  --max_steps_override $MAX --success_metric $SM \
  --num_steps_wait 10 --model_path $MODEL \
  --render_gpu_device_id 1 --model_gpu_device_id -1 \
  --output_dir $OUT --save_video_dir $VID/ketchup_s0_clean \
  --job_id 960000 --seed 0 \
  || echo "FAIL_ketchup_s0"

echo "[$(date +%H:%M:%S)] S20d SMOKE GPU(1,0) — ketchup s1"
$PY -u $S \
  --task ketchup --state_ids 1 --condition clean \
  --max_steps_override $MAX --success_metric $SM \
  --num_steps_wait 10 --model_path $MODEL \
  --render_gpu_device_id 1 --model_gpu_device_id -1 \
  --output_dir $OUT --save_video_dir $VID/ketchup_s1_clean \
  --job_id 960001 --seed 0 \
  || echo "FAIL_ketchup_s1"

echo "[$(date +%H:%M:%S)] S20d SMOKE GPU(1,0) — ketchup s3"
$PY -u $S \
  --task ketchup --state_ids 3 --condition clean \
  --max_steps_override $MAX --success_metric $SM \
  --num_steps_wait 10 --model_path $MODEL \
  --render_gpu_device_id 1 --model_gpu_device_id -1 \
  --output_dir $OUT --save_video_dir $VID/ketchup_s3_clean \
  --job_id 960002 --seed 0 \
  || echo "FAIL_ketchup_s3"

# ── GPU 2,6: tomato_sauce s3, s5 ──
echo "[$(date +%H:%M:%S)] S20d SMOKE GPU(2,6) — tomato_sauce s3"
export CUDA_VISIBLE_DEVICES=2,6
$PY -u $S \
  --task tomato_sauce --state_ids 3 --condition clean \
  --max_steps_override $MAX --success_metric $SM \
  --num_steps_wait 10 --model_path $MODEL \
  --render_gpu_device_id 1 --model_gpu_device_id -1 \
  --output_dir $OUT --save_video_dir $VID/tomato_sauce_s3_clean \
  --job_id 960003 --seed 0 \
  || echo "FAIL_tomato_sauce_s3"

echo "[$(date +%H:%M:%S)] S20d SMOKE GPU(2,6) — tomato_sauce s5"
$PY -u $S \
  --task tomato_sauce --state_ids 5 --condition clean \
  --max_steps_override $MAX --success_metric $SM \
  --num_steps_wait 10 --model_path $MODEL \
  --render_gpu_device_id 1 --model_gpu_device_id -1 \
  --output_dir $OUT --save_video_dir $VID/tomato_sauce_s5_clean \
  --job_id 960004 --seed 0 \
  || echo "FAIL_tomato_sauce_s5"

echo "[$(date +%H:%M:%S)] S20d SMOKE DONE"
