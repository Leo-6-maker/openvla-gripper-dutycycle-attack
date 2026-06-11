#!/bin/bash
# S20d SMOKE GPU 1,0 — physical GPUs 1 and 0
# States: ketchup s0, s1
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export OPENVLA_ATTN_IMPLEMENTATION=eager
export CUDA_VISIBLE_DEVICES=1,0

OUT=/data/liuyu/outputs/stageb_s20d_v4_official_l3_20260611/smoke
VID=/data/liuyu/outputs/stageb_s20d_v4_official_l3_20260611/smoke_videos
mkdir -p $OUT $VID

PY=/data/aviary/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py
MODEL=/data/aviary/models/openvla/openvla-7b-finetuned-libero-object
MAX=280; SM=done

# CUDA_VISIBLE_DEVICES=1,0 → local 0=Phys1, local 1=Phys0
# render_gpu_device_id=0 → uses Phys1 for render, model auto across both

echo "[$(date +%H:%M:%S)] GPU(1,0) — ketchup s0"
$PY -u $S \
  --task ketchup --state_ids 0 --condition clean \
  --max_steps_override $MAX --success_metric $SM \
  --num_steps_wait 10 --model_path $MODEL \
  --render_gpu_device_id 0 --model_gpu_device_id -1 \
  --output_dir $OUT --save_video_dir $VID/ketchup_s0_clean \
  --job_id 960000 --seed 0 \
  || echo "FAIL_ketchup_s0"

echo "[$(date +%H:%M:%S)] GPU(1,0) — ketchup s1"
$PY -u $S \
  --task ketchup --state_ids 1 --condition clean \
  --max_steps_override $MAX --success_metric $SM \
  --num_steps_wait 10 --model_path $MODEL \
  --render_gpu_device_id 0 --model_gpu_device_id -1 \
  --output_dir $OUT --save_video_dir $VID/ketchup_s1_clean \
  --job_id 960001 --seed 0 \
  || echo "FAIL_ketchup_s1"

echo "[$(date +%H:%M:%S)] GPU(1,0) DONE"
