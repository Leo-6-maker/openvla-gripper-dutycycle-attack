#!/bin/bash
# S13a GPU26 — tomato_s2_w90-100 RAND-only veto seeds 24,25,26
set +e
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s13a_rand_veto_and_milk_positive
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py

echo "[$(date +%H:%M:%S)] S13a GPU26 tomato_s2_w90-100 RAND veto START"
export CUDA_VISIBLE_DEVICES=2,6

echo "  RAND tomato_s2_w90-100 seed=24"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 --condition random_linf --open_duration 10 --attack_seed 24 --eps_raw_pixels 6 --job_id 952403 --pair_id tomato_s2_w90_100_s13a_randveto_seed24 --output_dir $OUT || echo "FAIL_RAND_s24"

echo "  RAND tomato_s2_w90-100 seed=25"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 --condition random_linf --open_duration 10 --attack_seed 25 --eps_raw_pixels 6 --job_id 952404 --pair_id tomato_s2_w90_100_s13a_randveto_seed25 --output_dir $OUT || echo "FAIL_RAND_s25"

echo "  RAND tomato_s2_w90-100 seed=26"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 --condition random_linf --open_duration 10 --attack_seed 26 --eps_raw_pixels 6 --job_id 952405 --pair_id tomato_s2_w90_100_s13a_randveto_seed26 --output_dir $OUT || echo "FAIL_RAND_s26"

echo "[$(date +%H:%M:%S)] S13a GPU26 tomato_s2_w90-100 RAND veto DONE"
