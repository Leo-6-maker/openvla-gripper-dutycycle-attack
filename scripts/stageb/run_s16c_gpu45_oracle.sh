#!/bin/bash
# S16c GPU45 — local ORACLE references for fresh candidates
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s16c_wave1_confirmation
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py

export CUDA_VISIBLE_DEVICES=4,5

echo "[$(date +%H:%M:%S)] S16c local ORACLE references START"

echo "  milk_s0_w240-250 clean"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 240 --window_end 250 --condition clean --open_duration 10 --attack_seed 0 --job_id 953109 --pair_id milk_s0_w240_250_s16c_oracle --output_dir $OUT || echo "FAIL_CLEAN_milk"

echo "  milk_s0_w240-250 ORACLE"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 240 --window_end 250 --condition oracle_open --open_duration 10 --attack_seed 0 --job_id 953110 --pair_id milk_s0_w240_250_s16c_oracle --output_dir $OUT || echo "FAIL_ORACLE_milk"

echo "  tomato_s2_w95-105 clean"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 95 --window_end 105 --condition clean --open_duration 10 --attack_seed 0 --job_id 953111 --pair_id tomato_s2_w95_105_s16c_oracle --output_dir $OUT || echo "FAIL_CLEAN_tomato"

echo "  tomato_s2_w95-105 ORACLE"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 95 --window_end 105 --condition oracle_open --open_duration 10 --attack_seed 0 --job_id 953112 --pair_id tomato_s2_w95_105_s16c_oracle --output_dir $OUT || echo "FAIL_ORACLE_tomato"

echo "[$(date +%H:%M:%S)] S16c GPU45 local ORACLE DONE"
