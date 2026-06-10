#!/bin/bash
# S8c Runner Parity Isolation — milk only, 4 jobs, GPU 1,0
# Tests: post_horizon=40 + half_open windows restore Phase1-like positive qpos?
set +e
export CUDA_VISIBLE_DEVICES=1,0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s8c_runner_parity_isolation
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_extended_visrand_physical.py

echo "[$(date +%H:%M:%S)] S8c RUNNER PARITY ISOLATION START (4 jobs, post_horizon=40, half_open)"

# ── short half_open (ws=70, we=80 → 10 steps) ──
echo "  CLEAN milk short half_open"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition clean --attack_seed 0 --env_seed 0 --job_id 950200 --pair_id milk_s0_w70_80_short_halfopen --original_window_start 70 --original_window_end 80 --oracle_ref_L10_pos_area 0.261452 --length_mode short_halfopen --post_horizon 40 --window_convention half_open --output_dir $OUT || echo "FAIL_CLEAN_milk_short"

echo "  ORACLE milk short half_open"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition oracle_open --attack_seed 0 --env_seed 0 --job_id 950201 --pair_id milk_s0_w70_80_short_halfopen --original_window_start 70 --original_window_end 80 --oracle_ref_L10_pos_area 0.261452 --length_mode short_halfopen --post_horizon 40 --window_convention half_open --output_dir $OUT || echo "FAIL_ORACLE_milk_short"

# ── extended20 half_open (ws=60, we=90 → 30 steps) ──
echo "  CLEAN milk ext20 half_open"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 60 --window_end 90 --condition clean --attack_seed 0 --env_seed 0 --job_id 950202 --pair_id milk_s0_w60_90_ext20_halfopen --original_window_start 70 --original_window_end 80 --oracle_ref_L10_pos_area 0.261452 --length_mode ext20_halfopen --post_horizon 40 --window_convention half_open --output_dir $OUT || echo "FAIL_CLEAN_milk_ext20"

echo "  ORACLE milk ext20 half_open"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 60 --window_end 90 --condition oracle_open --attack_seed 0 --env_seed 0 --job_id 950203 --pair_id milk_s0_w60_90_ext20_halfopen --original_window_start 70 --original_window_end 80 --oracle_ref_L10_pos_area 0.261452 --length_mode ext20_halfopen --post_horizon 40 --window_convention half_open --output_dir $OUT || echo "FAIL_ORACLE_milk_ext20"

echo "[$(date +%H:%M:%S)] S8c RUNNER PARITY ISOLATION DONE"
