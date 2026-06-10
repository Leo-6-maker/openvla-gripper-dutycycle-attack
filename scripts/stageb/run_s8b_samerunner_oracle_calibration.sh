#!/bin/bash
# S8b same-runner ORACLE calibration — milk only, 4 jobs, GPU 1,0
# Runner: run_extended_visrand_physical.py (v2, inclusive window)
set +e
export CUDA_VISIBLE_DEVICES=1,0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s8b_samerunner_oracle_calibration
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_extended_visrand_physical.py

echo "[$(date +%H:%M:%S)] S8b SAMERUNNER ORACLE CALIBRATION START (4 jobs)"

# ── short (ws=70, we=80, inclusive: 11 steps) ──
echo "  CLEAN milk short"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition clean --attack_seed 0 --env_seed 0 --job_id 950100 --pair_id milk_s0_w70_80_short_samerunner --original_window_start 70 --original_window_end 80 --oracle_ref_L10_pos_area 0.261452 --length_mode short --output_dir $OUT || echo "FAIL_CLEAN_milk_short"

echo "  ORACLE_OPEN milk short"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition oracle_open --attack_seed 0 --env_seed 0 --job_id 950101 --pair_id milk_s0_w70_80_short_samerunner --original_window_start 70 --original_window_end 80 --oracle_ref_L10_pos_area 0.261452 --length_mode short --output_dir $OUT || echo "FAIL_ORACLE_milk_short"

# ── extended20 (ws=60, we=90, inclusive: 31 steps) ──
echo "  CLEAN milk ext20"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 60 --window_end 90 --condition clean --attack_seed 0 --env_seed 0 --job_id 950102 --pair_id milk_s0_w60_90_extended20_samerunner --original_window_start 70 --original_window_end 80 --oracle_ref_L10_pos_area 0.261452 --length_mode extended20 --output_dir $OUT || echo "FAIL_CLEAN_milk_ext20"

echo "  ORACLE_OPEN milk ext20"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 60 --window_end 90 --condition oracle_open --attack_seed 0 --env_seed 0 --job_id 950103 --pair_id milk_s0_w60_90_extended20_samerunner --original_window_start 70 --original_window_end 80 --oracle_ref_L10_pos_area 0.261452 --length_mode extended20 --output_dir $OUT || echo "FAIL_ORACLE_milk_ext20"

echo "[$(date +%H:%M:%S)] S8b SAMERUNNER ORACLE CALIBRATION DONE"
