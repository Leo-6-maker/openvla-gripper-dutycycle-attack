#!/bin/bash
# S9a Init-Order Parity A/B — milk only, 2 jobs, GPU 1,0
# Tests: remove qvel[:]=0 + sim.forward() → restore Phase1-like ORACLE positive qpos?
set +e
export CUDA_VISIBLE_DEVICES=1,0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s9a_init_order_parity
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_extended_visrand_physical.py

echo "[$(date +%H:%M:%S)] S9a INIT-ORDER PARITY START (2 jobs, phase1_parity)"

echo "  CLEAN milk short phase1_parity"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition clean --attack_seed 0 --env_seed 0 --job_id 950300 --pair_id milk_s0_w70_80_short_s9a --original_window_start 70 --original_window_end 80 --oracle_ref_L10_pos_area 0.261452 --length_mode short_halfopen --post_horizon 40 --window_convention half_open --init_mode phase1_parity --output_dir $OUT || echo "FAIL_CLEAN"

echo "  ORACLE milk short phase1_parity"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition oracle_open --attack_seed 0 --env_seed 0 --job_id 950301 --pair_id milk_s0_w70_80_short_s9a --original_window_start 70 --original_window_end 80 --oracle_ref_L10_pos_area 0.261452 --length_mode short_halfopen --post_horizon 40 --window_convention half_open --init_mode phase1_parity --output_dir $OUT || echo "FAIL_ORACLE"

echo "[$(date +%H:%M:%S)] S9a INIT-ORDER PARITY DONE"
