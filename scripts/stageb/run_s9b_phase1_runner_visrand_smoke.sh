#!/bin/bash
# S9b Phase1-runner VIS/RAND physical bridge smoke — milk only, 4 jobs, pair-atomic
# ORACLE ref: sanity pos_area=+0.295
set +e
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s9b_phase1_runner_visrand_smoke
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py

# ── Seed 9: GPU 1,0 ──
echo "[$(date +%H:%M:%S)] S9b VIS/RAND SMOKE seed9 START"
export CUDA_VISIBLE_DEVICES=1,0

echo "  VIS milk L=10 seed=9"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition vis_pgd --open_duration 10 --attack_seed 9 --pgd_steps 20 --eps_raw_pixels 6 --job_id 950410 --pair_id milk_s0_w70_80_phase1port_seed9 --output_dir $OUT || echo "FAIL_VIS_seed9"

echo "  RAND milk L=10 seed=9"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition random_linf --open_duration 10 --attack_seed 9 --eps_raw_pixels 6 --job_id 950411 --pair_id milk_s0_w70_80_phase1port_seed9 --output_dir $OUT || echo "FAIL_RAND_seed9"

echo "[$(date +%H:%M:%S)] S9b VIS/RAND SMOKE seed9 DONE"

# ── Seed 10: GPU 4,5 ──
echo "[$(date +%H:%M:%S)] S9b VIS/RAND SMOKE seed10 START"
export CUDA_VISIBLE_DEVICES=4,5

echo "  VIS milk L=10 seed=10"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition vis_pgd --open_duration 10 --attack_seed 10 --pgd_steps 20 --eps_raw_pixels 6 --job_id 950412 --pair_id milk_s0_w70_80_phase1port_seed10 --output_dir $OUT || echo "FAIL_VIS_seed10"

echo "  RAND milk L=10 seed=10"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition random_linf --open_duration 10 --attack_seed 10 --eps_raw_pixels 6 --job_id 950413 --pair_id milk_s0_w70_80_phase1port_seed10 --output_dir $OUT || echo "FAIL_RAND_seed10"

echo "[$(date +%H:%M:%S)] S9b VIS/RAND SMOKE seed10 DONE"
echo "[$(date +%H:%M:%S)] S9b VIS/RAND SMOKE ALL DONE"
