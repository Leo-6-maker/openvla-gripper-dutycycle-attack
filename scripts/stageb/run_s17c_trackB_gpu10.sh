#!/bin/bash
# S17c Track B GPU10 — corrected command screen parents 1-2: tomato_sauce anchor + neighbor
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES=1,0
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s17c_trackB_command_screen
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
SEED=61; EPS=6; PGD=20

# P1: tomato_sauce_s0_w70-80 (positive calibration)
echo "[$(date +%H:%M:%S)] TrackB P1 tomato_sauce_s0_w70-80"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 0 --window_start 70 --window_end 80 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953400 --pair_id tomato_sauce_s0_w70_80_s17c_seed61 --output_dir $OUT || echo "FAIL_P1_VIS"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 0 --window_start 70 --window_end 80 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953401 --pair_id tomato_sauce_s0_w70_80_s17c_seed61 --output_dir $OUT || echo "FAIL_P1_RAND"

# P2: tomato_sauce_s0_w80-90 (anchor neighbor)
echo "[$(date +%H:%M:%S)] TrackB P2 tomato_sauce_s0_w80-90"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 0 --window_start 80 --window_end 90 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953402 --pair_id tomato_sauce_s0_w80_90_s17c_seed61 --output_dir $OUT || echo "FAIL_P2_VIS"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 0 --window_start 80 --window_end 90 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953403 --pair_id tomato_sauce_s0_w80_90_s17c_seed61 --output_dir $OUT || echo "FAIL_P2_RAND"

echo "[$(date +%H:%M:%S)] TrackB GPU10 DONE"
