#!/bin/bash
# S17c Track B GPU26 — corrected command screen parents 3-4: tomato_sauce w60-70 + bbq_sauce_s1
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES=2,6
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s17c_trackB_command_screen
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
SEED=61; EPS=6; PGD=20

# P3: tomato_sauce_s0_w60-70 (anchor neighbor)
echo "[$(date +%H:%M:%S)] TrackB P3 tomato_sauce_s0_w60-70"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 0 --window_start 60 --window_end 70 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953404 --pair_id tomato_sauce_s0_w60_70_s17c_seed61 --output_dir $OUT || echo "FAIL_P3_VIS"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 0 --window_start 60 --window_end 70 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953405 --pair_id tomato_sauce_s0_w60_70_s17c_seed61 --output_dir $OUT || echo "FAIL_P3_RAND"

# P4: bbq_sauce_s1_w50-60 (borderline retest)
echo "[$(date +%H:%M:%S)] TrackB P4 bbq_sauce_s1_w50-60"
$PY -u $S --gpu_pair 0,1 --task bbq_sauce --state-id 1 --window_start 50 --window_end 60 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953406 --pair_id bbq_sauce_s1_w50_60_s17c_seed61 --output_dir $OUT || echo "FAIL_P4_VIS"
$PY -u $S --gpu_pair 0,1 --task bbq_sauce --state-id 1 --window_start 50 --window_end 60 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953407 --pair_id bbq_sauce_s1_w50_60_s17c_seed61 --output_dir $OUT || echo "FAIL_P4_RAND"

echo "[$(date +%H:%M:%S)] TrackB GPU26 DONE"
