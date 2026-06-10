#!/bin/bash
# S16c GPU26 — tomato_s2_w95-105 RAND-only seeds 51,52,53 (3-seed RAND-veto)
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s16c_wave1_confirmation
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
TASK=tomato_sauce; SID=2; WS=95; WE=105; EPS=6

export CUDA_VISIBLE_DEVICES=2,6

echo "[$(date +%H:%M:%S)] S16c tomato_s2_w95-105 RAND-veto START"

echo "  RAND seed=51"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration 10 --attack_seed 51 --eps_raw_pixels $EPS --job_id 953106 --pair_id tomato_s2_w95_105_s16c_randveto_seed51 --output_dir $OUT || echo "FAIL_RAND_s51"

echo "  RAND seed=52"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration 10 --attack_seed 52 --eps_raw_pixels $EPS --job_id 953107 --pair_id tomato_s2_w95_105_s16c_randveto_seed52 --output_dir $OUT || echo "FAIL_RAND_s52"

echo "  RAND seed=53"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration 10 --attack_seed 53 --eps_raw_pixels $EPS --job_id 953108 --pair_id tomato_s2_w95_105_s16c_randveto_seed53 --output_dir $OUT || echo "FAIL_RAND_s53"

echo "[$(date +%H:%M:%S)] S16c GPU26 tomato RAND-veto DONE"
