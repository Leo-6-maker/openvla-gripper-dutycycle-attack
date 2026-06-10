#!/bin/bash
# S16c GPU10 — milk_s0_w240-250 VIS+RAND seeds 51,52,53 (parent-level confirmation)
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s16c_wave1_confirmation
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
TASK=milk; SID=0; WS=240; WE=250; EPS=6; PGD=20

export CUDA_VISIBLE_DEVICES=1,0

echo "[$(date +%H:%M:%S)] S16c milk_s0_w240-250 seed51"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition vis_pgd --open_duration 10 --attack_seed 51 --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953100 --pair_id milk_s0_w240_250_s16c_seed51 --output_dir $OUT || echo "FAIL_VIS_s51"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration 10 --attack_seed 51 --eps_raw_pixels $EPS --job_id 953101 --pair_id milk_s0_w240_250_s16c_seed51 --output_dir $OUT || echo "FAIL_RAND_s51"

echo "[$(date +%H:%M:%S)] S16c milk_s0_w240-250 seed52"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition vis_pgd --open_duration 10 --attack_seed 52 --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953102 --pair_id milk_s0_w240_250_s16c_seed52 --output_dir $OUT || echo "FAIL_VIS_s52"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration 10 --attack_seed 52 --eps_raw_pixels $EPS --job_id 953103 --pair_id milk_s0_w240_250_s16c_seed52 --output_dir $OUT || echo "FAIL_RAND_s52"

echo "[$(date +%H:%M:%S)] S16c milk_s0_w240-250 seed53"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition vis_pgd --open_duration 10 --attack_seed 53 --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953104 --pair_id milk_s0_w240_250_s16c_seed53 --output_dir $OUT || echo "FAIL_VIS_s53"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration 10 --attack_seed 53 --eps_raw_pixels $EPS --job_id 953105 --pair_id milk_s0_w240_250_s16c_seed53 --output_dir $OUT || echo "FAIL_RAND_s53"

echo "[$(date +%H:%M:%S)] S16c GPU10 milk_s0_w240-250 DONE"
