#!/bin/bash
# S17b Track A GPU10 — tomato_sauce_s0_w240-250 seed53 VIS+RAND + ORACLE
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES=1,0
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s17b_trackA_w240_250
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
TASK=tomato_sauce; SID=0; WS=240; WE=250; L=10; EPS=6; PGD=20
PAIR=tomato_sauce_s0_w240_250_s17b_seed53

echo "[$(date +%H:%M:%S)] S17b GPU10 — ORACLE sanity w240-250"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition oracle_open --open_duration $L --attack_seed 53 --job_id 953300 --pair_id ${PAIR} --output_dir $OUT || echo "FAIL_ORACLE"

echo "[$(date +%H:%M:%S)] S17b GPU10 — VIS seed53"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition vis_pgd --open_duration $L --attack_seed 53 --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953301 --pair_id ${PAIR} --output_dir $OUT || echo "FAIL_VIS_s53"

echo "[$(date +%H:%M:%S)] S17b GPU10 — RAND seed53"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration $L --attack_seed 53 --eps_raw_pixels $EPS --job_id 953302 --pair_id ${PAIR} --output_dir $OUT || echo "FAIL_RAND_s53"

echo "[$(date +%H:%M:%S)] S17b GPU10 DONE"
