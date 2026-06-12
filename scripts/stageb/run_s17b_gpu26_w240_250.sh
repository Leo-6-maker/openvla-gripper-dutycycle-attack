#!/bin/bash
# S17b Track A GPU26 — tomato_sauce_s0_w240-250 seed54 VIS+RAND
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES=2,6
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s17b_trackA_w240_250
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
TASK=tomato_sauce; SID=0; WS=240; WE=250; L=10; EPS=6; PGD=20
PAIR=tomato_sauce_s0_w240_250_s17b_seed54

echo "[$(date +%H:%M:%S)] S17b GPU26 — VIS seed54"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition vis_pgd --open_duration $L --attack_seed 54 --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953303 --pair_id ${PAIR} --output_dir $OUT || echo "FAIL_VIS_s54"

echo "[$(date +%H:%M:%S)] S17b GPU26 — RAND seed54"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration $L --attack_seed 54 --eps_raw_pixels $EPS --job_id 953304 --pair_id ${PAIR} --output_dir $OUT || echo "FAIL_RAND_s54"

echo "[$(date +%H:%M:%S)] S17b GPU26 DONE"
