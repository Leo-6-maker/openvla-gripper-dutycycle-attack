#!/bin/bash
# S20a GPU45 — ketchup w150-160 seed76 VIS+RAND
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES=4,5
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s20a_ketchup_confirmation
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
TASK=ketchup; SID=0; WS=150; WE=160; EPS=6; PGD=20; L=10

echo "[$(date +%H:%M:%S)] S20a seed76 VIS"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition vis_pgd --open_duration $L --attack_seed 76 --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 956005 --pair_id ketchup_s0_w150_160_s20a_seed76 --output_dir $OUT || echo "FAIL_VIS_s76"

echo "[$(date +%H:%M:%S)] S20a seed76 RAND (explicit random_control_seed=76)"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration $L --attack_seed 76 --eps_raw_pixels $EPS --random_control_seed 76 --job_id 956006 --pair_id ketchup_s0_w150_160_s20a_seed76 --output_dir $OUT || echo "FAIL_RAND_s76"

echo "[$(date +%H:%M:%S)] S20a GPU45 DONE"
