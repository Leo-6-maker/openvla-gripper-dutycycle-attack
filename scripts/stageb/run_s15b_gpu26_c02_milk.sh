#!/bin/bash
# S15b GPU26 — C02 milk_s0_w235-245 v0.4 forward screen
# Jobs: ORACLE + RAND×3 + VIS command-probe seed27
set +e
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s15b_v04_forward_screen
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py

TASK=milk; SID=0; WS=235; WE=245; L=10
PAIR="milk_s0_w235_245_s15b_v04_fwd"

echo "[$(date +%H:%M:%S)] S15b GPU26 ${PAIR} START"
export CUDA_VISIBLE_DEVICES=2,6

echo "  ORACLE"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition oracle_open --open_duration $L --attack_seed 0 --job_id 952705 --pair_id ${PAIR} --output_dir $OUT || echo "FAIL_ORACLE"

echo "  RAND seed=27"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration $L --attack_seed 27 --eps_raw_pixels 6 --job_id 952706 --pair_id ${PAIR} --output_dir $OUT || echo "FAIL_RAND_s27"

echo "  RAND seed=28"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration $L --attack_seed 28 --eps_raw_pixels 6 --job_id 952707 --pair_id ${PAIR} --output_dir $OUT || echo "FAIL_RAND_s28"

echo "  RAND seed=29"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration $L --attack_seed 29 --eps_raw_pixels 6 --job_id 952708 --pair_id ${PAIR} --output_dir $OUT || echo "FAIL_RAND_s29"

echo "  VIS cmd-probe seed=27"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition vis_pgd --open_duration $L --attack_seed 27 --pgd_steps 20 --eps_raw_pixels 6 --job_id 952709 --pair_id ${PAIR} --output_dir $OUT || echo "FAIL_VIS_s27"

echo "[$(date +%H:%M:%S)] S15b GPU26 ${PAIR} DONE"
