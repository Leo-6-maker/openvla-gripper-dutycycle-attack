#!/bin/bash
# S15b GPU10 — C01 cream_s0_w85-95 v0.4 forward screen
# Jobs: ORACLE + RAND×3 + VIS command-probe seed27
set +e
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s15b_v04_forward_screen
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py

TASK=cream_cheese; SID=0; WS=85; WE=95; L=10
PAIR="cream_s0_w85_95_s15b_v04_fwd"

echo "[$(date +%H:%M:%S)] S15b GPU10 ${PAIR} START"
export CUDA_VISIBLE_DEVICES=1,0

echo "  ORACLE"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition oracle_open --open_duration $L --attack_seed 0 --job_id 952700 --pair_id ${PAIR} --output_dir $OUT || echo "FAIL_ORACLE"

echo "  RAND seed=27"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration $L --attack_seed 27 --eps_raw_pixels 6 --job_id 952701 --pair_id ${PAIR} --output_dir $OUT || echo "FAIL_RAND_s27"

echo "  RAND seed=28"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration $L --attack_seed 28 --eps_raw_pixels 6 --job_id 952702 --pair_id ${PAIR} --output_dir $OUT || echo "FAIL_RAND_s28"

echo "  RAND seed=29"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration $L --attack_seed 29 --eps_raw_pixels 6 --job_id 952703 --pair_id ${PAIR} --output_dir $OUT || echo "FAIL_RAND_s29"

echo "  VIS cmd-probe seed=27"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition vis_pgd --open_duration $L --attack_seed 27 --pgd_steps 20 --eps_raw_pixels 6 --job_id 952704 --pair_id ${PAIR} --output_dir $OUT || echo "FAIL_VIS_s27"

echo "[$(date +%H:%M:%S)] S15b GPU10 ${PAIR} DONE"
