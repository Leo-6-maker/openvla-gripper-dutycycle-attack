#!/bin/bash
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES=4,5
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s17a_patched_runner_smoke
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
TASK=tomato_sauce; SID=0; WS=70; WE=80; L=10; EPS=6; PGD=20
PAIR=tomato_sauce_s0_w70_80_s17a_smoke_seed60

echo "[$(date +%H:%M:%S)] S17a GPU45 — VIS seed60"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition vis_pgd --open_duration $L --attack_seed 60 --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953202 --pair_id $PAIR --output_dir $OUT || echo "FAIL_VIS"

echo "[$(date +%H:%M:%S)] S17a GPU45 — RAND seed60"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration $L --attack_seed 60 --eps_raw_pixels $EPS --job_id 953203 --pair_id $PAIR --output_dir $OUT || echo "FAIL_RAND"

echo "[$(date +%H:%M:%S)] S17a GPU45 DONE"
