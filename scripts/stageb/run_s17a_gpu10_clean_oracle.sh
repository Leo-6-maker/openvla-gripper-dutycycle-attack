#!/bin/bash
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES=1,0
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s17a_patched_runner_smoke
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
TASK=tomato_sauce; SID=0; WS=70; WE=80; L=10
PAIR=tomato_sauce_s0_w70_80_s17a_smoke_seed60

echo "[$(date +%H:%M:%S)] S17a GPU10 — clean"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition clean --open_duration $L --attack_seed 60 --job_id 953200 --pair_id $PAIR --output_dir $OUT || echo "FAIL_CLEAN"

echo "[$(date +%H:%M:%S)] S17a GPU10 — ORACLE"
$PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition oracle_open --open_duration $L --attack_seed 60 --job_id 953201 --pair_id $PAIR --output_dir $OUT || echo "FAIL_ORACLE"

echo "[$(date +%H:%M:%S)] S17a GPU10 DONE"
