#!/bin/bash
# S9b Phase1-runner attack port — sanity only: clean + oracle_open
# Confirm Phase1 ORACLE physical reachability survives the port
set +e
export CUDA_VISIBLE_DEVICES=1,0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s9b_phase1_runner_port_sanity
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py

echo "[$(date +%H:%M:%S)] S9b PHASE1-PORT SANITY START (2 jobs)"

echo "  CLEAN milk L=10"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition clean --open_duration 10 --attack_seed 0 --job_id 950400 --pair_id milk_s0_w70_80_phase1port_sanity --output_dir $OUT || echo "FAIL_CLEAN"

echo "  ORACLE milk L=10"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition oracle_open --open_duration 10 --attack_seed 0 --job_id 950401 --pair_id milk_s0_w70_80_phase1port_sanity --output_dir $OUT || echo "FAIL_ORACLE"

echo "[$(date +%H:%M:%S)] S9b PHASE1-PORT SANITY DONE"
