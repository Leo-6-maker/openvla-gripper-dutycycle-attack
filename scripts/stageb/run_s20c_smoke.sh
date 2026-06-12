#!/bin/bash
# S20c L3 smoke: ketchup + tomato_sauce state0 clean under official-aligned runner
set +e
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s20c_l3_smoke
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s20c_official_l3_runner.py

echo "[$(date +%H:%M:%S)] S20c SMOKE — ketchup state0 clean (GPU 1,0)"
export CUDA_VISIBLE_DEVICES=1,0
$PY -u $S --gpu_pair 0,1 --task ketchup --state_id 0 --condition clean --attack_seed 0 --job_id 959000 --output_dir $OUT || echo "FAIL_ketchup"

echo "[$(date +%H:%M:%S)] S20c SMOKE — tomato_sauce state0 clean (GPU 2,6)"
export CUDA_VISIBLE_DEVICES=2,6
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state_id 0 --condition clean --attack_seed 0 --job_id 959001 --output_dir $OUT || echo "FAIL_tomato"

echo "[$(date +%H:%M:%S)] S20c SMOKE DONE"
