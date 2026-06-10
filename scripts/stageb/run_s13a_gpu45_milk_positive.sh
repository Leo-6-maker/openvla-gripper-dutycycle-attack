#!/bin/bash
# S13a GPU45 — milk_s0_w70-80 positive-control seed24 VIS+RAND matched pair
set +e
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s13a_rand_veto_and_milk_positive
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py

echo "[$(date +%H:%M:%S)] S13a GPU45 milk_s0_w70-80 positive-control seed24 START"
export CUDA_VISIBLE_DEVICES=4,5

echo "  VIS milk seed=24"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition vis_pgd --open_duration 10 --attack_seed 24 --pgd_steps 20 --eps_raw_pixels 6 --job_id 952406 --pair_id milk_s0_w70_80_s13a_posctrl_seed24 --output_dir $OUT || echo "FAIL_VIS_s24"

echo "  RAND milk seed=24"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition random_linf --open_duration 10 --attack_seed 24 --eps_raw_pixels 6 --job_id 952407 --pair_id milk_s0_w70_80_s13a_posctrl_seed24 --output_dir $OUT || echo "FAIL_RAND_s24"

echo "[$(date +%H:%M:%S)] S13a GPU45 milk positive-control seed24 DONE"
