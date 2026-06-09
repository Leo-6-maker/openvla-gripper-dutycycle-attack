#!/bin/bash
set +e
export CUDA_VISIBLE_DEVICES=2,6
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/gpu26_health_check
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py

echo "[$(date +%H:%M:%S)] GPU 2,6 HEALTH CHECK START"

# Single VIS smoke on known GOLD parent
echo "=== VIS milk[70,80] atk=0 on GPU 2,6 ==="
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 0 --job_id 999001 --pair_id gpu26_health_vis --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL on GPU 2,6"

# Single RAND smoke
echo "=== RAND milk[70,80] atk=0 on GPU 2,6 ==="
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 0 --job_id 999002 --pair_id gpu26_health_rand --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL on GPU 2,6"

echo "[$(date +%H:%M:%S)] GPU 2,6 HEALTH CHECK DONE"
