#!/bin/bash
set +e
export CUDA_VISIBLE_DEVICES=4,5
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py
echo "[$(date +%H:%M:%S)] K5C RETRY FAILED START (3 jobs on GPU 4,5)"

# 1. k5c_cmd_milk_neg VIS atk=0 (was job 520060, CUDA error on GPU 2,6)
echo "=== RETRY k5c_cmd_milk_neg VIS atk=0 ==="
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 235 --window_end 245 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 0 --job_id 520060 --pair_id k5c_cmd_milk_neg --output_dir $OUT --image_preprocess official_rot180 || echo "RETRY_FAIL k5c_cmd_milk_neg VIS atk=0"

# 2. k5c_cmd_milk_neg RAND atk=2 (was job 520065, CUDA error on GPU 2,6)
echo "=== RETRY k5c_cmd_milk_neg RAND atk=2 ==="
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 235 --window_end 245 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 2 --job_id 520065 --pair_id k5c_cmd_milk_neg --output_dir $OUT --image_preprocess official_rot180 || echo "RETRY_FAIL k5c_cmd_milk_neg RAND atk=2"

# 3. k5c_cmd_alpha VIS atk=0 (was job 520090, CUDA error on GPU 2,6)
echo "=== RETRY k5c_cmd_alpha VIS atk=0 ==="
$PY -u $S --gpu_pair 0,1 --task alphabet_soup --state-id 0 --window_start 65 --window_end 75 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 0 --job_id 520090 --pair_id k5c_cmd_alpha --output_dir $OUT --image_preprocess official_rot180 || echo "RETRY_FAIL k5c_cmd_alpha VIS atk=0"

echo "[$(date +%H:%M:%S)] K5C RETRY DONE"
