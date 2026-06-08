#!/bin/bash
set +e
export CUDA_VISIBLE_DEVICES=1,0
OUT=/data/liuyu/outputs/stageb_v1_1_k5b_targeted_stability_rc1a_0e3428f/k5b_smoke
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py
echo "[$(date +%H:%M:%S)] K5B SMOKE START"

# milk [240,250] — contrast, env=0, atk=0,1
echo "=== VIS milk_contrast atk0 ==="
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 240 --window_end 250 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 0 --job_id 520000 --pair_id k5b_smoke_milk_contrast_s0_w240_250_env0_atk0 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL milk_r0"
echo "=== RAND milk_contrast atk0 ==="
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 240 --window_end 250 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 0 --job_id 520001 --pair_id k5b_smoke_milk_contrast_s0_w240_250_env0_atk0 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL milk_r0"
echo "=== VIS milk_contrast atk1 ==="
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 240 --window_end 250 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 1 --job_id 520002 --pair_id k5b_smoke_milk_contrast_s0_w240_250_env0_atk1 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL milk_r1"
echo "=== RAND milk_contrast atk1 ==="
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 240 --window_end 250 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 1 --job_id 520003 --pair_id k5b_smoke_milk_contrast_s0_w240_250_env0_atk1 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL milk_r1"

# alphabet [60,70] — rand, env=0, atk=0,1
echo "=== VIS alphabet_rand atk0 ==="
$PY -u $S --gpu_pair 0,1 --task alphabet_soup --state-id 0 --window_start 60 --window_end 70 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 0 --job_id 520004 --pair_id k5b_smoke_rand_alpha_s0_w60_70_env0_atk0 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL alpha_r0"
echo "=== RAND alphabet_rand atk0 ==="
$PY -u $S --gpu_pair 0,1 --task alphabet_soup --state-id 0 --window_start 60 --window_end 70 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 0 --job_id 520005 --pair_id k5b_smoke_rand_alpha_s0_w60_70_env0_atk0 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL alpha_r0"
echo "=== VIS alphabet_rand atk1 ==="
$PY -u $S --gpu_pair 0,1 --task alphabet_soup --state-id 0 --window_start 60 --window_end 70 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 1 --job_id 520006 --pair_id k5b_smoke_rand_alpha_s0_w60_70_env0_atk1 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL alpha_r1"
echo "=== RAND alphabet_rand atk1 ==="
$PY -u $S --gpu_pair 0,1 --task alphabet_soup --state-id 0 --window_start 60 --window_end 70 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 1 --job_id 520007 --pair_id k5b_smoke_rand_alpha_s0_w60_70_env0_atk1 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL alpha_r1"

echo "[$(date +%H:%M:%S)] K5B SMOKE DONE"
