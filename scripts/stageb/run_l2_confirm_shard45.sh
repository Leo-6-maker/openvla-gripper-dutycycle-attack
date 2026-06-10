#!/bin/bash
set +e
export CUDA_VISIBLE_DEVICES=4,5
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/layer2_hiddensafe_confirmation/shard45
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py

echo "[$(date +%H:%M:%S)] L2_CONFIRM_shard45 START (5 pairs, 10 jobs)"

echo "  VIS milk_s0_w70_80__atk9 atk=9"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 9 --job_id 810000 --pair_id milk_s0_w70_80__atk9 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL milk_s0_w70_80__atk9 atk=9"
echo "  RAND milk_s0_w70_80__atk9 atk=9"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition random_linf --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 9 --job_id 810001 --pair_id milk_s0_w70_80__atk9 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL milk_s0_w70_80__atk9 atk=9"
echo "  VIS butter_s0_w90_100__atk9 atk=9"
$PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 90 --window_end 100 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 9 --job_id 810002 --pair_id butter_s0_w90_100__atk9 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL butter_s0_w90_100__atk9 atk=9"
echo "  RAND butter_s0_w90_100__atk9 atk=9"
$PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 90 --window_end 100 --condition random_linf --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 9 --job_id 810003 --pair_id butter_s0_w90_100__atk9 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL butter_s0_w90_100__atk9 atk=9"
echo "  VIS salad_dressing_s1_w50_60__atk10 atk=10"
$PY -u $S --gpu_pair 0,1 --task salad_dressing --state-id 1 --window_start 50 --window_end 60 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 1 --attack_seed 10 --job_id 810004 --pair_id salad_dressing_s1_w50_60__atk10 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL salad_dressing_s1_w50_60__atk10 atk=10"
echo "  RAND salad_dressing_s1_w50_60__atk10 atk=10"
$PY -u $S --gpu_pair 0,1 --task salad_dressing --state-id 1 --window_start 50 --window_end 60 --condition random_linf --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 1 --attack_seed 10 --job_id 810005 --pair_id salad_dressing_s1_w50_60__atk10 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL salad_dressing_s1_w50_60__atk10 atk=10"
echo "  VIS cream_cheese_s0_w85_95__atk10 atk=10"
$PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 85 --window_end 95 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 10 --job_id 810006 --pair_id cream_cheese_s0_w85_95__atk10 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL cream_cheese_s0_w85_95__atk10 atk=10"
echo "  RAND cream_cheese_s0_w85_95__atk10 atk=10"
$PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 85 --window_end 95 --condition random_linf --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 10 --job_id 810007 --pair_id cream_cheese_s0_w85_95__atk10 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL cream_cheese_s0_w85_95__atk10 atk=10"
echo "  VIS salad_dressing_s1_w50_60__atk9 atk=9"
$PY -u $S --gpu_pair 0,1 --task salad_dressing --state-id 1 --window_start 50 --window_end 60 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 1 --attack_seed 9 --job_id 810008 --pair_id salad_dressing_s1_w50_60__atk9 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL salad_dressing_s1_w50_60__atk9 atk=9"
echo "  RAND salad_dressing_s1_w50_60__atk9 atk=9"
$PY -u $S --gpu_pair 0,1 --task salad_dressing --state-id 1 --window_start 50 --window_end 60 --condition random_linf --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 1 --attack_seed 9 --job_id 810009 --pair_id salad_dressing_s1_w50_60__atk9 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL salad_dressing_s1_w50_60__atk9 atk=9"

echo "[$(date +%H:%M:%S)] L2_CONFIRM_shard45 DONE"
