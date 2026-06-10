#!/bin/bash
set +e
export CUDA_VISIBLE_DEVICES=1,0
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/layer2_hiddensafe_confirmation/shard10
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py

echo "[$(date +%H:%M:%S)] L2_CONFIRM_shard10 START (6 pairs, 12 jobs)"

echo "  VIS cream_cheese_s0_w65_75__atk10 atk=10"
$PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 65 --window_end 75 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 10 --job_id 800000 --pair_id cream_cheese_s0_w65_75__atk10 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL cream_cheese_s0_w65_75__atk10 atk=10"
echo "  RAND cream_cheese_s0_w65_75__atk10 atk=10"
$PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 65 --window_end 75 --condition random_linf --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 10 --job_id 800001 --pair_id cream_cheese_s0_w65_75__atk10 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL cream_cheese_s0_w65_75__atk10 atk=10"
echo "  VIS bbq_sauce_s2_w100_110__atk10 atk=10"
$PY -u $S --gpu_pair 0,1 --task bbq_sauce --state-id 2 --window_start 100 --window_end 110 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed 10 --job_id 800002 --pair_id bbq_sauce_s2_w100_110__atk10 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL bbq_sauce_s2_w100_110__atk10 atk=10"
echo "  RAND bbq_sauce_s2_w100_110__atk10 atk=10"
$PY -u $S --gpu_pair 0,1 --task bbq_sauce --state-id 2 --window_start 100 --window_end 110 --condition random_linf --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed 10 --job_id 800003 --pair_id bbq_sauce_s2_w100_110__atk10 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL bbq_sauce_s2_w100_110__atk10 atk=10"
echo "  VIS cream_cheese_s0_w65_75__atk9 atk=9"
$PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 65 --window_end 75 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 9 --job_id 800004 --pair_id cream_cheese_s0_w65_75__atk9 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL cream_cheese_s0_w65_75__atk9 atk=9"
echo "  RAND cream_cheese_s0_w65_75__atk9 atk=9"
$PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 65 --window_end 75 --condition random_linf --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 9 --job_id 800005 --pair_id cream_cheese_s0_w65_75__atk9 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL cream_cheese_s0_w65_75__atk9 atk=9"
echo "  VIS bbq_sauce_s2_w100_110__atk9 atk=9"
$PY -u $S --gpu_pair 0,1 --task bbq_sauce --state-id 2 --window_start 100 --window_end 110 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed 9 --job_id 800006 --pair_id bbq_sauce_s2_w100_110__atk9 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL bbq_sauce_s2_w100_110__atk9 atk=9"
echo "  RAND bbq_sauce_s2_w100_110__atk9 atk=9"
$PY -u $S --gpu_pair 0,1 --task bbq_sauce --state-id 2 --window_start 100 --window_end 110 --condition random_linf --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed 9 --job_id 800007 --pair_id bbq_sauce_s2_w100_110__atk9 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL bbq_sauce_s2_w100_110__atk9 atk=9"
echo "  VIS milk_s0_w70_80__atk10 atk=10"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 10 --job_id 800008 --pair_id milk_s0_w70_80__atk10 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL milk_s0_w70_80__atk10 atk=10"
echo "  RAND milk_s0_w70_80__atk10 atk=10"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition random_linf --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 10 --job_id 800009 --pair_id milk_s0_w70_80__atk10 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL milk_s0_w70_80__atk10 atk=10"
echo "  VIS butter_s0_w90_100__atk10 atk=10"
$PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 90 --window_end 100 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 10 --job_id 800010 --pair_id butter_s0_w90_100__atk10 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL butter_s0_w90_100__atk10 atk=10"
echo "  RAND butter_s0_w90_100__atk10 atk=10"
$PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 90 --window_end 100 --condition random_linf --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 10 --job_id 800011 --pair_id butter_s0_w90_100__atk10 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL butter_s0_w90_100__atk10 atk=10"

echo "[$(date +%H:%M:%S)] L2_CONFIRM_shard10 DONE"
