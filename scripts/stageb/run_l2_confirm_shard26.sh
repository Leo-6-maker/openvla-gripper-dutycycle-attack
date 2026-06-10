#!/bin/bash
set +e
export CUDA_VISIBLE_DEVICES=2,6
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/layer2_hiddensafe_confirmation/shard26
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py

echo "[$(date +%H:%M:%S)] L2_CONFIRM_shard26 START (5 pairs, 10 jobs)"

echo "  VIS cream_cheese_s0_w85_95__atk9 atk=9"
$PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 85 --window_end 95 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 9 --job_id 820000 --pair_id cream_cheese_s0_w85_95__atk9 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL cream_cheese_s0_w85_95__atk9 atk=9"
echo "  RAND cream_cheese_s0_w85_95__atk9 atk=9"
$PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 85 --window_end 95 --condition random_linf --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 9 --job_id 820001 --pair_id cream_cheese_s0_w85_95__atk9 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL cream_cheese_s0_w85_95__atk9 atk=9"
echo "  VIS tomato_sauce_s2_w165_175__atk10 atk=10"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 165 --window_end 175 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed 10 --job_id 820002 --pair_id tomato_sauce_s2_w165_175__atk10 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL tomato_sauce_s2_w165_175__atk10 atk=10"
echo "  RAND tomato_sauce_s2_w165_175__atk10 atk=10"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 165 --window_end 175 --condition random_linf --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed 10 --job_id 820003 --pair_id tomato_sauce_s2_w165_175__atk10 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL tomato_sauce_s2_w165_175__atk10 atk=10"
echo "  VIS milk_s0_w230_240__atk10 atk=10"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 230 --window_end 240 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 10 --job_id 820004 --pair_id milk_s0_w230_240__atk10 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL milk_s0_w230_240__atk10 atk=10"
echo "  RAND milk_s0_w230_240__atk10 atk=10"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 230 --window_end 240 --condition random_linf --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 10 --job_id 820005 --pair_id milk_s0_w230_240__atk10 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL milk_s0_w230_240__atk10 atk=10"
echo "  VIS tomato_sauce_s2_w165_175__atk9 atk=9"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 165 --window_end 175 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed 9 --job_id 820006 --pair_id tomato_sauce_s2_w165_175__atk9 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL tomato_sauce_s2_w165_175__atk9 atk=9"
echo "  RAND tomato_sauce_s2_w165_175__atk9 atk=9"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 165 --window_end 175 --condition random_linf --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed 9 --job_id 820007 --pair_id tomato_sauce_s2_w165_175__atk9 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL tomato_sauce_s2_w165_175__atk9 atk=9"
echo "  VIS milk_s0_w230_240__atk9 atk=9"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 230 --window_end 240 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 9 --job_id 820008 --pair_id milk_s0_w230_240__atk9 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL milk_s0_w230_240__atk9 atk=9"
echo "  RAND milk_s0_w230_240__atk9 atk=9"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 230 --window_end 240 --condition random_linf --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 9 --job_id 820009 --pair_id milk_s0_w230_240__atk9 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL milk_s0_w230_240__atk9 atk=9"

echo "[$(date +%H:%M:%S)] L2_CONFIRM_shard26 DONE"
