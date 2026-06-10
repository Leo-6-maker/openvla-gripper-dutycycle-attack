#!/bin/bash
set +e
export CUDA_VISIBLE_DEVICES=2,6
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/layer2_hiddensafe_confirmation/shard26_retry_tomato
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py

echo "TOMATO RETRY START"

echo "  VIS tomato_s2_w165_175 atk=9"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 165 --window_end 175 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed 9 --job_id 820100 --pair_id tomato_sauce_s2_w165_175__atk9 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL tomato atk=9"

echo "  RAND tomato_s2_w165_175 atk=9"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 165 --window_end 175 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed 9 --job_id 820101 --pair_id tomato_sauce_s2_w165_175__atk9 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL tomato atk=9"

echo "  VIS tomato_s2_w165_175 atk=10"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 165 --window_end 175 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed 10 --job_id 820102 --pair_id tomato_sauce_s2_w165_175__atk10 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL tomato atk=10"

echo "  RAND tomato_s2_w165_175 atk=10"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 165 --window_end 175 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed 10 --job_id 820103 --pair_id tomato_sauce_s2_w165_175__atk10 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL tomato atk=10"

echo "TOMATO RETRY DONE"
