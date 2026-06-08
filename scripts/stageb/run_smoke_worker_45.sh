#!/bin/bash
# Smoke worker: worker_45 GPU=4,5
# 2 parents, 4 jobs
set +e

export CUDA_VISIBLE_DEVICES=4,5

echo "[$(date +%H:%M:%S)] worker_45 SMOKE START: 2 parents"

echo "=== VIS 200009: bbq_sauce s2 [100,110] seed=2 hard_negative ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task bbq_sauce --state-id 2 --window_start 100 --window_end 110 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 400 --seed 2 \
  --job_id 200009 --pair_id smoke_hard_negative_bbq_sauce_s2_w100_110_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_smoke_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 200009 smoke_hard_negative_bbq_sauce_s2_w100_110_seed2"

echo "=== RAND 200010: bbq_sauce s2 [100,110] seed=2 hard_negative ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task bbq_sauce --state-id 2 --window_start 100 --window_end 110 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 400 --seed 2 \
  --job_id 200010 --pair_id smoke_hard_negative_bbq_sauce_s2_w100_110_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_smoke_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 200010 smoke_hard_negative_bbq_sauce_s2_w100_110_seed2"

echo "=== VIS 200013: tomato_sauce s2 [90,100] seed=2 rand_abstain ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 400 --seed 2 \
  --job_id 200013 --pair_id smoke_rand_abstain_tomato_sauce_s2_w90_100_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_smoke_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 200013 smoke_rand_abstain_tomato_sauce_s2_w90_100_seed2"

echo "=== RAND 200014: tomato_sauce s2 [90,100] seed=2 rand_abstain ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 400 --seed 2 \
  --job_id 200014 --pair_id smoke_rand_abstain_tomato_sauce_s2_w90_100_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_smoke_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 200014 smoke_rand_abstain_tomato_sauce_s2_w90_100_seed2"

echo "[$(date +%H:%M:%S)] worker_45 SMOKE DONE"

