#!/bin/bash
# Smoke worker: worker_10 GPU=1,0
# 2 parents, 4 jobs
set +e

export CUDA_VISIBLE_DEVICES=1,0

echo "[$(date +%H:%M:%S)] worker_10 SMOKE START: 2 parents"

echo "=== VIS 200001: alphabet_soup s1 [50,60] seed=1 cmd_expansion ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task alphabet_soup --state-id 1 --window_start 50 --window_end 60 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 400 --seed 1 \
  --job_id 200001 --pair_id smoke_cmd_expansion_alphabet_soup_s1_w50_60_seed1 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_smoke_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 200001 smoke_cmd_expansion_alphabet_soup_s1_w50_60_seed1"

echo "=== RAND 200002: alphabet_soup s1 [50,60] seed=1 cmd_expansion ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task alphabet_soup --state-id 1 --window_start 50 --window_end 60 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 400 --seed 1 \
  --job_id 200002 --pair_id smoke_cmd_expansion_alphabet_soup_s1_w50_60_seed1 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_smoke_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 200002 smoke_cmd_expansion_alphabet_soup_s1_w50_60_seed1"

echo "=== VIS 200005: bbq_sauce s1 [55,65] seed=1 cmd_expansion ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task bbq_sauce --state-id 1 --window_start 55 --window_end 65 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 400 --seed 1 \
  --job_id 200005 --pair_id smoke_cmd_expansion_bbq_sauce_s1_w55_65_seed1 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_smoke_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 200005 smoke_cmd_expansion_bbq_sauce_s1_w55_65_seed1"

echo "=== RAND 200006: bbq_sauce s1 [55,65] seed=1 cmd_expansion ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task bbq_sauce --state-id 1 --window_start 55 --window_end 65 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 400 --seed 1 \
  --job_id 200006 --pair_id smoke_cmd_expansion_bbq_sauce_s1_w55_65_seed1 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_smoke_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 200006 smoke_cmd_expansion_bbq_sauce_s1_w55_65_seed1"

echo "[$(date +%H:%M:%S)] worker_10 SMOKE DONE"

