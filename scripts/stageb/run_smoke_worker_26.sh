#!/bin/bash
# Smoke worker: worker_26 GPU=2,6
# 2 parents, 4 jobs
set +e

export CUDA_VISIBLE_DEVICES=2,6

echo "[$(date +%H:%M:%S)] worker_26 SMOKE START: 2 parents"

echo "=== VIS 200005: cream_cheese s2 [50,60] seed=2 phys_enrichment ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task cream_cheese --state-id 2 --window_start 50 --window_end 60 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 400 --seed 2 \
  --job_id 200005 --pair_id smoke_phys_enrichment_cream_cheese_s2_w50_60_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_smoke_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 200005 smoke_phys_enrichment_cream_cheese_s2_w50_60_seed2"

echo "=== RAND 200006: cream_cheese s2 [50,60] seed=2 phys_enrichment ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task cream_cheese --state-id 2 --window_start 50 --window_end 60 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 400 --seed 2 \
  --job_id 200006 --pair_id smoke_phys_enrichment_cream_cheese_s2_w50_60_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_smoke_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 200006 smoke_phys_enrichment_cream_cheese_s2_w50_60_seed2"

echo "=== VIS 200009: orange_juice s2 [20,30] seed=2 phys_enrichment ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task orange_juice --state-id 2 --window_start 20 --window_end 30 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 400 --seed 2 \
  --job_id 200009 --pair_id smoke_phys_enrichment_orange_juice_s2_w20_30_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_smoke_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 200009 smoke_phys_enrichment_orange_juice_s2_w20_30_seed2"

echo "=== RAND 200010: orange_juice s2 [20,30] seed=2 phys_enrichment ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task orange_juice --state-id 2 --window_start 20 --window_end 30 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 400 --seed 2 \
  --job_id 200010 --pair_id smoke_phys_enrichment_orange_juice_s2_w20_30_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_smoke_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 200010 smoke_phys_enrichment_orange_juice_s2_w20_30_seed2"

echo "[$(date +%H:%M:%S)] worker_26 SMOKE DONE"

