Warning: Permanently added '10.60.133.3' (ED25519) to the list of known hosts.
Warning: Permanently added '10.60.133.4' (ED25519) to the list of known hosts.
#!/bin/bash
# Expansion worker: worker_10 GPU=1,0
# 7 parents, 14 jobs (job_id range 300000-300013)
set +e

export CUDA_VISIBLE_DEVICES=1,0

echo "[$(date +%H:%M:%S)] worker_10 EXPANSION START: 7 parents"

echo "=== VIS 300000: bbq_sauce s2 [200,210] seed=2 cmd_expansion ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task bbq_sauce --state-id 2 --window_start 200 --window_end 210 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 299 --seed 2 \
  --job_id 300000 --pair_id exp_cmd_expansion_bbq_sauce_s2_w200_210_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300000 exp_cmd_expansion_bbq_sauce_s2_w200_210_seed2"

echo "=== RAND 300001: bbq_sauce s2 [200,210] seed=2 cmd_expansion ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task bbq_sauce --state-id 2 --window_start 200 --window_end 210 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 299 --seed 2 \
  --job_id 300001 --pair_id exp_cmd_expansion_bbq_sauce_s2_w200_210_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300001 exp_cmd_expansion_bbq_sauce_s2_w200_210_seed2"

echo "=== VIS 300002: salad_dressing s2 [110,120] seed=2 cmd_expansion ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task salad_dressing --state-id 2 --window_start 110 --window_end 120 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 262 --seed 2 \
  --job_id 300002 --pair_id exp_cmd_expansion_salad_dressing_s2_w110_120_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300002 exp_cmd_expansion_salad_dressing_s2_w110_120_seed2"

echo "=== RAND 300003: salad_dressing s2 [110,120] seed=2 cmd_expansion ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task salad_dressing --state-id 2 --window_start 110 --window_end 120 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 262 --seed 2 \
  --job_id 300003 --pair_id exp_cmd_expansion_salad_dressing_s2_w110_120_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300003 exp_cmd_expansion_salad_dressing_s2_w110_120_seed2"

echo "=== VIS 300004: cream_cheese s0 [85,95] seed=0 phys_enrichment ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task cream_cheese --state-id 0 --window_start 85 --window_end 95 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 147 --seed 0 \
  --job_id 300004 --pair_id exp_phys_enrichment_cream_cheese_s0_w85_95_seed0 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300004 exp_phys_enrichment_cream_cheese_s0_w85_95_seed0"

echo "=== RAND 300005: cream_cheese s0 [85,95] seed=0 phys_enrichment ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task cream_cheese --state-id 0 --window_start 85 --window_end 95 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 147 --seed 0 \
  --job_id 300005 --pair_id exp_phys_enrichment_cream_cheese_s0_w85_95_seed0 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300005 exp_phys_enrichment_cream_cheese_s0_w85_95_seed0"

echo "=== VIS 300006: alphabet_soup s1 [65,75] seed=1 hard_neg_candidate ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task alphabet_soup --state-id 1 --window_start 65 --window_end 75 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 299 --seed 1 \
  --job_id 300006 --pair_id exp_hard_neg_candidate_alphabet_soup_s1_w65_75_seed1 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300006 exp_hard_neg_candidate_alphabet_soup_s1_w65_75_seed1"

echo "=== RAND 300007: alphabet_soup s1 [65,75] seed=1 hard_neg_candidate ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task alphabet_soup --state-id 1 --window_start 65 --window_end 75 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 299 --seed 1 \
  --job_id 300007 --pair_id exp_hard_neg_candidate_alphabet_soup_s1_w65_75_seed1 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300007 exp_hard_neg_candidate_alphabet_soup_s1_w65_75_seed1"

echo "=== VIS 300008: milk s0 [240,250] seed=0 hard_neg_candidate ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task milk --state-id 0 --window_start 240 --window_end 250 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 299 --seed 0 \
  --job_id 300008 --pair_id exp_hard_neg_candidate_milk_s0_w240_250_seed0 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300008 exp_hard_neg_candidate_milk_s0_w240_250_seed0"

echo "=== RAND 300009: milk s0 [240,250] seed=0 hard_neg_candidate ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task milk --state-id 0 --window_start 240 --window_end 250 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 299 --seed 0 \
  --job_id 300009 --pair_id exp_hard_neg_candidate_milk_s0_w240_250_seed0 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300009 exp_hard_neg_candidate_milk_s0_w240_250_seed0"

echo "=== VIS 300010: salad_dressing s2 [120,130] seed=2 hard_neg_candidate ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task salad_dressing --state-id 2 --window_start 120 --window_end 130 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 262 --seed 2 \
  --job_id 300010 --pair_id exp_hard_neg_candidate_salad_dressing_s2_w120_130_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300010 exp_hard_neg_candidate_salad_dressing_s2_w120_130_seed2"

echo "=== RAND 300011: salad_dressing s2 [120,130] seed=2 hard_neg_candidate ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task salad_dressing --state-id 2 --window_start 120 --window_end 130 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 262 --seed 2 \
  --job_id 300011 --pair_id exp_hard_neg_candidate_salad_dressing_s2_w120_130_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300011 exp_hard_neg_candidate_salad_dressing_s2_w120_130_seed2"

echo "=== VIS 300012: milk s0 [70,80] seed=0 sentinel_repeat ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task milk --state-id 0 --window_start 70 --window_end 80 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 299 --seed 0 \
  --job_id 300012 --pair_id exp_sentinel_repeat_milk_s0_w70_80_seed0 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300012 exp_sentinel_repeat_milk_s0_w70_80_seed0"

echo "=== RAND 300013: milk s0 [70,80] seed=0 sentinel_repeat ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task milk --state-id 0 --window_start 70 --window_end 80 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 299 --seed 0 \
  --job_id 300013 --pair_id exp_sentinel_repeat_milk_s0_w70_80_seed0 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300013 exp_sentinel_repeat_milk_s0_w70_80_seed0"

echo "[$(date +%H:%M:%S)] worker_10 EXPANSION DONE"

