Warning: Permanently added '10.60.133.3' (ED25519) to the list of known hosts.
Warning: Permanently added '10.60.133.4' (ED25519) to the list of known hosts.
#!/bin/bash
# Expansion worker: worker_45 GPU=4,5
# 7 parents, 14 jobs (job_id range 300028-300041)
set +e

export CUDA_VISIBLE_DEVICES=4,5

echo "[$(date +%H:%M:%S)] worker_45 EXPANSION START: 7 parents"

echo "=== VIS 300028: salad_dressing s1 [50,60] seed=1 cmd_expansion ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task salad_dressing --state-id 1 --window_start 50 --window_end 60 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 114 --seed 1 \
  --job_id 300028 --pair_id exp_cmd_expansion_salad_dressing_s1_w50_60_seed1 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300028 exp_cmd_expansion_salad_dressing_s1_w50_60_seed1"

echo "=== RAND 300029: salad_dressing s1 [50,60] seed=1 cmd_expansion ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task salad_dressing --state-id 1 --window_start 50 --window_end 60 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 114 --seed 1 \
  --job_id 300029 --pair_id exp_cmd_expansion_salad_dressing_s1_w50_60_seed1 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300029 exp_cmd_expansion_salad_dressing_s1_w50_60_seed1"

echo "=== VIS 300030: milk s0 [230,240] seed=0 cmd_expansion ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task milk --state-id 0 --window_start 230 --window_end 240 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 299 --seed 0 \
  --job_id 300030 --pair_id exp_cmd_expansion_milk_s0_w230_240_seed0 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300030 exp_cmd_expansion_milk_s0_w230_240_seed0"

echo "=== RAND 300031: milk s0 [230,240] seed=0 cmd_expansion ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task milk --state-id 0 --window_start 230 --window_end 240 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 299 --seed 0 \
  --job_id 300031 --pair_id exp_cmd_expansion_milk_s0_w230_240_seed0 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300031 exp_cmd_expansion_milk_s0_w230_240_seed0"

echo "=== VIS 300032: tomato_sauce s2 [150,160] seed=2 phys_enrichment ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task tomato_sauce --state-id 2 --window_start 150 --window_end 160 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 299 --seed 2 \
  --job_id 300032 --pair_id exp_phys_enrichment_tomato_sauce_s2_w150_160_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300032 exp_phys_enrichment_tomato_sauce_s2_w150_160_seed2"

echo "=== RAND 300033: tomato_sauce s2 [150,160] seed=2 phys_enrichment ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task tomato_sauce --state-id 2 --window_start 150 --window_end 160 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 299 --seed 2 \
  --job_id 300033 --pair_id exp_phys_enrichment_tomato_sauce_s2_w150_160_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300033 exp_phys_enrichment_tomato_sauce_s2_w150_160_seed2"

echo "=== VIS 300034: orange_juice s2 [40,50] seed=2 hard_neg_candidate ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task orange_juice --state-id 2 --window_start 40 --window_end 50 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 299 --seed 2 \
  --job_id 300034 --pair_id exp_hard_neg_candidate_orange_juice_s2_w40_50_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300034 exp_hard_neg_candidate_orange_juice_s2_w40_50_seed2"

echo "=== RAND 300035: orange_juice s2 [40,50] seed=2 hard_neg_candidate ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task orange_juice --state-id 2 --window_start 40 --window_end 50 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 299 --seed 2 \
  --job_id 300035 --pair_id exp_hard_neg_candidate_orange_juice_s2_w40_50_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300035 exp_hard_neg_candidate_orange_juice_s2_w40_50_seed2"

echo "=== VIS 300036: salad_dressing s2 [80,90] seed=2 hard_neg_candidate ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task salad_dressing --state-id 2 --window_start 80 --window_end 90 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 262 --seed 2 \
  --job_id 300036 --pair_id exp_hard_neg_candidate_salad_dressing_s2_w80_90_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300036 exp_hard_neg_candidate_salad_dressing_s2_w80_90_seed2"

echo "=== RAND 300037: salad_dressing s2 [80,90] seed=2 hard_neg_candidate ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task salad_dressing --state-id 2 --window_start 80 --window_end 90 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 262 --seed 2 \
  --job_id 300037 --pair_id exp_hard_neg_candidate_salad_dressing_s2_w80_90_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300037 exp_hard_neg_candidate_salad_dressing_s2_w80_90_seed2"

echo "=== VIS 300038: orange_juice s1 [15,25] seed=1 rand_abstain ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task orange_juice --state-id 1 --window_start 15 --window_end 25 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 299 --seed 1 \
  --job_id 300038 --pair_id exp_rand_abstain_orange_juice_s1_w15_25_seed1 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300038 exp_rand_abstain_orange_juice_s1_w15_25_seed1"

echo "=== RAND 300039: orange_juice s1 [15,25] seed=1 rand_abstain ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task orange_juice --state-id 1 --window_start 15 --window_end 25 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 299 --seed 1 \
  --job_id 300039 --pair_id exp_rand_abstain_orange_juice_s1_w15_25_seed1 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300039 exp_rand_abstain_orange_juice_s1_w15_25_seed1"

echo "=== VIS 300040: alphabet_soup s0 [60,70] seed=0 sentinel_repeat ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task alphabet_soup --state-id 0 --window_start 60 --window_end 70 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 299 --seed 0 \
  --job_id 300040 --pair_id exp_sentinel_repeat_alphabet_soup_s0_w60_70_seed0 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300040 exp_sentinel_repeat_alphabet_soup_s0_w60_70_seed0"

echo "=== RAND 300041: alphabet_soup s0 [60,70] seed=0 sentinel_repeat ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task alphabet_soup --state-id 0 --window_start 60 --window_end 70 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 299 --seed 0 \
  --job_id 300041 --pair_id exp_sentinel_repeat_alphabet_soup_s0_w60_70_seed0 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300041 exp_sentinel_repeat_alphabet_soup_s0_w60_70_seed0"

echo "[$(date +%H:%M:%S)] worker_45 EXPANSION DONE"

