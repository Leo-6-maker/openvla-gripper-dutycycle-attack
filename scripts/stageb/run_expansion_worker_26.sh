Warning: Permanently added '10.60.133.3' (ED25519) to the list of known hosts.
Warning: Permanently added '10.60.133.4' (ED25519) to the list of known hosts.
#!/bin/bash
# Expansion worker: worker_26 GPU=2,6
# 7 parents, 14 jobs (job_id range 300014-300027)
set +e

export CUDA_VISIBLE_DEVICES=2,6

echo "[$(date +%H:%M:%S)] worker_26 EXPANSION START: 7 parents"

echo "=== VIS 300014: cream_cheese s1 [145,155] seed=1 cmd_expansion ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task cream_cheese --state-id 1 --window_start 145 --window_end 155 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 299 --seed 1 \
  --job_id 300014 --pair_id exp_cmd_expansion_cream_cheese_s1_w145_155_seed1 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300014 exp_cmd_expansion_cream_cheese_s1_w145_155_seed1"

echo "=== RAND 300015: cream_cheese s1 [145,155] seed=1 cmd_expansion ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task cream_cheese --state-id 1 --window_start 145 --window_end 155 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 299 --seed 1 \
  --job_id 300015 --pair_id exp_cmd_expansion_cream_cheese_s1_w145_155_seed1 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300015 exp_cmd_expansion_cream_cheese_s1_w145_155_seed1"

echo "=== VIS 300016: orange_juice s2 [25,35] seed=2 cmd_expansion ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task orange_juice --state-id 2 --window_start 25 --window_end 35 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 299 --seed 2 \
  --job_id 300016 --pair_id exp_cmd_expansion_orange_juice_s2_w25_35_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300016 exp_cmd_expansion_orange_juice_s2_w25_35_seed2"

echo "=== RAND 300017: orange_juice s2 [25,35] seed=2 cmd_expansion ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task orange_juice --state-id 2 --window_start 25 --window_end 35 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 299 --seed 2 \
  --job_id 300017 --pair_id exp_cmd_expansion_orange_juice_s2_w25_35_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300017 exp_cmd_expansion_orange_juice_s2_w25_35_seed2"

echo "=== VIS 300018: bbq_sauce s0 [60,70] seed=0 phys_enrichment ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task bbq_sauce --state-id 0 --window_start 60 --window_end 70 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 140 --seed 0 \
  --job_id 300018 --pair_id exp_phys_enrichment_bbq_sauce_s0_w60_70_seed0 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300018 exp_phys_enrichment_bbq_sauce_s0_w60_70_seed0"

echo "=== RAND 300019: bbq_sauce s0 [60,70] seed=0 phys_enrichment ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task bbq_sauce --state-id 0 --window_start 60 --window_end 70 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 140 --seed 0 \
  --job_id 300019 --pair_id exp_phys_enrichment_bbq_sauce_s0_w60_70_seed0 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300019 exp_phys_enrichment_bbq_sauce_s0_w60_70_seed0"

echo "=== VIS 300020: cream_cheese s1 [195,205] seed=1 hard_neg_candidate ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task cream_cheese --state-id 1 --window_start 195 --window_end 205 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 299 --seed 1 \
  --job_id 300020 --pair_id exp_hard_neg_candidate_cream_cheese_s1_w195_205_seed1 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300020 exp_hard_neg_candidate_cream_cheese_s1_w195_205_seed1"

echo "=== RAND 300021: cream_cheese s1 [195,205] seed=1 hard_neg_candidate ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task cream_cheese --state-id 1 --window_start 195 --window_end 205 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 299 --seed 1 \
  --job_id 300021 --pair_id exp_hard_neg_candidate_cream_cheese_s1_w195_205_seed1 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300021 exp_hard_neg_candidate_cream_cheese_s1_w195_205_seed1"

echo "=== VIS 300022: tomato_sauce s2 [155,165] seed=2 hard_neg_candidate ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task tomato_sauce --state-id 2 --window_start 155 --window_end 165 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 299 --seed 2 \
  --job_id 300022 --pair_id exp_hard_neg_candidate_tomato_sauce_s2_w155_165_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300022 exp_hard_neg_candidate_tomato_sauce_s2_w155_165_seed2"

echo "=== RAND 300023: tomato_sauce s2 [155,165] seed=2 hard_neg_candidate ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task tomato_sauce --state-id 2 --window_start 155 --window_end 165 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 299 --seed 2 \
  --job_id 300023 --pair_id exp_hard_neg_candidate_tomato_sauce_s2_w155_165_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300023 exp_hard_neg_candidate_tomato_sauce_s2_w155_165_seed2"

echo "=== VIS 300024: alphabet_soup s0 [15,25] seed=0 rand_abstain ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task alphabet_soup --state-id 0 --window_start 15 --window_end 25 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 299 --seed 0 \
  --job_id 300024 --pair_id exp_rand_abstain_alphabet_soup_s0_w15_25_seed0 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300024 exp_rand_abstain_alphabet_soup_s0_w15_25_seed0"

echo "=== RAND 300025: alphabet_soup s0 [15,25] seed=0 rand_abstain ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task alphabet_soup --state-id 0 --window_start 15 --window_end 25 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 299 --seed 0 \
  --job_id 300025 --pair_id exp_rand_abstain_alphabet_soup_s0_w15_25_seed0 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300025 exp_rand_abstain_alphabet_soup_s0_w15_25_seed0"

echo "=== VIS 300026: tomato_sauce s2 [95,105] seed=2 sentinel_repeat ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task tomato_sauce --state-id 2 --window_start 95 --window_end 105 \
  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \
  --max_steps 299 --seed 2 \
  --job_id 300026 --pair_id exp_sentinel_repeat_tomato_sauce_s2_w95_105_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "VIS_FAIL 300026 exp_sentinel_repeat_tomato_sauce_s2_w95_105_seed2"

echo "=== RAND 300027: tomato_sauce s2 [95,105] seed=2 sentinel_repeat ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py \
  --gpu_pair 0,1 \
  --task tomato_sauce --state-id 2 --window_start 95 --window_end 105 \
  --condition random_linf --eps_raw_pixels 6 \
  --max_steps 299 --seed 2 \
  --job_id 300027 --pair_id exp_sentinel_repeat_tomato_sauce_s2_w95_105_seed2 \
  --output_dir /data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827 \
  --image_preprocess official_rot180 \
  || echo "RAND_FAIL 300027 exp_sentinel_repeat_tomato_sauce_s2_w95_105_seed2"

echo "[$(date +%H:%M:%S)] worker_26 EXPANSION DONE"

