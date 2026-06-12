#!/bin/bash
# Silver confirmation: worker_45 GPU=4,5
# 4 parents x 2 repeats = 16 jobs
set +e

export CUDA_VISIBLE_DEVICES=4,5

echo "data_anchor=d4a3827 code_commit=e33b5e4 batch=silver_confirmation"
echo "[$(date +%H:%M:%S)] worker_45 CONFIRMATION START: 4 parents"

echo "=== VIS 400100: milk s0 [230,240] seed=1 confounded_both r0 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task milk --state-id 0 --window_start 230 --window_end 240 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 1 --job_id 400100 --pair_id silver_confounded_both_milk_s0_w230_240_seed0_r0 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "VIS_FAIL 400100 silver_confounded_both_milk_s0_w230_240_seed0_r0"

echo "=== RAND 400101: milk s0 [230,240] seed=1 confounded_both r0 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task milk --state-id 0 --window_start 230 --window_end 240 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 1 --job_id 400101 --pair_id silver_confounded_both_milk_s0_w230_240_seed0_r0 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "RAND_FAIL 400101 silver_confounded_both_milk_s0_w230_240_seed0_r0"

echo "=== VIS 400102: milk s0 [230,240] seed=2 confounded_both r1 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task milk --state-id 0 --window_start 230 --window_end 240 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 2 --job_id 400102 --pair_id silver_confounded_both_milk_s0_w230_240_seed0_r1 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "VIS_FAIL 400102 silver_confounded_both_milk_s0_w230_240_seed0_r1"

echo "=== RAND 400103: milk s0 [230,240] seed=2 confounded_both r1 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task milk --state-id 0 --window_start 230 --window_end 240 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 2 --job_id 400103 --pair_id silver_confounded_both_milk_s0_w230_240_seed0_r1 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "RAND_FAIL 400103 silver_confounded_both_milk_s0_w230_240_seed0_r1"

echo "=== VIS 400104: tomato_sauce s2 [150,160] seed=201 rand_command r0 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 150 --window_end 160 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 201 --job_id 400104 --pair_id silver_rand_command_tomato_sauce_s2_w150_160_seed2_r0 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "VIS_FAIL 400104 silver_rand_command_tomato_sauce_s2_w150_160_seed2_r0"

echo "=== RAND 400105: tomato_sauce s2 [150,160] seed=201 rand_command r0 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 150 --window_end 160 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 201 --job_id 400105 --pair_id silver_rand_command_tomato_sauce_s2_w150_160_seed2_r0 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "RAND_FAIL 400105 silver_rand_command_tomato_sauce_s2_w150_160_seed2_r0"

echo "=== VIS 400106: tomato_sauce s2 [150,160] seed=202 rand_command r1 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 150 --window_end 160 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 202 --job_id 400106 --pair_id silver_rand_command_tomato_sauce_s2_w150_160_seed2_r1 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "VIS_FAIL 400106 silver_rand_command_tomato_sauce_s2_w150_160_seed2_r1"

echo "=== RAND 400107: tomato_sauce s2 [150,160] seed=202 rand_command r1 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 150 --window_end 160 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 202 --job_id 400107 --pair_id silver_rand_command_tomato_sauce_s2_w150_160_seed2_r1 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "RAND_FAIL 400107 silver_rand_command_tomato_sauce_s2_w150_160_seed2_r1"

echo "=== VIS 400108: tomato_sauce s2 [90,100] seed=201 rand_phys r0 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 201 --job_id 400108 --pair_id silver_rand_phys_tomato_sauce_s2_w90_100_seed2_r0 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "VIS_FAIL 400108 silver_rand_phys_tomato_sauce_s2_w90_100_seed2_r0"

echo "=== RAND 400109: tomato_sauce s2 [90,100] seed=201 rand_phys r0 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 201 --job_id 400109 --pair_id silver_rand_phys_tomato_sauce_s2_w90_100_seed2_r0 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "RAND_FAIL 400109 silver_rand_phys_tomato_sauce_s2_w90_100_seed2_r0"

echo "=== VIS 400110: tomato_sauce s2 [90,100] seed=202 rand_phys r1 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 202 --job_id 400110 --pair_id silver_rand_phys_tomato_sauce_s2_w90_100_seed2_r1 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "VIS_FAIL 400110 silver_rand_phys_tomato_sauce_s2_w90_100_seed2_r1"

echo "=== RAND 400111: tomato_sauce s2 [90,100] seed=202 rand_phys r1 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 202 --job_id 400111 --pair_id silver_rand_phys_tomato_sauce_s2_w90_100_seed2_r1 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "RAND_FAIL 400111 silver_rand_phys_tomato_sauce_s2_w90_100_seed2_r1"

echo "=== VIS 400112: salad_dressing s2 [120,130] seed=201 clean_negative r0 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 120 --window_end 130 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 201 --job_id 400112 --pair_id silver_clean_negative_salad_dressing_s2_w120_130_seed2_r0 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "VIS_FAIL 400112 silver_clean_negative_salad_dressing_s2_w120_130_seed2_r0"

echo "=== RAND 400113: salad_dressing s2 [120,130] seed=201 clean_negative r0 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 120 --window_end 130 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 201 --job_id 400113 --pair_id silver_clean_negative_salad_dressing_s2_w120_130_seed2_r0 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "RAND_FAIL 400113 silver_clean_negative_salad_dressing_s2_w120_130_seed2_r0"

echo "=== VIS 400114: salad_dressing s2 [120,130] seed=202 clean_negative r1 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 120 --window_end 130 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 202 --job_id 400114 --pair_id silver_clean_negative_salad_dressing_s2_w120_130_seed2_r1 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "VIS_FAIL 400114 silver_clean_negative_salad_dressing_s2_w120_130_seed2_r1"

echo "=== RAND 400115: salad_dressing s2 [120,130] seed=202 clean_negative r1 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 120 --window_end 130 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 202 --job_id 400115 --pair_id silver_clean_negative_salad_dressing_s2_w120_130_seed2_r1 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "RAND_FAIL 400115 silver_clean_negative_salad_dressing_s2_w120_130_seed2_r1"

echo "[$(date +%H:%M:%S)] worker_45 CONFIRMATION DONE"

