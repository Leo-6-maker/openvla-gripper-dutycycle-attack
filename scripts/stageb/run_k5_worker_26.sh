#!/bin/bash
# K5 repeat stability: worker_26 GPU=2,6
# 2 parents × 5 seeds = 20 jobs
set +e

export CUDA_VISIBLE_DEVICES=2,6

echo "S5_K5_REPEAT code=a20379f anchor=d4a3827"
echo "[$(date +%H:%M:%S)] worker_26 K5 START: 2 parents, 20 jobs"

echo "=== VIS 500030: tomato_sauce s2 [90,100] env=2 atk=0 rand_phys_flip ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 0 --job_id 500030 --pair_id k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk0 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "VIS_FAIL 500030 k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk0"

echo "=== RAND 500031: tomato_sauce s2 [90,100] env=2 atk=0 rand_phys_flip ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 0 --job_id 500031 --pair_id k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk0 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "RAND_FAIL 500031 k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk0"

echo "=== VIS 500032: tomato_sauce s2 [90,100] env=2 atk=1 rand_phys_flip ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 1 --job_id 500032 --pair_id k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk1 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "VIS_FAIL 500032 k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk1"

echo "=== RAND 500033: tomato_sauce s2 [90,100] env=2 atk=1 rand_phys_flip ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 1 --job_id 500033 --pair_id k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk1 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "RAND_FAIL 500033 k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk1"

echo "=== VIS 500034: tomato_sauce s2 [90,100] env=2 atk=2 rand_phys_flip ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 2 --job_id 500034 --pair_id k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk2 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "VIS_FAIL 500034 k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk2"

echo "=== RAND 500035: tomato_sauce s2 [90,100] env=2 atk=2 rand_phys_flip ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 2 --job_id 500035 --pair_id k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk2 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "RAND_FAIL 500035 k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk2"

echo "=== VIS 500036: tomato_sauce s2 [90,100] env=2 atk=3 rand_phys_flip ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 3 --job_id 500036 --pair_id k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk3 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "VIS_FAIL 500036 k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk3"

echo "=== RAND 500037: tomato_sauce s2 [90,100] env=2 atk=3 rand_phys_flip ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 3 --job_id 500037 --pair_id k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk3 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "RAND_FAIL 500037 k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk3"

echo "=== VIS 500038: tomato_sauce s2 [90,100] env=2 atk=4 rand_phys_flip ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 4 --job_id 500038 --pair_id k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk4 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "VIS_FAIL 500038 k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk4"

echo "=== RAND 500039: tomato_sauce s2 [90,100] env=2 atk=4 rand_phys_flip ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 90 --window_end 100 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 4 --job_id 500039 --pair_id k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk4 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "RAND_FAIL 500039 k5_rand_phys_flip_tomato_sauce_s2_w90_100_env2_atk4"

echo "=== VIS 500040: salad_dressing s2 [120,130] env=2 atk=0 neg_drift ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 120 --window_end 130 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 0 --job_id 500040 --pair_id k5_neg_drift_salad_dressing_s2_w120_130_env2_atk0 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "VIS_FAIL 500040 k5_neg_drift_salad_dressing_s2_w120_130_env2_atk0"

echo "=== RAND 500041: salad_dressing s2 [120,130] env=2 atk=0 neg_drift ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 120 --window_end 130 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 0 --job_id 500041 --pair_id k5_neg_drift_salad_dressing_s2_w120_130_env2_atk0 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "RAND_FAIL 500041 k5_neg_drift_salad_dressing_s2_w120_130_env2_atk0"

echo "=== VIS 500042: salad_dressing s2 [120,130] env=2 atk=1 neg_drift ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 120 --window_end 130 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 1 --job_id 500042 --pair_id k5_neg_drift_salad_dressing_s2_w120_130_env2_atk1 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "VIS_FAIL 500042 k5_neg_drift_salad_dressing_s2_w120_130_env2_atk1"

echo "=== RAND 500043: salad_dressing s2 [120,130] env=2 atk=1 neg_drift ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 120 --window_end 130 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 1 --job_id 500043 --pair_id k5_neg_drift_salad_dressing_s2_w120_130_env2_atk1 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "RAND_FAIL 500043 k5_neg_drift_salad_dressing_s2_w120_130_env2_atk1"

echo "=== VIS 500044: salad_dressing s2 [120,130] env=2 atk=2 neg_drift ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 120 --window_end 130 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 2 --job_id 500044 --pair_id k5_neg_drift_salad_dressing_s2_w120_130_env2_atk2 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "VIS_FAIL 500044 k5_neg_drift_salad_dressing_s2_w120_130_env2_atk2"

echo "=== RAND 500045: salad_dressing s2 [120,130] env=2 atk=2 neg_drift ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 120 --window_end 130 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 2 --job_id 500045 --pair_id k5_neg_drift_salad_dressing_s2_w120_130_env2_atk2 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "RAND_FAIL 500045 k5_neg_drift_salad_dressing_s2_w120_130_env2_atk2"

echo "=== VIS 500046: salad_dressing s2 [120,130] env=2 atk=3 neg_drift ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 120 --window_end 130 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 3 --job_id 500046 --pair_id k5_neg_drift_salad_dressing_s2_w120_130_env2_atk3 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "VIS_FAIL 500046 k5_neg_drift_salad_dressing_s2_w120_130_env2_atk3"

echo "=== RAND 500047: salad_dressing s2 [120,130] env=2 atk=3 neg_drift ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 120 --window_end 130 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 3 --job_id 500047 --pair_id k5_neg_drift_salad_dressing_s2_w120_130_env2_atk3 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "RAND_FAIL 500047 k5_neg_drift_salad_dressing_s2_w120_130_env2_atk3"

echo "=== VIS 500048: salad_dressing s2 [120,130] env=2 atk=4 neg_drift ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 120 --window_end 130 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 4 --job_id 500048 --pair_id k5_neg_drift_salad_dressing_s2_w120_130_env2_atk4 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "VIS_FAIL 500048 k5_neg_drift_salad_dressing_s2_w120_130_env2_atk4"

echo "=== RAND 500049: salad_dressing s2 [120,130] env=2 atk=4 neg_drift ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 120 --window_end 130 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 4 --job_id 500049 --pair_id k5_neg_drift_salad_dressing_s2_w120_130_env2_atk4 --output_dir /data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f --image_preprocess official_rot180 || echo "RAND_FAIL 500049 k5_neg_drift_salad_dressing_s2_w120_130_env2_atk4"

echo "[$(date +%H:%M:%S)] worker_26 K5 DONE"

