#!/bin/bash
# Silver confirmation: worker_26 GPU=2,6
# 4 parents x 2 repeats = 16 jobs
set +e

export CUDA_VISIBLE_DEVICES=2,6

echo "data_anchor=d4a3827 code_commit=e33b5e4 batch=silver_confirmation"
echo "[$(date +%H:%M:%S)] worker_26 CONFIRMATION START: 4 parents"

echo "=== VIS 400000: bbq_sauce s2 [100,110] seed=201 cmd_phys_surprise r0 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task bbq_sauce --state-id 2 --window_start 100 --window_end 110 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 201 --job_id 400000 --pair_id silver_cmd_phys_surprise_bbq_sauce_s2_w100_110_seed2_r0 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "VIS_FAIL 400000 silver_cmd_phys_surprise_bbq_sauce_s2_w100_110_seed2_r0"

echo "=== RAND 400001: bbq_sauce s2 [100,110] seed=201 cmd_phys_surprise r0 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task bbq_sauce --state-id 2 --window_start 100 --window_end 110 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 201 --job_id 400001 --pair_id silver_cmd_phys_surprise_bbq_sauce_s2_w100_110_seed2_r0 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "RAND_FAIL 400001 silver_cmd_phys_surprise_bbq_sauce_s2_w100_110_seed2_r0"

echo "=== VIS 400002: bbq_sauce s2 [100,110] seed=202 cmd_phys_surprise r1 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task bbq_sauce --state-id 2 --window_start 100 --window_end 110 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 202 --job_id 400002 --pair_id silver_cmd_phys_surprise_bbq_sauce_s2_w100_110_seed2_r1 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "VIS_FAIL 400002 silver_cmd_phys_surprise_bbq_sauce_s2_w100_110_seed2_r1"

echo "=== RAND 400003: bbq_sauce s2 [100,110] seed=202 cmd_phys_surprise r1 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task bbq_sauce --state-id 2 --window_start 100 --window_end 110 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 202 --job_id 400003 --pair_id silver_cmd_phys_surprise_bbq_sauce_s2_w100_110_seed2_r1 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "RAND_FAIL 400003 silver_cmd_phys_surprise_bbq_sauce_s2_w100_110_seed2_r1"

echo "=== VIS 400004: bbq_sauce s2 [200,210] seed=201 cmd_phys_new r0 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task bbq_sauce --state-id 2 --window_start 200 --window_end 210 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 201 --job_id 400004 --pair_id silver_cmd_phys_new_bbq_sauce_s2_w200_210_seed2_r0 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "VIS_FAIL 400004 silver_cmd_phys_new_bbq_sauce_s2_w200_210_seed2_r0"

echo "=== RAND 400005: bbq_sauce s2 [200,210] seed=201 cmd_phys_new r0 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task bbq_sauce --state-id 2 --window_start 200 --window_end 210 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 201 --job_id 400005 --pair_id silver_cmd_phys_new_bbq_sauce_s2_w200_210_seed2_r0 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "RAND_FAIL 400005 silver_cmd_phys_new_bbq_sauce_s2_w200_210_seed2_r0"

echo "=== VIS 400006: bbq_sauce s2 [200,210] seed=202 cmd_phys_new r1 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task bbq_sauce --state-id 2 --window_start 200 --window_end 210 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 202 --job_id 400006 --pair_id silver_cmd_phys_new_bbq_sauce_s2_w200_210_seed2_r1 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "VIS_FAIL 400006 silver_cmd_phys_new_bbq_sauce_s2_w200_210_seed2_r1"

echo "=== RAND 400007: bbq_sauce s2 [200,210] seed=202 cmd_phys_new r1 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task bbq_sauce --state-id 2 --window_start 200 --window_end 210 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 202 --job_id 400007 --pair_id silver_cmd_phys_new_bbq_sauce_s2_w200_210_seed2_r1 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "RAND_FAIL 400007 silver_cmd_phys_new_bbq_sauce_s2_w200_210_seed2_r1"

echo "=== VIS 400008: cream_cheese s1 [145,155] seed=101 phys_only r0 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task cream_cheese --state-id 1 --window_start 145 --window_end 155 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 101 --job_id 400008 --pair_id silver_phys_only_cream_cheese_s1_w145_155_seed1_r0 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "VIS_FAIL 400008 silver_phys_only_cream_cheese_s1_w145_155_seed1_r0"

echo "=== RAND 400009: cream_cheese s1 [145,155] seed=101 phys_only r0 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task cream_cheese --state-id 1 --window_start 145 --window_end 155 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 101 --job_id 400009 --pair_id silver_phys_only_cream_cheese_s1_w145_155_seed1_r0 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "RAND_FAIL 400009 silver_phys_only_cream_cheese_s1_w145_155_seed1_r0"

echo "=== VIS 400010: cream_cheese s1 [145,155] seed=102 phys_only r1 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task cream_cheese --state-id 1 --window_start 145 --window_end 155 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 102 --job_id 400010 --pair_id silver_phys_only_cream_cheese_s1_w145_155_seed1_r1 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "VIS_FAIL 400010 silver_phys_only_cream_cheese_s1_w145_155_seed1_r1"

echo "=== RAND 400011: cream_cheese s1 [145,155] seed=102 phys_only r1 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task cream_cheese --state-id 1 --window_start 145 --window_end 155 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 102 --job_id 400011 --pair_id silver_phys_only_cream_cheese_s1_w145_155_seed1_r1 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "RAND_FAIL 400011 silver_phys_only_cream_cheese_s1_w145_155_seed1_r1"

echo "=== VIS 400012: milk s0 [70,80] seed=1 cmd_phys_anchor r0 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 1 --job_id 400012 --pair_id silver_cmd_phys_anchor_milk_s0_w70_80_seed0_r0 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "VIS_FAIL 400012 silver_cmd_phys_anchor_milk_s0_w70_80_seed0_r0"

echo "=== RAND 400013: milk s0 [70,80] seed=1 cmd_phys_anchor r0 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 1 --job_id 400013 --pair_id silver_cmd_phys_anchor_milk_s0_w70_80_seed0_r0 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "RAND_FAIL 400013 silver_cmd_phys_anchor_milk_s0_w70_80_seed0_r0"

echo "=== VIS 400014: milk s0 [70,80] seed=2 cmd_phys_anchor r1 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 2 --job_id 400014 --pair_id silver_cmd_phys_anchor_milk_s0_w70_80_seed0_r1 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "VIS_FAIL 400014 silver_cmd_phys_anchor_milk_s0_w70_80_seed0_r1"

echo "=== RAND 400015: milk s0 [70,80] seed=2 cmd_phys_anchor r1 ==="
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 2 --job_id 400015 --pair_id silver_cmd_phys_anchor_milk_s0_w70_80_seed0_r1 --output_dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4 --image_preprocess official_rot180 || echo "RAND_FAIL 400015 silver_cmd_phys_anchor_milk_s0_w70_80_seed0_r1"

echo "[$(date +%H:%M:%S)] worker_26 CONFIRMATION DONE"

