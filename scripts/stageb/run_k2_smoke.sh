#!/bin/bash
set +e
export CUDA_VISIBLE_DEVICES=1,0
OUT=/data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f/k5_smoke
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py
echo "[$(date +%H:%M:%S)] K2 SMOKE START"

# milk env_seed=0, attack_seed=0 (VIS done manually, just do RAND)
echo "=== RAND milk r0 ==="
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 0 --job_id 600001 --pair_id k5_smoke_milk_s0_w70_80_env0_atk0 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL milk_r0"

# milk r1
echo "=== VIS milk r1 ==="
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 1 --job_id 600002 --pair_id k5_smoke_milk_s0_w70_80_env0_atk1 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL milk_r1"
echo "=== RAND milk r1 ==="
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 1 --job_id 600003 --pair_id k5_smoke_milk_s0_w70_80_env0_atk1 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL milk_r1"

# salad r0
echo "=== VIS salad r0 ==="
$PY -u $S --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 120 --window_end 130 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 0 --job_id 600004 --pair_id k5_smoke_salad_s2_w120_130_env2_atk0 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL salad_r0"
echo "=== RAND salad r0 ==="
$PY -u $S --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 120 --window_end 130 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 0 --job_id 600005 --pair_id k5_smoke_salad_s2_w120_130_env2_atk0 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL salad_r0"

# salad r1
echo "=== VIS salad r1 ==="
$PY -u $S --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 120 --window_end 130 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 1 --job_id 600006 --pair_id k5_smoke_salad_s2_w120_130_env2_atk1 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL salad_r1"
echo "=== RAND salad r1 ==="
$PY -u $S --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 120 --window_end 130 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 2 --env_seed 2 --attack_seed 1 --job_id 600007 --pair_id k5_smoke_salad_s2_w120_130_env2_atk1 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL salad_r1"

echo "[$(date +%H:%M:%S)] K2 SMOKE DONE"
