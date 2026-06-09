#!/bin/bash
set +e
export CUDA_VISIBLE_DEVICES=1,0
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/k5c_smoke
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py
echo "[$(date +%H:%M:%S)] K5C SMOKE START"

# ── Parent 1: k5c_rand_butter1 — butter s=0 w=90-100 env=0 ──
echo "=== VIS k5c_rand_butter1 atk0 ==="
$PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 90 --window_end 100 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 0 --job_id 520900 --pair_id k5c_smoke_rand_butter1_s0_w90_100_env0 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL k5c_rand_butter1 atk0"
echo "=== RAND k5c_rand_butter1 atk0 ==="
$PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 90 --window_end 100 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 0 --job_id 520901 --pair_id k5c_smoke_rand_butter1_s0_w90_100_env0 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL k5c_rand_butter1 atk0"
echo "=== VIS k5c_rand_butter1 atk1 ==="
$PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 90 --window_end 100 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 1 --job_id 520902 --pair_id k5c_smoke_rand_butter1_s0_w90_100_env0 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL k5c_rand_butter1 atk1"
echo "=== RAND k5c_rand_butter1 atk1 ==="
$PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 90 --window_end 100 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 1 --job_id 520903 --pair_id k5c_smoke_rand_butter1_s0_w90_100_env0 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL k5c_rand_butter1 atk1"

# ── Parent 2: k5c_phys_butter — butter s=0 w=135-145 env=0 ──
echo "=== VIS k5c_phys_butter atk0 ==="
$PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 135 --window_end 145 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 0 --job_id 520904 --pair_id k5c_smoke_phys_butter_s0_w135_145_env0 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL k5c_phys_butter atk0"
echo "=== RAND k5c_phys_butter atk0 ==="
$PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 135 --window_end 145 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 0 --job_id 520905 --pair_id k5c_smoke_phys_butter_s0_w135_145_env0 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL k5c_phys_butter atk0"
echo "=== VIS k5c_phys_butter atk1 ==="
$PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 135 --window_end 145 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 1 --job_id 520906 --pair_id k5c_smoke_phys_butter_s0_w135_145_env0 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL k5c_phys_butter atk1"
echo "=== RAND k5c_phys_butter atk1 ==="
$PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 135 --window_end 145 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed 1 --job_id 520907 --pair_id k5c_smoke_phys_butter_s0_w135_145_env0 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL k5c_phys_butter atk1"

echo "[$(date +%H:%M:%S)] K5C SMOKE DONE"
