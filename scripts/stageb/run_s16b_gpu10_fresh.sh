#!/bin/bash
# S16b GPU10 — fresh Layer1 parents 1-4: VIS+RAND command screen seed50
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s16b_command_level_visrand_screen
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
SEED=50; EPS=6; PGD=20

export CUDA_VISIBLE_DEVICES=1,0
# Parent 1: milk_s0_w240-250 (fresh milk late)
echo "[$(date +%H:%M:%S)] P1 milk_s0_w240-250"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 240 --window_end 250 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953000 --pair_id milk_s0_w240_250_s16b --output_dir $OUT || echo "FAIL_P1_VIS"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 240 --window_end 250 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953001 --pair_id milk_s0_w240_250_s16b --output_dir $OUT || echo "FAIL_P1_RAND"

# Parent 2: tomato_s2_w95-105 (fresh tomato)
echo "[$(date +%H:%M:%S)] P2 tomato_s2_w95-105"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 95 --window_end 105 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953002 --pair_id tomato_s2_w95_105_s16b --output_dir $OUT || echo "FAIL_P2_VIS"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 95 --window_end 105 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953003 --pair_id tomato_s2_w95_105_s16b --output_dir $OUT || echo "FAIL_P2_RAND"

# Parent 3: tomato_s0_w50-60 (fresh tomato early)
echo "[$(date +%H:%M:%S)] P3 tomato_s0_w50-60"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 0 --window_start 50 --window_end 60 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953004 --pair_id tomato_s0_w50_60_s16b --output_dir $OUT || echo "FAIL_P3_VIS"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 0 --window_start 50 --window_end 60 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953005 --pair_id tomato_s0_w50_60_s16b --output_dir $OUT || echo "FAIL_P3_RAND"

# Parent 4: salad_s1_w50-60 (fresh salad, first salad test)
echo "[$(date +%H:%M:%S)] P4 salad_s1_w50-60"
$PY -u $S --gpu_pair 0,1 --task salad_dressing --state-id 1 --window_start 50 --window_end 60 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953006 --pair_id salad_s1_w50_60_s16b --output_dir $OUT || echo "FAIL_P4_VIS"
$PY -u $S --gpu_pair 0,1 --task salad_dressing --state-id 1 --window_start 50 --window_end 60 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953007 --pair_id salad_s1_w50_60_s16b --output_dir $OUT || echo "FAIL_P4_RAND"

echo "[$(date +%H:%M:%S)] S16b GPU10 DONE"
