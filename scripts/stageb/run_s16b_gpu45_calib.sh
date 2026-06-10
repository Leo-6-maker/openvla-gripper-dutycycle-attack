#!/bin/bash
# S16b GPU45 — calibration parents 9-12: VIS+RAND command screen seed50
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s16b_command_level_visrand_screen
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
SEED=50; EPS=6; PGD=20

export CUDA_VISIBLE_DEVICES=4,5
# Parent 9: tomato_s2_w155-165 (calibration RAND confounded)
echo "[$(date +%H:%M:%S)] P9 tomato_s2_w155-165 calibration"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 155 --window_end 165 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953016 --pair_id tomato_s2_w155_165_s16b_calib --output_dir $OUT || echo "FAIL_P9_VIS"
$PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 155 --window_end 165 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953017 --pair_id tomato_s2_w155_165_s16b_calib --output_dir $OUT || echo "FAIL_P9_RAND"

# Parent 10: butter_s0_w80-90 (calibration RAND REJECTED)
echo "[$(date +%H:%M:%S)] P10 butter_s0_w80-90 calibration"
$PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 80 --window_end 90 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953018 --pair_id butter_s0_w80_90_s16b_calib --output_dir $OUT || echo "FAIL_P10_VIS"
$PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 80 --window_end 90 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953019 --pair_id butter_s0_w80_90_s16b_calib --output_dir $OUT || echo "FAIL_P10_RAND"

# Parent 11: milk_s0_w230-240 (calibration borderline)
echo "[$(date +%H:%M:%S)] P11 milk_s0_w230-240 calibration"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 230 --window_end 240 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953020 --pair_id milk_s0_w230_240_s16b_calib --output_dir $OUT || echo "FAIL_P11_VIS"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 230 --window_end 240 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953021 --pair_id milk_s0_w230_240_s16b_calib --output_dir $OUT || echo "FAIL_P11_RAND"

# Parent 12: milk_s0_w235-245 (calibration borderline)
echo "[$(date +%H:%M:%S)] P12 milk_s0_w235-245 calibration"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 235 --window_end 245 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953022 --pair_id milk_s0_w235_245_s16b_calib --output_dir $OUT || echo "FAIL_P12_VIS"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 235 --window_end 245 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953023 --pair_id milk_s0_w235_245_s16b_calib --output_dir $OUT || echo "FAIL_P12_RAND"

echo "[$(date +%H:%M:%S)] S16b GPU45 DONE"
