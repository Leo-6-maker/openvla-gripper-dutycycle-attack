#!/bin/bash
# S16b GPU26 — fresh parent 5 + calibration 6-8: VIS+RAND command screen seed50
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s16b_command_level_visrand_screen
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
SEED=50; EPS=6; PGD=20

export CUDA_VISIBLE_DEVICES=2,6
# Parent 5: salad_s0_w55-65 (fresh salad clean)
echo "[$(date +%H:%M:%S)] P5 salad_s0_w55-65"
$PY -u $S --gpu_pair 0,1 --task salad_dressing --state-id 0 --window_start 55 --window_end 65 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953008 --pair_id salad_s0_w55_65_s16b --output_dir $OUT || echo "FAIL_P5_VIS"
$PY -u $S --gpu_pair 0,1 --task salad_dressing --state-id 0 --window_start 55 --window_end 65 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953009 --pair_id salad_s0_w55_65_s16b --output_dir $OUT || echo "FAIL_P5_RAND"

# Parent 6: milk_s0_w70-80 (calibration positive anchor)
echo "[$(date +%H:%M:%S)] P6 milk_s0_w70-80 calibration"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953010 --pair_id milk_s0_w70_80_s16b_calib --output_dir $OUT || echo "FAIL_P6_VIS"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953011 --pair_id milk_s0_w70_80_s16b_calib --output_dir $OUT || echo "FAIL_P6_RAND"

# Parent 7: cream_s2_w50-60 (calibration cmd_weak)
echo "[$(date +%H:%M:%S)] P7 cream_s2_w50-60 calibration"
$PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 2 --window_start 50 --window_end 60 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953012 --pair_id cream_s2_w50_60_s16b_calib --output_dir $OUT || echo "FAIL_P7_VIS"
$PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 2 --window_start 50 --window_end 60 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953013 --pair_id cream_s2_w50_60_s16b_calib --output_dir $OUT || echo "FAIL_P7_RAND"

# Parent 8: cream_s0_w85-95 (calibration RAND confounded)
echo "[$(date +%H:%M:%S)] P8 cream_s0_w85-95 calibration"
$PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 85 --window_end 95 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953014 --pair_id cream_s0_w85_95_s16b_calib --output_dir $OUT || echo "FAIL_P8_VIS"
$PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 85 --window_end 95 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953015 --pair_id cream_s0_w85_95_s16b_calib --output_dir $OUT || echo "FAIL_P8_RAND"

echo "[$(date +%H:%M:%S)] S16b GPU26 DONE"
