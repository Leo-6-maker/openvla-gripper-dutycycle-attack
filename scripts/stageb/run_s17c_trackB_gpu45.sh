#!/bin/bash
# S17c Track B GPU45 — corrected command screen parents 5-6: bbq_sauce_s0 + salad_dressing (weak control)
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES=4,5
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s17c_trackB_command_screen
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
SEED=61; EPS=6; PGD=20

# P5: bbq_sauce_s0_w55-65 (weak retest)
echo "[$(date +%H:%M:%S)] TrackB P5 bbq_sauce_s0_w55-65"
$PY -u $S --gpu_pair 0,1 --task bbq_sauce --state-id 0 --window_start 55 --window_end 65 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953408 --pair_id bbq_sauce_s0_w55_65_s17c_seed61 --output_dir $OUT || echo "FAIL_P5_VIS"
$PY -u $S --gpu_pair 0,1 --task bbq_sauce --state-id 0 --window_start 55 --window_end 65 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953409 --pair_id bbq_sauce_s0_w55_65_s17c_seed61 --output_dir $OUT || echo "FAIL_P5_RAND"

# P6: salad_dressing_s2_w50-60 (known weak control)
echo "[$(date +%H:%M:%S)] TrackB P6 salad_dressing_s2_w50-60"
$PY -u $S --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 50 --window_end 60 --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 953410 --pair_id salad_dressing_s2_w50_60_s17c_seed61 --output_dir $OUT || echo "FAIL_P6_VIS"
$PY -u $S --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 50 --window_end 60 --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --job_id 953411 --pair_id salad_dressing_s2_w50_60_s17c_seed61 --output_dir $OUT || echo "FAIL_P6_RAND"

echo "[$(date +%H:%M:%S)] TrackB GPU45 DONE"
