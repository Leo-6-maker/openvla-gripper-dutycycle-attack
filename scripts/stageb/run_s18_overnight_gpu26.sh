#!/bin/bash
# S18 Overnight GPU26 — actual tasks 4-6: ketchup, tomato_sauce, butter
# 3 tasks × 5 windows × 2 conditions = 30 jobs, seed70
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES=2,6
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s18_overnight_census
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
SEED=70; EPS=6; PGD=20; SID=0; L=10

TASKS=("ketchup" "tomato_sauce" "butter")
WINDOWS=("50 60" "70 80" "90 100" "150 160" "230 240")
JOB=954100

for TASK in "${TASKS[@]}"; do
  for WIN in "${WINDOWS[@]}"; do
    WS=$(echo $WIN | cut -d' ' -f1); WE=$(echo $WIN | cut -d' ' -f2)
    PAIR="${TASK}_s${SID}_w${WS}_${WE}_s18_seed${SEED}"
    echo "[$(date +%H:%M:%S)] S18 ${TASK} w${WS}-${WE}"
    $PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition vis_pgd --open_duration $L --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id $((JOB++)) --pair_id $PAIR --output_dir $OUT || echo "FAIL_VIS_${TASK}_w${WS}"
    $PY -u $S --gpu_pair 0,1 --task $TASK --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration $L --attack_seed $SEED --eps_raw_pixels $EPS --job_id $((JOB++)) --pair_id $PAIR --output_dir $OUT || echo "FAIL_RAND_${TASK}_w${WS}"
  done
done
echo "[$(date +%H:%M:%S)] S18 GPU26 ALL DONE"
