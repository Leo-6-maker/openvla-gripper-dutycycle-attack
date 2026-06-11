#!/bin/bash
# S19 Phase2 GPU26 — milk w90-100 + orange_juice w50-60: VIS+RAND seeds 71/72/73
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES=2,6
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s19_phase2_multiseed_confirmation
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
EPS=6; PGD=20; L=10

# C3: milk_s0_w90-100
for SEED in 71 72 73; do
  echo "[$(date +%H:%M:%S)] S19 C3 milk w90-100 seed=$SEED VIS"
  $PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 90 --window_end 100 --condition vis_pgd --open_duration $L --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 95520$SEED --pair_id milk_s0_w90_100_s19_seed$SEED --output_dir $OUT || echo "FAIL_C3_VIS_s$SEED"
  echo "[$(date +%H:%M:%S)] S19 C3 milk w90-100 seed=$SEED RAND"
  $PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 90 --window_end 100 --condition random_linf --open_duration $L --attack_seed $SEED --eps_raw_pixels $EPS --job_id 95521$SEED --pair_id milk_s0_w90_100_s19_seed$SEED --output_dir $OUT || echo "FAIL_C3_RAND_s$SEED"
done

# C4: orange_juice_s0_w50-60
for SEED in 71 72 73; do
  echo "[$(date +%H:%M:%S)] S19 C4 orange_juice w50-60 seed=$SEED VIS"
  $PY -u $S --gpu_pair 0,1 --task orange_juice --state-id 0 --window_start 50 --window_end 60 --condition vis_pgd --open_duration $L --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 95522$SEED --pair_id orange_juice_s0_w50_60_s19_seed$SEED --output_dir $OUT || echo "FAIL_C4_VIS_s$SEED"
  echo "[$(date +%H:%M:%S)] S19 C4 orange_juice w50-60 seed=$SEED RAND"
  $PY -u $S --gpu_pair 0,1 --task orange_juice --state-id 0 --window_start 50 --window_end 60 --condition random_linf --open_duration $L --attack_seed $SEED --eps_raw_pixels $EPS --job_id 95523$SEED --pair_id orange_juice_s0_w50_60_s19_seed$SEED --output_dir $OUT || echo "FAIL_C4_RAND_s$SEED"
done

echo "[$(date +%H:%M:%S)] S19 Phase2 GPU26 DONE"
