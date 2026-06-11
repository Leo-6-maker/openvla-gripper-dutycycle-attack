#!/bin/bash
# S19 Phase2 GPU10 — ketchup w150-160 + milk w230-240: VIS+RAND seeds 71/72/73
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES=1,0
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s19_phase2_multiseed_confirmation
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
EPS=6; PGD=20; L=10

# C1: ketchup_s0_w150-160
for SEED in 71 72 73; do
  echo "[$(date +%H:%M:%S)] S19 C1 ketchup w150-160 seed=$SEED VIS"
  $PY -u $S --gpu_pair 0,1 --task ketchup --state-id 0 --window_start 150 --window_end 160 --condition vis_pgd --open_duration $L --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 95510$SEED --pair_id ketchup_s0_w150_160_s19_seed$SEED --output_dir $OUT || echo "FAIL_C1_VIS_s$SEED"
  echo "[$(date +%H:%M:%S)] S19 C1 ketchup w150-160 seed=$SEED RAND"
  $PY -u $S --gpu_pair 0,1 --task ketchup --state-id 0 --window_start 150 --window_end 160 --condition random_linf --open_duration $L --attack_seed $SEED --eps_raw_pixels $EPS --job_id 95511$SEED --pair_id ketchup_s0_w150_160_s19_seed$SEED --output_dir $OUT || echo "FAIL_C1_RAND_s$SEED"
done

# C2: milk_s0_w230-240
for SEED in 71 72 73; do
  echo "[$(date +%H:%M:%S)] S19 C2 milk w230-240 seed=$SEED VIS"
  $PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 230 --window_end 240 --condition vis_pgd --open_duration $L --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 95512$SEED --pair_id milk_s0_w230_240_s19_seed$SEED --output_dir $OUT || echo "FAIL_C2_VIS_s$SEED"
  echo "[$(date +%H:%M:%S)] S19 C2 milk w230-240 seed=$SEED RAND"
  $PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 230 --window_end 240 --condition random_linf --open_duration $L --attack_seed $SEED --eps_raw_pixels $EPS --job_id 95513$SEED --pair_id milk_s0_w230_240_s19_seed$SEED --output_dir $OUT || echo "FAIL_C2_RAND_s$SEED"
done

echo "[$(date +%H:%M:%S)] S19 Phase2 GPU10 DONE"
