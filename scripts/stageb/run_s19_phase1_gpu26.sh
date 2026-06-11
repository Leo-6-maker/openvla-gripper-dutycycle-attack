#!/bin/bash
# S19 Phase1 GPU26 — milk w90-100 + orange_juice w50-60: RAND 71/72/73 + ORACLE
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES=2,6
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s19_phase1_randveto_oracle
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
EPS=6; PGD=20; L=10

# C3: milk_s0_w90-100
T=milk; SID=0; WS=90; WE=100
echo "[$(date +%H:%M:%S)] S19 C3 milk w90-100 ORACLE"
$PY -u $S --gpu_pair 0,1 --task $T --state-id $SID --window_start $WS --window_end $WE --condition oracle_open --open_duration $L --attack_seed 0 --job_id 955020 --pair_id milk_s0_w90_100_s19 --output_dir $OUT || echo "FAIL_C3_ORACLE"
for SEED in 71 72 73; do
  echo "[$(date +%H:%M:%S)] S19 C3 milk w90-100 RAND seed=$SEED"
  $PY -u $S --gpu_pair 0,1 --task $T --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration $L --attack_seed $SEED --eps_raw_pixels $EPS --job_id 95502$SEED --pair_id milk_s0_w90_100_s19_seed$SEED --output_dir $OUT || echo "FAIL_C3_RAND_s$SEED"
done

# C4: orange_juice_s0_w50-60
T=orange_juice; SID=0; WS=50; WE=60
echo "[$(date +%H:%M:%S)] S19 C4 orange_juice w50-60 ORACLE"
$PY -u $S --gpu_pair 0,1 --task $T --state-id $SID --window_start $WS --window_end $WE --condition oracle_open --open_duration $L --attack_seed 0 --job_id 955030 --pair_id orange_juice_s0_w50_60_s19 --output_dir $OUT || echo "FAIL_C4_ORACLE"
for SEED in 71 72 73; do
  echo "[$(date +%H:%M:%S)] S19 C4 orange_juice w50-60 RAND seed=$SEED"
  $PY -u $S --gpu_pair 0,1 --task $T --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration $L --attack_seed $SEED --eps_raw_pixels $EPS --job_id 95503$SEED --pair_id orange_juice_s0_w50_60_s19_seed$SEED --output_dir $OUT || echo "FAIL_C4_RAND_s$SEED"
done

echo "[$(date +%H:%M:%S)] S19 Phase1 GPU26 DONE"
