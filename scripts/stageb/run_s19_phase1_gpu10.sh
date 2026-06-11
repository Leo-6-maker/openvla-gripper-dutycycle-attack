#!/bin/bash
# S19 Phase1 GPU10 — ketchup w150-160 + milk w230-240: RAND 71/72/73 + ORACLE
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES=1,0
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s19_phase1_randveto_oracle
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
EPS=6; PGD=20; L=10

# C1: ketchup_s0_w150-160
T=ketchup; SID=0; WS=150; WE=160
echo "[$(date +%H:%M:%S)] S19 C1 ketchup w150-160 ORACLE"
$PY -u $S --gpu_pair 0,1 --task $T --state-id $SID --window_start $WS --window_end $WE --condition oracle_open --open_duration $L --attack_seed 0 --job_id 955000 --pair_id ketchup_s0_w150_160_s19 --output_dir $OUT || echo "FAIL_C1_ORACLE"
for SEED in 71 72 73; do
  echo "[$(date +%H:%M:%S)] S19 C1 ketchup w150-160 RAND seed=$SEED"
  $PY -u $S --gpu_pair 0,1 --task $T --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration $L --attack_seed $SEED --eps_raw_pixels $EPS --job_id 95500$SEED --pair_id ketchup_s0_w150_160_s19_seed$SEED --output_dir $OUT || echo "FAIL_C1_RAND_s$SEED"
done

# C2: milk_s0_w230-240
T=milk; SID=0; WS=230; WE=240
echo "[$(date +%H:%M:%S)] S19 C2 milk w230-240 ORACLE"
$PY -u $S --gpu_pair 0,1 --task $T --state-id $SID --window_start $WS --window_end $WE --condition oracle_open --open_duration $L --attack_seed 0 --job_id 955010 --pair_id milk_s0_w230_240_s19 --output_dir $OUT || echo "FAIL_C2_ORACLE"
for SEED in 71 72 73; do
  echo "[$(date +%H:%M:%S)] S19 C2 milk w230-240 RAND seed=$SEED"
  $PY -u $S --gpu_pair 0,1 --task $T --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration $L --attack_seed $SEED --eps_raw_pixels $EPS --job_id 95501$SEED --pair_id milk_s0_w230_240_s19_seed$SEED --output_dir $OUT || echo "FAIL_C2_RAND_s$SEED"
done

echo "[$(date +%H:%M:%S)] S19 Phase1 GPU10 DONE"
