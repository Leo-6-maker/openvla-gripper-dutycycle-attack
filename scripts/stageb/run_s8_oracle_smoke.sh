#!/bin/bash
set +e
export CUDA_VISIBLE_DEVICES=1,0
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s8_oracle_open_physical_scan/smoke
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_oracle_open_physical_scan.py

echo "[$(date +%H:%M:%S)] S8 ORACLE SMOKE START (4 jobs)"

echo "  CLEAN milk_s0_w70_80 L=10"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition clean --open_duration 10 --job_id 900100 --output_dir $OUT || echo "FAIL_CLEAN_milk"
echo "  ORACLE milk_s0_w70_80 L=10"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition oracle_open --open_duration 10 --job_id 900101 --output_dir $OUT || echo "FAIL_ORACLE_milk"

echo "  CLEAN cream_cheese_s0_w65_75 L=10"
$PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 65 --window_end 75 --condition clean --open_duration 10 --job_id 900102 --output_dir $OUT || echo "FAIL_CLEAN_cream"
echo "  ORACLE cream_cheese_s0_w65_75 L=10"
$PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 65 --window_end 75 --condition oracle_open --open_duration 10 --job_id 900103 --output_dir $OUT || echo "FAIL_ORACLE_cream"

echo "[$(date +%H:%M:%S)] S8 ORACLE SMOKE DONE"
