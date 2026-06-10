#!/bin/bash
set +e
export CUDA_VISIBLE_DEVICES=2,6
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s8_oracle_open_physical_scan/shard26
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_oracle_open_physical_scan.py

echo "[$(date +%H:%M:%S)] S8_ORACLE_shard26 START (10 jobs, 5 pairs)"

echo "  CLEAN milk_s0_w70_80_L40"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition clean --open_duration 40 --job_id 920000 --output_dir $OUT || echo "FAIL_CLEAN_milk_s0_w70_80_L40"
echo "  ORACLE milk_s0_w70_80_L40"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition oracle_open --open_duration 40 --job_id 920001 --output_dir $OUT || echo "FAIL_ORACLE_milk_s0_w70_80_L40"
echo "  CLEAN butter_s0_w90_100_L40"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 90 --window_end 100 --condition clean --open_duration 40 --job_id 920002 --output_dir $OUT || echo "FAIL_CLEAN_butter_s0_w90_100_L40"
echo "  ORACLE butter_s0_w90_100_L40"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 90 --window_end 100 --condition oracle_open --open_duration 40 --job_id 920003 --output_dir $OUT || echo "FAIL_ORACLE_butter_s0_w90_100_L40"
echo "  CLEAN tomato_sauce_s2_w165_175_L40"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 165 --window_end 175 --condition clean --open_duration 40 --job_id 920004 --output_dir $OUT || echo "FAIL_CLEAN_tomato_sauce_s2_w165_175_L40"
echo "  ORACLE tomato_sauce_s2_w165_175_L40"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 165 --window_end 175 --condition oracle_open --open_duration 40 --job_id 920005 --output_dir $OUT || echo "FAIL_ORACLE_tomato_sauce_s2_w165_175_L40"
echo "  CLEAN cream_cheese_s0_w65_75_L30"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 65 --window_end 75 --condition clean --open_duration 30 --job_id 920006 --output_dir $OUT || echo "FAIL_CLEAN_cream_cheese_s0_w65_75_L30"
echo "  ORACLE cream_cheese_s0_w65_75_L30"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 65 --window_end 75 --condition oracle_open --open_duration 30 --job_id 920007 --output_dir $OUT || echo "FAIL_ORACLE_cream_cheese_s0_w65_75_L30"
echo "  CLEAN cream_cheese_s0_w65_75_L40"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 65 --window_end 75 --condition clean --open_duration 40 --job_id 920008 --output_dir $OUT || echo "FAIL_CLEAN_cream_cheese_s0_w65_75_L40"
echo "  ORACLE cream_cheese_s0_w65_75_L40"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 65 --window_end 75 --condition oracle_open --open_duration 40 --job_id 920009 --output_dir $OUT || echo "FAIL_ORACLE_cream_cheese_s0_w65_75_L40"

echo "[$(date +%H:%M:%S)] S8_ORACLE_shard26 DONE"
