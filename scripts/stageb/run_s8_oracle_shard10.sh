#!/bin/bash
set +e
export CUDA_VISIBLE_DEVICES=1,0
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s8_oracle_open_physical_scan/shard10
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_oracle_open_physical_scan.py

echo "[$(date +%H:%M:%S)] S8_ORACLE_shard10 START (12 jobs, 6 pairs)"

echo "  CLEAN milk_s0_w70_80_L10"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition clean --open_duration 10 --job_id 900000 --output_dir $OUT || echo "FAIL_CLEAN_milk_s0_w70_80_L10"
echo "  ORACLE milk_s0_w70_80_L10"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition oracle_open --open_duration 10 --job_id 900001 --output_dir $OUT || echo "FAIL_ORACLE_milk_s0_w70_80_L10"
echo "  CLEAN milk_s0_w70_80_L20"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition clean --open_duration 20 --job_id 900002 --output_dir $OUT || echo "FAIL_CLEAN_milk_s0_w70_80_L20"
echo "  ORACLE milk_s0_w70_80_L20"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition oracle_open --open_duration 20 --job_id 900003 --output_dir $OUT || echo "FAIL_ORACLE_milk_s0_w70_80_L20"
echo "  CLEAN butter_s0_w90_100_L10"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 90 --window_end 100 --condition clean --open_duration 10 --job_id 900004 --output_dir $OUT || echo "FAIL_CLEAN_butter_s0_w90_100_L10"
echo "  ORACLE butter_s0_w90_100_L10"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 90 --window_end 100 --condition oracle_open --open_duration 10 --job_id 900005 --output_dir $OUT || echo "FAIL_ORACLE_butter_s0_w90_100_L10"
echo "  CLEAN butter_s0_w90_100_L20"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 90 --window_end 100 --condition clean --open_duration 20 --job_id 900006 --output_dir $OUT || echo "FAIL_CLEAN_butter_s0_w90_100_L20"
echo "  ORACLE butter_s0_w90_100_L20"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 90 --window_end 100 --condition oracle_open --open_duration 20 --job_id 900007 --output_dir $OUT || echo "FAIL_ORACLE_butter_s0_w90_100_L20"
echo "  CLEAN tomato_sauce_s2_w165_175_L10"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 165 --window_end 175 --condition clean --open_duration 10 --job_id 900008 --output_dir $OUT || echo "FAIL_CLEAN_tomato_sauce_s2_w165_175_L10"
echo "  ORACLE tomato_sauce_s2_w165_175_L10"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 165 --window_end 175 --condition oracle_open --open_duration 10 --job_id 900009 --output_dir $OUT || echo "FAIL_ORACLE_tomato_sauce_s2_w165_175_L10"
echo "  CLEAN cream_cheese_s0_w65_75_L10"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 65 --window_end 75 --condition clean --open_duration 10 --job_id 900010 --output_dir $OUT || echo "FAIL_CLEAN_cream_cheese_s0_w65_75_L10"
echo "  ORACLE cream_cheese_s0_w65_75_L10"
PYTHONPATH=src $PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 65 --window_end 75 --condition oracle_open --open_duration 10 --job_id 900011 --output_dir $OUT || echo "FAIL_ORACLE_cream_cheese_s0_w65_75_L10"

echo "[$(date +%H:%M:%S)] S8_ORACLE_shard10 DONE"
