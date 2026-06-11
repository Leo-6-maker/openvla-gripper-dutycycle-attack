#!/bin/bash
# S20d V4 clean clone smoke — 3-way parallel launcher
# Usage: launch each GPU script in a separate tmux window with 30s stagger:
#   tmux new-session -d -s s20d_gpu10 "bash scripts/stageb/run_s20d_smoke_gpu10.sh 2>&1 | tee /data/liuyu/outputs/stageb_s20d_v4_official_l3_20260611/smoke/log_gpu10.txt"
#   sleep 30
#   tmux new-window -t s20d_gpu26 "bash scripts/stageb/run_s20d_smoke_gpu26.sh 2>&1 | tee /data/liuyu/outputs/stageb_s20d_v4_official_l3_20260611/smoke/log_gpu26.txt"
#   sleep 30
#   tmux new-window -t s20d_gpu45 "bash scripts/stageb/run_s20d_smoke_gpu45.sh 2>&1 | tee /data/liuyu/outputs/stageb_s20d_v4_official_l3_20260611/smoke/log_gpu45.txt"
set +e

OUT=/data/liuyu/outputs/stageb_s20d_v4_official_l3_20260611/smoke
mkdir -p $OUT

echo "[$(date +%H:%M:%S)] S20d SMOKE MASTER — launching 3 GPU workers with 30s stagger"

REPO=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607

# GPU 1,0 (physical 1+0): ketchup s0, s1
bash $REPO/scripts/stageb/run_s20d_smoke_gpu10.sh &
PID10=$!
echo "[$(date +%H:%M:%S)] GPU(1,0) PID=$PID10"

sleep 30

# GPU 2,6 (physical 2+6): ketchup s3, tomato_sauce s3
bash $REPO/scripts/stageb/run_s20d_smoke_gpu26.sh &
PID26=$!
echo "[$(date +%H:%M:%S)] GPU(2,6) PID=$PID26"

sleep 30

# GPU 4,5 (physical 4+5): tomato_sauce s5
bash $REPO/scripts/stageb/run_s20d_smoke_gpu45.sh &
PID45=$!
echo "[$(date +%H:%M:%S)] GPU(4,5) PID=$PID45"

echo "[$(date +%H:%M:%S)] All 3 GPU workers launched: PIDs $PID10 $PID26 $PID45"
wait $PID10 $PID26 $PID45
echo "[$(date +%H:%M:%S)] S20d SMOKE DONE — all workers finished"
