#!/bin/bash
# Single-GPU continuous worker: races for remaining Phase 3 jobs
GPU=$1
LAUNCHER=/mnt/sdc/dty_user/openvla_attack/tmp/phase3_refresh_launcher.sh
LOG_DIR=/mnt/sdc/dty_user/openvla_attack/tmp
TOTAL=108
LOCKDIR=/mnt/sdc/dty_user/openvla_attack/tmp/p3_lock

for IDX in $(seq 0 $((TOTAL - 1))); do
  if mkdir $LOCKDIR/idx_$IDX 2>/dev/null; then
    echo "$(date +%H:%M:%S) GPU$GPU claimed idx=$IDX"
    bash $LAUNCHER $GPU $IDX > $LOG_DIR/p3_gpu${GPU}_idx${IDX}.log 2>&1
    echo "$(date +%H:%M:%S) GPU$GPU done idx=$IDX"
  fi
done
echo "$(date +%H:%M:%S) GPU$GPU: all jobs done"
