#!/bin/bash
# Continuous per-GPU dispatch — each GPU races for remaining jobs via mkdir lock
LAUNCHER=/mnt/sdc/dty_user/openvla_attack/tmp/phase3_refresh_launcher.sh
LOG_DIR=/mnt/sdc/dty_user/openvla_attack/tmp
TOTAL=108
LOCKDIR=/mnt/sdc/dty_user/openvla_attack/tmp/p3_lock
mkdir -p $LOCKDIR

claim_and_run() {
  local GPU=$1
  for IDX in $(seq 0 $((TOTAL - 1))); do
    if mkdir $LOCKDIR/idx_$IDX 2>/dev/null; then
      echo "$(date +%H:%M:%S) GPU$GPU claimed idx=$IDX"
      bash $LAUNCHER $GPU $IDX > $LOG_DIR/p3_gpu${GPU}_idx${IDX}.log 2>&1
      echo "$(date +%H:%M:%S) GPU$GPU done idx=$IDX"
    fi
  done
  echo "$(date +%H:%M:%S) GPU$GPU: all jobs done"
}

for GPU in 1 2 3 4 5 6; do
  nohup bash -c "$(declare -f claim_and_run); claim_and_run $GPU" > $LOG_DIR/p3_gpu${GPU}_loop.log 2>&1 &
  echo "GPU$GPU loop launched PID=$!"
done
echo "6 GPUs racing for $TOTAL jobs with mkdir atomic claims."
