#!/bin/bash
# Phase 3 auto-handoff: sequential batch launch for 108 metric refresh runs
LAUNCHER=/mnt/sdc/dty_user/openvla_attack/tmp/phase3_refresh_launcher.sh
LOG_DIR=/mnt/sdc/dty_user/openvla_attack/tmp
WAIT_DIR=/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/metric_refresh_v2
TOTAL=108
BATCH_SIZE=12

launch_batch() {
  local START=$1
  local END=$((START + BATCH_SIZE - 1))
  if [ $END -ge $TOTAL ]; then END=$((TOTAL - 1)); fi
  echo "$(date +%H:%M:%S) Launching batch indices $START-$END"
  for IDX in $(seq $START $END); do
    local GPU=$(( IDX % 6 + 1 ))
    bash $LAUNCHER $GPU $IDX > $LOG_DIR/p3_idx${IDX}.log 2>&1 &
  done
}

# Skip first batch (0-11 already launched)
NEXT=12
echo "$(date +%H:%M:%S) Starting auto-handoff from index $NEXT"

while [ $NEXT -lt $TOTAL ]; do
  while true; do
    STILL=$(ps -eo pid,args --no-headers 2>/dev/null | grep -c 'run_v2_vis_sc5_mlp_bridge_telemetry' 2>/dev/null || echo 0)
    DONE=$(find $WAIT_DIR -name COMPLETE.json 2>/dev/null | wc -l)
    echo "$(date +%H:%M:%S) Done: $DONE/$TOTAL, Running: $STILL, Next starts at $NEXT"
    if [ $STILL -le 2 ]; then break; fi
    sleep 30
  done
  launch_batch $NEXT
  NEXT=$((NEXT + BATCH_SIZE))
  sleep 5
done

# Final wait
while true; do
  DONE=$(find $WAIT_DIR -name COMPLETE.json 2>/dev/null | wc -l)
  STILL=$(ps -eo pid,args --no-headers 2>/dev/null | grep -c 'run_v2_vis_sc5_mlp_bridge_telemetry' 2>/dev/null || echo 0)
  echo "$(date +%H:%M:%S) Final — Done: $DONE/$TOTAL, Running: $STILL"
  if [ $DONE -ge $TOTAL ]; then break; fi
  sleep 30
done
echo "$(date +%H:%M:%S) ALL $TOTAL METRIC REFRESH RUNS COMPLETE"
