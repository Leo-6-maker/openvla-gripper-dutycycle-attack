#!/bin/bash
# Phase 8 Dispatcher — counter-based atomic dispatch (same as Object breadth pattern)
# Usage: bash phase8_dispatcher.sh <GPU> <SUITE>
# Launch N copies per GPU for parallel workers
set -e
GPU=$1; SUITE=$2
BASE=/mnt/sdc/dty_user/openvla_attack
COUNTER=$BASE/evidence/phase8_cross_suite_v1/queue/counter_${SUITE}_gpu${GPU}
mkdir -p $(dirname $COUNTER)

# Initialize counter
if [ ! -f "$COUNTER" ]; then echo 0 > "$COUNTER"; fi

while true; do
    # Atomic increment
    IDX=$(cat "$COUNTER")
    NEXT=$((IDX + 1))
    echo $NEXT > "$COUNTER"

    if [ $IDX -ge 210 ]; then
        echo "$(date) GPU$GPU $SUITE: all 210 done"
        break
    fi

    bash $BASE/tmp/phase8_simple_launch.sh $GPU $SUITE $IDX || true
done
