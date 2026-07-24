#!/bin/bash
# Phase 6 Clean Qualification: find first clean-success state for 8 reference slots
# Parallel dispatch across all available GPUs
set -e
cd /mnt/sdc/dty_user/openvla_attack
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager
export OPENVLA_MODEL_PATH=/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object
export HOME=/mnt/sdc/dty_user/openvla_attack/sandbox_home TMPDIR=/mnt/sdc/dty_user/openvla_attack/tmp
export TF_FORCE_GPU_ALLOW_GROWTH=true
PY=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3
B=/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_v2_vis_sc5_mlp_bridge_telemetry_v2.py
C=/mnt/sdc/dty_user/openvla_attack/outputs/sc5_v2_seed42/sc5_mlp_v2.pt
COMMIT=$(git rev-parse HEAD)
BASE=/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/breadth_clean_qualify

# 8 reference slots: CELL TASK USED_STATES
# butter has 2 slots (A and B) — need 2 different unused states
declare -A SLOT_TASK
SLOT_TASK["salad"]="2"
SLOT_TASK["bbq"]="3"
SLOT_TASK["ketchup"]="4"
SLOT_TASK["milk"]="7"
SLOT_TASK["butterA"]="6"
SLOT_TASK["orange"]="9"
SLOT_TASK["butterB"]="6"
SLOT_TASK["tomato"]="5"

declare -A USED
USED["2"]="0"
USED["3"]="0"
USED["4"]="0"
USED["7"]="4"
USED["6"]="0,2"
USED["9"]="0"
USED["5"]="0"

run_clean() {
  local GPU=$1 SLOT=$2 TASK=$3 STATE=$4
  local OUT=${BASE}/${SLOT}_s${STATE}
  if [ -f "$OUT/COMPLETE.json" ]; then
    local S=$(python3 -c "import json; print(json.load(open('$OUT/episode_summary.json')).get('task_success',False))" 2>/dev/null)
    echo "EXISTING $SLOT s$STATE success=$S"
    return 0
  fi
  echo "=== GPU$GPU: $SLOT s$STATE $(date) ==="
  rm -rf "$OUT"; mkdir -p "$OUT"
  $PY -u $B --condition CLEAN --state_id $STATE --anchor 0 --seed_id 42 --task_idx $TASK \
    --attack_objective vanilla_tma_gripper_open_ce \
    --output_dir "$OUT" --render_gpu $GPU --mlp_path $C \
    --libero_preprocess_backend upstream_tf_jpeg \
    --save_video --source_commit $COMMIT --video_fps 10 --frame_stride 2 \
    > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  local S=$(python3 -c "import json; print(json.load(open('$OUT/episode_summary.json')).get('task_success',False))" 2>/dev/null)
  echo "=== DONE $SLOT s$STATE success=$S $(date) ==="
}

# Usage: bash phase6_clean_qualify.sh <GPU> <SLOT> <START_STATE>
# Will try states starting from START_STATE, return first success

GPU=$1; SLOT=$2; STATE=$3
TASK=${SLOT_TASK[$SLOT]}
USED_STR="${USED[$TASK]}"

while true; do
  # Skip used states
  SKIP=0
  for US in ${USED_STR//,/ }; do
    if [ "$STATE" = "$US" ]; then SKIP=1; break; fi
  done
  if [ $SKIP -eq 0 ]; then
    run_clean $GPU $SLOT $TASK $STATE
    SUCCESS=$(python3 -c "import json; print(json.load(open('$BASE/${SLOT}_s${STATE}/episode_summary.json')).get('task_success',False))" 2>/dev/null)
    if [ "$SUCCESS" = "True" ]; then
      echo "FROZEN: $SLOT s$STATE"
      break
    else
      echo "INELIGIBLE: $SLOT s$STATE"
    fi
  fi
  STATE=$((STATE + 1))
  if [ $STATE -gt 20 ]; then echo "FAILED: $SLOT no valid state found"; break; fi
done
