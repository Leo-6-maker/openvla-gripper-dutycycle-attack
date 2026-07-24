#!/bin/bash
# Phase 4 (P30/P31): Adapted Action-Discrepancy PGD
# Uses untargeted_clean_token_ce — closest available match for UADA
# 3 canaries + 27 full runs = 30 total
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
OBJ="untargeted_clean_token_ce"

# Cell mapping: CELL TASK STATE ANCHOR
declare -A ANCHORS
ANCHORS["salad_dressing_s0"]="2 0 84"
ANCHORS["bbq_sauce_s0"]="3 0 128"
ANCHORS["ketchup_s0"]="4 0 95"
ANCHORS["milk_s4"]="7 4 92"
ANCHORS["butter_s2"]="6 2 100"
ANCHORS["alphabet_soup_s0"]="0 0 86"
ANCHORS["orange_juice_s0"]="9 0 167"
ANCHORS["butter_s0"]="6 0 85"
ANCHORS["tomato_sauce_s0"]="5 0 176"

CELLS=("salad_dressing_s0" "bbq_sauce_s0" "ketchup_s0" "milk_s4" "butter_s2" "alphabet_soup_s0" "orange_juice_s0" "butter_s0" "tomato_sauce_s0")
C_CANARY=("salad_dressing_s0" "butter_s0" "tomato_sauce_s0")

run_one() {
  local GPU=$1 CELL=$2 TASK=$3 STATE=$4 ANCHOR=$5 SEED=$6 TAG=$7
  export CUDA_VISIBLE_DEVICES=$GPU
  local BASE_DIR=/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object
  local OUT
  if [ "$TAG" = "canary" ]; then
    OUT=${BASE_DIR}/uada_canary/${CELL}_s${SEED}
  else
    OUT=${BASE_DIR}/uada_full/${CELL}_s${SEED}
  fi
  if [ -f "$OUT/COMPLETE.json" ]; then echo "SKIP $TAG/$CELL s$SEED"; return 0; fi
  echo "=== GPU$GPU: $TAG $CELL s$SEED $(date) ==="
  rm -rf "$OUT"; mkdir -p "$OUT"
  $PY -u $B --condition TRUE_T10 --state_id $STATE --anchor $ANCHOR --seed_id $SEED --task_idx $TASK \
    --attack_objective "$OBJ" \
    --output_dir "$OUT" --render_gpu $GPU --mlp_path $C \
    --libero_preprocess_backend upstream_tf_jpeg \
    --save_video --source_commit $COMMIT --video_fps 10 --frame_stride 2 \
    > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  echo "=== DONE GPU$GPU: $TAG $CELL s$SEED $(date) ==="
}

# Usage: bash phase4_uada_launcher.sh <GPU> <JOB_INDEX>
# 0-2:   canaries (salad, butter_s0, tomato s42)
# 3-29:  full matrix (9 cells x 3 seeds)

IDX=${2:-0}
GPU=$1

if [ $IDX -lt 3 ]; then
    TAG="canary"
    CELL=${C_CANARY[$IDX]}
    SEED=42
else
    TAG="uada_full"
    BASE_IDX=$((IDX - 3))
    CELL_IDX=$((BASE_IDX / 3))
    SEED_IDX=$((BASE_IDX % 3))
    CELL=${CELLS[$CELL_IDX]}
    SEEDS=(42 123 456)
    SEED=${SEEDS[$SEED_IDX]}
fi

read TASK STATE ANCHOR <<< "${ANCHORS[$CELL]}"
run_one $GPU "$CELL" $TASK $STATE $ANCHOR $SEED "$TAG"
