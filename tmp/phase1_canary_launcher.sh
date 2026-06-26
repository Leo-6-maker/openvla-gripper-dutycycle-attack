#!/bin/bash
# Phase 1 Telemetry V2 Canary Launcher
# Reuses battle-tested run_one() from phaseA_gpu*.sh
# Each GPU runs 1 canary on salad_dressing_s0 seed42
set -e
cd /mnt/sdc/dty_user/openvla_attack
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager
export OPENVLA_MODEL_PATH=/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object
export HOME=/mnt/sdc/dty_user/openvla_attack/sandbox_home TMPDIR=/mnt/sdc/dty_user/openvla_attack/tmp
export TF_FORCE_GPU_ALLOW_GROWTH=true
PY=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3
# Telemetry V2 bridge
B=/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_v2_vis_sc5_mlp_bridge_telemetry_v2.py
C=/mnt/sdc/dty_user/openvla_attack/outputs/sc5_v2_seed42/sc5_mlp_v2.pt
COMMIT=$(git rev-parse HEAD)
CANARY_BASE=/mnt/sdc/dty_user/openvla_attack/evidence/canary/telemetry_v2

# Canary cell: salad_dressing_s0, task=2, state=0, anchor=84, seed=42
CELL=salad_dressing_s0; TASK=2; STATE=0; ANCHOR=84; SEED=42

run_one() {
  local GPU=$1 COND=$2 OBJ=$3 LOCK=$4 EXTRA_FLAGS=$5 TAG=$6
  export CUDA_VISIBLE_DEVICES=$GPU
  local OUT=${CANARY_BASE}/${TAG}/${CELL}_s${SEED}
  if [ -f "$OUT/COMPLETE.json" ] || [ -f "$OUT/.done" ]; then echo "SKIP GPU$GPU $TAG — already done"; return 0; fi
  echo "=== GPU$GPU: $TAG $(date) ==="
  rm -rf "$OUT"; mkdir -p "$OUT"
  local LOCK_FLAG=""; [ "$LOCK" = "1" ] && LOCK_FLAG="--arm_lock"
  $PY -u $B --condition $COND --state_id $STATE --anchor $ANCHOR --seed_id $SEED --task_idx $TASK \
    --attack_objective "$OBJ" $LOCK_FLAG $EXTRA_FLAGS \
    --output_dir "$OUT" --render_gpu $GPU --mlp_path $C \
    --libero_preprocess_backend upstream_tf_jpeg \
    --save_video --source_commit $COMMIT --video_fps 10 --frame_stride 2 \
    > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  echo "=== GPU$GPU: $TAG DONE $(date) ==="
}

# Usage: bash phase1_canary_launcher.sh <GPU> <MODE>
# MODE: tma_nolock | tma_armlock | prefix_nolock | prefix_armlock | tma_early | tma_random | noemit

TMA="vanilla_tma_gripper_open_ce"
PREFIX="autoregressive_prefix_gripper_target_token_logratio_arm_v3"

case "${2:-help}" in
  tma_nolock)
    run_one $1 TRUE_T10 "$TMA" 0 "" tma_student_nolock ;;
  tma_armlock)
    run_one $1 TRUE_T10 "$TMA" 1 "" tma_student_armlock ;;
  prefix_nolock)
    run_one $1 TRUE_T10 "$PREFIX" 0 "" prefix_student_nolock ;;
  prefix_armlock)
    run_one $1 TRUE_T10 "$PREFIX" 1 "" prefix_student_armlock ;;
  tma_early)
    run_one $1 TRUE_T10 "$TMA" 0 "--trigger_step_override 64 --keep_running" tma_early_shift ;;
  tma_random)
    run_one $1 TRUE_T10 "$TMA" 0 "--trigger_step_override 78 --keep_running" tma_random_time ;;
  noemit)
    run_one $1 TRUE_T10 "$PREFIX" 0 "" formal_noemit
    # Note: cream_cheese (task=1, anchor=116) for no-emit pass
    ;;
  *)
    echo "Usage: $0 <GPU_ID> <MODE>"
    echo "MODES: tma_nolock tma_armlock prefix_nolock prefix_armlock tma_early tma_random noemit"
    exit 1 ;;
esac
