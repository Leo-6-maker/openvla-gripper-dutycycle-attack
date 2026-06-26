#!/bin/bash
# Stage A: 14 diagnostic repeat runs
# 8 discordant-no-lock × 2 repeats + 3 matched-no-lock × 1 + 3 ArmLock × 1 = 14
set -e
cd /mnt/sdc/dty_user/openvla_attack
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager TF_FORCE_GPU_ALLOW_GROWTH=true
export HOME=/mnt/sdc/dty_user/openvla_attack/sandbox_home TMPDIR=/mnt/sdc/dty_user/openvla_attack/tmp
PY=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3
B=/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_v2_vis_sc5_mlp_bridge_telemetry_v2.py
C=/mnt/sdc/dty_user/openvla_attack/outputs/sc5_v2_seed42/sc5_mlp_v2.pt
COMMIT=$(git rev-parse HEAD)
BASE=/mnt/sdc/dty_user/openvla_attack/evidence/repeatability_diagnostic_v3
TMA="vanilla_tma_gripper_open_ce"
PREFIX="autoregressive_prefix_gripper_target_token_logratio_arm_v3"

declare -A A
A["alphabet_soup_s0"]="0 0 86"
A["bbq_sauce_s0"]="3 0 128"
A["butter_s0"]="6 0 85"
A["butter_s2"]="6 2 100"
A["orange_juice_s0"]="9 0 167"
A["tomato_sauce_s0"]="5 0 176"
A["salad_dressing_s0"]="2 0 84"

# 14 jobs: CELL TASK STATE ANCHOR SEED OBJ LOCK REPEAT_ID ROLE
JOBS=(
  "alphabet_soup_s0 0 0 86 456 $TMA 1 1 discordant_armlock"
  "alphabet_soup_s0 0 0 86 456 $TMA 1 2 discordant_armlock"
  "bbq_sauce_s0 3 0 128 42 $PREFIX 0 1 discordant_nolock"
  "bbq_sauce_s0 3 0 128 42 $PREFIX 0 2 discordant_nolock"
  "butter_s0 6 0 85 123 $TMA 0 1 discordant_nolock"
  "butter_s0 6 0 85 123 $TMA 0 2 discordant_nolock"
  "butter_s2 6 2 100 42 $PREFIX 0 1 discordant_nolock"
  "butter_s2 6 2 100 42 $PREFIX 0 2 discordant_nolock"
  "bbq_sauce_s0 3 0 128 456 $TMA 0 1 matched_nolock_ctrl"
  "butter_s0 6 0 85 42 $TMA 0 1 matched_nolock_ctrl"
  "tomato_sauce_s0 5 0 176 42 $PREFIX 0 1 matched_nolock_ctrl"
  "butter_s2 6 2 100 456 $TMA 1 1 armlock_ctrl"
  "tomato_sauce_s0 5 0 176 456 $TMA 1 1 armlock_ctrl"
  "salad_dressing_s0 2 0 84 42 $TMA 1 1 armlock_ctrl"
)

GPU=$1; IDX=$2
read CELL TASK STATE ANCHOR SEED OBJ LOCK RID ROLE <<< "${JOBS[$IDX]}"
LOCK_FLAG=""; [ "$LOCK" = "1" ] && LOCK_FLAG="--arm_lock"
TAG=$( [ "$LOCK" = "1" ] && echo "armlock" || echo "nolock" )
OUT=${BASE}/${CELL}_s${SEED}_${TAG}_r${RID}_${ROLE}

if [ -f "$OUT/COMPLETE.json" ]; then echo "SKIP $CELL s$SEED $ROLE r$RID"; exit 0; fi
echo "=== GPU$GPU: $CELL s$SEED $ROLE r$RID $(date) ==="
rm -rf "$OUT"; mkdir -p "$OUT"
env CUDA_VISIBLE_DEVICES=$GPU $PY -u $B --condition TRUE_T10 --state_id $STATE \
  --anchor $ANCHOR --seed_id $SEED --task_idx $TASK --attack_objective "$OBJ" $LOCK_FLAG \
  --output_dir "$OUT" --render_gpu $GPU --mlp_path $C \
  --libero_preprocess_backend upstream_tf_jpeg --save_video --source_commit $COMMIT \
  --video_fps 10 --frame_stride 2 > "$OUT/stdout.log" 2> "$OUT/stderr.log"
echo "=== DONE GPU$GPU: $CELL s$SEED $ROLE r$RID $(date) ==="
