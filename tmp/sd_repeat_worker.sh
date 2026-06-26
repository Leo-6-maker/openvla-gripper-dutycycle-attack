#!/bin/bash
# Stage D repeat: 16 nolock repeat panel (8 states x 2 objectives x seed42)
set -e
cd /mnt/sdc/dty_user/openvla_attack
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager TF_FORCE_GPU_ALLOW_GROWTH=true
export HOME=/mnt/sdc/dty_user/openvla_attack/sandbox_home TMPDIR=/mnt/sdc/dty_user/openvla_attack/tmp
PY=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3
B=/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_v2_vis_sc5_mlp_bridge_telemetry_v2.py
C=/mnt/sdc/dty_user/openvla_attack/outputs/sc5_v2_seed42/sc5_mlp_v2.pt
COMMIT=$(git rev-parse HEAD)
BASE=/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/breadth_120
TMA="vanilla_tma_gripper_open_ce"
PREFIX="autoregressive_prefix_gripper_target_token_logratio_arm_v3"
STATES=("salad_dressing 2 1" "bbq_sauce 3 4" "ketchup 4 1" "milk 7 5" "butter 6 5" "orange_juice 9 2" "tomato_sauce 5 1" "butter 6 6")

run_one() {
  local G=$1 C=$2 T=$3 S=$4 O=$5 TG=$6 RID=$7
  local OUT=${BASE}/${TG}/${C}_s${S}_s42_r${RID}
  if [ -f "$OUT/COMPLETE.json" ]; then return 0; fi
  mkdir -p "$OUT"
  echo "GPU$G: $TG $C s$S r$RID $(date)"
  env CUDA_VISIBLE_DEVICES=$G $PY -u $B --condition TRUE_T10 --state_id $S \
    --anchor 0 --seed_id 42 --task_idx $T --attack_objective "$O" \
    --eval_seed 0 --output_dir "$OUT" --render_gpu $G --mlp_path $C \
    --libero_preprocess_backend upstream_tf_jpeg --save_video --source_commit $COMMIT \
    --video_fps 10 --frame_stride 2 > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  echo "GPU$G: $TG $C s$S r$RID DONE $(date)"
}

GPU=$1; IDX=$2
SI=$((IDX / 2)); OI=$((IDX % 2))
read C T S <<< "${STATES[$SI]}"
if [ $OI -eq 0 ]; then TG=tma_nolock; O=$TMA; else TG=prefix_nolock; O=$PREFIX; fi
RID=$((IDX + 1))
run_one $GPU "$C" $T $S "$O" "$TG" "$RID"
